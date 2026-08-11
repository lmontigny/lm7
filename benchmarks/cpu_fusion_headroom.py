"""Ask how much of a CPU workload a fusing compiler could possibly win.

[`benchmarks/local.py`](local.py) is this repo's CPU baseline, and on Arm it
reports eager and Inductor as a tie at every batch size tried -- see
[docs/cpu.md](../docs/cpu.md#latency-on-a-neoverse-n3). That reads like a
compiler failure, and is not one.

TorchInductor fuses pointwise work; it does not write GEMM kernels. Both the
eager and the compiled path hand `Linear` to the same BLAS, so the most fusion
can ever recover is whatever time the workload spends *outside* its matmuls.
This measures that ceiling directly, by timing each layer of the `local.py` MLP
on its own. When the GEMM share is 97%, a 1.0x Inductor result is the arithmetic
working out, not a backend falling back -- and no amount of compiler work moves
it.

    python benchmarks/cpu_fusion_headroom.py --batch-size 1 8 64 512

Run it on an otherwise idle machine, and read the share rather than the absolute
milliseconds: CPU benchmarks are far more sensitive to a busy host than GPU ones.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def _median_ms(call: Callable[[], Any], *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples)


def measure(batch_size: int, *, warmup: int, repeats: int) -> dict[str, Any]:
    # The same seed and the same shapes as benchmarks/local.py, so the parts sum
    # to something comparable with the total that harness reports.
    torch.manual_seed(0)
    first = torch.nn.Linear(1024, 4096).eval()
    activation = torch.nn.GELU()
    second = torch.nn.Linear(4096, 1024).eval()

    example = torch.randn(batch_size, 1024)
    with torch.no_grad():
        hidden = first(example)
        activated = activation(hidden)

        first_ms = _median_ms(lambda: first(example), warmup=warmup, repeats=repeats)
        activation_ms = _median_ms(lambda: activation(hidden), warmup=warmup, repeats=repeats)
        second_ms = _median_ms(lambda: second(activated), warmup=warmup, repeats=repeats)

    total_ms = first_ms + activation_ms + second_ms
    return {
        "batch_size": batch_size,
        "linear1_ms": first_ms,
        "gelu_ms": activation_ms,
        "linear2_ms": second_ms,
        "sum_ms": total_ms,
        # What is left for a fusing compiler once the GEMMs are excluded, which
        # is the ceiling on any Inductor speedup for this workload.
        "gemm_share": (first_ms + second_ms) / total_ms,
        "fusion_headroom": activation_ms / total_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure how much of the local.py MLP is GEMM, and so how much fusion can win."
    )
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 8, 64, 512])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--threads", type=int, help="Pin torch to this many threads.")
    parser.add_argument("--output", type=Path, help="Write machine-readable results as JSON.")
    arguments = parser.parse_args()

    if any(size < 1 for size in arguments.batch_size):
        parser.error("--batch-size values must be at least 1")
    if arguments.threads is not None:
        torch.set_num_threads(arguments.threads)

    results = []
    for batch_size in arguments.batch_size:
        result = measure(batch_size, warmup=arguments.warmup, repeats=arguments.repeats)
        results.append(result)
        print(
            f"batch={result['batch_size']:<5} "
            f"linear1={result['linear1_ms']:8.3f} ms  "
            f"gelu={result['gelu_ms']:7.3f} ms  "
            f"linear2={result['linear2_ms']:8.3f} ms  "
            f"sum={result['sum_ms']:8.3f} ms  "
            f"GEMM={result['gemm_share'] * 100:5.1f}%  "
            f"fusion headroom={result['fusion_headroom'] * 100:4.1f}%"
        )

    report = {
        "schema_version": 1,
        "host": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "threads": torch.get_num_threads(),
            "torch": torch.__version__,
        },
        "workload": {"model": "mlp", "dtype": "float32"},
        "results": results,
    }
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON: {output}")


if __name__ == "__main__":
    main()
