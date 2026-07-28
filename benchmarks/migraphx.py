"""Side-by-side evaluation of TorchInductor and Torch-MIGraphX on an AMD GPU.

This is the first implementation slice of the MIGraphX evaluation plan in
``docs/amd-migraphx.md``. It does not add an LM7 backend: it runs a model
through eager (the correctness reference), TorchInductor, and a manually
installed Torch-MIGraphX ``torch.compile`` path under one measurement harness,
so first-call cost, steady-state latency, throughput, peak memory, and accuracy
against eager are directly comparable.

Run it on a ROCm-enabled PyTorch build with Torch-MIGraphX installed:

    python benchmarks/migraphx.py --dtype float16 --batch-size 8 \
      --output artifacts/benchmarks/migraphx-mlp-fp16-b8.json

Paths whose runtime is missing are reported as unavailable and skipped rather
than failing the run, so an inductor-only baseline still works before
Torch-MIGraphX is installed.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

# torch.testing.assert_close defaults are tight for float32 but unrealistic for
# compiled low-precision inference; these atols match the looser policy used for
# validated float16/bfloat16 paths elsewhere in LM7.
_DEFAULT_ATOL = {"float32": 1e-4, "float16": 2e-2, "bfloat16": 5e-2}


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _amd_available() -> bool:
    return torch.cuda.is_available() and bool(getattr(torch.version, "hip", None))


def _migraphx_available() -> bool:
    try:
        import torch_migraphx  # noqa: F401  (registers the "migraphx" dynamo backend)
    except ImportError:
        return False
    return True


def _mlp(batch_size: int, dtype: torch.dtype) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096),
        torch.nn.GELU(),
        torch.nn.Linear(4096, 1024),
    ).eval()
    inputs = (torch.randn(batch_size, 1024, device="cuda", dtype=dtype),)
    return model.to(device="cuda", dtype=dtype), inputs


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _build(path: str, model: torch.nn.Module) -> Any:
    if path == "eager":
        return model
    return torch.compile(model, backend=path)


def _measure(fn: Any, args: tuple[torch.Tensor, ...], warmup: int, repeats: int) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        started = time.perf_counter()
        reference = fn(*args)
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - started) * 1000

        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()

        latencies_ms: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            fn(*args)
            torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - started) * 1000)

    median_ms = statistics.median(latencies_ms)
    batch_size = args[0].shape[0] if args and args[0].ndim else 1
    return {
        "output": reference,
        "first_call_ms": first_call_ms,
        "latency_median_ms": median_ms,
        "latency_p95_ms": _percentile(latencies_ms, 0.95),
        "samples_per_second": batch_size * 1000 / median_ms if median_ms else float("inf"),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare eager, TorchInductor, and Torch-MIGraphX on an AMD GPU."
    )
    parser.add_argument(
        "--path",
        nargs="+",
        default=["eager", "inductor", "migraphx"],
        help="Execution paths to evaluate; 'eager' is always the correctness reference.",
    )
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--atol",
        type=float,
        help="Max absolute difference from eager allowed (default depends on dtype).",
    )
    parser.add_argument("--output", type=Path, help="Write machine-readable results as JSON.")
    arguments = parser.parse_args()

    if not _amd_available():
        raise SystemExit(
            'Expected a ROCm-enabled PyTorch build with an AMD GPU; install with ".[dev]" '
            "on a ROCm host. See docs/amd-rocm.md and docs/amd-migraphx.md."
        )
    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    dtype = _dtype(arguments.dtype)
    atol = arguments.atol if arguments.atol is not None else _DEFAULT_ATOL[arguments.dtype]

    # Build the model and inputs once so every path runs identical weights and
    # inputs; only then is the accuracy comparison against eager meaningful.
    torch.manual_seed(0)
    base_model, inputs = _mlp(arguments.batch_size, dtype)
    with torch.inference_mode():
        reference_output = base_model(*inputs)
        torch.cuda.synchronize()

    paths = ["eager", *[p for p in arguments.path if p != "eager"]]
    results = []
    for path in paths:
        if path == "migraphx" and not _migraphx_available():
            print(f"{path:>10}  unavailable: install Torch-MIGraphX in this ROCm environment")
            results.append({"path": path, "available": False})
            continue

        model = copy.deepcopy(base_model)
        measured = _measure(_build(path, model), inputs, arguments.warmup, arguments.repeats)
        output = measured.pop("output")
        max_abs_diff = (reference_output - output.to(reference_output.dtype)).abs().max().item()
        measured.update(
            {
                "path": path,
                "available": True,
                "max_abs_diff_vs_eager": max_abs_diff,
                "within_tolerance": max_abs_diff <= atol,
            }
        )
        results.append(measured)

        peak = f"{measured['peak_memory_bytes'] / 1024**2:8.1f} MiB"
        accuracy = "reference" if path == "eager" else f"maxdiff={max_abs_diff:.3e}"
        print(
            f"{path:>10}  first={measured['first_call_ms']:9.2f} ms  "
            f"median={measured['latency_median_ms']:8.3f} ms  "
            f"p95={measured['latency_p95_ms']:8.3f} ms  "
            f"throughput={measured['samples_per_second']:10.2f} samples/s  "
            f"peak={peak}  {accuracy}"
        )
        del model
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "workload": {
            "model": "mlp",
            "dtype": arguments.dtype,
            "batch_size": arguments.batch_size,
            "target": "amd",
            "atol": atol,
        },
        "results": results,
    }
    if arguments.output is not None:
        out = arguments.output.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON: {out}")


if __name__ == "__main__":
    main()
