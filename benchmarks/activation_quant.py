"""Compare weight-only against dynamic-activation quantization on plain linears.

A single `nn.Linear` isolates the quantization path from everything a real model
adds -- attention, normalization, layer selection, graph breaks -- which matters
here because the question is narrow: does quantizing *activations* as well as
weights make the matmul faster, and at which shapes.

The distinction the numbers turn on: a weight-only mode stores narrow weights and
unpacks them to BF16 inside the kernel, so it can only ever save bytes moved. A
dynamic mode quantizes the activations too, so the multiply itself happens in the
narrow format. Only the second can cut arithmetic, and only the second has ever
beaten the BF16 baseline here.

    python benchmarks/activation_quant.py --output artifacts/activation_quant.json

TorchAO's fused Triton activation-scaling kernel for NVFP4 requires MSLK
(github.com/pytorch/MSLK), which is not installable from PyPI -- that name is an
empty placeholder. Without it the config falls back to a torch implementation,
which this script reports rather than hides.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from lm7.huggingface import (
    FP8,
    FP8_DYNAMIC,
    INT8,
    NO_QUANTIZATION,
    NVFP4,
    NVFP4_DYNAMIC,
    _apply_quantization,
    nvfp4_dynamic_kernel,
)
from lm7.targets import parse_target

# TorchAO's fused Triton path wants M % 128 == 0 and K % 64 == 0; these shapes
# satisfy both so a fallback is a missing package rather than a bad shape.
SHAPES = ((128, 4096, 4096), (256, 4096, 4096), (1024, 4096, 4096), (128, 8192, 8192))

MODES = (NO_QUANTIZATION, INT8, FP8, NVFP4, FP8_DYNAMIC, NVFP4_DYNAMIC)


class _MLP(torch.nn.Module):
    def __init__(self, K: int, N: int) -> None:
        super().__init__()
        self.down_proj = torch.nn.Linear(K, N, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(x)


class _Block(torch.nn.Module):
    """One linear, reached through a transformer-shaped module path.

    The path matters and a bare `nn.Linear` will not do. LM7's FP8 filter selects
    linears whose fully-qualified name contains `.mlp.`, so a top-level linear
    matches nothing and the mode raises "matched no quantizable layers" -- which
    looks like an unsupported format and is really a naming mismatch. Nesting the
    layer here makes the benchmark exercise the same filters a real model does.
    """

    def __init__(self, K: int, N: int) -> None:
        super().__init__()
        self.mlp = _MLP(K, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _build(K: int, N: int) -> torch.nn.Module:
    model = torch.nn.Sequential()
    model.add_module("layer", _Block(K, N))
    return model.cuda().to(torch.bfloat16).eval()


def _measure(
    model: torch.nn.Module, x: torch.Tensor, *, warmup: int, repeats: int
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        output = model(x)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - started) * 1000.0
    captured = output.detach().float().cpu()

    for _ in range(warmup):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "first_call_ms": first_call_ms,
        "latency_median_ms": statistics.median(samples),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "output": captured,
    }


def run_shape(
    M: int, K: int, N: int, modes: tuple[str, ...], *, warmup: int, repeats: int
) -> list[dict[str, Any]]:
    target = parse_target("nvidia")
    results = []
    reference: torch.Tensor | None = None
    baseline_latency: float | None = None

    for mode in modes:
        torch.manual_seed(0)
        linear = _build(K, N)
        record: dict[str, Any] = {"M": M, "K": K, "N": N, "quantization": mode}
        converted = 0
        if mode != NO_QUANTIZATION:
            try:
                _, converted = _apply_quantization(linear, target, mode)
            except Exception as error:  # noqa: BLE001 - a rejected mode is a result
                record.update(
                    {"works": False, "error_type": type(error).__name__, "error": str(error)[:300]}
                )
                results.append(record)
                continue
        record["converted_modules"] = converted

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        torch._dynamo.reset()
        compiled = torch.compile(linear, fullgraph=True)
        try:
            measured = _measure(compiled, x, warmup=warmup, repeats=repeats)
        except Exception as error:  # noqa: BLE001 - so is a kernel that will not run
            record.update(
                {"works": False, "error_type": type(error).__name__, "error": str(error)[:300]}
            )
            results.append(record)
            continue

        output = measured.pop("output")
        if mode == NO_QUANTIZATION:
            reference = output
            baseline_latency = measured["latency_median_ms"]
        record.update(measured)
        record["works"] = True
        if reference is not None:
            difference = (output - reference).abs().max().item()
            scale = reference.abs().max().item()
            record["max_abs_diff"] = difference
            record["relative_error"] = difference / scale if scale else None
        if baseline_latency:
            record["latency_ratio"] = measured["latency_median_ms"] / baseline_latency
        results.append(record)

        ratio = record.get("latency_ratio")
        print(
            f"  {mode:<16} median={measured['latency_median_ms']:8.4f} ms"
            + (f"  {ratio:5.2f}x baseline" if ratio else "  baseline      ")
            + f"  vram={measured['peak_memory_bytes'] / 1e6:7.1f} MB"
            + (
                f"  rel_err={record['relative_error']:6.2%}"
                if record.get("relative_error") is not None
                else ""
            )
        )
        del linear, compiled
        torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weight-only versus dynamic-activation quantization on plain linears."
    )
    parser.add_argument("--mode", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA GPU.")

    import torchao

    results = []
    for M, K, N in SHAPES:
        print(f"M={M} K={K} N={N}")
        results.extend(
            run_shape(
                M,
                K,
                N,
                tuple(arguments.mode),
                warmup=arguments.warmup,
                repeats=arguments.repeats,
            )
        )

    report = {
        "schema_version": 1,
        "environment": {
            "torch": torch.__version__,
            "torchao": torchao.__version__,
            "device": torch.cuda.get_device_name(0),
            "capability": "sm{}{}".format(*torch.cuda.get_device_capability(0)),
            "nvfp4_dynamic_kernel": nvfp4_dynamic_kernel(),
            "host": platform.node(),
        },
        "shapes": [list(shape) for shape in SHAPES],
        "results": results,
    }
    print(f"nvfp4 activation kernel: {report['environment']['nvfp4_dynamic_kernel']}")
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON: {output}")


if __name__ == "__main__":
    main()
