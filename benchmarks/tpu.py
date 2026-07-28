from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import lm7


def _runtime_available() -> bool:
    try:
        import torch_xla.runtime as xr
    except ImportError:
        return False
    return xr.device_type() == "TPU" and xr.addressable_device_count() > 0


def _mlp(batch_size: int, dtype: torch.dtype) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096),
        torch.nn.GELU(),
        torch.nn.Linear(4096, 1024),
    ).eval()
    return model.to(dtype=dtype), (torch.randn(batch_size, 1024, dtype=dtype),)


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LM7 inference on a Google TPU VM.")
    parser.add_argument("--backend", nargs="+", default=["eager", "openxla"])
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, help="Write machine-readable results as JSON.")
    arguments = parser.parse_args()

    if not _runtime_available():
        raise SystemExit('Expected a TPU PJRT runtime; install LM7 with ".[openxla]" on a TPU VM.')
    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    results = []
    for backend in arguments.backend:
        model, args = _mlp(arguments.batch_size, _dtype(arguments.dtype))
        result = lm7.benchmark(
            model,
            args=args,
            target="tpu",
            backend=backend,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        results.append(result.to_dict())
        print(
            f"{backend:>10}  first={result.first_call_ms:9.2f} ms  "
            f"median={result.latency_median_ms:8.3f} ms  "
            f"p95={result.latency_p95_ms:8.3f} ms  "
            f"throughput={result.samples_per_second:10.2f} samples/s"
        )

    report = {
        "schema_version": 1,
        "workload": {
            "model": "mlp",
            "dtype": arguments.dtype,
            "batch_size": arguments.batch_size,
            "target": "tpu",
        },
        "results": results,
    }
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON: {output}")


if __name__ == "__main__":
    main()
