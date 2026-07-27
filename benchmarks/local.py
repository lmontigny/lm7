from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import lm7


def _model_and_input(batch_size: int) -> tuple[torch.nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096),
        torch.nn.GELU(),
        torch.nn.Linear(4096, 1024),
    ).eval()
    return model, torch.randn(batch_size, 1024)


def _target_available(target: str) -> bool:
    if target == "cpu":
        return True
    if target == "apple":
        return torch.backends.mps.is_available()
    return torch.cuda.is_available() and not getattr(torch.version, "hip", None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the same LM7 workload on local CPU, NVIDIA, and Apple GPUs."
    )
    parser.add_argument(
        "--target",
        nargs="+",
        choices=("cpu", "nvidia", "apple"),
        default=["cpu", "nvidia"],
    )
    parser.add_argument(
        "--backend",
        nargs="+",
        choices=("eager", "inductor"),
        default=["eager", "inductor"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, help="Write machine-readable results as JSON.")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any requested target is unavailable instead of skipping it.",
    )
    arguments = parser.parse_args()

    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    results = []
    for target in arguments.target:
        if not _target_available(target):
            if arguments.require_all:
                raise SystemExit(f"Requested target {target!r} is unavailable.")
            print(f"{target:>7}: skipped (unavailable)")
            continue
        for backend in arguments.backend:
            model, example_input = _model_and_input(arguments.batch_size)
            result = lm7.benchmark(
                model,
                args=(example_input,),
                target=target,
                backend=backend,
                warmup=arguments.warmup,
                repeats=arguments.repeats,
            )
            results.append(result.to_dict())
            peak = (
                f"{result.peak_memory_bytes / 1024**2:.1f} MiB"
                if result.peak_memory_bytes is not None
                else "n/a"
            )
            print(
                f"{target:>7}/{backend:<8} first={result.first_call_ms:9.2f} ms  "
                f"median={result.latency_median_ms:8.3f} ms  "
                f"p95={result.latency_p95_ms:8.3f} ms  "
                f"throughput={result.samples_per_second:10.2f} samples/s  "
                f"peak={peak}"
            )
            del model
            if target == "nvidia":
                torch.cuda.empty_cache()
            elif target == "apple":
                torch.mps.empty_cache()

    report = {
        "schema_version": 1,
        "workload": {
            "model": "mlp",
            "dtype": "float32",
            "batch_size": arguments.batch_size,
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
