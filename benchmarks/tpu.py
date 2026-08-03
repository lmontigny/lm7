from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import lm7

HF_MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "llama32-1b": "unsloth/Llama-3.2-1B-Instruct",
    "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
}


def _runtime_available() -> bool:
    try:
        import torch_xla.runtime as xr
    except ImportError:
        return False
    return xr.device_type() == "TPU" and xr.addressable_device_count() > 0


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _workload(
    name: str,
    *,
    batch_size: int,
    dtype: torch.dtype,
    prompt: str,
) -> tuple[torch.nn.Module, tuple[Any, ...], dict[str, Any]]:
    if name == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 1024),
        ).eval()
        return model.to(dtype=dtype), (torch.randn(batch_size, 1024, dtype=dtype),), {}
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit('Install Hugging Face support with: pip install -e ".[hf]"') from None
    model_id = HF_MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
    inputs = tokenizer([prompt] * batch_size, return_tensors="pt")
    return model, (), {**inputs, "use_cache": False}


def _apply_mat_mul_precision(precision: str) -> None:
    """Set XLA's fp32 matmul precision for the whole run.

    Deliberately not routed through ``options={"mat_mul_precision": ...}``. The
    setting is process-global and XLA reads it while lowering the first
    computation, so in a run that benchmarks several backends only the first one
    could ever set it -- and that is usually ``eager``, which takes no options.
    Applying it once up front is the only way it can describe every row of the
    report. See docs/google-tpu.md.
    """
    import torch_xla.backends

    torch_xla.backends.set_mat_mul_precision(precision)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LM7 inference on a Google TPU VM.")
    parser.add_argument(
        "--model",
        choices=("mlp", *HF_MODELS),
        default="mlp",
        help="Workload to benchmark.",
    )
    parser.add_argument("--backend", nargs="+", default=["eager", "openxla"])
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--mat-mul-precision",
        choices=("default", "high", "highest"),
        help=(
            "XLA fp32 matmul precision, applied once for every backend in this run. "
            "Only meaningful with --dtype float32."
        ),
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output", type=Path, help="Write machine-readable results as JSON.")
    arguments = parser.parse_args()

    if not _runtime_available():
        raise SystemExit('Expected a TPU PJRT runtime; install LM7 with ".[openxla]" on a TPU VM.')
    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if arguments.mat_mul_precision is not None:
        _apply_mat_mul_precision(arguments.mat_mul_precision)

    results = []
    for backend in arguments.backend:
        model, args, kwargs = _workload(
            arguments.model,
            batch_size=arguments.batch_size,
            dtype=_dtype(arguments.dtype),
            prompt=arguments.prompt,
        )
        result = lm7.benchmark(
            model,
            args=args,
            kwargs=kwargs,
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
        del model

    report = {
        "schema_version": 1,
        "workload": {
            "model": arguments.model,
            "model_id": HF_MODELS.get(arguments.model),
            "dtype": arguments.dtype,
            "batch_size": arguments.batch_size,
            "prompt": arguments.prompt if arguments.model in HF_MODELS else None,
            "mat_mul_precision": arguments.mat_mul_precision,
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
