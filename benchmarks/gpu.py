from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import lm7

HF_MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "lfm25": "LiquidAI/LFM2.5-230M",
}


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LM7 inference on a local GPU.")
    parser.add_argument(
        "--model",
        choices=("mlp", *HF_MODELS),
        default="mlp",
        help="Workload to benchmark.",
    )
    parser.add_argument("--backend", nargs="+", default=["eager", "inductor"])
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="Optional torch.compile mode for the Inductor backend.",
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output", type=Path, help="Write machine-readable results as JSON.")
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this PyTorch environment.")
    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1")

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
            target="nvidia",
            backend=backend,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
            options=(
                {"compile_mode": arguments.compile_mode}
                if backend == "inductor" and arguments.compile_mode
                else None
            ),
        )
        results.append(result.to_dict())
        print(
            f"{backend:>10}  first={result.first_call_ms:9.2f} ms  "
            f"median={result.latency_median_ms:8.3f} ms  "
            f"p95={result.latency_p95_ms:8.3f} ms  "
            f"throughput={result.samples_per_second:10.2f} samples/s  "
            f"peak={result.peak_memory_bytes / 1024**2:8.1f} MiB"
        )
        del model
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "workload": {
            "model": arguments.model,
            "model_id": HF_MODELS.get(arguments.model),
            "dtype": arguments.dtype,
            "batch_size": arguments.batch_size,
            "prompt": arguments.prompt if arguments.model in HF_MODELS else None,
            "compile_mode": arguments.compile_mode,
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
