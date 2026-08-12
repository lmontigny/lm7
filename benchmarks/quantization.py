"""Sweep weight-only quantization modes on one accelerator, measuring both halves
of the trade: what the footprint buys, and what the latency costs.

The numbers in ``docs/quantization.md`` were originally collected by hand, one
``lm7 model run --quantize ...`` invocation at a time, which made them awkward to
reproduce on a second machine and impossible to reproduce prompt-for-prompt. This
script is the repeatable version: it runs every mode against the same baseline in
one process and writes a JSON report.

Accuracy and latency are deliberately measured through different backends:

- **Accuracy runs eager.** Weight-only quantization changes the *weights*, so the
  logit error it introduces is a property of the dequantized weight and not of the
  kernel that consumes it. Eager isolates that, and avoids paying a per-prompt
  compile for a number that does not depend on the compiler.
- **Latency runs the compiled backend.** That is the only configuration anyone
  should deploy — eager NVFP4 pays an unfused unpack on every matmul and is
  pathologically slow.

Quantization goes through ``lm7.huggingface`` rather than calling TorchAO here, so
the benchmark cannot drift away from what ``lm7 model run`` actually does.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

import lm7
from lm7.detection import resolve_target, synchronize
from lm7.huggingface import (
    FP8_DYNAMIC,
    FP8_DYNAMIC_ROWWISE,
    NO_QUANTIZATION,
    NVFP4_DYNAMIC,
    _apply_quantization,
    _model_storage_bytes,
    _peak_memory,
    _reset_peak_memory,
    normalize_quantization,
)

HF_MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "lfm25": "LiquidAI/LFM2.5-230M",
    "llama32-1b": "unsloth/Llama-3.2-1B-Instruct",
    "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "llama31-8b": "unsloth/Llama-3.1-8B-Instruct",
    # The dense validation ladder, reachable by name and *not yet measured
    # through this harness* -- see docs/limitations.md#model-coverage. Each
    # answers something the entries above do not:
    #
    #   a second size of an architecture already here, so scale varies alone
    "lfm25-350m": "LiquidAI/LFM2.5-350M",
    #   Qwen3 rather than the Qwen3.5-0.8B elsewhere in these dicts; 2.03B
    #   parameters including embeddings, despite the name
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    #   the first *dense* 7B -- Mixtral-8x7B is sparse and lives in moe.py.
    #   7.25B parameters, ~14.5 GB at BF16 and ~29 GB at FP32, so it does not
    #   fit either CPU host in docs/tested-hardware.md at this repo's FP32 CPU
    #   compute dtype
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
}

# The four prompts behind the sm89 validation table were never recorded, so these
# are a stated set rather than a reconstruction of it. Method matches -- top-1
# agreement and maximum last-token logit difference against the unquantized
# baseline -- but a prompt-for-prompt comparison against that table is not
# available for any model.
PROMPTS = (
    "The capital of France is",
    "def add(a, b):",
    "The three laws of robotics state that",
    "In 1969, humans first landed on",
)

# The dynamic modes quantize activations too, so the matmul runs in the narrow
# format instead of dequantizing to BF16 first -- the only family here that can
# cut arithmetic rather than only bytes moved.
MODES = ("none", "int8", "fp8", "nvfp4", FP8_DYNAMIC, FP8_DYNAMIC_ROWWISE, NVFP4_DYNAMIC)


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _load(model_id: str, dtype: torch.dtype) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit('Install Hugging Face support with: pip install -e ".[hf]"') from None
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
    return model, tokenizer


def _last_token_logits(callable_model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        output = callable_model(**inputs, use_cache=False)
    logits = output.logits if hasattr(output, "logits") else output[0]
    return logits[0, -1, :].float().cpu()


def _accuracy(
    model: torch.nn.Module,
    tokenizer: Any,
    target: Any,
    baseline: list[torch.Tensor] | None,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Per-prompt last-token logits, plus agreement against the baseline when given."""
    eager = lm7.compile(model, target=target, backend="eager", transfers="automatic", cache=False)
    logits = [
        _last_token_logits(eager, dict(tokenizer(prompt, return_tensors="pt")))
        for prompt in PROMPTS
    ]
    if baseline is None:
        return logits, {
            "top1_agreement": None,
            "max_logit_difference": None,
            "next_tokens": [tokenizer.decode(row.argmax().item()) for row in logits],
        }
    agreed = sum(
        int(row.argmax().item() == reference.argmax().item())
        for row, reference in zip(logits, baseline, strict=True)
    )
    difference = max(
        (row - reference).abs().max().item()
        for row, reference in zip(logits, baseline, strict=True)
    )
    return logits, {
        "top1_agreement": f"{agreed}/{len(PROMPTS)}",
        "max_logit_difference": difference,
        "next_tokens": [tokenizer.decode(row.argmax().item()) for row in logits],
    }


def _latency(
    model: torch.nn.Module,
    tokenizer: Any,
    target: Any,
    backend: str,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """First-call and steady-state milliseconds on the first prompt, compiled."""
    inputs = dict(tokenizer(PROMPTS[0], return_tensors="pt"))
    compiled = lm7.compile(
        model, target=target, backend=backend, transfers="automatic", fallback="error", cache=False
    )
    synchronize(target)
    started = time.perf_counter()
    with torch.no_grad():
        compiled(**inputs, use_cache=False)
    synchronize(target)
    first_call_ms = (time.perf_counter() - started) * 1000.0

    for _ in range(warmup):
        with torch.no_grad():
            compiled(**inputs, use_cache=False)
    synchronize(target)

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        with torch.no_grad():
            compiled(**inputs, use_cache=False)
        synchronize(target)
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "first_call_ms": first_call_ms,
        "latency_median_ms": statistics.median(samples),
        "latency_min_ms": min(samples),
        "input_tokens": int(inputs["input_ids"].shape[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep weight-only quantization modes on one accelerator."
    )
    parser.add_argument("--model", choices=sorted(HF_MODELS), default="llama32-1b")
    parser.add_argument("--mode", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--target", default="auto")
    parser.add_argument("--backend", default="inductor", help="backend for the latency measurement")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--output", type=Path, help="Write machine-readable results as JSON.")
    arguments = parser.parse_args()

    model_id = HF_MODELS[arguments.model]
    target = resolve_target(arguments.target)
    dtype = _dtype(arguments.dtype)

    if "none" not in arguments.mode and not arguments.skip_accuracy:
        raise SystemExit(
            "Accuracy is measured against the unquantized baseline, so 'none' must be "
            "one of --mode. Pass --skip-accuracy to measure latency alone."
        )

    results = []
    baseline_logits: list[torch.Tensor] | None = None
    baseline_storage: int | None = None
    baseline_latency: float | None = None

    for mode in arguments.mode:
        mode = normalize_quantization(mode)
        model, tokenizer = _load(model_id, dtype)
        _reset_peak_memory(target)
        try:
            quantization_ms, quantized_modules = (
                (0.0, 0) if mode == NO_QUANTIZATION else _apply_quantization(model, target, mode)
            )
        except Exception as error:
            # A mode this hardware cannot run is a result, not a crash. torchao
            # asserts on NVFP4 activation quantization below sm100, so the default
            # `--mode` list ends the process on every pre-Blackwell card -- after
            # the other five modes have been measured and before anything is
            # written, because the JSON is serialized once at the end. Record the
            # refusal and keep the run.
            #
            # `none` is the exception: the accuracy and ratio baselines come from
            # it, so continuing past a failure there would silently report every
            # later row against nothing.
            if mode == NO_QUANTIZATION:
                raise
            results.append(
                {
                    "quantization": mode,
                    "unsupported": f"{type(error).__name__}: {error}",
                }
            )
            print(f"{mode:>7}  unsupported on {target.architecture}: {type(error).__name__}")
            del model
            continue
        storage_bytes = _model_storage_bytes(model)

        accuracy: dict[str, Any] = {}
        if not arguments.skip_accuracy:
            logits, accuracy = _accuracy(model, tokenizer, target, baseline_logits)
            if mode == NO_QUANTIZATION:
                baseline_logits = logits

        timing = _latency(
            model,
            tokenizer,
            target,
            arguments.backend,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        if mode == NO_QUANTIZATION:
            baseline_storage = storage_bytes
            baseline_latency = timing["latency_median_ms"]

        result = {
            "quantization": mode,
            "quantized_modules": quantized_modules,
            "quantization_ms": quantization_ms,
            "model_storage_bytes": storage_bytes,
            "storage_ratio": (baseline_storage / storage_bytes) if baseline_storage else None,
            "latency_ratio": (
                (timing["latency_median_ms"] / baseline_latency) if baseline_latency else None
            ),
            "peak_memory_bytes": _peak_memory(target),
            **timing,
            **accuracy,
        }
        results.append(result)
        ratio = result["latency_ratio"]
        storage_ratio = result["storage_ratio"]
        print(
            f"{mode:>7}  median={timing['latency_median_ms']:8.3f} ms"
            f"  ({'baseline' if ratio is None or ratio == 1.0 else f'{ratio:.2f}x'})"
            f"  first={timing['first_call_ms'] / 1000.0:7.2f} s"
            f"  storage={storage_bytes / 1e9:6.3f} GB"
            f"  ({'baseline' if storage_ratio == 1.0 else f'{storage_ratio:.2f}x smaller'})"
            f"  top1={accuracy.get('top1_agreement') or '-'}"
            f"  max_logit_diff={accuracy.get('max_logit_difference') or 0.0:6.2f}"
        )
        del model

    report = {
        "schema_version": 1,
        "workload": {
            "model": arguments.model,
            "model_id": model_id,
            "dtype": arguments.dtype,
            "target": str(target),
            "latency_backend": arguments.backend,
            "accuracy_backend": None if arguments.skip_accuracy else "eager",
            "prompts": list(PROMPTS),
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
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
