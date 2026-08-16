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
    "llama32-1b": "unsloth/Llama-3.2-1B-Instruct",
    "qwen35-0.8b": "Qwen/Qwen3.5-0.8B",
    "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
    # The dense validation ladder, reachable by name and not yet measured here.
    # See docs/limitations.md#model-coverage.
    "lfm25-350m": "LiquidAI/LFM2.5-350M",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    # The first dense 7B in these dicts; ~14.5 GB at BF16, so it needs a card
    # larger than the 12 GiB sm89 dev GPU.
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
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


# Arms whose inputs are placed on the device once, up front, rather than on
# every call. They exist to answer "what does the orchestration layer cost over
# the toolchain underneath it", and they are timed by `benchmark_callable` --
# the same loop, warmup, synchronization and statistics the ordinary arms get --
# so that what differs between them is what happens per call and nothing else.
#
#   torch-eager / torch-compile   PyTorch called directly, no LM7 in the path
#   eager-placed / inductor-placed  LM7 with transfers="explicit"
#
# The pair matters more than either arm alone. LM7's ordinary arms use
# `transfers="automatic"` and so copy their inputs to the device on every call,
# which is a real cost but a *chosen* one -- the placed arms hold everything
# else equal and isolate what dispatch itself costs. Comparing `inductor`
# against `torch-compile` without `inductor-placed` in between charges LM7 for
# a host-to-device copy the vendor arm never makes.
VENDOR_ARMS = ("torch-eager", "torch-compile")
PLACED_ARMS = ("eager-placed", "inductor-placed")
DIRECT_ARMS = VENDOR_ARMS + PLACED_ARMS


def _direct_arm(
    arm: str,
    model: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    target: str,
    warmup: int,
    repeats: int,
    compile_mode: str | None,
    record_latencies: bool = False,
) -> Any:
    """Time an arm whose model and inputs are already on the device."""
    import lm7
    from lm7.benchmarking import benchmark_callable
    from lm7.detection import inference_context, resolve_target, torch_device
    from lm7.module import _map_tensors

    resolved = resolve_target(target)
    device = torch_device(resolved)
    # Moved here for every arm in this function, including the LM7 ones:
    # `transfers="explicit"` is a promise the caller places its own tensors, and
    # LM7's backends only move a model when transfers are automatic.
    model = model.to(device)
    args = _map_tensors(args, lambda tensor: tensor.to(device))
    kwargs = _map_tensors(kwargs, lambda tensor: tensor.to(device))

    if arm in PLACED_ARMS:
        compiled = lm7.compile(
            model,
            target=target,
            backend=arm.removesuffix("-placed"),
            transfers="explicit",
            fallback="error",
            cache=False,
        )

        # No inference context here: LM7 enters its own, and wrapping it twice
        # would time a context this arm does not actually pay for.
        def run() -> Any:
            return compiled(*args, **kwargs)

    else:
        # `mode=None` unless --compile-mode says otherwise, which is exactly what
        # LM7's Inductor backend passes, so the two compile the same way and the
        # difference between them is dispatch rather than codegen.
        call = model if arm == "torch-eager" else torch.compile(model, mode=compile_mode)

        def run() -> Any:
            with inference_context(resolved):
                return call(*args, **kwargs)

    return benchmark_callable(
        run,
        target=resolved,
        backend=arm,
        warmup=warmup,
        repeats=repeats,
        batch_size=_batch_size(args, kwargs),
        record_latencies=record_latencies,
    )


def _batch_size(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            return int(value.shape[0])
    return 1


def _target_available(target: str) -> bool:
    if target == "apple":
        return torch.backends.mps.is_available()
    if target == "auto":
        return torch.cuda.is_available() or torch.backends.mps.is_available()
    return torch.cuda.is_available()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LM7 inference on a local GPU.")
    parser.add_argument(
        "--model",
        choices=("mlp", *HF_MODELS),
        default="mlp",
        help="Workload to benchmark.",
    )
    parser.add_argument(
        "--backend",
        nargs="+",
        default=["eager", "inductor"],
        help=(
            "LM7 backend names, plus the arms that place their inputs up front "
            f"instead of per call: {', '.join(VENDOR_ARMS)} bypass LM7 entirely, "
            f"and {', '.join(PLACED_ARMS)} are LM7 with transfers='explicit'. "
            "Run a vendor arm and its placed counterpart together to separate "
            "dispatch cost from input-transfer cost"
        ),
    )
    parser.add_argument(
        "--target",
        choices=("auto", "nvidia", "amd", "apple"),
        default="auto",
        help="GPU vendor to benchmark; auto uses the locally detected GPU.",
    )
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        help="Optional torch.compile mode for the Inductor backend.",
    )
    parser.add_argument(
        "--record-latencies",
        action="store_true",
        help=(
            "keep every per-call measurement in the JSON, not only median and p95. "
            "Off by default because a 300-repeat arm would otherwise put 300 floats "
            "in a report that gets quoted; docs/figures/overhead.py needs them"
        ),
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output", type=Path, help="Write machine-readable results as JSON.")
    arguments = parser.parse_args()

    if not _target_available(arguments.target):
        raise SystemExit(f"Target {arguments.target!r} is unavailable in this PyTorch environment.")
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
        if backend in DIRECT_ARMS:
            result = _direct_arm(
                backend,
                model,
                args,
                kwargs,
                target=arguments.target,
                warmup=arguments.warmup,
                repeats=arguments.repeats,
                compile_mode=arguments.compile_mode,
                record_latencies=arguments.record_latencies,
            )
        else:
            result = lm7.benchmark(
                model,
                args=args,
                kwargs=kwargs,
                target=arguments.target,
                backend=backend,
                warmup=arguments.warmup,
                repeats=arguments.repeats,
                record_latencies=arguments.record_latencies,
                options=(
                    {"compile_mode": arguments.compile_mode}
                    if backend == "inductor" and arguments.compile_mode
                    else None
                ),
            )
        results.append(result.to_dict())
        peak = (
            f"{result.peak_memory_bytes / 1024**2:8.1f} MiB"
            if result.peak_memory_bytes is not None
            else "n/a"
        )
        print(
            f"{backend:>10}  first={result.first_call_ms:9.2f} ms  "
            f"median={result.latency_median_ms:8.3f} ms  "
            f"p95={result.latency_p95_ms:8.3f} ms  "
            f"throughput={result.samples_per_second:10.2f} samples/s  "
            f"peak={peak}"
        )
        resolved_vendor = result.target.split(":", 1)[0]
        del model
        if resolved_vendor == "apple":
            torch.mps.empty_cache()
        elif resolved_vendor in {"nvidia", "amd"}:
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
            "target": arguments.target,
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
