"""Separate what `torch.compile` wins from how the timing loop measures it.

Three arms, none of which put LM7 in the timed call:

    eager             PyTorch as written
    inductor          torch.compile(mode="default")          fused kernels
    reduce-overhead   torch.compile(mode="reduce-overhead")  fused kernels + CUDA Graphs

The eager -> inductor step is what better kernels buy. The inductor ->
reduce-overhead step is what removing per-launch CPU work buys, because CUDA
Graphs change nothing else.

Each arm is timed under both synchronization policies this repo uses, because
they do not agree:

    per-call   synchronize before and after every call -- `lm7.benchmark_callable`,
               which is what benchmarks/gpu.py reports
    batched    synchronize once around a loop of `--repeats` calls

A per-call barrier is the honest measure of one isolated call's latency, and it
is also precisely the pattern that denies CUDA Graphs their advantage: what they
remove is CPU launch overhead, and a loop that blocks on the GPU after every
iteration never lets that overhead overlap with anything. Reporting both is the
point of this script -- on a launch-bound model the two policies disagree by
enough to change the conclusion.

Warmup is time-based rather than a fixed iteration count. A fixed five
iterations leaves an idle GPU at its low-power clock (300 MHz on the sm89 dev
card, against a 3105 MHz boost), so an eager arm measured first is charged for
a clock ramp that a compiled arm measured after a long compile never pays.

    python benchmarks/compile_modes.py --model smollm2 \
      --output artifacts/compile-modes-smollm2.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from lm7.benchmarking import benchmark_callable
from lm7.detection import inference_context, resolve_target

# Copied from benchmarks/gpu.py so the two scripts name the same checkpoints.
HF_MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "lfm25": "LiquidAI/LFM2.5-230M",
    "llama32-1b": "unsloth/Llama-3.2-1B-Instruct",
    "qwen35-0.8b": "Qwen/Qwen3.5-0.8B",
    "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
}

ARMS = ("eager", "inductor", "reduce-overhead")
PROMPT = "The capital of France is"


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


class _CausalLMForward(torch.nn.Module):
    """Call an HF causal LM with a single positional tensor, returning logits.

    `use_cache=False` keeps a KV cache out of a forward-pass measurement, which
    is how benchmarks/gpu.py builds its inputs too. This is a full forward over
    a short prompt, not autoregressive decode; decode is benchmarks/decode.py.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, use_cache=False).logits


def _workload(
    name: str, *, dtype: torch.dtype, batch_size: int
) -> tuple[torch.nn.Module, torch.Tensor]:
    model: torch.nn.Module
    if name == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 1024),
        ).to(dtype=dtype)
        return model, torch.randn(batch_size, 1024, dtype=dtype)
    if name == "resnet18":
        try:
            from torchvision.models import resnet18
        except ImportError:
            raise SystemExit("Install torchvision for the resnet18 workload.") from None
        return resnet18(weights=None).to(dtype=dtype), torch.randn(
            batch_size, 3, 224, 224, dtype=dtype
        )
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit('Install Hugging Face support with: pip install -e ".[hf]"') from None
    model_id = HF_MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    input_ids = tokenizer([PROMPT] * batch_size, return_tensors="pt").input_ids
    return _CausalLMForward(model), input_ids


def _warm(call: Callable[[], Any], seconds: float) -> int:
    """Run until the GPU clocks have settled, and report how many calls that took."""
    calls = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        call()
        calls += 1
    torch.cuda.synchronize()
    return calls


def _batched_ms(call: Callable[[], Any], repeats: int) -> float:
    """Mean latency with a single synchronization around the whole loop."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000 / repeats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare torch.compile modes under both synchronization policies."
    )
    parser.add_argument("--model", default="smollm2", choices=("mlp", "resnet18", *HF_MODELS))
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--warm-seconds",
        type=float,
        default=10.0,
        help="Time-based warmup per arm, so an idle GPU's clock ramp is not measured.",
    )
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path, help="Write machine-readable results as JSON.")
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA GPU.")

    dtype = _dtype(arguments.dtype)
    model, example = _workload(arguments.model, dtype=dtype, batch_size=arguments.batch_size)
    model = model.to("cuda").eval()
    example = example.to("cuda")
    target = resolve_target("nvidia")

    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        if arm == "eager":
            runner: Any = model
        else:
            mode = "default" if arm == "inductor" else arm
            runner = torch.compile(model, mode=mode)

        # The inference context lives inside the timed callable, exactly as
        # benchmarks/gpu.py builds its direct arms. Both policies below then
        # time identical work, so what differs between them is only where the
        # synchronization happens.
        def call(runner: Any = runner) -> Any:
            with inference_context(target):
                return runner(example)

        torch.cuda.reset_peak_memory_stats()
        warm_calls = _warm(call, arguments.warm_seconds)
        batched = _batched_ms(call, arguments.repeats)
        per_call = benchmark_callable(
            call,
            target=target,
            backend=arm,
            warmup=5,
            repeats=arguments.repeats,
            batch_size=arguments.batch_size,
        )
        rows.append(
            {
                "arm": arm,
                "warm_calls": warm_calls,
                "batched_mean_ms": batched,
                "per_call_median_ms": per_call.latency_median_ms,
                "per_call_p95_ms": per_call.latency_p95_ms,
                "peak_memory_bytes": per_call.peak_memory_bytes,
            }
        )

    gpu = torch.cuda.get_device_name(0)
    base_batched = rows[0]["batched_mean_ms"]
    base_per_call = rows[0]["per_call_median_ms"]
    print(f"\n{arguments.model} | {arguments.dtype} | batch {arguments.batch_size} | {gpu}")
    print(f"{'arm':<17}{'batched ms':>12}{'speedup':>9}{'per-call ms':>13}{'speedup':>9}")
    for row in rows:
        print(
            f"{row['arm']:<17}{row['batched_mean_ms']:>12.3f}"
            f"{base_batched / row['batched_mean_ms']:>8.2f}x"
            f"{row['per_call_median_ms']:>13.3f}"
            f"{base_per_call / row['per_call_median_ms']:>8.2f}x"
        )

    if arguments.output:
        payload = {
            "model": arguments.model,
            "model_id": HF_MODELS.get(arguments.model),
            "dtype": arguments.dtype,
            "batch_size": arguments.batch_size,
            "gpu": gpu,
            "capability": ".".join(str(part) for part in torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "warm_seconds": arguments.warm_seconds,
            "repeats": arguments.repeats,
            "arms": rows,
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote {arguments.output}")


if __name__ == "__main__":
    main()
