"""Measure what `torch.compile` actually does to a sparse Mixture-of-Experts model.

`docs/limitations.md` said that compiling buys almost nothing on an MoE because
"Dynamo graph-breaks around the routing regardless of backend". That was asserted
from a latency measurement rather than from Dynamo, and it turns out to be two
separate claims with two different answers, so this script measures both directly:

- **How many graphs Dynamo produces, and why it breaks**, via
  ``torch._dynamo.explain``. The break reason matters: it is
  ``aten.nonzero.default``'s data-dependent output shape in the router, not the
  ``for expert_idx in expert_hit`` Python loop that breaks ``torch.export``. Both
  architectures hit the first; only pre-5.x Mixtral hits the second.
- **What that costs**, by timing ``eager`` against ``inductor`` on the same model.

The answer depends on the transformers version, which is why the report records
it. Run the same command in two environments to compare.

    python benchmarks/moe.py --architecture mixtral olmoe
    python benchmarks/moe.py --model olmoe-1b-7b --target nvidia
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch._dynamo

import lm7
from lm7.detection import resolve_target, synchronize, torch_device

HF_MODELS = {
    "olmoe-1b-7b": "allenai/OLMoE-1B-7B-0924-Instruct",
    # 30.5B total / 3.3B active, ~61 GB at bf16.
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    # 46.7B total / 12.9B active, ~93 GB at bf16 -- which is most of a 96 GB card,
    # and the reason this entry exists. Every Mixtral claim in this repo until now
    # came from a hand-built 2-layer config; this is the real one.
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

# Mirrors examples/sparse_moe.py: 2 layers and a handful of experts, sized so a
# CPU runner compiles it quickly. Dimensions are multiples of 16 because the
# transformers 5.x `grouped_mm` path requires strides that are multiples of 16
# bytes and raises in eager otherwise.
_TINY = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 64,
}

ARCHITECTURES = ("mixtral", "olmoe")


def _tiny_model(architecture: str) -> torch.nn.Module:
    if architecture == "olmoe":
        from transformers import OlmoeConfig, OlmoeForCausalLM

        built = OlmoeForCausalLM(OlmoeConfig(num_experts=8, num_experts_per_tok=2, **_TINY))
    else:
        from transformers import MixtralConfig, MixtralForCausalLM

        built = MixtralForCausalLM(
            MixtralConfig(num_local_experts=4, num_experts_per_tok=2, **_TINY)
        )
    built = built.eval()
    built.config.use_cache = False
    return built


def _explain(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Dynamo's own account of the capture: how many graphs, how many breaks, why.

    One eager call first, because transformers emits `logger.warning_once` from
    inside `forward` and Dynamo breaks on `logging.Logger` methods. Those breaks
    are real but one-shot -- they disappear on the second call and have nothing to
    do with MoE routing. Tracing without flushing them first reports a model as
    badly broken up when it is not: an early revision of this script measured 14
    breaks on a model that actually captures as one graph. The device matters for
    the same reason, so callers move the model before calling this.
    """
    with torch.no_grad():
        model(**inputs, use_cache=False)
    torch._dynamo.reset()
    explained = torch._dynamo.explain(lambda **kw: model(**kw, use_cache=False))(**inputs)
    reasons = Counter(
        str(getattr(reason, "reason", reason)).strip().splitlines()[0]
        for reason in explained.break_reasons
    )
    return {
        "graph_count": explained.graph_count,
        "graph_break_count": explained.graph_break_count,
        "op_count": explained.op_count,
        "break_reasons": [{"reason": text, "count": n} for text, n in reasons.most_common()],
    }


def _time(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    target: Any,
    backend: str,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    torch._dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    compiled = lm7.compile(
        model, target=target, backend=backend, transfers="automatic", fallback="error", cache=False
    )
    synchronize(target)
    started = time.perf_counter()
    with torch.no_grad():
        output = compiled(**inputs, use_cache=False)
    synchronize(target)
    first_call_ms = (time.perf_counter() - started) * 1000.0
    logits = output.logits if hasattr(output, "logits") else output[0]
    next_token_id = int(logits[0, -1].argmax().item())

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
        "backend": backend,
        # Peak device memory matters for a model sized against the card rather than
        # against a workload: a 46.7B MoE at bf16 is ~93 GB of weights on a 95 GiB
        # card, so how much headroom is left decides whether compiling is possible
        # at all, not merely whether it is fast.
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else None,
        "first_call_ms": first_call_ms,
        "latency_median_ms": statistics.median(samples),
        "next_token_id": next_token_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure graph breaks and speedup on sparse MoE.")
    parser.add_argument("--architecture", nargs="+", choices=ARCHITECTURES, default=[])
    parser.add_argument("--model", choices=sorted(HF_MODELS), help="a real MoE from Hugging Face")
    parser.add_argument("--target", default="auto")
    parser.add_argument("--backend", nargs="+", default=["eager", "inductor"])
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, help="Write machine-readable results as JSON.")
    arguments = parser.parse_args()

    if not arguments.architecture and arguments.model is None:
        arguments.architecture = list(ARCHITECTURES)

    import transformers

    target = resolve_target(arguments.target)
    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    subjects: list[tuple[str, torch.nn.Module, dict[str, torch.Tensor]]] = []

    for architecture in arguments.architecture:
        model = _tiny_model(architecture).to(dtype=dtype)
        subjects.append((architecture, model, {"input_ids": torch.randint(0, 256, (1, 16))}))

    if arguments.model is not None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = HF_MODELS[arguments.model]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
        model.config.use_cache = False
        subjects.append(
            (arguments.model, model, dict(tokenizer(arguments.prompt, return_tensors="pt")))
        )

    results = []
    device = torch_device(target)
    for name, model, inputs in subjects:
        # Dynamo traces what it is given, so a model left on the host would be
        # measured in a configuration that never runs -- bfloat16 on CPU takes a
        # different path through transformers than bfloat16 on CUDA.
        model = model.to(device)
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        capture = _explain(model, inputs)
        timings = [
            _time(
                model,
                inputs,
                target,
                backend,
                warmup=arguments.warmup,
                repeats=arguments.repeats,
            )
            for backend in arguments.backend
        ]
        by_backend = {timing["backend"]: timing for timing in timings}
        speedup = None
        if "eager" in by_backend and "inductor" in by_backend:
            speedup = (
                by_backend["eager"]["latency_median_ms"]
                / by_backend["inductor"]["latency_median_ms"]
            )
        agree = len({timing["next_token_id"] for timing in timings}) == 1
        results.append(
            {
                "subject": name,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                **capture,
                "timings": timings,
                "inductor_speedup": speedup,
                "backends_agree_on_next_token": agree,
            }
        )
        print(
            f"{name:>12}  graphs={capture['graph_count']:>3}"
            f"  breaks={capture['graph_break_count']:>3}"
            f"  ops={capture['op_count']:>5}"
            + "".join(
                f"  {timing['backend']}={timing['latency_median_ms']:8.3f} ms" for timing in timings
            )
            + (f"  speedup={speedup:.2f}x" if speedup is not None else "")
            + "".join(
                f"  vram[{timing['backend']}]={timing['peak_memory_bytes'] / 1e9:.1f}GB"
                for timing in timings
                if timing.get("peak_memory_bytes")
            )
            + f"  agree={agree}"
        )
        for reason in capture["break_reasons"]:
            print(f"{'':>12}  x{reason['count']} {reason['reason'][:100]}")
        del model

    report = {
        "schema_version": 1,
        "environment": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "target": str(target),
            "dtype": arguments.dtype,
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
