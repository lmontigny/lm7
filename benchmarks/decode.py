"""Measure the compiled prefill/decode split: eager against Inductor against CUDA Graphs.

A forward pass benchmark answers "how fast is this model". Generation asks two
different questions, and they have different answers:

    prefill   one large matmul-bound pass over the whole prompt
    decode    a thousand tiny memory-bound passes, one token each

Compiling matters far more for the second, because a decode step spends most of
its wall clock launching kernels rather than running them -- which is also why
CUDA Graphs, which remove the launches and nothing else, are worth separating
from Inductor's codegen.

    python benchmarks/decode.py --output artifacts/decode.json
    python benchmarks/decode.py --sequence-length 512 8192 --batch-size 1 4 8 \
        --quantization none fp8-dynamic --decode-steps 1000

Every configuration reports what compiled and how often. `steady` counters that
are anything but zero mean a token triggered a compile, which is the failure this
path is built to make visible rather than to explain away as jitter.
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

import lm7
from lm7.detection import resolve_target, synchronize
from lm7.huggingface import (
    FP8,
    FP8_DYNAMIC,
    FP8_DYNAMIC_ROWWISE,
    INT8,
    NO_QUANTIZATION,
    _apply_quantization,
)

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"

QUANTIZATIONS = (NO_QUANTIZATION, INT8, FP8, FP8_DYNAMIC, FP8_DYNAMIC_ROWWISE)

# The execution arms. `eager` is the control: a static cache and the same
# two-graph structure with no compiler under it, which is what separates "the
# split helped" from "the compiler helped". `cudagraphs` is Inductor plus capture,
# not capture on its own -- `reduce-overhead` is the only preset that requests it.
# `decode-only` is the same as `cudagraphs` with the prompt pass left eager, which
# is the boundary Transformers' own compiled generation draws; comparing the two
# prices compiling prefill at all.
ARMS: dict[str, dict[str, Any]] = {
    "eager": {"backend": "eager", "compile_mode": None},
    "inductor": {"backend": "inductor", "compile_mode": None},
    "cudagraphs": {"backend": "inductor", "compile_mode": "reduce-overhead"},
    "decode-only": {
        "backend": "inductor",
        "compile_mode": "reduce-overhead",
        "compile_prefill": False,
    },
}

SEQUENCE_LENGTHS = (512, 1024, 2048, 4096, 8192)
BATCH_SIZES = (1, 4, 8)


def _prompt(batch: int, length: int, vocab: int, device: torch.device) -> torch.Tensor:
    """A prompt of exactly `length` tokens, identical across arms.

    Random ids rather than real text on purpose: the point is the shape, every arm
    sees the same tensor from the same seed, and no tokenizer can be asked for a
    prompt that happens to be 8192 tokens long.
    """
    generator = torch.Generator(device="cpu").manual_seed(length * 1000 + batch)
    ids = torch.randint(0, vocab, (batch, length), generator=generator, dtype=torch.long)
    return ids.to(device)


def _load(model_id: str, quantization: str, target: Any) -> tuple[Any, float, int]:
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval()
    quantization_ms, converted = 0.0, 0
    if quantization != NO_QUANTIZATION:
        quantization_ms, converted = _apply_quantization(model, target, quantization)
    return model, quantization_ms, converted


def _measure(
    model_id: str,
    record: dict[str, Any],
    *,
    arm: str,
    quantization: str,
    batch: int,
    length: int,
    decode_steps: int,
    warmup_steps: int,
    repeats: int,
    target: Any,
) -> None:
    """One configuration, start to finish, in a frame that then goes away.

    The model and the runner are deliberately local to this call: a fresh model is
    loaded per configuration so quantization never carries over, and `empty_cache`
    can only return the last one's weights to the allocator once this frame has
    exited.
    """
    model, quantization_ms, converted = _load(model_id, quantization, target)
    record["quantization_ms"] = quantization_ms
    record["quantized_modules"] = converted
    # The cache has to hold the prompt and everything decoded from it, and nothing
    # more. Sizing it to exactly that rather than to a round number keeps the KV
    # column comparable across sequence lengths.
    runner = lm7.compile_generation(
        model,
        target=target,
        max_batch_size=batch,
        max_sequence_length=length + decode_steps + 1,
        **ARMS[arm],
    )
    vocab = int(model.config.get_text_config(decoder=True).vocab_size)
    prompt = _prompt(batch, length, vocab, runner.device)

    # The runner compiles when it is built and when it first sees a prompt length,
    # but CUDA Graph capture happens on the first replay after that -- so a short
    # unmeasured run comes first.
    runner.generate(prompt, max_new_tokens=warmup_steps + 1)
    record["compile"] = {
        "prefill": runner.counters["prefill"],
        "decode": runner.counters["decode"],
    }
    record["cudagraphs"] = runner.cudagraphs
    record["cache_bytes"] = runner.cache_bytes

    torch.cuda.reset_peak_memory_stats()
    prefill_samples: list[float] = []
    decode_samples: list[float] = []
    steady_before = runner.counters["steady"]
    tokens: list[int] = []
    for _ in range(repeats):
        result = runner.generate(prompt, max_new_tokens=decode_steps + 1)
        prefill_samples.append(result.prefill_ms)
        decode_samples.append(result.decode_ms)
        tokens = result.tokens[0, :16].tolist()
    steady_after = runner.counters["steady"]

    record["steady_counters"] = {
        key: steady_after[key] - steady_before[key] for key in steady_after
    }
    record["recompiled_during_decode"] = record["steady_counters"]["frames"] > 0
    record["prefill_ms"] = statistics.median(prefill_samples)
    record["decode_ms"] = statistics.median(decode_samples)
    record["ms_per_token"] = record["decode_ms"] / decode_steps
    record["decode_tokens_per_second"] = decode_steps * batch / (record["decode_ms"] / 1000.0)
    record["prefill_tokens_per_second"] = length * batch / (record["prefill_ms"] / 1000.0)
    record["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    # The cache's own count, not the runner's bookkeeping. They agree only while
    # every execution of a graph is one the caller asked for -- see
    # GenerationRunner._warm -- and a mismatch means every token after the first
    # was computed against a shifted cache.
    record["cache_sequence_length"] = runner.cache_sequence_length
    record["expected_cache_sequence_length"] = length + decode_steps
    record["tokens"] = tokens
    record["works"] = True


def run_configuration(
    model_id: str,
    *,
    arm: str,
    quantization: str,
    batch: int,
    length: int,
    decode_steps: int,
    warmup_steps: int,
    repeats: int,
    target: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "arm": arm,
        "quantization": quantization,
        "batch_size": batch,
        "sequence_length": length,
        "decode_steps": decode_steps,
        **ARMS[arm],
    }
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        _measure(
            model_id,
            record,
            arm=arm,
            quantization=quantization,
            batch=batch,
            length=length,
            decode_steps=decode_steps,
            warmup_steps=warmup_steps,
            repeats=repeats,
            target=target,
        )
    except Exception as error:  # noqa: BLE001 - a configuration that will not run is a result
        record.update(
            {"works": False, "error_type": type(error).__name__, "error": str(error)[:400]}
        )
    torch.cuda.empty_cache()
    return record


def _compare_arms(group: list[dict[str, Any]]) -> None:
    """Score each arm against the uncompiled one, on tokens and on latency.

    A compiler that changes the tokens has not made decoding faster, it has made
    it something else, so the speedup column is only meaningful next to this one.
    The reference is the first arm that ran, which is `eager` unless the caller
    narrowed `--arm`.
    """
    reference = next((record for record in group if record.get("works")), None)
    if reference is None:
        return
    for record in group:
        if not record.get("works"):
            continue
        record["reference_arm"] = reference["arm"]
        record["same_tokens_as_reference"] = record["tokens"] == reference["tokens"]
        record["decode_speedup"] = reference["ms_per_token"] / record["ms_per_token"]
        record["prefill_speedup"] = reference["prefill_ms"] / record["prefill_ms"]
        agreement = "same tokens" if record["same_tokens_as_reference"] else "DIFFERENT TOKENS"
        print(
            f"  {record['arm']:<11} vs {reference['arm']}: "
            f"decode {record['decode_speedup']:5.2f}x  prefill {record['prefill_speedup']:5.2f}x"
            f"  {agreement}"
        )


def _print(record: dict[str, Any]) -> None:
    head = (
        f"  {record['arm']:<11} {record['quantization']:<20} "
        f"b{record['batch_size']:<2} s{record['sequence_length']:<5}"
    )
    if not record.get("works"):
        print(f"{head} FAIL {record.get('error_type')}: {record.get('error', '')[:90]}")
        return
    captured = record["cudagraphs"]["decode"].get("cudagraphs_active")
    print(
        f"{head} prefill={record['prefill_ms']:8.1f}ms"
        f" decode={record['ms_per_token']:6.3f}ms/tok"
        f" {record['decode_tokens_per_second']:8.1f}tok/s"
        f" vram={record['peak_memory_bytes'] / 2**30:5.1f}GiB"
        f" kv={record['cache_bytes'] / 2**30:4.2f}GiB"
        f" recompiled={record['recompiled_during_decode']!s:<5}"
        f" captured={captured}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compiled prefill and KV-cache decode across eager, Inductor and CUDA Graphs."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target", default="nvidia")
    parser.add_argument("--arm", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument(
        "--quantization", nargs="+", choices=QUANTIZATIONS, default=[NO_QUANTIZATION]
    )
    parser.add_argument("--sequence-length", nargs="+", type=int, default=list(SEQUENCE_LENGTHS))
    parser.add_argument("--batch-size", nargs="+", type=int, default=list(BATCH_SIZES))
    parser.add_argument("--decode-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA GPU.")
    target = resolve_target(arguments.target)
    synchronize(target)

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for quantization in arguments.quantization:
        for length in arguments.sequence_length:
            for batch in arguments.batch_size:
                print(f"{quantization} sequence={length} batch={batch}")
                group: list[dict[str, Any]] = []
                for arm in arguments.arm:
                    record = run_configuration(
                        arguments.model,
                        arm=arm,
                        quantization=quantization,
                        batch=batch,
                        length=length,
                        decode_steps=arguments.decode_steps,
                        warmup_steps=arguments.warmup_steps,
                        repeats=arguments.repeats,
                        target=target,
                    )
                    group.append(record)
                    _print(record)
                _compare_arms(group)
                results.extend(group)

    report = {
        "schema_version": 1,
        "environment": {
            "torch": torch.__version__,
            "transformers": _version("transformers"),
            "torchao": _version("torchao"),
            "device": torch.cuda.get_device_name(0),
            "capability": "sm{}{}".format(*torch.cuda.get_device_capability(0)),
            "host": platform.node(),
        },
        "model": arguments.model,
        "decode_steps": arguments.decode_steps,
        "repeats": arguments.repeats,
        "elapsed_s": time.perf_counter() - started,
        "results": results,
    }
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON: {output}")


def _version(name: str) -> str | None:
    try:
        return str(__import__(name).__version__)
    except Exception:  # noqa: BLE001 - an absent optional package is not a failure
        return None


if __name__ == "__main__":
    main()
