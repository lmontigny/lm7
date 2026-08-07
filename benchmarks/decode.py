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
    python benchmarks/decode.py --target cpu --sequence-length 128 512

A CPU target runs the two arms that do not want CUDA Graphs, in float32 rather
than bfloat16 -- see `DTYPES`. Peak memory is not reported there, because a host
allocator number is not the same measurement as `max_memory_allocated`.

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
from lm7.detection import detect_targets, resolve_target, synchronize
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

# CUDA Graphs are a CUDA feature, so the two arms that request them have nothing
# to offer a CPU run -- `reduce-overhead` there is `inductor` under another name.
CPU_ARMS = ("eager", "inductor")

# The same policy `lm7.huggingface` applies to quantized runs, for the same
# reason: x86 without AVX-512 has no native bfloat16, so a CPU run in bfloat16
# measures emulation rather than the decode path.
DTYPES = {"cpu": torch.float32}
DEFAULT_DTYPE = torch.bfloat16


def _is_cuda(target: Any) -> bool:
    return target.vendor in {"nvidia", "amd"}


def _reset_peak_memory(target: Any) -> None:
    if _is_cuda(target):
        torch.cuda.reset_peak_memory_stats()


def _peak_memory(target: Any) -> int | None:
    """Device bytes at peak, or None where the concept does not apply.

    None rather than a host RSS reading. A CPU allocator number is not comparable
    to `max_memory_allocated`, and reporting one under the same key would invite
    exactly that comparison.
    """
    return int(torch.cuda.max_memory_allocated()) if _is_cuda(target) else None


def _release(target: Any) -> None:
    if _is_cuda(target):
        torch.cuda.empty_cache()


# A paragraph of ordinary English, tokenized and then tiled to whatever length is
# asked for. Latency does not care what the tokens say, but token *agreement*
# between arms does: see `_prompt`.
PROMPT_TEXT = (
    "The capital of France is Paris, a city on the Seine whose history runs from a "
    "Gaulish settlement through a Roman garrison to the seat of a modern republic. "
    "Serving a language model is two problems rather than one. Reading a prompt is "
    "a single wide pass over many tokens at once, bounded by arithmetic. Writing an "
    "answer is a long sequence of narrow passes, one token at a time, bounded by "
    "how fast weights can be read out of memory. "
)


def _prompt(
    batch: int, length: int, vocab: int, device: torch.device, source: str, tokenizer: Any
) -> torch.Tensor:
    """A prompt of exactly `length` tokens, identical across arms.

    Two sources, because they answer different questions. `random` ids need no
    tokenizer and no text long enough to reach 8192 tokens, and for *latency* they
    are as good as anything: the shapes are what cost time.

    They are not good enough to compare tokens with. On input it was never trained
    on, the model's next-token distribution is nearly flat, so the greedy argmax
    sits on a near-tie and BF16 rounding differences between an eager and an
    Inductor prefill are enough to flip it. `text` tiles real prose instead, which
    keeps the distribution peaked and makes "did every arm produce the same
    tokens" a question about the compiler rather than about tie-breaking.
    """
    if source == "text":
        tokens = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids[0]
        repeats = -(-length // int(tokens.numel()))
        ids = tokens.repeat(repeats)[:length].unsqueeze(0).expand(batch, -1).contiguous()
        return ids.to(device)
    generator = torch.Generator(device="cpu").manual_seed(length * 1000 + batch)
    ids = torch.randint(0, vocab, (batch, length), generator=generator, dtype=torch.long)
    return ids.to(device)


def _load(model_id: str, quantization: str, target: Any) -> tuple[Any, Any, float, int]:
    import transformers

    dtype = DTYPES.get(target.vendor, DEFAULT_DTYPE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
    quantization_ms, converted = 0.0, 0
    if quantization != NO_QUANTIZATION:
        quantization_ms, converted = _apply_quantization(model, target, quantization)
    return model, tokenizer, quantization_ms, converted


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
    prompt_source: str,
    target: Any,
) -> None:
    """One configuration, start to finish, in a frame that then goes away.

    The model and the runner are deliberately local to this call: a fresh model is
    loaded per configuration so quantization never carries over, and `empty_cache`
    can only return the last one's weights to the allocator once this frame has
    exited.
    """
    model, tokenizer, quantization_ms, converted = _load(model_id, quantization, target)
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
    prompt = _prompt(batch, length, vocab, runner.device, prompt_source, tokenizer)

    # The runner compiles when it first sees each shape, and CUDA Graph capture
    # happens on a replay after that -- so a short run comes first and is not part
    # of the steady numbers. It is still worth timing: this is the cold start a
    # serving process pays before its first token, and it is the cost the counters
    # below say is paid once.
    cold = runner.generate(prompt, max_new_tokens=warmup_steps + 1)
    record["cold_prefill_ms"] = cold.prefill_ms
    record["cold_decode_ms"] = cold.decode_ms
    record["compile"] = {
        "prefill": runner.counters["prefill"],
        "decode": runner.counters["decode"],
    }
    record["cudagraphs"] = runner.cudagraphs
    record["cache_bytes"] = runner.cache_bytes

    _reset_peak_memory(target)
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
    record["peak_memory_bytes"] = _peak_memory(target)
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
    prompt_source: str,
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
    _release(target)
    _reset_peak_memory(target)
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
            prompt_source=prompt_source,
            target=target,
        )
    except Exception as error:  # noqa: BLE001 - a configuration that will not run is a result
        record.update(
            {"works": False, "error_type": type(error).__name__, "error": str(error)[:400]}
        )
    _release(target)
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
        + (
            f" vram={record['peak_memory_bytes'] / 2**30:5.1f}GiB"
            if record.get("peak_memory_bytes")
            else ""
        )
        + f" kv={record['cache_bytes'] / 2**30:4.2f}GiB"
        f" recompiled={record['recompiled_during_decode']!s:<5}"
        f" captured={captured}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compiled prefill and KV-cache decode across eager, Inductor and CUDA Graphs."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target", default="nvidia")
    # No default: an unset `--arm` means "whatever this target can run", which is
    # every arm on a GPU and the two that do not want CUDA Graphs on a CPU.
    parser.add_argument("--arm", nargs="+", choices=tuple(ARMS), default=None)
    parser.add_argument(
        "--quantization", nargs="+", choices=QUANTIZATIONS, default=[NO_QUANTIZATION]
    )
    parser.add_argument("--sequence-length", nargs="+", type=int, default=list(SEQUENCE_LENGTHS))
    parser.add_argument("--batch-size", nargs="+", type=int, default=list(BATCH_SIZES))
    parser.add_argument("--decode-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--prompt-source",
        choices=("text", "random"),
        default="text",
        help="tiled English prose, or random token ids (see _prompt)",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    target = resolve_target(arguments.target)
    arms = arguments.arm or (list(CPU_ARMS) if target.vendor == "cpu" else list(ARMS))
    unavailable = [arm for arm in arms if ARMS[arm]["compile_mode"] and target.vendor == "cpu"]
    if unavailable:
        raise SystemExit(
            f"{', '.join(unavailable)} request CUDA Graphs, which {target} does not have. "
            f"On a CPU target the arms are: {', '.join(CPU_ARMS)}."
        )
    synchronize(target)

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    output = arguments.output.expanduser().resolve() if arguments.output else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)

    def report() -> dict[str, Any]:
        return {
            # 2 adds the cold-start timings: how long the first prompt and the
            # first token take, which is what the steady numbers are net of.
            "schema_version": 2,
            "environment": {
                "torch": torch.__version__,
                "transformers": _version("transformers"),
                "torchao": _version("torchao"),
                "target": str(target),
                "device": _device_name(target),
                "dtype": str(DTYPES.get(target.vendor, DEFAULT_DTYPE)).removeprefix("torch."),
                "host": platform.node(),
            },
            "model": arguments.model,
            "prompt_source": arguments.prompt_source,
            "decode_steps": arguments.decode_steps,
            "repeats": arguments.repeats,
            "complete": False,
            "elapsed_s": time.perf_counter() - started,
            "results": results,
        }

    def write(final: bool = False) -> None:
        # Rewritten after every configuration rather than once at the end. A full
        # sweep is an hour of metered GPU on a host that can be reclaimed under it,
        # and a run that loses everything to a restart it was 90% through is not a
        # measurement. `complete` says whether the file is the whole sweep.
        if output is None:
            return
        current = report()
        current["complete"] = final
        output.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for quantization in arguments.quantization:
        for length in arguments.sequence_length:
            for batch in arguments.batch_size:
                print(f"{quantization} sequence={length} batch={batch}", flush=True)
                group: list[dict[str, Any]] = []
                for arm in arms:
                    record = run_configuration(
                        arguments.model,
                        arm=arm,
                        quantization=quantization,
                        batch=batch,
                        length=length,
                        decode_steps=arguments.decode_steps,
                        warmup_steps=arguments.warmup_steps,
                        repeats=arguments.repeats,
                        prompt_source=arguments.prompt_source,
                        target=target,
                    )
                    group.append(record)
                    _print(record)
                    results.append(record)
                    write()
                _compare_arms(group)
                write()

    write(final=True)
    if output is not None:
        print(f"JSON: {output}")


def _device_name(target: Any) -> str:
    """Whatever this target calls itself, without assuming it is a GPU."""
    for device in detect_targets():
        if device.target.vendor == target.vendor and device.target.kind == target.kind:
            return device.name
    return str(target)


def _version(name: str) -> str | None:
    try:
        return str(__import__(name).__version__)
    except Exception:  # noqa: BLE001 - an absent optional package is not a failure
        return None


if __name__ == "__main__":
    main()
