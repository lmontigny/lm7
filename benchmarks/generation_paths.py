"""Four ways to generate the same tokens, in one harness.

The question this answers is "what does LM7 buy over just calling
``model.generate()`` behind a web server", and it is not the same answer on
every target:

    eager        model.generate()                         a plain FastAPI server
    hf-static    + StaticCache + CompileConfig            `lm7 model generate`
    hf-forced    + CompileConfig(_compile_all_devices)    the same, past the allowlist
    lm7          lm7.compile_generation()                 `lm7 model serve`

``hf-static`` is the interesting arm. Transformers gates compiled generation on a
hardcoded device allowlist -- ``["cuda", "xpu", "neuron", "tpu"]`` in
``generation/utils.py`` -- so on Apple Silicon and on CPU it logs "unable to meet
the criteria for compilation" and decodes eagerly. ``hf-forced`` flips the
private ``_compile_all_devices`` escape hatch to separate *cannot* from *does
not*, which is the whole point of running it: an arm that is merely disallowed
should get faster when allowed.

    python benchmarks/generation_paths.py --target apple
    python benchmarks/generation_paths.py --target cpu --output artifacts/paths.json

One script rather than four, because `benchmarks/moe.py` and
`benchmarks/nvidia_matrix.py` disagree by 2.3x on the same card purely from
building inputs differently. Every arm here shares one tokenized prompt, one
token budget, one timing function, and one freshly loaded copy of the same
checkpoint. Outputs are compared across arms, so a "faster" arm that quietly
decoded something else is reported rather than believed.
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

from lm7.detection import resolve_target, synchronize, torch_device
from lm7.generation import compile_generation

HF_MODELS = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "lfm2.5-230m": "LiquidAI/LFM2.5-230M",
    "llama-3.2-1b": "unsloth/Llama-3.2-1B-Instruct",
    # The dense validation ladder, reachable by name and not yet measured here.
    # See docs/limitations.md#model-coverage.
    "lfm2.5-350m": "LiquidAI/LFM2.5-350M",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
}


def load(model_id: str, dtype: torch.dtype, device: torch.device) -> Any:
    """A fresh copy per arm.

    Not shared: `compile_generation` moves and wraps the module it is given, and
    a compiled arm handing its already-traced model to the next one would measure
    the previous arm.
    """
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
    return model.to(device)


def timed(call: Any, target: Any, repeats: int) -> tuple[float, Any]:
    """Warm up once -- that call pays for compilation -- then take the median."""
    result = call()
    samples = []
    for _ in range(repeats):
        synchronize(target)
        started = time.perf_counter()
        result = call()
        synchronize(target)
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples), result


def time_generate(
    model: Any, input_ids: torch.Tensor, kwargs: dict[str, Any], target: Any, repeats: int
) -> tuple[float, Any]:
    """``timed`` around ``model.generate``, with the model bound as a parameter.

    A closure over a loop variable would be timing whichever model the loop had
    reached by the time it ran, which is the kind of bug that produces a
    plausible number rather than an error.
    """

    def call() -> Any:
        return model.generate(input_ids, **kwargs)

    return timed(call, target, repeats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="smollm2-135m", help=f"one of {', '.join(HF_MODELS)}")
    parser.add_argument("--target", default="auto")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None, help="write JSON here")
    args = parser.parse_args()

    import transformers

    model_id = HF_MODELS.get(args.model, args.model)
    target = resolve_target(args.target)
    device = torch_device(target)
    # float32 on CPU for the same reason benchmarks/decode.py uses it: a CPU
    # float16 matmul is emulated, so it would measure the emulation.
    dtype = torch.float32 if target.vendor == "cpu" else torch.float16

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)
    budget: dict[str, Any] = {"max_new_tokens": args.max_new_tokens, "do_sample": False}

    def decode(tokens: torch.Tensor) -> str:
        return str(tokenizer.decode(tokens, skip_special_tokens=True))

    arms: dict[str, dict[str, Any]] = {}

    prompt_tokens = int(input_ids.shape[-1])
    ms, out = time_generate(load(model_id, dtype, device), input_ids, budget, target, args.repeats)
    arms["eager"] = {"ms": ms, "text": decode(out[0, prompt_tokens:])}

    for name, force in (("hf-static", False), ("hf-forced", True)):
        config = transformers.CompileConfig(backend="inductor", mode="reduce-overhead")
        if force:
            # Private, and the only way to tell "this device cannot compile"
            # apart from "this device is not on the list".
            config._compile_all_devices = True
        kwargs = {**budget, "cache_implementation": "static", "compile_config": config}
        ms, out = time_generate(
            load(model_id, dtype, device), input_ids, kwargs, target, args.repeats
        )
        arms[name] = {"ms": ms, "text": decode(out[0, prompt_tokens:])}

    runner = compile_generation(
        load(model_id, dtype, device),
        target,
        backend="auto",
        max_batch_size=1,
        max_sequence_length=args.max_sequence_length,
    )

    def run_lm7() -> Any:
        return runner.generate(input_ids, max_new_tokens=args.max_new_tokens)

    ms, out = timed(run_lm7, target, args.repeats)
    arms["lm7"] = {
        "ms": ms,
        "text": decode(out.tokens[0]),
        "backend": runner.cudagraphs["decode"]["backend"],
        "counters": runner.counters,
    }

    baseline = arms["eager"]["ms"]
    for arm in arms.values():
        arm["ms_per_token"] = arm["ms"] / args.max_new_tokens
        arm["speedup_vs_eager"] = baseline / arm["ms"]

    texts = {name: arm["text"] for name, arm in arms.items()}
    agree = len(set(texts.values())) == 1

    data = {
        "model": model_id,
        "target": str(target),
        "dtype": str(dtype).removeprefix("torch."),
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "identical_output": agree,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "arms": arms,
    }

    print(f"\n{model_id} · {target} · {data['dtype']} · {args.max_new_tokens} tokens")
    print(f"median of {args.repeats}, total wall clock divided by tokens\n")
    for name, arm in arms.items():
        print(
            f"  {name:<12}{arm['ms']:9.1f} ms{arm['ms_per_token']:8.2f} ms/token"
            f"{arm['speedup_vs_eager']:8.2f}x"
        )
    print(f"\n  identical output across all arms: {agree}")
    if not agree:
        for name, text in texts.items():
            print(f"    {name:<12}{text[:56]!r}")
    print(f"  lm7 backend: {arms['lm7']['backend']}")
    print(f"  lm7 steady frames (must be 0): {arms['lm7']['counters']['steady']['frames']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2))
        print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
