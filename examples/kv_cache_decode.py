"""Generate with separately compiled prefill and KV-cache decode graphs.

The two phases of autoregressive generation want different treatment: the prompt
pass is one wide, arithmetic-bound call whose shape changes per request, and the
decode step is a thousand narrow, memory-bound calls at one fixed shape.
``lm7.compile_generation`` compiles them apart against one static cache and
reports what each phase cost to compile.

    python examples/kv_cache_decode.py --target nvidia --compile-mode reduce-overhead

The script checks itself: it generates once through the runner and once through
``model.generate`` and requires identical tokens. That check is the point. A
decode loop can be fast and wrong in ways that read as fluent text -- see
docs/kv-cache-decode.md -- so a timing without it proves nothing.

It runs in float32 by default for that reason. Greedy decoding is token-exact
only when the arithmetic is: in bfloat16 an argmax turns any rounding difference
into a different word, and Inductor's kernels do not round like eager's. Pass
``--dtype bfloat16`` for timings representative of how a GPU would actually serve
this, and the token comparison becomes a report rather than a gate.
"""

from __future__ import annotations

import argparse

import torch

import lm7
from lm7.detection import resolve_target

DEFAULT_MODEL = "hf://HuggingFaceTB/SmolLM2-135M-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compiled prefill and KV-cache decode through LM7."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target", default="auto", help="target selector (default: auto)")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--backend", choices=("auto", "eager", "inductor"), default="auto")
    parser.add_argument(
        "--compile-mode",
        default=None,
        help="Inductor preset; 'reduce-overhead' is how CUDA Graphs are requested",
    )
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="float32 keeps the token comparison exact; see the module docstring",
    )
    arguments = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit('Install Hugging Face support with: pip install -e ".[hf]"') from None

    target = resolve_target(arguments.target)
    model_id = arguments.model.removeprefix("hf://")
    dtype = getattr(torch, arguments.dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
    input_ids = tokenizer(arguments.prompt, return_tensors="pt").input_ids

    runner = lm7.compile_generation(
        model,
        target=target,
        backend=arguments.backend,
        compile_mode=arguments.compile_mode,
        max_batch_size=1,
        max_sequence_length=arguments.max_sequence_length,
    )
    print(runner)
    print(f"KV cache: {runner.cache_bytes / 2**20:.1f} MiB on {runner.device}")

    # First call compiles; the numbers worth quoting come from the second.
    runner.generate(input_ids, max_new_tokens=arguments.max_new_tokens)
    result = runner.generate(input_ids, max_new_tokens=arguments.max_new_tokens)

    print(f"\n{arguments.prompt}{tokenizer.decode(result.tokens[0])}")
    print(
        f"\nprefill {result.prefill_ms:.1f} ms"
        f"  decode {result.ms_per_decoded_token:.3f} ms/token"
        f" over {result.decode_steps} steps"
    )
    for phase, counts in runner.counters.items():
        print(f"  {phase:<8} {counts}")
    print(f"  cuda graphs: {runner.cudagraphs['decode']}")

    with torch.inference_mode():
        reference = model.generate(
            input_ids.to(runner.device),
            max_new_tokens=arguments.max_new_tokens,
            do_sample=False,
        )
    expected = reference[:, input_ids.shape[-1] :]
    agree = result.tokens.tolist() == expected.tolist()
    print(f"\nsame tokens as model.generate: {agree}")
    if not agree and dtype is torch.float32:
        raise SystemExit("The compiled decode loop disagreed with model.generate.")
    if not agree:
        print(
            f"  {arguments.dtype} greedy decoding is not token-exact across kernels; rerun "
            "with --dtype float32 to check correctness. See docs/kv-cache-decode.md."
        )
    if runner.counters["steady"]["frames"]:
        raise SystemExit("A token triggered a compile; the decode graph is not stable.")


if __name__ == "__main__":
    main()
