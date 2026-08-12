"""Export a KV-cache decode step, then generate from the artifact in this process.

Every other artifact LM7 writes is a pure function: same inputs, same answer,
forever. This one is not. A decode artifact carries its KV cache as buffers
inside the exported program and writes into them, so calling it twice is two
different things happening, and the order matters.

    python examples/exported_decode.py --backend aot_inductor

The script exports, reloads the artifact from disk, drives a greedy loop through
it, and requires the tokens to equal an eager run of the same weights. That last
part is the point rather than a flourish: a backend that functionalizes the cache
writes away does not raise. It returns a correct first token -- the cache is
empty then, so there is nothing to have lost -- and diverges from the second.
Timing such an artifact without comparing its tokens measures a model that is not
answering the question.

There is one graph here, not two, and that is not a shortcut: each exported
program carries its own cache buffers, so a separate prefill artifact would fill
a cache this one never sees. Sharing a cache means sharing a graph, and a bounded
dynamic sequence dimension is what lets that one graph take a whole prompt in a
single call and then one token at a time. Pass ``--decode-shape single-token``
to compare against the fixed capture, which feeds the prompt a token at a time.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

import lm7
from lm7.huggingface import _decode_module, _load_transformers, _model_id, export_hf_model

DEFAULT_MODEL = "hf://HuggingFaceTB/SmolLM2-135M-Instruct"


def greedy(step, prompt_ids: torch.Tensor, new_tokens: int, one_call: bool) -> list[int]:
    """Prefill, then decode a token at a time from the logits.

    ``one_call`` is the difference a dynamic capture makes: the whole prompt goes
    through the graph at once instead of a forward pass per token. Both fill the
    same cache to the same state, so the tokens that follow are identical -- only
    the time taken differs.
    """
    logits = None
    if one_call:
        logits = step(prompt_ids.unsqueeze(0), torch.arange(len(prompt_ids)))
    else:
        for position, token_id in enumerate(prompt_ids.tolist()):
            logits = step(torch.tensor([[token_id]]), torch.tensor([position]))
    token = int(logits[:, -1].argmax(-1))
    tokens = [token]
    for position in range(len(prompt_ids), len(prompt_ids) + new_tokens - 1):
        logits = step(torch.tensor([[token]]), torch.tensor([position]))
        token = int(logits[:, -1].argmax(-1))
        tokens.append(token)
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and run a KV-cache decode artifact.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--backend",
        default="export",
        choices=("export", "aot_inductor"),
        help="export backend (default: export)",
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument(
        "--decode-shape",
        default="dynamic",
        choices=("dynamic", "single-token"),
        help="dynamic prefills the whole prompt in one call (default: dynamic)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-cache-len", type=int, default=128)
    parser.add_argument("--output", default="build/decode.lm7")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)

    result = export_hf_model(
        args.model,
        output=str(output),
        prompt=args.prompt,
        backend=args.backend,
        target="cpu",
        # float32 on purpose: greedy decoding is token-exact only when the
        # arithmetic is, and this script gates on token equality.
        dtype="float32",
        decode=True,
        decode_shape=args.decode_shape,
        max_cache_len=args.max_cache_len,
    )
    print(f"Exported {result.output} ({result.artifact_bytes / 1024**2:.1f} MiB)")
    print(f"Cache: {result.max_cache_len} tokens, batch 1")
    print(f"Shape: {result.decode_shape}, up to {result.max_tokens_per_call} token(s) per call")
    one_call = result.decode_shape == "dynamic"

    transformers = _load_transformers()
    model_id = _model_id(args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    prompt_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"][0]

    artifact = lm7.load_artifact(result.output)
    with torch.inference_mode():
        tokens = greedy(
            lambda ids, pos: artifact(input_ids=ids, cache_position=pos),
            prompt_ids,
            args.max_new_tokens,
            one_call,
        )

    # The reference: the same weights, the same one-token-at-a-time loop, no
    # export anywhere in it.
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    reference_module = _decode_module(model, batch_size=1, max_cache_len=args.max_cache_len)
    with torch.inference_mode():
        # The reference always feeds the prompt a token at a time, so a dynamic
        # artifact is checked against a *different* way of filling the cache
        # rather than against itself.
        expected = greedy(
            lambda ids, pos: reference_module(ids, pos), prompt_ids, args.max_new_tokens, False
        )

    print(f"Artifact: {tokenizer.decode(tokens)!r}")
    print(f"Eager:    {tokenizer.decode(expected)!r}")
    if tokens != expected:
        matched = next((i for i, (a, b) in enumerate(zip(tokens, expected)) if a != b), len(tokens))
        raise SystemExit(
            f"Artifact and eager diverged at token {matched}. The cache writes did not "
            "survive this backend; see docs/exported-decode.md."
        )
    print(f"{len(tokens)} tokens match eager exactly.")


if __name__ == "__main__":
    main()
