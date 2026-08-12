"""A real decode artifact, reloaded from disk, against an eager reference.

This is the test the whole feature rests on, because the failure it catches is
silent. A backend that functionalizes the KV cache writes away raises nothing: it
returns the correct first token, because the cache is empty then and there is
nothing to have lost, and diverges from the second onward into fluent, wrong
text. Only comparing every token against eager finds that.

The model is `hf-internal-testing/tiny-random-LlamaForCausalLM` -- 15 MB, random
weights, four layers. Its text is gibberish and nothing here says otherwise; what
matters is that both sides produce the *same* gibberish, which they only do if
the cache accumulated correctly on both. Token equality against a real
checkpoint's real sentences belongs to docs/exported-decode.md.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers", reason="the hf extra is not installed")

from lm7 import load_artifact
from lm7.huggingface import _decode_module, _load_transformers, export_hf_model

pytestmark = pytest.mark.export_decode

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
MAX_CACHE_LEN = 32
PROMPT_IDS = [1, 4, 9, 16]
NEW_TOKENS = 6


def greedy(step, prompt_ids: list[int], new_tokens: int, one_call: bool = False) -> list[int]:
    """Prefill, then decode a token at a time.

    ``one_call`` sends the whole prompt through in a single call, which only a
    dynamic capture accepts. The reference never uses it, so a dynamic artifact
    is always compared against a *different* way of filling the same cache rather
    than against itself — which is what makes the comparison worth running.
    """
    logits = None
    if one_call:
        logits = step(torch.tensor([prompt_ids]), torch.arange(len(prompt_ids), dtype=torch.long))
    else:
        for position, token_id in enumerate(prompt_ids):
            logits = step(torch.tensor([[token_id]]), torch.tensor([position]))
    token = int(logits[:, -1].argmax(-1))
    tokens = [token]
    for position in range(len(prompt_ids), len(prompt_ids) + new_tokens - 1):
        logits = step(torch.tensor([[token]]), torch.tensor([position]))
        token = int(logits[:, -1].argmax(-1))
        tokens.append(token)
    return tokens


@pytest.fixture(scope="module")
def eager_tokens() -> list[int]:
    transformers = _load_transformers()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        TINY_MODEL, dtype=torch.float32
    ).eval()
    module = _decode_module(model, batch_size=1, max_cache_len=MAX_CACHE_LEN)
    with torch.inference_mode():
        return greedy(lambda ids, pos: module(ids, pos), PROMPT_IDS, NEW_TOKENS)


@pytest.mark.parametrize("backend", ("export", "aot_inductor"))
@pytest.mark.parametrize("shape", ("dynamic", "single-token"))
def test_a_reloaded_decode_artifact_matches_eager_token_for_token(
    backend, shape, tmp_path, eager_tokens
):
    result = export_hf_model(
        f"hf://{TINY_MODEL}",
        output=str(tmp_path / f"{backend}-{shape}.lm7"),
        target="cpu",
        backend=backend,
        # float32 because the comparison is an argmax: in a narrower format a
        # rounding difference becomes a different token and this gate would be
        # measuring the number format instead of the cache.
        dtype="float32",
        decode=True,
        decode_shape=shape,
        max_cache_len=MAX_CACHE_LEN,
    )
    assert result.decode is True
    assert result.max_cache_len == MAX_CACHE_LEN
    assert result.decode_shape == shape
    assert result.max_tokens_per_call == (MAX_CACHE_LEN - 1 if shape == "dynamic" else 1)

    artifact = load_artifact(result.output)
    assert artifact.manifest.decode is not None
    assert artifact.manifest.decode["max_cache_len"] == MAX_CACHE_LEN
    assert artifact.manifest.decode["shape"] == shape
    assert artifact.manifest.decode["cache_bytes"] > 0

    with torch.inference_mode():
        tokens = greedy(
            lambda ids, pos: artifact(input_ids=ids, cache_position=pos),
            PROMPT_IDS,
            NEW_TOKENS,
            one_call=shape == "dynamic",
        )
    assert tokens == eager_tokens


def test_one_dynamic_capture_serves_many_prompt_lengths(tmp_path):
    """The traced length must not be the only one the artifact accepts.

    Exported with a 4-token example and asked for 1, 2, 9 and 17 — each compared
    against a token-at-a-time eager run of that same prompt, so a length that
    silently prefills the wrong number of slots shows up as different tokens
    rather than as a shape error.
    """
    transformers = _load_transformers()
    result = export_hf_model(
        f"hf://{TINY_MODEL}",
        output=str(tmp_path / "dynamic.lm7"),
        target="cpu",
        backend="export",
        dtype="float32",
        decode=True,
        decode_shape="dynamic",
        max_cache_len=MAX_CACHE_LEN,
    )
    artifact = load_artifact(result.output)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        TINY_MODEL, dtype=torch.float32
    ).eval()
    reference = _decode_module(model, batch_size=1, max_cache_len=MAX_CACHE_LEN)

    for length in (1, 2, 9, 17):
        prompt = [(index * 7) % 900 + 3 for index in range(length)]
        with torch.inference_mode():
            expected = greedy(lambda ids, pos: reference(ids, pos), prompt, NEW_TOKENS)
            tokens = greedy(
                lambda ids, pos: artifact(input_ids=ids, cache_position=pos),
                prompt,
                NEW_TOKENS,
                one_call=True,
            )
        assert tokens == expected, f"diverged at prompt length {length}"


def test_the_cache_actually_accumulates(tmp_path):
    """The same token at the same position twice must not be two different answers.

    `cumulative_length` is re-anchored from `cache_position` on every call rather
    than advanced per execution, which is what makes this graph safe to call more
    than once per slot -- and is the property the JIT path has to work around
    with `warmup: False`. Decoding *past* that position must still differ, or the
    cache is not accumulating at all and every step is a fresh prefill.
    """
    result = export_hf_model(
        f"hf://{TINY_MODEL}",
        output=str(tmp_path / "decode.lm7"),
        target="cpu",
        backend="export",
        dtype="float32",
        decode=True,
        max_cache_len=MAX_CACHE_LEN,
    )
    artifact = load_artifact(result.output)
    with torch.inference_mode():
        first = artifact(input_ids=torch.tensor([[1]]), cache_position=torch.tensor([0])).clone()
        again = artifact(input_ids=torch.tensor([[1]]), cache_position=torch.tensor([0])).clone()
        # Same token, same slot: idempotent.
        assert torch.equal(first, again)

        artifact(input_ids=torch.tensor([[4]]), cache_position=torch.tensor([1]))
        later = artifact(input_ids=torch.tensor([[1]]), cache_position=torch.tensor([2])).clone()
        # Same token, a slot further on, with context behind it: different.
        assert not torch.equal(first, later)
