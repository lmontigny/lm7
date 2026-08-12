"""Greedy generation from an exported KV-cache decode artifact.

`lm7 model export --decode` writes an artifact that can generate; until this,
nothing in LM7 would drive it, so using one meant writing the loop yourself
against `lm7.load_artifact`. The loop is short -- prefill, then argmax and step --
but three of its details are easy to get silently wrong, and all three are the
artifact's own properties rather than the caller's business:

  * **which tokenizer.** The graph emits token ids, and ids are only words under
    the tokenizer the model was trained with. The manifest's `source` records it.
  * **where in the cache.** `cache_position` is an input, not state the artifact
    tracks, so the caller owns the count and an off-by-one is fluent, wrong text.
  * **how many tokens per call.** A `dynamic` artifact takes a whole prompt at
    once; a `single-token` one must be fed a token at a time.

This module reads all three off the manifest. See docs/exported-decode.md.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .errors import UnsupportedModelError
from .exporting import load_artifact


@dataclass(frozen=True)
class ArtifactGenerateResult:
    artifact: str
    backend: str
    target: str
    tokenizer_id: str
    prompt: str
    decode_shape: str
    max_cache_len: int
    input_tokens: int
    generated_tokens: int
    prefill_ms: float
    decode_ms: float
    generated_token_ids: tuple[int, ...]
    generated_text: str
    stopped_at_eos: bool

    @property
    def ms_per_decoded_token(self) -> float:
        return self.decode_ms / self.generated_tokens if self.generated_tokens else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ms_per_decoded_token": self.ms_per_decoded_token}


def generate_from_artifact(
    path: str | Path,
    *,
    prompt: str,
    max_new_tokens: int = 32,
    tokenizer_id: str | None = None,
) -> ArtifactGenerateResult:
    """Greedily generate from a decode artifact, reading its shape off the manifest.

    ``tokenizer_id`` overrides what the artifact recorded, for an artifact
    exported before manifests carried a source or one whose tokenizer genuinely
    differs. It is an override rather than the usual path because getting it wrong
    is not an error the run can detect.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1.")

    artifact = load_artifact(path)
    manifest = artifact.manifest
    decode = dict(manifest.decode or {})
    if not decode:
        raise UnsupportedModelError(
            f"{path} is not a decode artifact: its graph is a single forward pass with no KV "
            "cache, so there is no sequence to continue. Export one with "
            "`lm7 model export --decode`, or call this artifact directly for its logits."
        )

    source = dict(manifest.source or {})
    resolved_tokenizer = tokenizer_id or source.get("tokenizer_id")
    if not resolved_tokenizer:
        raise UnsupportedModelError(
            "This artifact does not record which tokenizer it was built with, so its token ids "
            "cannot be turned into text. It predates manifests carrying a source. Pass "
            "--tokenizer with the model id it was exported from."
        )

    max_cache_len = int(decode.get("max_cache_len", 0))
    per_call = int(decode.get("max_tokens_per_call", 1))
    shape = str(decode.get("shape", "single-token"))

    # Checked before the tokenizer is fetched, because it needs no prompt to
    # answer: a budget this large cannot fit even an empty one, and finding out
    # after a Hub round trip is a slower way to learn the same thing. The exact
    # check, which needs the tokenized length, follows below.
    if max_cache_len and max_new_tokens >= max_cache_len:
        raise UnsupportedModelError(
            f"This artifact holds {max_cache_len} tokens of KV cache, and {max_new_tokens} new "
            "tokens would not fit even with an empty prompt. The cache is buffers inside the "
            "artifact and cannot grow; export one with a larger --max-cache-len, or ask for "
            "fewer tokens."
        )

    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise UnsupportedModelError(
            "Generating text needs a tokenizer, which comes from Transformers. "
            'Install it with: pip install "lm7[hf]". The artifact itself does not need it.'
        ) from exc
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(resolved_tokenizer)
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    except Exception as exc:
        raise UnsupportedModelError(
            f"Could not tokenize with {resolved_tokenizer!r}: {exc}."
        ) from exc

    prompt_tokens = int(input_ids.shape[-1])
    # Refused up front rather than part-way through: the cache cannot grow, and
    # finding out at token 900 means throwing away 900 tokens of work.
    if prompt_tokens + max_new_tokens > max_cache_len:
        raise UnsupportedModelError(
            f"This artifact holds {max_cache_len} tokens of KV cache, and {prompt_tokens} prompt "
            f"tokens plus {max_new_tokens} new ones is {prompt_tokens + max_new_tokens}. The cache "
            "is buffers inside the artifact and cannot grow; export one with a larger "
            "--max-cache-len, or ask for fewer tokens."
        )

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    generated: list[int] = []
    stopped_at_eos = False

    with torch.inference_mode():
        started = time.perf_counter()
        logits = _prefill(artifact, input_ids, prompt_tokens, per_call)
        prefill_ms = (time.perf_counter() - started) * 1000

        filled = prompt_tokens
        started = time.perf_counter()
        for _ in range(max_new_tokens):
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
            token_id = int(token)
            if eos_token_id is not None and token_id == eos_token_id:
                stopped_at_eos = True
                break
            generated.append(token_id)
            if filled >= max_cache_len:
                break
            logits = artifact(
                input_ids=token,
                cache_position=torch.tensor([filled], dtype=torch.long),
            )
            filled += 1
        decode_ms = (time.perf_counter() - started) * 1000

    return ArtifactGenerateResult(
        artifact=str(artifact.path),
        backend=manifest.backend,
        target=str(manifest.target.get("vendor", "unknown")),
        tokenizer_id=resolved_tokenizer,
        prompt=prompt,
        decode_shape=shape,
        max_cache_len=max_cache_len,
        input_tokens=prompt_tokens,
        generated_tokens=len(generated),
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        generated_token_ids=tuple(generated),
        generated_text=tokenizer.decode(generated, skip_special_tokens=True),
        stopped_at_eos=stopped_at_eos,
    )


def _prefill(
    artifact: Any, input_ids: torch.Tensor, prompt_tokens: int, per_call: int
) -> torch.Tensor:
    """Fill the cache with the prompt, in as few calls as the graph allows.

    A `dynamic` artifact takes the whole prompt at once. A `single-token` one, or
    a prompt longer than one call may carry, is chunked at consecutive cache
    positions -- which fills the cache to the same state, just more slowly.
    """
    logits = None
    for start in range(0, prompt_tokens, max(per_call, 1)):
        chunk = input_ids[:, start : start + per_call]
        logits = artifact(
            input_ids=chunk,
            cache_position=torch.arange(start, start + chunk.shape[-1], dtype=torch.long),
        )
    assert logits is not None  # a prompt is never empty: the tokenizer emits at least one id
    return logits


__all__ = ["ArtifactGenerateResult", "generate_from_artifact"]
