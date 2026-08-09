"""What LM7 refuses to pretend it implemented.

Separate from ``schemas.py`` because this is the half that has no Pydantic in it:
the check runs against the raw request body, and keeping it importable without
the ``serve`` extra means it is covered by the portable test suite rather than
only by the marked HTTP job.

An OpenAI client sends fields this server cannot honour. Pydantic would drop the
unknown ones silently, so a caller could ask for four completions, or a JSON
schema, or a tool call, and get a plain single answer with nothing to indicate
that the request was not the one that ran. Every field named here is refused
with a 400 instead.
"""

from __future__ import annotations

from typing import Any

UNSUPPORTED_FIELDS = (
    "n",
    "best_of",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "tools",
    "functions",
    "tool_choice",
    "function_call",
    "response_format",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
)

# The values that mean "I did not actually ask for this". OpenAI SDKs send `n=1`
# and zero penalties as a matter of course, so treating presence alone as a
# request would refuse every request that came from an SDK rather than curl.
_INERT: dict[str, tuple[Any, ...]] = {
    "n": (None, 1),
    "best_of": (None, 1),
    "logprobs": (None, False, 0),
    "top_logprobs": (None, 0),
    "frequency_penalty": (None, 0, 0.0),
    "presence_penalty": (None, 0, 0.0),
    "repetition_penalty": (None, 1, 1.0),
    "tool_choice": (None, "none", "auto"),
    "function_call": (None, "none", "auto"),
    "tools": (None,),
    "functions": (None,),
    "response_format": (None, {"type": "text"}),
}


def unsupported_fields(body: dict[str, Any]) -> list[str]:
    """Which unimplementable fields this request body set to a meaningful value."""
    named = []
    for field in UNSUPPORTED_FIELDS:
        if field not in body:
            continue
        value = body[field]
        if any(value == inert for inert in _INERT.get(field, (None,))):
            continue
        if field in ("tools", "functions") and not value:
            continue
        named.append(field)
    return named


__all__ = ["UNSUPPORTED_FIELDS", "unsupported_fields"]
