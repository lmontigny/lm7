"""Pydantic models for the subset of the OpenAI API this server implements.

Deliberately a subset, and deliberately strict about it. Everything named here
is implemented; a request that asks for anything else is refused by
:mod:`lm7.serve.validation` rather than quietly served as something narrower.

The response models exist for the same reason as the request models: FastAPI
generates ``/docs`` from them, so the schema *is* the statement of what these
endpoints return.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str


class ChatCompletionRequest(BaseModel):
    """``POST /v1/chat/completions``.

    ``model`` is accepted and ignored beyond being echoed back: this server holds
    exactly one model, loaded and compiled at startup, and swapping it would mean
    recompiling both graphs and reallocating the KV cache.
    """

    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None


class CompletionRequest(BaseModel):
    """``POST /v1/completions``, the pre-chat endpoint some local clients still use."""

    prompt: str
    model: str | None = None
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class ChatCompletionDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(BaseModel):
    """One Server-Sent Event of a streamed chat completion."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Literal["stop", "length"] | None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class CompletionChunk(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "lm7"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model: str
    target: str
    backend: str


class MetricsResponse(BaseModel):
    """What a single-stream server can honestly report.

    No queue depth, no running/waiting split, no cache-utilization percentage:
    there is one request in flight at a time and one static cache that is fully
    allocated whether it is used or not.
    """

    model: str
    target: str
    backend: str
    requests: int
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    tpot_ms: float
    kv_cache_bytes: int
    max_model_len: int
    # False until the first request has compiled the graphs, which is why that
    # request is slow. A client can say "compiling" instead of looking hung.
    warm: bool
    # How many distinct prompt lengths the prefill graph has compiled for.
    prefill_lengths: int
    # Compiles triggered *after* warmup. Must stay 0: anything else means a
    # token caused a recompile, which is the regression the prefill/decode
    # split exists to prevent -- see docs/kv-cache-decode.md.
    steady_frames: int


__all__ = [
    "ChatCompletionChoice",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionDelta",
    "ChatCompletionMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "CompletionChoice",
    "CompletionChunk",
    "CompletionRequest",
    "CompletionResponse",
    "HealthResponse",
    "MetricsResponse",
    "ModelCard",
    "ModelList",
    "Usage",
]
