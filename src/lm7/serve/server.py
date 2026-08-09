"""The FastAPI application: OpenAI-compatible routes over one :class:`LM7ServeEngine`.

Deliberately without ``from __future__ import annotations``. FastAPI resolves
handler annotations with ``get_type_hints`` against module globals to work out
what is a body and what is a query parameter, and postponed annotations that
name anything not importable at module scope become required query parameters --
so every request fails with a 422 rather than anything that points at the cause.
Keeping FastAPI's imports at module scope here, and importing *this* module
lazily from everywhere else, is what lets the rest of ``lm7`` stay importable
without the ``serve`` extra installed.
"""

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .engine import LM7ServeEngine
from .schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    HealthResponse,
    MetricsResponse,
    ModelCard,
    ModelList,
    Usage,
)
from .validation import unsupported_fields

FinishReason = Literal["stop", "length"]

# The engine reports why it stopped in its own vocabulary; OpenAI's has two
# words. A client that hung up gets "stop" because there is nobody left to tell.
_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "cancelled": "stop",
}


def _finish(reason: str | None) -> FinishReason:
    return _FINISH_REASONS.get(reason or "", "stop")


def build_app(engine: LM7ServeEngine) -> FastAPI:
    """An application bound to one already-loaded engine.

    The engine is a constructor argument rather than a startup hook so that the
    process fails at ``lm7 model serve``, with the model's own error, instead of
    binding a port and then 500-ing every request.
    """
    app = FastAPI(
        title="lm7 serve",
        summary="An OpenAI-compatible endpoint over LM7's compiled static KV-cache decode loop.",
        version=_version(),
    )
    app.state.engine = engine

    def model_name(requested: str | None) -> str:
        # Echoed back if the client named one, because some SDKs assert that the
        # response's model matches the request's. This server holds exactly one.
        return requested or engine.model_id

    async def guard(http: Request, max_tokens: int, prompt: str) -> None:
        """Refuse, before a response type is chosen, anything that cannot be served."""
        named = unsupported_fields(await _raw_body(http))
        if named:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"lm7 serve does not implement {', '.join(named)}. It runs one sequence at a "
                    "time through a compiled decode loop; these fields would change the answer, "
                    "so they are refused rather than ignored."
                ),
            )
        try:
            engine.check_capacity(prompt, max_tokens)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def stream_chat(
        body: ChatCompletionRequest, prompt: str, http: Request
    ) -> AsyncIterator[str]:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        name = model_name(body.model)

        def chunk(delta: ChatCompletionDelta, finish: FinishReason | None) -> str:
            payload = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=name,
                choices=[ChatCompletionChunkChoice(delta=delta, finish_reason=finish)],
            )
            return _event(_chunk_payload(payload.model_dump(exclude_none=True)))

        # The role arrives in its own chunk with no content, which is what the
        # OpenAI streaming format specifies and what clients key their
        # "assistant is typing" state off.
        yield chunk(ChatCompletionDelta(role="assistant", content=""), None)
        async for token in engine.generate(
            prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            seed=body.seed,
            stop=body.stop,
            is_disconnected=http.is_disconnected,
        ):
            if token.finished:
                yield chunk(ChatCompletionDelta(), _finish(token.finish_reason))
            else:
                yield chunk(ChatCompletionDelta(content=token.text), None)
        yield "data: [DONE]\n\n"

    async def stream_completion(body: CompletionRequest, http: Request) -> AsyncIterator[str]:
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        name = model_name(body.model)

        def chunk(text: str, finish: FinishReason | None) -> str:
            payload = CompletionChunk(
                id=completion_id,
                created=created,
                model=name,
                choices=[CompletionChoice(text=text, finish_reason=finish)],
            )
            return _event(_chunk_payload(payload.model_dump(exclude_none=True)))

        async for token in engine.generate(
            body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            seed=body.seed,
            stop=body.stop,
            is_disconnected=http.is_disconnected,
        ):
            if token.finished:
                yield chunk("", _finish(token.finish_reason))
            else:
                yield chunk(token.text, None)
        yield "data: [DONE]\n\n"

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Answerable while a generation is in flight, which is the point.

        Every PyTorch call in the engine runs in a worker thread, so the event
        loop is free to serve this even mid-decode. If this ever starts timing
        out under load, something has gone back to blocking the loop.
        """
        return HealthResponse(model=engine.model_id, target=engine.target, backend=engine.backend)

    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics() -> MetricsResponse:
        return MetricsResponse(**engine.metrics_snapshot())

    @app.get("/v1/models", response_model=ModelList)
    async def models() -> ModelList:
        return ModelList(data=[ModelCard(id=engine.model_id)])

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, http: Request) -> Any:
        prompt = engine.apply_chat_template(body.messages)
        await guard(http, body.max_tokens, prompt)
        if body.stream:
            return StreamingResponse(
                stream_chat(body, prompt, http),
                media_type="text/event-stream",
                headers=_STREAM_HEADERS,
            )
        prompt_tokens = int(engine.encode(prompt).shape[-1])
        text, reason, generated = await engine.complete(
            prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            seed=body.seed,
            stop=body.stop,
            is_disconnected=http.is_disconnected,
        )
        return ChatCompletionResponse(
            model=model_name(body.model),
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(content=text),
                    finish_reason=_finish(reason),
                )
            ],
            usage=_usage(prompt_tokens, generated),
        )

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, http: Request) -> Any:
        await guard(http, body.max_tokens, body.prompt)
        if body.stream:
            return StreamingResponse(
                stream_completion(body, http),
                media_type="text/event-stream",
                headers=_STREAM_HEADERS,
            )
        prompt_tokens = int(engine.encode(body.prompt).shape[-1])
        text, reason, generated = await engine.complete(
            body.prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            seed=body.seed,
            stop=body.stop,
            is_disconnected=http.is_disconnected,
        )
        return CompletionResponse(
            model=model_name(body.model),
            choices=[CompletionChoice(text=text, finish_reason=_finish(reason))],
            usage=_usage(prompt_tokens, generated),
        )

    return app


def run_server(config: Any, engine: LM7ServeEngine | None = None) -> None:
    """Load the model if needed, then block in Uvicorn until interrupted.

    ``engine`` is a parameter so a caller that has already paid for the load --
    or a test -- does not pay for it twice.
    """
    import uvicorn

    if engine is None:
        engine = LM7ServeEngine.load(config)
    uvicorn.run(
        build_app(engine),
        host=config.host,
        port=config.port,
        # `warning`, not `info`: LM7's own startup line has already said what is
        # being served and where, and Uvicorn's banner repeats it less usefully.
        log_level="warning",
    )


# Both matter for streaming through anything that buffers by default: nginx
# reads the second, and the first stops an intermediary from serving a cached
# half-completion to the next caller.
_STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _chunk_payload(payload: dict) -> dict:
    """Match OpenAI's streaming shape, which is not simply "drop the nulls".

    OpenAI omits ``delta.role`` from every chunk after the first, but *keeps*
    ``finish_reason: null`` on every chunk before the last, and keeps ``delta``
    as an empty object on that last one. A plain ``exclude_none`` drops all
    three; this puts the two that belong back.
    """
    for choice in payload.get("choices", []):
        choice.setdefault("finish_reason", None)
        if "text" not in choice:
            choice.setdefault("delta", {})
    return payload


def _usage(prompt_tokens: int, completion_tokens: int) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


async def _raw_body(http: Request) -> dict:
    """The request body as the client sent it, extras included.

    Pydantic has already discarded unknown fields by the time a handler runs, so
    "did the caller ask for something we cannot do" can only be answered here.
    Starlette caches the body, so this does not consume the stream twice.
    """
    try:
        body = await http.json()
    except Exception:  # noqa: BLE001 - a body FastAPI already parsed; unreachable in practice
        return {}
    return body if isinstance(body, dict) else {}


def _version() -> str:
    from ..api import version

    return str(version())


__all__ = ["build_app", "run_server"]
