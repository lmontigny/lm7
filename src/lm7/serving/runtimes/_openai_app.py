"""The OpenAI-compatible routes for LM7's reference runtime.

Deliberately without ``from __future__ import annotations`` and with FastAPI
imported at module scope. FastAPI resolves handler annotations with
``get_type_hints`` against module globals, so a postponed ``Request``
annotation naming a function-local import is unresolvable -- it becomes a
required query parameter and every request fails with a 422. Keeping this
module separate is what lets the annotations be real objects while
``eager.py`` still imports FastAPI lazily and stays importable without it.
"""

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


def _created() -> int:
    return int(time.time())


def build_app(server: Any) -> FastAPI:
    app = FastAPI(title="lm7 serve (reference runtime)")

    async def collect(prompt: str, max_tokens: int, http: Request) -> str:
        chunks = []
        async for delta, finished in server.generate(prompt, max_tokens, http.is_disconnected):
            if finished:
                break
            chunks.append(delta)
        return "".join(chunks)

    async def stream(
        prompt: str, max_tokens: int, http: Request, chat: bool, model: str
    ) -> AsyncIterator[str]:
        completion_id = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}"
        async for delta, finished in server.generate(prompt, max_tokens, http.is_disconnected):
            if chat:
                choice: dict[str, Any] = (
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                    if finished
                    else {"index": 0, "delta": {"content": delta}, "finish_reason": None}
                )
            else:
                choice = {
                    "index": 0,
                    "text": "" if finished else delta,
                    "finish_reason": "stop" if finished else None,
                }
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk" if chat else "text_completion",
                "created": _created(),
                "model": model,
                "choices": [choice],
            }
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return {
            "runtime": server.runtime_name,
            "model": server.model_id,
            "memory": server.budget.to_dict(),
            "compilation": server.compilation(),
            **server.metrics.to_dict(),
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": server.model_id, "object": "model", "owned_by": "lm7"}],
        }

    @app.post("/v1/completions")
    async def completions(body: dict[str, Any], http: Request) -> Any:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise HTTPException(status_code=400, detail="'prompt' must be a string.")
        max_tokens = int(body.get("max_tokens", 16))
        try:
            # Checked here, before the response type is chosen, so an oversized
            # request is a 400 and not a stream that dies mid-flight.
            server.check_capacity(prompt, max_tokens)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if body.get("stream"):
            return StreamingResponse(
                stream(prompt, max_tokens, http, False, server.model_id),
                media_type="text/event-stream",
            )
        text = await collect(prompt, max_tokens, http)
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": _created(),
            "model": server.model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict[str, Any], http: Request) -> Any:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="'messages' must be a non-empty list.")
        prompt = server.apply_chat_template(messages)
        max_tokens = int(body.get("max_tokens", 16))
        try:
            # Checked here, before the response type is chosen, so an oversized
            # request is a 400 and not a stream that dies mid-flight.
            server.check_capacity(prompt, max_tokens)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if body.get("stream"):
            return StreamingResponse(
                stream(prompt, max_tokens, http, True, server.model_id),
                media_type="text/event-stream",
            )
        text = await collect(prompt, max_tokens, http)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": _created(),
            "model": server.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    return app
