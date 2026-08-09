"""An OpenAI-compatible HTTP endpoint over LM7's compiled decode loop.

    lm7 model serve hf://HuggingFaceTB/SmolLM2-135M-Instruct --target auto

This is a local, single-user server. It holds one model, one pair of compiled
graphs and one static KV cache, and it serves one request at a time behind a
lock. It has no continuous batching, no paged attention and no prefix caching,
because implementing those would mean writing a serving engine -- and LM7 does
not write compilers either. When throughput matters, ``--backend vllm`` hands
the port to vLLM and steps out of the request path entirely.

What it is for is the other case: a model on whatever hardware is in front of
you, routed through the same ``target``/``backend`` matrix as ``lm7.compile``,
answering the API every local client already speaks. See docs/serving.md.

Nothing here is imported by ``import lm7``. FastAPI, Uvicorn and Pydantic live
behind the optional ``serve`` extra, and this package's submodules import them
only where they are used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .engine import LM7ServeEngine, ServeConfig, TokenDelta

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from fastapi import FastAPI


def build_app(engine: LM7ServeEngine) -> FastAPI:
    """The FastAPI application for an already-loaded engine.

    A function rather than a re-export so that ``import lm7.serve`` works without
    FastAPI installed: ``serve.server`` imports it at module scope, deliberately,
    and is therefore imported here only when asked for.
    """
    from .server import build_app as _build_app

    return _build_app(engine)


def run_server(config: ServeConfig, engine: LM7ServeEngine | None = None) -> None:
    """Load the model if needed and block in Uvicorn until interrupted."""
    from .server import run_server as _run_server

    _run_server(config, engine)


__all__ = [
    "LM7ServeEngine",
    "ServeConfig",
    "TokenDelta",
    "build_app",
    "run_server",
]
