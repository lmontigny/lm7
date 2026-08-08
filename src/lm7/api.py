from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch

from .backends import registry
from .backends.base import CompileRequest
from .cache import clear_cache
from .detection import detect_targets, resolve_target
from .module import CompiledModule
from .planner import plan
from .serving import registry as serving_registry
from .serving.base import ServeRequest, ServerHandle
from .serving.planner import plan_serving
from .targets import TargetSpec


def compile(
    model: torch.nn.Module,
    target: str | TargetSpec | None = None,
    *,
    backend: str | None = None,
    mode: str = "lazy",
    transfers: str = "automatic",
    fallback: str | None = None,
    cache: bool = True,
    options: Mapping[str, Any] | None = None,
) -> CompiledModule:
    target = target or os.environ.get("LM7_TARGET", "auto")
    backend = backend or os.environ.get("LM7_BACKEND", "auto")
    fallback = fallback or os.environ.get("LM7_FALLBACK", "warn")
    if mode not in {"lazy", "eager"}:
        raise ValueError("mode must be 'lazy' or 'eager' in LM7 v0.1.")
    if transfers not in {"automatic", "explicit"}:
        raise ValueError("transfers must be 'automatic' or 'explicit'.")
    if fallback not in {"warn", "error"}:
        raise ValueError("fallback must be 'warn' or 'error'.")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if mode == "eager":
        backend = "eager"
    return CompiledModule(model, target, backend, mode, transfers, fallback, cache, options or {})


def explain(
    model: torch.nn.Module | None = None, target: str = "auto", backend: str = "auto"
) -> str:
    resolved = resolve_target(target)
    request = CompileRequest(
        model or torch.nn.Identity(), resolved, "lazy", "automatic", "warn", {}
    )
    _, selected_plan = plan(request, backend, registry)
    lines = [f"Selected {selected_plan.selected} for {resolved}", "", "Candidates:"]
    for candidate in selected_plan.candidates:
        status = "supported" if candidate.support.supported else "unavailable"
        lines.append(
            f"- {candidate.backend}: {status} (priority {candidate.support.priority}) — "
            f"{candidate.support.reason}"
        )
    return "\n".join(lines)


def serve(
    model: str,
    target: str | TargetSpec | None = None,
    *,
    runtime: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    dtype: str = "auto",
    max_model_len: int = 2048,
    max_num_seqs: int = 1,
    max_batched_tokens: int | None = None,
    tensor_parallel_size: int = 1,
    kv_cache_fraction: float | None = None,
    prefix_caching: bool = False,
    lora_adapters: tuple[str, ...] = (),
    speculative_model: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> ServerHandle:
    """Start an OpenAI-compatible server and return a handle to it.

    LM7 does not implement a serving engine. It resolves the target, checks the
    request against what each registered runtime can actually do, and hands the
    work to whichever one wins -- vLLM where it is installed, and LM7's own
    single-stream reference server otherwise.

    The handle is a context manager, so the server stops when the block exits.
    """
    request, runtime_name = _serve_request(
        model,
        target,
        runtime=runtime,
        host=host,
        port=port,
        dtype=dtype,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_batched_tokens=max_batched_tokens,
        tensor_parallel_size=tensor_parallel_size,
        kv_cache_fraction=kv_cache_fraction,
        prefix_caching=prefix_caching,
        lora_adapters=lora_adapters,
        speculative_model=speculative_model,
        options=options,
    )
    selected, _ = plan_serving(request, runtime_name, serving_registry)
    return selected.launch(request)


def _serve_request(
    model: str,
    target: str | TargetSpec | None,
    *,
    runtime: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    dtype: str = "auto",
    max_model_len: int = 2048,
    max_num_seqs: int = 1,
    max_batched_tokens: int | None = None,
    tensor_parallel_size: int = 1,
    kv_cache_fraction: float | None = None,
    prefix_caching: bool = False,
    lora_adapters: tuple[str, ...] = (),
    speculative_model: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> tuple[ServeRequest, str]:
    resolved = resolve_target(target or os.environ.get("LM7_TARGET", "auto"))
    request = ServeRequest(
        model=model,
        target=resolved,
        host=host,
        port=port,
        dtype=dtype,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_batched_tokens=max_batched_tokens,
        tensor_parallel_size=tensor_parallel_size,
        kv_cache_fraction=kv_cache_fraction,
        prefix_caching=prefix_caching,
        lora_adapters=tuple(lora_adapters),
        speculative_model=speculative_model,
        extra=options or {},
    )
    return request, runtime or os.environ.get("LM7_RUNTIME", "auto")


def explain_serving(model: str, target: str = "auto", runtime: str = "auto") -> str:
    request, runtime_name = _serve_request(model, target, runtime=runtime)
    _, selected_plan = plan_serving(request, runtime_name, serving_registry)
    lines = [f"Selected {selected_plan.selected} for {request.target}", "", "Candidates:"]
    for candidate in selected_plan.candidates:
        status = "supported" if candidate.support.supported else "unavailable"
        lines.append(
            f"- {candidate.runtime}: {status} (priority {candidate.support.priority}) — "
            f"{candidate.support.reason}"
        )
    return "\n".join(lines)


def runtimes() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "name": runtime.name,
            "available": (info := runtime.probe()).available,
            "version": info.version,
            "reason": info.reason,
            "capabilities": runtime.capabilities().to_dict(),
        }
        for runtime in serving_registry.all()
    )


def backends() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "name": backend.name,
            "available": (info := backend.probe()).available,
            "version": info.version,
            "reason": info.reason,
        }
        for backend in registry.all()
    )


def version() -> str:
    from . import __version__

    return __version__


__all__ = [
    "backends",
    "clear_cache",
    "compile",
    "detect_targets",
    "explain",
    "explain_serving",
    "runtimes",
    "serve",
    "version",
]
