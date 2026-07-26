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


def compile(
    model: torch.nn.Module,
    target: str | None = None,
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


__all__ = ["backends", "clear_cache", "compile", "detect_targets", "explain", "version"]
