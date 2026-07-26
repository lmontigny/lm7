from __future__ import annotations

from dataclasses import dataclass

from .backends.base import Backend, CompileRequest, Support
from .backends.registry import BackendRegistry
from .errors import BackendUnavailableError


@dataclass(frozen=True)
class Candidate:
    backend: str
    support: Support


@dataclass(frozen=True)
class Plan:
    selected: str
    candidates: tuple[Candidate, ...]


def plan(
    request: CompileRequest, backend_name: str, registry: BackendRegistry
) -> tuple[Backend, Plan]:
    candidates = tuple(
        Candidate(backend.name, backend.supports(request)) for backend in registry.all()
    )
    by_name = {candidate.backend: candidate for candidate in candidates}
    if backend_name != "auto":
        if backend_name not in by_name:
            raise BackendUnavailableError(
                f"Requested backend {backend_name!r} is not registered. "
                f"Available: {', '.join(by_name)}."
            )
        candidate = by_name[backend_name]
        if not candidate.support.supported:
            raise BackendUnavailableError(
                f"Backend {backend_name} is unavailable for target {request.target}: "
                f"{candidate.support.reason}"
            )
        return registry.get(backend_name), Plan(backend_name, candidates)
    supported = [c for c in candidates if c.support.supported]
    if not supported:
        raise BackendUnavailableError(f"No backend supports target {request.target}.")
    selected = min(supported, key=lambda c: (-c.support.priority, c.backend))
    return registry.get(selected.backend), Plan(selected.backend, candidates)
