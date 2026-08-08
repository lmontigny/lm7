from __future__ import annotations

from collections.abc import Sequence
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


def select_name(
    candidates: Sequence[tuple[str, Support]],
    requested: str,
    *,
    kind: str,
    subject: str,
) -> str:
    """Pick the highest-priority supported candidate, or say why none was.

    Shared by the compile planner and the serving planner: both rank the same
    ``Support`` and owe the caller the same three answers (you asked for one
    that does not exist, you asked for one that cannot run this, nothing can
    run this). ``kind`` is the noun for the message -- "backend" or "runtime".
    """
    by_name = dict(candidates)
    if requested != "auto":
        if requested not in by_name:
            raise BackendUnavailableError(
                f"Requested {kind} {requested!r} is not registered. "
                f"Available: {', '.join(by_name)}."
            )
        support = by_name[requested]
        if not support.supported:
            raise BackendUnavailableError(
                f"{kind.capitalize()} {requested} is unavailable for {subject}: {support.reason}"
            )
        return requested
    supported = [(name, support) for name, support in candidates if support.supported]
    if not supported:
        # The reasons are the answer here, not decoration: "nothing supports
        # this" sends the reader to the docs, whereas "eager declines because it
        # does not implement continuous batching" sends them to the flag they
        # passed.
        detail = "; ".join(f"{name} ({support.reason})" for name, support in candidates)
        raise BackendUnavailableError(f"No {kind} supports {subject}. {detail}")
    return min(supported, key=lambda item: (-item[1].priority, item[0]))[0]


def plan(
    request: CompileRequest, backend_name: str, registry: BackendRegistry
) -> tuple[Backend, Plan]:
    candidates = tuple(
        Candidate(backend.name, backend.supports(request)) for backend in registry.all()
    )
    selected = select_name(
        [(candidate.backend, candidate.support) for candidate in candidates],
        backend_name,
        kind="backend",
        subject=f"target {request.target}",
    )
    return registry.get(selected), Plan(selected, candidates)
