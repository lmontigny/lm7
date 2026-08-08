from __future__ import annotations

from dataclasses import dataclass

from ..backends.base import Support
from ..planner import select_name
from .base import ServeRequest, ServingRuntime
from .registry import RuntimeRegistry


@dataclass(frozen=True)
class RuntimeCandidate:
    runtime: str
    support: Support


@dataclass(frozen=True)
class ServePlan:
    selected: str
    candidates: tuple[RuntimeCandidate, ...]


def plan_serving(
    request: ServeRequest, runtime_name: str, registry: RuntimeRegistry
) -> tuple[ServingRuntime, ServePlan]:
    candidates = tuple(
        RuntimeCandidate(runtime.name, runtime.supports(request)) for runtime in registry.all()
    )
    selected = select_name(
        [(candidate.runtime, candidate.support) for candidate in candidates],
        runtime_name,
        kind="runtime",
        subject=f"target {request.target}",
    )
    return registry.get(selected), ServePlan(selected, candidates)
