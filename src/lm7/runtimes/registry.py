from __future__ import annotations

from .base import Runtime
from .tensorrt_llm import TensorRTLLMRuntime


class RuntimeRegistry:
    """Mirrors `BackendRegistry`, deliberately.

    Runtimes are not ranked by priority the way backends are. A backend is chosen
    for you when you say `backend="auto"`; a serving runtime is an explicit
    decision with its own dependency set and its own operational behaviour, so
    `lm7 serve` requires `--runtime` rather than guessing.
    """

    def __init__(self) -> None:
        self._runtimes: dict[str, Runtime] = {}

    def register(self, runtime: Runtime) -> None:
        if runtime.name in self._runtimes:
            raise ValueError(f"Runtime {runtime.name!r} is already registered.")
        self._runtimes[runtime.name] = runtime

    def get(self, name: str) -> Runtime:
        if name not in self._runtimes:
            known = ", ".join(sorted(self._runtimes)) or "none"
            raise KeyError(f"Unknown runtime {name!r}; registered: {known}.")
        return self._runtimes[name]

    def all(self) -> tuple[Runtime, ...]:
        return tuple(self._runtimes[name] for name in sorted(self._runtimes))


registry = RuntimeRegistry()
registry.register(TensorRTLLMRuntime())


def inspect_runtimes() -> tuple[dict[str, object], ...]:
    """What `lm7 runtimes` and `lm7 doctor --json` report."""
    return tuple(
        {
            "name": info.name,
            "version": info.version,
            "available": info.available,
            "reason": info.reason,
            "pinned": dict(info.pinned),
        }
        for info in (runtime.probe() for runtime in registry.all())
    )
