from __future__ import annotations

from .base import ServingRuntime


class RuntimeRegistry:
    def __init__(self) -> None:
        self._runtimes: dict[str, ServingRuntime] = {}

    def register(self, runtime: ServingRuntime) -> None:
        if runtime.name in self._runtimes:
            raise ValueError(f"Serving runtime {runtime.name!r} is already registered.")
        self._runtimes[runtime.name] = runtime

    def get(self, name: str) -> ServingRuntime:
        return self._runtimes[name]

    def all(self) -> tuple[ServingRuntime, ...]:
        return tuple(self._runtimes[name] for name in sorted(self._runtimes))
