from __future__ import annotations

from .base import Backend


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {}

    def register(self, backend: Backend) -> None:
        if backend.name in self._backends:
            raise ValueError(f"Backend {backend.name!r} is already registered.")
        self._backends[backend.name] = backend

    def get(self, name: str) -> Backend:
        return self._backends[name]

    def all(self) -> tuple[Backend, ...]:
        return tuple(self._backends[name] for name in sorted(self._backends))
