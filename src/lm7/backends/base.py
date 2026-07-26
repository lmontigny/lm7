from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from ..targets import TargetSpec


@dataclass(frozen=True)
class BackendInfo:
    name: str
    version: str | None
    available: bool
    reason: str


@dataclass(frozen=True)
class CompileRequest:
    model: torch.nn.Module
    target: TargetSpec
    mode: str
    transfers: str
    fallback: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Support:
    supported: bool
    reason: str
    priority: int = 0


@dataclass
class Artifact:
    backend: str
    target: TargetSpec
    callable: Callable[..., Any] | None = None
    path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def probe(self) -> BackendInfo: ...
    def supports(self, request: CompileRequest) -> Support: ...
    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact: ...
    def load(self, artifact: Artifact) -> Callable[..., Any]: ...
