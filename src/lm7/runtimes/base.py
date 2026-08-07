"""Serving runtimes, which are deliberately not `Backend`s.

A `Backend` compiles one module and hands back something callable:
`probe`/`supports`/`compile`/`load`, and the caller drives it. That shape cannot
express continuous batching, because scheduling is a property of a *server*
holding many in-flight requests, not of a callable one caller invokes.

So a runtime is a separate concept with its own protocol, and the split of
responsibility is explicit: LM7 owns target resolution, runtime selection,
dependency checks, configuration, engine-cache metadata, and measurement. The
runtime owns attention kernels, KV-cache management, the batch scheduler, the
decode loop, and engine execution. Anything in the second list that starts
appearing in this package is a bug in the boundary.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ..targets import TargetSpec


@dataclass(frozen=True)
class RuntimeInfo:
    """Whether a runtime can run here, and if not, what is missing.

    Mirrors `BackendInfo`. `reason` is user-facing and should name the install
    step, because a serving runtime's dependency set is large enough that "not
    available" on its own is unactionable.
    """

    name: str
    version: str | None
    available: bool
    reason: str
    # Versions an engine is pinned to. Recorded even when unavailable, so a
    # `doctor` on a machine without the runtime still reports what it would need.
    pinned: Mapping[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class ServeConfig:
    """Everything LM7 decides before the runtime is handed control.

    Deliberately small and declarative: these are the knobs that change what
    engine gets built, which is why they are also what the cache key is computed
    from. Runtime-internal tuning does not belong here.
    """

    dtype: str = "bfloat16"
    max_batch_size: int = 8
    max_input_len: int = 1024
    max_output_len: int = 256
    # Fraction of *free* device memory the runtime may hold for paged KV cache.
    # The runtime owns the paging; LM7 owns the budget it is given.
    kv_cache_free_gpu_memory_fraction: float = 0.85
    quantization: str = "none"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeSupport:
    supported: bool
    reason: str


@dataclass(frozen=True)
class GenerationChunk:
    """One streamed step.

    `text` is the delta rather than the running total, because a caller that
    wants the total can accumulate and one that wants to print cannot subtract.
    """

    text: str
    token_id: int | None = None
    finished: bool = False


class Runtime(Protocol):
    name: str

    def probe(self) -> RuntimeInfo: ...
    def supports(
        self, target: TargetSpec, model_id: str, config: ServeConfig
    ) -> RuntimeSupport: ...
    def prepare(self, target: TargetSpec, model_id: str, config: ServeConfig) -> Any: ...
    def generate(
        self, prepared: Any, prompt: str, *, max_new_tokens: int
    ) -> Iterator[GenerationChunk]: ...
