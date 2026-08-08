from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ..backends.base import Support
from ..targets import TargetSpec


@dataclass(frozen=True)
class RuntimeInfo:
    """The ``BackendInfo`` analogue: can this runtime run here, and why not."""

    name: str
    version: str | None
    available: bool
    reason: str


@dataclass(frozen=True)
class Capabilities:
    """What a serving runtime actually implements.

    This has no counterpart in ``lm7.backends`` and is the reason the serving
    layer earns its own protocol. A compile backend either compiles a model or
    declines it; a serving runtime can accept a model and silently ignore half
    the request. Asking for LoRA adapters from a runtime that serves none must
    be a refusal, not a flag that evaporates -- so the constraints LM7 accepts
    are checked against this before anything is loaded.
    """

    continuous_batching: bool = False
    paged_kv_cache: bool = False
    prefix_caching: bool = False
    chunked_prefill: bool = False
    speculative_decoding: bool = False
    lora: bool = False
    streaming: bool = False
    cancellation: bool = False
    metrics: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ServeRequest:
    """A model, a target, and the constraints the runtime has to honour.

    Deliberately not a superset of every engine's flags. Anything LM7 does not
    model itself goes through ``extra`` untouched and unvalidated, which keeps
    this dataclass from growing a field per vLLM release.
    """

    model: str
    target: TargetSpec
    host: str = "127.0.0.1"
    port: int = 8000
    dtype: str = "auto"
    # Which LM7 compile backend the built-in runtime drives its decode graph
    # with. Only that runtime honours it: a third-party engine compiles
    # internally and is handed a checkpoint, not a compiled module, so a
    # runtime that cannot act on this refuses rather than ignoring it.
    compile_backend: str = "auto"
    max_model_len: int = 2048
    max_num_seqs: int = 1
    max_batched_tokens: int | None = None
    tensor_parallel_size: int = 1
    kv_cache_fraction: float | None = None
    prefix_caching: bool = False
    lora_adapters: tuple[str, ...] = ()
    speculative_model: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def requested_capabilities(self) -> tuple[str, ...]:
        """The capability names this request needs a runtime to actually have.

        Only constraints that were asked for appear here. A request that never
        mentions LoRA does not require a runtime to serve adapters, so the
        fallback runtime stays usable for the common case.
        """
        required: list[str] = []
        if self.prefix_caching:
            required.append("prefix_caching")
        if self.lora_adapters:
            required.append("lora")
        if self.speculative_model is not None:
            required.append("speculative_decoding")
        if self.max_batched_tokens is not None:
            required.append("chunked_prefill")
        if self.max_num_seqs > 1:
            required.append("continuous_batching")
        return tuple(required)


@dataclass
class ServerHandle:
    """A running server: where it is, what chose it, and how to stop it."""

    runtime: str
    base_url: str
    target: TargetSpec
    config: Mapping[str, Any] = field(default_factory=dict)
    _stop: Any = None

    def stop(self) -> None:
        if self._stop is not None:
            self._stop()

    # `Self` would need typing_extensions on the 3.10 this project still
    # supports, and ServerHandle is not subclassed.
    def __enter__(self) -> ServerHandle:  # noqa: PYI034
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


class ServingRuntime(Protocol):
    name: str

    def probe(self) -> RuntimeInfo: ...
    def capabilities(self) -> Capabilities: ...
    def supports(self, request: ServeRequest) -> Support: ...
    def describe(self, request: ServeRequest) -> Mapping[str, Any]: ...
    def launch(self, request: ServeRequest) -> ServerHandle: ...


def unmet_capabilities(request: ServeRequest, capabilities: Capabilities) -> tuple[str, ...]:
    """Requested capabilities the runtime does not have."""
    have = capabilities.to_dict()
    return tuple(name for name in request.requested_capabilities() if not have.get(name, False))
