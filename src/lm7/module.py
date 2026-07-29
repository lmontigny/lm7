from __future__ import annotations

import logging
import threading
import warnings
from collections.abc import Mapping
from typing import Any

import torch

from .backends import registry
from .backends.base import Artifact, CompileRequest
from .cache import input_signature, memory_cache
from .detection import resolve_target, torch_device
from .errors import CompilationError, InputDeviceError
from .planner import Plan, plan
from .targets import TargetSpec

logger = logging.getLogger("lm7")


def _map_tensors(value: Any, fn: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_map_tensors(v, fn) for v in value))
    if isinstance(value, tuple):
        return tuple(_map_tensors(v, fn) for v in value)
    if isinstance(value, list):
        return [_map_tensors(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _map_tensors(v, fn) for k, v in value.items()}
    return value


class CompiledModule(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        target: str | TargetSpec,
        backend: str,
        mode: str,
        transfers: str,
        fallback: str,
        cache: bool,
        options: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.requested_target = target
        self.backend_override = backend
        self.mode = mode
        self.transfers = transfers
        self.fallback = fallback
        self.cache_enabled = cache
        self.options = dict(options)
        self.state = "uncompiled"
        self.selected_backend: str | None = None
        self.target: TargetSpec | None = None
        self.plan: Plan | None = None
        # Retained so callers can find a backend's on-disk artifact, such as the
        # OpenVINO IR or the AOTInductor package, after the first compile.
        self.artifact: Artifact | None = None
        self._variants: dict[tuple[Any, ...], Any] = {}
        self._compile_lock = threading.RLock()
        if self.model.training:
            warnings.warn(
                "LM7 is inference-first; compiling a model in training mode.", stacklevel=2
            )

    def _prepare_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any]:
        assert self.target is not None
        device = torch_device(self.target)
        if self.transfers == "automatic":
            return _map_tensors(args, lambda x: x.to(device)), _map_tensors(
                kwargs, lambda x: x.to(device)
            )

        def validate(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.device != device:
                raise InputDeviceError(
                    f"Input transfer stage failed for target {self.target}: expected {device}, "
                    f"got {tensor.device}. Move inputs explicitly or use transfers='automatic'."
                )
            return tensor

        return _map_tensors(args, validate), _map_tensors(kwargs, validate)

    def _compile_variant(
        self, signature: tuple[Any, ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        self.target = resolve_target(self.requested_target)
        request = CompileRequest(
            self.model,
            self.target,
            self.mode,
            self.transfers,
            self.fallback,
            self.options,
        )
        backend, selected_plan = plan(request, self.backend_override, registry)
        self.plan = selected_plan
        self.selected_backend = backend.name
        logger.info("Selected backend %s for target %s", backend.name, self.target)
        try:
            artifact: Artifact = backend.compile(request, args, kwargs)
        except CompilationError:
            if self.fallback == "error" or backend.name == "eager":
                self.state = "failed"
                raise
            warnings.warn(
                f"Backend {backend.name} compilation failed for {self.target}; "
                "falling back to PyTorch eager.",
                RuntimeWarning,
                stacklevel=2,
            )
            logger.warning("Falling back from %s to eager", backend.name)
            backend = registry.get("eager")
            artifact = backend.compile(request, args, kwargs)
            self.selected_backend = "eager"
        callable_variant = backend.load(artifact)
        self.artifact = artifact
        self._variants[signature] = callable_variant
        if self.cache_enabled:
            memory_cache.put(
                (id(self.model), signature, str(self.target), backend.name), callable_variant
            )
        self.state = "compiled"
        return callable_variant

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        signature = input_signature(args, kwargs)
        variant = self._variants.get(signature)
        if variant is None:
            with self._compile_lock:
                variant = self._variants.get(signature)
                if variant is None:
                    variant = self._compile_variant(signature, args, kwargs)
        prepared_args, prepared_kwargs = self._prepare_inputs(args, kwargs)
        context = (
            torch.no_grad()
            if self.target is not None and self.target.vendor == "tpu"
            else torch.inference_mode()
        )
        with context:
            return variant(*prepared_args, **prepared_kwargs)
