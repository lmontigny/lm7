from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..detection import torch_device
from ..errors import CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support


class InductorBackend:
    name = "inductor"

    def probe(self) -> BackendInfo:
        available = callable(getattr(torch, "compile", None))
        reason = (
            "torch.compile is available."
            if available
            else "This PyTorch build has no torch.compile."
        )
        return BackendInfo(self.name, torch.__version__, available, reason)

    def supports(self, request: CompileRequest) -> Support:
        probe = self.probe()
        if not probe.available:
            return Support(False, probe.reason)
        if request.target.vendor not in {"cpu", "nvidia", "amd", "intel"}:
            return Support(False, f"Inductor does not support target {request.target} in LM7 v0.1.")
        return Support(
            True, f"torch.compile supports {request.target.kind} execution.", priority=100
        )

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        try:
            device = torch_device(request.target)
            if request.transfers == "automatic":
                request.model.to(device)
            options = dict(request.options)
            compile_mode = options.pop("compile_mode", None)
            dynamic = options.pop("dynamic", None)
            compiled = torch.compile(
                request.model,
                backend="inductor",
                mode=compile_mode,
                dynamic=dynamic,
                options=options or None,
            )
            # torch.compile is lazy: the first call is part of compilation and must
            # remain inside this error boundary so configured fallback can work.
            warmup_args = _map_tensors(example_args, lambda tensor: tensor.to(device))
            warmup_kwargs = _map_tensors(example_kwargs, lambda tensor: tensor.to(device))
            with torch.inference_mode():
                compiled(*warmup_args, **warmup_kwargs)
            return Artifact(self.name, request.target, compiled, metadata={"compiled": True})
        except Exception as exc:
            raise CompilationError(
                f"Compilation stage failed for target {request.target} with backend inductor: {exc}. "
                "Try backend='eager' or fallback='warn'."
            ) from exc

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        assert artifact.callable is not None
        return artifact.callable


def _map_tensors(value: Any, fn: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_tensors(item, fn) for key, item in value.items()}
    return value
