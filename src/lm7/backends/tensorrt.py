from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..detection import torch_device
from ..errors import CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support


class TensorRTBackend:
    """Optional NVIDIA TensorRT backend powered by Torch-TensorRT."""

    name = "tensorrt"

    def probe(self) -> BackendInfo:
        try:
            installed = importlib.util.find_spec("torch_tensorrt") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            return BackendInfo(
                self.name,
                None,
                False,
                'Torch-TensorRT is not installed; install LM7 with ".[tensorrt]".',
            )
        try:
            version = importlib.metadata.version("torch-tensorrt")
        except importlib.metadata.PackageNotFoundError:
            version = None
        if not torch.cuda.is_available():
            return BackendInfo(
                self.name,
                version,
                False,
                "Torch-TensorRT is installed, but CUDA is unavailable.",
            )
        return BackendInfo(
            self.name,
            version,
            True,
            "Torch-TensorRT and CUDA are available.",
        )

    def supports(self, request: CompileRequest) -> Support:
        probe = self.probe()
        if not probe.available:
            return Support(False, probe.reason)
        if request.target.vendor != "nvidia":
            return Support(False, "TensorRT supports NVIDIA GPU targets only.")
        return Support(
            True,
            "Torch-TensorRT supports NVIDIA execution; explicit selection is experimental.",
            priority=90,
        )

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        try:
            # Importing Torch-TensorRT registers its public ``tensorrt``
            # torch.compile backend. Keep this optional dependency lazy.
            torch_tensorrt = importlib.import_module("torch_tensorrt")
            device = torch_device(request.target)
            if request.transfers == "automatic":
                request.model.to(device)
            options = dict(request.options)
            dynamic = options.pop("dynamic", False)
            compiled = torch.compile(
                request.model,
                backend="tensorrt",
                dynamic=dynamic,
                options=options or None,
            )
            # torch.compile is lazy, so include the first call in the error
            # boundary and preserve LM7's configured fallback behavior.
            warmup_args = _map_tensors(example_args, lambda tensor: tensor.to(device))
            warmup_kwargs = _map_tensors(example_kwargs, lambda tensor: tensor.to(device))
            with torch.inference_mode():
                compiled(*warmup_args, **warmup_kwargs)
            return Artifact(
                self.name,
                request.target,
                compiled,
                metadata={
                    "compiled": True,
                    "torch_tensorrt_version": getattr(torch_tensorrt, "__version__", None),
                },
            )
        except Exception as exc:
            raise CompilationError(
                f"Compilation stage failed for target {request.target} with backend tensorrt: "
                f"{exc}. Try backend='inductor', backend='eager', or fallback='warn'."
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
