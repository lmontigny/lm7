from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..errors import CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support


class OpenXLABackend:
    """Optional Google TPU backend powered by PyTorch/XLA and OpenXLA."""

    name = "openxla"

    def probe(self) -> BackendInfo:
        try:
            installed = importlib.util.find_spec("torch_xla") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            return BackendInfo(
                self.name,
                None,
                False,
                'PyTorch/XLA is not installed; install LM7 with ".[openxla]" on a TPU VM.',
            )
        try:
            version = importlib.metadata.version("torch-xla")
        except importlib.metadata.PackageNotFoundError:
            version = None
        try:
            runtime = importlib.import_module("torch_xla.runtime")
            device_type = runtime.device_type()
            device_count = runtime.addressable_device_count() if device_type == "TPU" else 0
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
            return BackendInfo(
                self.name,
                version,
                False,
                f"PyTorch/XLA could not initialize a TPU runtime: {exc}",
            )
        if device_type != "TPU" or device_count < 1:
            return BackendInfo(
                self.name,
                version,
                False,
                f"PyTorch/XLA is installed, but the PJRT device is {device_type or 'unset'}, not TPU.",
            )
        return BackendInfo(
            self.name,
            version,
            True,
            f"PyTorch/XLA found {device_count} addressable TPU device(s).",
        )

    def supports(self, request: CompileRequest) -> Support:
        probe = self.probe()
        if not probe.available:
            return Support(False, probe.reason)
        if request.target.vendor != "tpu":
            return Support(False, "OpenXLA supports Google TPU targets only in LM7.")
        return Support(
            True,
            "PyTorch/XLA provides the OpenXLA torch.compile backend for TPU inference.",
            priority=100,
        )

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        try:
            torch_xla = importlib.import_module("torch_xla")
            device = torch_xla.device(request.target.ordinal)
            if request.transfers == "automatic":
                request.model.to(device)
            options = dict(request.options)
            dynamic = options.pop("dynamic", None)
            compile_kwargs: dict[str, Any] = {"backend": "openxla"}
            if dynamic is not None:
                compile_kwargs["dynamic"] = dynamic
            if options:
                compile_kwargs["options"] = options
            compiled = torch.compile(request.model, **compile_kwargs)
            warmup_args = _map_tensors(example_args, lambda tensor: tensor.to(device))
            warmup_kwargs = _map_tensors(example_kwargs, lambda tensor: tensor.to(device))
            # PyTorch/XLA requires tensor version counters while tracing, which
            # torch.inference_mode() disables. no_grad() is inference-safe here.
            with torch.no_grad():
                compiled(*warmup_args, **warmup_kwargs)
            torch_xla.sync(wait=True)
            return Artifact(
                self.name,
                request.target,
                compiled,
                metadata={
                    "compiled": True,
                    "torch_xla_version": getattr(torch_xla, "__version__", None),
                },
            )
        except Exception as exc:
            raise CompilationError(
                f"Compilation stage failed for target {request.target} with backend openxla: "
                f"{exc}. Verify the TPU PJRT runtime or use fallback='warn'."
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
