from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..errors import CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support

# XLA lowers an fp32 matmul to bf16 passes on TPU unless told otherwise, so a
# model that matches CPU eager to 1e-6 on every other target matches it to only
# ~1e-3 here. The three settings are XLA's own; measured on a TPU v6e, a bare
# matmul lands at 3.6e-02, 1.6e-04, and 1.9e-06 respectively. LM7 surfaces the
# choice rather than picking for the caller -- see docs/google-tpu.md.
MAT_MUL_PRECISIONS = ("default", "high", "highest")


def _apply_mat_mul_precision(value: Any) -> str:
    """Set XLA's fp32 matmul precision, refusing to do so once it is too late.

    ``set_mat_mul_precision`` is process-global and XLA reads it while lowering
    the first computation. Calling it afterwards still updates what
    ``get_mat_mul_precision`` reports while the numerics stay on the old
    setting, so the getter cannot be used to confirm the change took. A silent
    no-op is the one outcome LM7 must not produce for an accuracy control, so
    this checks whether anything has executed yet and says so instead.
    """
    if value not in MAT_MUL_PRECISIONS:
        raise CompilationError(
            f"Unsupported mat_mul_precision {value!r}; expected one of "
            f"{', '.join(MAT_MUL_PRECISIONS)}."
        )
    metrics = importlib.import_module("torch_xla.debug.metrics")
    if metrics.counter_value("ExecuteComputation") is not None:
        raise CompilationError(
            f"mat_mul_precision={value!r} cannot be applied: this process has already "
            "run an XLA computation, and the setting is process-global and read once. "
            "Set it before the first compile or execution in the process, for example "
            "by making this the first lm7.compile call."
        )
    backends = importlib.import_module("torch_xla.backends")
    backends.set_mat_mul_precision(value)
    return str(value)


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
            options = dict(request.options)
            # Applied before anything touches the device, because the first XLA
            # computation freezes it for the rest of the process.
            precision = options.pop("mat_mul_precision", None)
            applied_precision = (
                _apply_mat_mul_precision(precision) if precision is not None else None
            )
            device = torch_xla.device(request.target.ordinal)
            if request.transfers == "automatic":
                request.model.to(device)
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
                    "mat_mul_precision": applied_precision,
                },
            )
        except CompilationError:
            # Already a precise diagnosis; re-wrapping would bury it.
            raise
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
