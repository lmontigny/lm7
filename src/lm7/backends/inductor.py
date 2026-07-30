from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..detection import torch_device
from ..errors import CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support

# TorchInductor generates GPU kernels with Triton, and Triton's Intel GPU code
# generator is out-of-tree: it ships as the separate `pytorch-triton-xpu` wheel
# and registers itself as the "intel" entry in Triton's backend registry. A
# stock `triton` has no such entry, so torch.compile raises inside the first
# call -- which the default fallback="warn" turns into a quiet eager run on a
# machine the user picked for its GPU. LM7 checks up front instead.
_XPU_TRITON_BACKEND = "intel"


def triton_backends() -> frozenset[str] | None:
    """Vendor keys Triton has a code generator for, or ``None`` if unreadable.

    ``None`` means "cannot tell", which is deliberately different from an empty
    set: an unreadable registry must never be reported as a missing backend.
    """
    try:
        registry = importlib.import_module("triton.backends").backends
        return frozenset(registry)
    except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError):
        return None


def _triton_version() -> str | None:
    try:
        return str(importlib.import_module("triton").__version__)
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        return None


def _triton_summary() -> str:
    """One sentence on Triton's state, for `lm7 doctor`."""
    version = _triton_version()
    if version is None:
        return "Triton is not installed, so GPU kernel generation is unavailable."
    names = triton_backends()
    if names is None:
        return f"Triton {version} is installed."
    return f"Triton {version} generates for {', '.join(sorted(names)) or 'nothing'}."


def _xpu_triton_reason() -> str | None:
    """Why Inductor cannot reach an Intel GPU here, or ``None`` if it can."""
    try:
        installed = importlib.util.find_spec("triton") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if not installed:
        return (
            "TorchInductor generates Intel GPU kernels with Triton, and no triton "
            "package is installed. Install the Intel build with "
            '"pip install pytorch-triton-xpu", or use backend="eager".'
        )
    names = triton_backends()
    if names is None or _XPU_TRITON_BACKEND in names:
        return None
    return (
        "The installed Triton has no Intel GPU code generator (it registers: "
        f"{', '.join(sorted(names)) or 'none'}). That backend is out-of-tree and "
        'ships as pytorch-triton-xpu; install it with "pip install '
        'pytorch-triton-xpu", or use backend="eager". See docs/intel-gpu.md.'
    )


class InductorBackend:
    name = "inductor"

    def probe(self) -> BackendInfo:
        available = callable(getattr(torch, "compile", None))
        reason = (
            "torch.compile is available."
            if available
            else "This PyTorch build has no torch.compile."
        )
        if available:
            # `lm7 doctor` is where someone looks after a fallback warning, so
            # the kernel generator's own state belongs in the report.
            reason += f" {_triton_summary()}"
        return BackendInfo(self.name, torch.__version__, available, reason)

    def supports(self, request: CompileRequest) -> Support:
        probe = self.probe()
        if not probe.available:
            return Support(False, probe.reason)
        if request.target.vendor not in {"cpu", "nvidia", "amd", "intel", "apple"}:
            return Support(False, f"Inductor does not support target {request.target} in LM7 v0.1.")
        if request.target.kind == "npu":
            # torch.compile needs a torch device to lower to, and there is no
            # NPU one. Claiming this target would silently compile for the CPU.
            return Support(
                False,
                "PyTorch has no NPU device for TorchInductor to lower to; "
                f"{request.target} is reached through backend='openvino'.",
            )
        if request.target.vendor == "intel" and request.target.kind == "gpu":
            xpu_reason = _xpu_triton_reason()
            if xpu_reason is not None:
                return Support(False, xpu_reason)
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
