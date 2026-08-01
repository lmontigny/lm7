from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..errors import CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support

# Below `inductor` (100) and every export backend, so `backend="auto"` never
# selects it, and above `eager`/`tvm` (0), because it is a real optimizing
# compiler whose numerics matched Inductor's exactly. Kept as a named constant so
# the value and the reason string cannot drift apart -- see `supports()`.
ZENTORCH_PRIORITY = 50

# zentorch registers its Dynamo backend as an import side effect, so the module
# has to be imported before torch.compile can resolve the name. Importing it is
# the whole integration: LM7 adds no kernels and no configuration of its own.
_BACKEND_NAME = "zentorch"


class ZenTorchBackend:
    """AMD's ZenDNN PyTorch extension, as a torch.compile backend for AMD CPUs.

    This is the AMD-CPU counterpart to what `openvino` is for Intel CPUs: a
    vendor's own CPU compiler, reachable by name. Like that one it is opt-in and
    never chosen implicitly, and for the same kind of reason -- on the one part
    LM7 has measured it did not beat TorchInductor. See docs/zentorch.md.
    """

    name = "zentorch"

    def probe(self) -> BackendInfo:
        version = _package_version()
        if importlib.util.find_spec("zentorch") is None:
            return BackendInfo(
                self.name,
                version,
                False,
                'zentorch is not installed; install LM7 with ".[zentorch]" on an AMD '
                "EPYC host. It ships x86-64 Linux wheels only.",
            )
        # zentorch is a Dynamo backend, so a PyTorch without torch.compile cannot
        # reach it however well the package itself imports.
        if not callable(getattr(torch, "compile", None)):
            return BackendInfo(
                self.name, version, False, "This PyTorch build has no torch.compile."
            )
        return BackendInfo(
            self.name, version, True, "zentorch provides a ZenDNN torch.compile backend."
        )

    def supports(self, request: CompileRequest) -> Support:
        probe = self.probe()
        if not probe.available:
            return Support(False, probe.reason)
        if request.target.vendor != "cpu":
            # "amd" in an LM7 target means the ROCm GPU. zentorch is a CPU
            # extension and shares nothing with that path, so it must not claim
            # a target whose name merely looks like AMD's.
            return Support(
                False,
                "zentorch is an AMD CPU extension and compiles for cpu targets only; "
                f"{request.target} is a GPU target reached through inductor.",
            )
        return Support(
            True,
            "zentorch compiles through ZenDNN for AMD CPUs; select it explicitly, as "
            "Inductor outranks it by default. See docs/zentorch.md.",
            priority=ZENTORCH_PRIORITY,
        )

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        options = dict(request.options)
        dynamic = options.pop("dynamic", None)
        if options:
            # Inductor's `options` are inductor config keys, which mean nothing to
            # another Dynamo backend. Refusing beats forwarding them and having
            # them silently ignored. Checked before the import so a caller error
            # is reported as one rather than as a missing package.
            unexpected = ", ".join(sorted(options))
            raise CompilationError(
                f"The zentorch backend accepts the 'dynamic' option only; got: "
                f"{unexpected}. Inductor's compile options do not carry over to it."
            )
        try:
            importlib.import_module("zentorch")
            device = torch.device("cpu")
            if request.transfers == "automatic":
                request.model.to(device)
            compiled = torch.compile(request.model, backend=_BACKEND_NAME, dynamic=dynamic)
            # torch.compile is lazy: the first call is part of compilation and must
            # remain inside this error boundary so configured fallback can work.
            warmup_args = _map_tensors(example_args, lambda tensor: tensor.to(device))
            warmup_kwargs = _map_tensors(example_kwargs, lambda tensor: tensor.to(device))
            with torch.inference_mode():
                compiled(*warmup_args, **warmup_kwargs)
            return Artifact(self.name, request.target, compiled, metadata={"compiled": True})
        except Exception as exc:
            raise CompilationError(
                f"Compilation stage failed for target {request.target} with backend "
                f"zentorch: {exc}. Try backend='inductor' or fallback='warn'."
            ) from exc

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        assert artifact.callable is not None
        return artifact.callable


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("zentorch")
    except importlib.metadata.PackageNotFoundError:
        return None


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
