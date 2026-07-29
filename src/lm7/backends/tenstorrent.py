from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..detection import activate_tenstorrent_pjrt, tenstorrent_device_nodes
from ..errors import CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support

_INSTALL_HINT = (
    'install it with "uv pip install pjrt-plugin-tt '
    '--extra-index-url https://pypi.eng.aws.tenstorrent.com/" and run "tt-forge-install"'
)


class TenstorrentBackend:
    """Optional Tenstorrent backend powered by the tt-xla PJRT plugin and tt-mlir.

    `torch.compile(..., backend="tt")` hands the captured FX graph to tt-xla,
    which lowers it to StableHLO, compiles it with tt-mlir, and executes the
    result on Wormhole or Blackhole silicon through tt-metal.
    """

    name = "tenstorrent"

    def probe(self) -> BackendInfo:
        try:
            installed = importlib.util.find_spec("torch_plugin_tt") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            return BackendInfo(
                self.name,
                None,
                False,
                f"The Tenstorrent PJRT plugin is not installed; {_INSTALL_HINT}.",
            )
        version = _plugin_version()
        try:
            runtime = importlib.import_module("torch_xla.runtime")
            device_type = activate_tenstorrent_pjrt(runtime)
            device_count = runtime.addressable_device_count() if device_type == "TT" else 0
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
            return BackendInfo(
                self.name,
                version,
                False,
                f"tt-xla could not initialize a Tenstorrent PJRT runtime: {exc}",
            )
        if device_type != "TT":
            return BackendInfo(
                self.name,
                version,
                False,
                f"The Tenstorrent PJRT plugin is installed, but the PJRT device is "
                f"{device_type or 'unset'}, not TT. Set PJRT_DEVICE=TT.",
            )
        if device_count < 1:
            nodes = tenstorrent_device_nodes()
            detail = (
                f"tt-kmd published {', '.join(nodes)}, so check the tt-metal runtime"
                if nodes
                else "no /dev/tenstorrent node exists, so check the card and the tt-kmd driver"
            )
            return BackendInfo(
                self.name,
                version,
                False,
                f"tt-xla reported no addressable Tenstorrent device; {detail}.",
            )
        return BackendInfo(
            self.name,
            version,
            True,
            f"tt-xla found {device_count} addressable Tenstorrent device(s).",
        )

    def supports(self, request: CompileRequest) -> Support:
        probe = self.probe()
        if not probe.available:
            return Support(False, probe.reason)
        if request.target.vendor != "tenstorrent":
            return Support(False, "tt-xla supports Tenstorrent targets only in LM7.")
        return Support(
            True,
            "tt-xla provides the tt torch.compile backend for Tenstorrent inference.",
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
            compile_kwargs: dict[str, Any] = {"backend": "tt"}
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
                    "pjrt_plugin_tt_version": _plugin_version(),
                },
            )
        except Exception as exc:
            raise CompilationError(
                f"Compilation stage failed for target {request.target} with backend tenstorrent: "
                f"{exc}. Verify the Tenstorrent PJRT runtime or use fallback='warn'."
            ) from exc

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        assert artifact.callable is not None
        return artifact.callable


def _plugin_version() -> str | None:
    try:
        return importlib.metadata.version("pjrt-plugin-tt")
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
