from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..errors import ArtifactLoadError, CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support

_REQUIRED_MODULES = {
    "litert_torch": "litert-torch",
    "ai_edge_litert": "ai-edge-litert",
}


@dataclass(frozen=True)
class LiteRTOptions:
    strict_export: str | bool
    lightweight_conversion: bool
    enable_x64: bool
    runtime_constant_folding: bool | None

    @property
    def converter_options(self) -> Mapping[str, Any]:
        return {
            "strict_export": self.strict_export,
            "lightweight_conversion": self.lightweight_conversion,
            "enable_x64": self.enable_x64,
            "runtime_constant_folding": self.runtime_constant_folding,
        }


class LiteRTBackend:
    """Export-only PyTorch-to-LiteRT backend using the CPU/XNNPACK runtime."""

    name = "litert"

    def probe(self) -> BackendInfo:
        missing = [
            package for module, package in _REQUIRED_MODULES.items() if not _has_module(module)
        ]
        if missing:
            if "litert-torch" in missing and _is_linux_aarch64():
                return BackendInfo(
                    self.name,
                    None,
                    False,
                    "LiteRT export support is not currently installable on Linux aarch64: "
                    "litert-torch depends on litert-converter==0.3.*, which has no "
                    "Linux aarch64 distribution. Export on a supported host and run "
                    "the resulting .tflite artifact with a device LiteRT runtime.",
                )
            return BackendInfo(
                self.name,
                None,
                False,
                'LiteRT export support is not installed; install LM7 with ".[litert]". '
                f"Missing: {', '.join(missing)}.",
            )
        torch_version = _torch_major_minor()
        if torch_version < (2, 4) or torch_version >= (2, 13):
            return BackendInfo(
                self.name,
                runtime_version(),
                False,
                "LiteRT Torch requires PyTorch >=2.4,<2.13; this environment has "
                f"PyTorch {torch.__version__}. Use a separate LiteRT export environment.",
            )
        try:
            _import_module("litert_torch")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            return BackendInfo(
                self.name,
                runtime_version(),
                False,
                f"LiteRT Torch could not initialize: {exc}.",
            )
        return BackendInfo(
            self.name,
            runtime_version(),
            True,
            "LiteRT Torch can convert static CPU artifacts for LiteRT/XNNPACK execution.",
        )

    def supports(self, request: CompileRequest) -> Support:
        del request
        probe = self.probe()
        reason = (
            "LiteRT is an AOT export backend. Use lm7.export(..., backend='litert') "
            "instead of lm7.compile()."
            if probe.available
            else probe.reason
        )
        return Support(False, reason)

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        del request, example_args, example_kwargs
        raise CompilationError("LiteRT is export-only; use lm7.export(..., backend='litert').")

    def convert_module(
        self,
        model: torch.nn.Module,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
        output_path: Path,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Path:
        probe = self.probe()
        if not probe.available:
            raise CompilationError(probe.reason)
        settings = parse_options(options)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            litert_torch = _import_module("litert_torch")
            cpu_args = _map_tensors(example_args, lambda tensor: tensor.detach().cpu())
            cpu_kwargs = _map_tensors(dict(example_kwargs), lambda tensor: tensor.detach().cpu())
            model = model.to("cpu").eval()
            with torch.no_grad():
                converted = litert_torch.convert(
                    model,
                    sample_args=cpu_args,
                    sample_kwargs=cpu_kwargs,
                    **settings.converter_options,
                )
            converted.export(str(output_path))
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("LiteRT Torch did not write a non-empty flatbuffer")
            return output_path
        except CompilationError:
            raise
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            raise CompilationError(
                f"LiteRT conversion failed for {output_path}: {exc}. Check that the "
                "model is torch.export-compatible and that its operators have LiteRT "
                "lowerings. Dynamic shapes are outside the initial backend scope."
            ) from exc

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        if artifact.callable is not None:
            return artifact.callable
        if artifact.path is None:
            raise ArtifactLoadError("LiteRT artifact has no .tflite model path.")
        return self.load_tflite(artifact.path)

    def load_tflite(self, path: Path) -> Callable[..., Any]:
        probe = self.probe()
        if not probe.available:
            raise ArtifactLoadError(probe.reason)
        path = Path(path)
        if not path.is_file():
            raise ArtifactLoadError(f"LiteRT model {path} does not exist.")
        try:
            model = _import_module("litert_torch").load(str(path))
            return _LiteRTCallable(model)
        except Exception as exc:
            raise ArtifactLoadError(f"Failed to load LiteRT model {path}: {exc}.") from exc


class _LiteRTCallable:
    def __init__(self, model: Any) -> None:
        self._model = model

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            inputs = _map_tensors(
                (args, kwargs),
                lambda tensor: tensor.detach().cpu().contiguous().numpy(),
            )
            converted_args, converted_kwargs = inputs
            outputs = self._model(*converted_args, **converted_kwargs)
            return _to_torch(outputs)
        except Exception as exc:
            raise RuntimeError(f"LiteRT execution failed: {exc}.") from exc


def parse_options(options: Mapping[str, Any] | None) -> LiteRTOptions:
    values = dict(options or {})
    strict_export = values.pop("strict_export", "auto")
    if strict_export not in {"auto", True, False}:
        raise CompilationError("LiteRT strict_export must be 'auto', True, or False.")
    lightweight_conversion = bool(values.pop("lightweight_conversion", False))
    enable_x64 = bool(values.pop("enable_x64", True))
    runtime_constant_folding = values.pop("runtime_constant_folding", None)
    if runtime_constant_folding is not None:
        runtime_constant_folding = bool(runtime_constant_folding)
    if values:
        raise CompilationError(f"Unsupported LiteRT options: {', '.join(sorted(values))}.")
    return LiteRTOptions(
        strict_export,
        lightweight_conversion,
        enable_x64,
        runtime_constant_folding,
    )


def runtime_version() -> str | None:
    try:
        return importlib.metadata.version("litert-torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def _is_linux_aarch64() -> bool:
    return platform.system() == "Linux" and platform.machine().lower() in {
        "aarch64",
        "arm64",
    }


def _to_torch(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_to_torch(item) for item in value)
    if isinstance(value, list):
        return [_to_torch(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_torch(item) for key, item in value.items()}
    try:
        return torch.as_tensor(value).clone()
    except Exception as exc:
        raise TypeError(f"Unexpected LiteRT output type {type(value).__name__}.") from exc


def _map_tensors(value: Any, fn: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_tensors(item, fn) for key, item in value.items()}
    if value is not None:
        raise TypeError(f"LiteRT accepts tensor inputs only; got {type(value).__name__}.")
    return value


def _torch_major_minor() -> tuple[int, int]:
    release = torch.__version__.split("+", 1)[0]
    major, minor, *_ = release.split(".")
    return int(major), int(minor)


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _import_module(name: str) -> Any:
    return importlib.import_module(name)
