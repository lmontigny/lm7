from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..cache import cache_dir
from ..errors import ArtifactLoadError, CompilationError
from ..targets import TargetSpec
from .base import Artifact, BackendInfo, CompileRequest, Support

_REQUIRED_MODULES = {
    "onnx": "onnx",
    "onnxscript": "onnxscript",
    "onnxruntime": "onnxruntime",
}
_TARGET_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "nvidia": "CUDAExecutionProvider",
}


@dataclass(frozen=True)
class ONNXRuntimeOptions:
    provider: str
    provider_options: Mapping[str, Any]
    disable_cpu_fallback: bool
    opset_version: int | None
    optimize: bool

    @property
    def compiler_options(self) -> Mapping[str, Any]:
        return {
            "opset_version": self.opset_version,
            "optimize": self.optimize,
        }


class ONNXRuntimeBackend:
    """ONNX artifact backend executed through an explicit ORT provider."""

    name = "onnxruntime"

    def probe(self) -> BackendInfo:
        missing = [
            package for module, package in _REQUIRED_MODULES.items() if not _has_module(module)
        ]
        if missing:
            return BackendInfo(
                self.name,
                None,
                False,
                'ONNX Runtime support is not installed; install LM7 with ".[onnxruntime]". '
                f"Missing: {', '.join(missing)}.",
            )
        try:
            runtime = _import_module("onnxruntime")
            providers = tuple(runtime.get_available_providers())
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            return BackendInfo(
                self.name,
                runtime_version(),
                False,
                f"ONNX Runtime could not initialize: {exc}.",
            )
        reason = "ONNX Runtime is available with providers: " + ", ".join(providers) + "."
        return BackendInfo(self.name, runtime_version(), True, reason)

    def supports(self, request: CompileRequest) -> Support:
        probe = self.probe()
        if not probe.available:
            return Support(False, probe.reason)
        provider = _TARGET_PROVIDERS.get(request.target.vendor)
        if provider is None:
            return Support(
                False,
                "ONNX Runtime is initially validated for CPU and NVIDIA targets only.",
            )
        available = available_providers()
        if provider not in available:
            package = "onnxruntime-gpu" if provider == "CUDAExecutionProvider" else "onnxruntime"
            return Support(
                False,
                f"ONNX Runtime provider {provider!r} is unavailable; install {package}. "
                f"Available providers: {', '.join(available) or 'none'}.",
            )
        return Support(
            True,
            f"ONNX Runtime can execute through {provider}.",
            priority=70,
        )

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        settings = parse_options(request.target, request.options)
        model_path: Path | None = None
        try:
            export_args = _map_tensors(example_args, lambda tensor: tensor.detach().cpu())
            export_kwargs = _map_tensors(dict(example_kwargs), lambda tensor: tensor.detach().cpu())
            request.model.to("cpu")
            with torch.no_grad():
                exported_program = torch.export.export(
                    request.model,
                    export_args,
                    export_kwargs,
                    strict=False,
                )

            artifact_root = cache_dir() / "onnxruntime"
            artifact_root.mkdir(parents=True, exist_ok=True)
            handle, stem = tempfile.mkstemp(suffix=".onnx", dir=artifact_root)
            os.close(handle)
            model_path = Path(stem)
            model_path.unlink(missing_ok=True)
            self.compile_exported(
                exported_program,
                model_path,
                options=settings.compiler_options,
            )
            compiled = self.load_onnx(
                model_path,
                provider=settings.provider,
                provider_options=settings.provider_options,
                disable_cpu_fallback=settings.disable_cpu_fallback,
            )
            return Artifact(
                self.name,
                request.target,
                callable=compiled,
                path=model_path,
                metadata={
                    "compiled": True,
                    "format": "onnx",
                    "provider": settings.provider,
                    "provider_options": dict(settings.provider_options),
                    "disable_cpu_fallback": settings.disable_cpu_fallback,
                    "opset_version": settings.opset_version,
                    "optimize": settings.optimize,
                },
            )
        except (ArtifactLoadError, CompilationError):
            if model_path is not None:
                model_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            if model_path is not None:
                model_path.unlink(missing_ok=True)
            raise CompilationError(
                f"Compilation stage failed for target {request.target} with backend "
                f"onnxruntime: {exc}. Check ONNX operator coverage or use "
                "backend='inductor'."
            ) from exc

    def compile_exported(
        self,
        exported_program: torch.export.ExportedProgram,
        model_path: Path,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Path:
        probe = self.probe()
        if not probe.available:
            raise CompilationError(probe.reason)
        compiler_options = dict(options or {})
        opset_version = compiler_options.pop("opset_version", None)
        optimize = bool(compiler_options.pop("optimize", True))
        if compiler_options:
            raise CompilationError(
                f"Unsupported ONNX compiler options: {', '.join(sorted(compiler_options))}."
            )
        if opset_version is not None:
            try:
                opset_version = int(opset_version)
            except (TypeError, ValueError) as exc:
                raise CompilationError("ONNX opset_version must be an integer.") from exc

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            onnx_program = torch.onnx.export(
                exported_program,
                args=(),
                f=None,
                dynamo=True,
                external_data=False,
                opset_version=opset_version,
                optimize=optimize,
            )
            if onnx_program is None or not hasattr(onnx_program, "save"):
                raise RuntimeError("torch.onnx.export did not return an ONNXProgram")
            onnx_program.save(model_path, external_data=False)
            if not model_path.is_file() or model_path.stat().st_size == 0:
                raise RuntimeError("the ONNX exporter did not write a non-empty model")
            onnx = _import_module("onnx")
            onnx.checker.check_model(onnx.load(str(model_path)))
            return model_path
        except Exception as exc:
            model_path.unlink(missing_ok=True)
            raise CompilationError(
                f"ONNX conversion failed for {model_path}: {exc}. Check that the "
                "model's operators are supported by the torch.export-based ONNX exporter. "
                "Models larger than 2 GiB are outside the initial embedded-weight scope."
            ) from exc

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        if artifact.callable is not None:
            return artifact.callable
        if artifact.path is None:
            raise ArtifactLoadError("ONNX Runtime artifact has no ONNX model path.")
        provider = str(
            artifact.metadata.get("provider")
            or _TARGET_PROVIDERS.get(artifact.target.vendor, "CPUExecutionProvider")
        )
        return self.load_onnx(
            artifact.path,
            provider=provider,
            provider_options=artifact.metadata.get("provider_options"),
            disable_cpu_fallback=bool(
                artifact.metadata.get("disable_cpu_fallback", provider != "CPUExecutionProvider")
            ),
        )

    def load_onnx(
        self,
        model_path: Path,
        *,
        provider: str,
        provider_options: Mapping[str, Any] | None = None,
        disable_cpu_fallback: bool = True,
    ) -> Callable[..., Any]:
        probe = self.probe()
        if not probe.available:
            raise ArtifactLoadError(probe.reason)
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ArtifactLoadError(f"ONNX model {model_path} does not exist.")
        runtime = _import_module("onnxruntime")
        available = tuple(runtime.get_available_providers())
        if provider not in available:
            raise ArtifactLoadError(
                f"ONNX Runtime provider {provider!r} is unavailable; available providers: "
                f"{', '.join(available) or 'none'}."
            )
        try:
            session_options = runtime.SessionOptions()
            if disable_cpu_fallback and provider != "CPUExecutionProvider":
                session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            providers: list[Any] = [provider]
            if provider_options:
                providers = [(provider, dict(provider_options))]
            session = runtime.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers,
            )
            if disable_cpu_fallback and hasattr(session, "disable_fallback"):
                session.disable_fallback()
            if provider not in session.get_providers():
                raise RuntimeError(
                    f"session did not activate requested provider {provider!r}: "
                    f"{session.get_providers()}"
                )
            return _ONNXRuntimeCallable(session)
        except Exception as exc:
            raise ArtifactLoadError(
                f"Failed to load ONNX model {model_path} with provider {provider}: {exc}."
            ) from exc


class _ONNXRuntimeCallable:
    def __init__(self, session: Any) -> None:
        self._session = session
        self._input_names = tuple(value.name for value in session.get_inputs())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        feeds = {
            name: value.detach().cpu().contiguous().numpy()
            for name, value in kwargs.items()
            if name in self._input_names and isinstance(value, torch.Tensor)
        }
        unmatched_kwargs = {name: value for name, value in kwargs.items() if name not in feeds}
        tensors = _flatten_tensors((args, unmatched_kwargs))
        remaining_names = tuple(name for name in self._input_names if name not in feeds)
        if len(tensors) != len(remaining_names):
            raise ValueError(
                f"ONNX artifact expects {len(self._input_names)} tensor inputs, "
                f"got {len(feeds) + len(tensors)}."
            )
        feeds.update(
            {
                name: tensor.detach().cpu().contiguous().numpy()
                for name, tensor in zip(remaining_names, tensors, strict=True)
            }
        )
        try:
            outputs = tuple(
                torch.as_tensor(value).clone() for value in self._session.run(None, feeds)
            )
        except Exception as exc:
            raise RuntimeError(f"ONNX Runtime execution failed: {exc}.") from exc
        return outputs[0] if len(outputs) == 1 else outputs


def parse_options(
    target: TargetSpec,
    options: Mapping[str, Any] | None,
) -> ONNXRuntimeOptions:
    values = dict(options or {})
    default_provider = _TARGET_PROVIDERS.get(target.vendor)
    provider = str(values.pop("provider", default_provider or ""))
    if not provider:
        raise CompilationError(
            f"ONNX Runtime has no default execution provider for target {target}."
        )
    provider_options = dict(values.pop("provider_options", {}))
    disable_cpu_fallback = bool(
        values.pop("disable_cpu_fallback", provider != "CPUExecutionProvider")
    )
    opset_version = values.pop("opset_version", None)
    optimize = bool(values.pop("optimize", True))
    if values:
        raise CompilationError(f"Unsupported ONNX Runtime options: {', '.join(sorted(values))}.")
    return ONNXRuntimeOptions(
        provider,
        provider_options,
        disable_cpu_fallback,
        opset_version,
        optimize,
    )


def runtime_version() -> str | None:
    for distribution in ("onnxruntime", "onnxruntime-gpu"):
        version = _package_version(distribution)
        if version is not None:
            return version
    return None


def available_providers() -> tuple[str, ...]:
    if not _has_module("onnxruntime"):
        return ()
    try:
        return tuple(_import_module("onnxruntime").get_available_providers())
    except (ImportError, OSError, RuntimeError, ValueError):
        return ()


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []

    def walk(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensors.append(item)
        elif isinstance(item, (tuple, list)):
            for child in item:
                walk(child)
        elif isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif item is not None:
            raise TypeError(f"ONNX Runtime accepts tensor inputs only; got {type(item).__name__}.")

    walk(value)
    return tensors


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


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _import_module(name: str) -> Any:
    return importlib.import_module(name)
