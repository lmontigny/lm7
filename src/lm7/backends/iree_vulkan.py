from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from ..errors import ArtifactLoadError, CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support

SUPPORTED_TARGET_VENDORS = frozenset({"amd", "arm", "intel", "nvidia"})
_REQUIRED_MODULES = {
    "iree.compiler": "iree-base-compiler",
    "iree.runtime": "iree-base-runtime",
    "iree.turbine.aot": "iree-turbine",
}
_OPT_LEVELS = frozenset({"O0", "O1", "O2", "O3"})


class IREEVulkanBackend:
    """Export-only IREE backend producing Vulkan/SPIR-V VMFB artifacts."""

    name = "iree_vulkan"

    def probe(self) -> BackendInfo:
        missing = [
            package for module, package in _REQUIRED_MODULES.items() if not _has_module(module)
        ]
        if missing:
            return BackendInfo(
                self.name,
                None,
                False,
                "IREE Vulkan support is not installed; install LM7 with "
                '"iree-vulkan" support: pip install "lm7[iree-vulkan]". '
                f"Missing: {', '.join(missing)}.",
            )
        version = _package_version("iree-base-compiler")
        devices = query_vulkan_devices()
        reason = (
            f"IREE {version or 'unknown'} can compile and execute Vulkan VMFB artifacts."
            if devices
            else "IREE can compile Vulkan VMFB artifacts, but its runtime currently "
            "enumerates no Vulkan devices."
        )
        return BackendInfo(self.name, version, True, reason)

    def supports(self, request: CompileRequest) -> Support:
        probe = self.probe()
        if not probe.available:
            return Support(False, probe.reason)
        return Support(
            False,
            "IREE Vulkan is an AOT export backend. Use lm7.export(..., "
            "backend='iree_vulkan') instead of lm7.compile().",
        )

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        del request, example_args, example_kwargs
        raise CompilationError(
            "IREE Vulkan is export-only; use lm7.export(..., backend='iree_vulkan')."
        )

    def compile_exported(
        self,
        exported_program: torch.export.ExportedProgram,
        output_path: Path,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Path:
        probe = self.probe()
        if not probe.available:
            raise CompilationError(probe.reason)

        compiler_options = dict(options or {})
        vulkan_target = compiler_options.pop("vulkan_target", None)
        opt_level = str(compiler_options.pop("opt_level", "O2")).upper()
        if opt_level not in _OPT_LEVELS:
            raise CompilationError(
                f"Invalid IREE optimization level {opt_level!r}; expected one of "
                f"{', '.join(sorted(_OPT_LEVELS))}."
            )
        if compiler_options:
            raise CompilationError(
                f"Unsupported IREE Vulkan compiler options: {', '.join(sorted(compiler_options))}."
            )

        flags = [
            "--iree-hal-target-device=vulkan",
            f"--iree-opt-level={opt_level}",
        ]
        if vulkan_target:
            flags.append(f"--iree-vulkan-target={vulkan_target}")

        output_path = Path(output_path)
        try:
            turbine_aot = _import_module("iree.turbine.aot")
            export_output = turbine_aot.export(exported_program)
            export_output.session.set_flags(*flags)
            export_output.compile(output_path, target_backends=None)
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise RuntimeError("IREE did not produce a non-empty VMFB file")
            return output_path
        except CompilationError:
            raise
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            raise CompilationError(
                f"IREE Vulkan compilation failed for {output_path}: {exc}. "
                "Try the portable target by removing options['vulkan_target'], "
                "or inspect the exported graph for unsupported operations."
            ) from exc

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        if artifact.callable is not None:
            return artifact.callable
        if artifact.path is None:
            raise ArtifactLoadError("IREE Vulkan artifact has no VMFB path.")
        return self.load_vmfb(
            artifact.path,
            device_uri=artifact.metadata.get("device_uri"),
        )

    def load_vmfb(
        self,
        path: Path,
        *,
        device_uri: str | None = None,
        function_name: str = "main",
    ) -> Callable[..., Any]:
        return _IREEVulkanCallable(
            Path(path),
            device_uri=device_uri,
            function_name=function_name,
        )


class _IREEVulkanCallable:
    """Lazily load a VMFB so compilation works on hosts without Vulkan."""

    def __init__(self, path: Path, *, device_uri: str | None, function_name: str) -> None:
        self._path = path
        self._device_uri = device_uri
        self._function_name = function_name
        self._config: Any = None
        self._vm_module: Any = None
        self._bound_module: Any = None
        self._function: Callable[..., Any] | None = None

    def _ensure_loaded(self) -> Callable[..., Any]:
        if self._function is not None:
            return self._function
        if not self._path.is_file():
            raise ArtifactLoadError(f"IREE VMFB file {self._path} does not exist.")
        try:
            runtime = _import_module("iree.runtime")
        except ImportError as exc:
            raise ArtifactLoadError(
                'IREE runtime is not installed; install LM7 with ".[iree-vulkan]".'
            ) from exc
        try:
            if self._device_uri:
                device = runtime.get_device(self._device_uri)
                config = runtime.Config(device=device)
            else:
                devices = runtime.get_driver("vulkan").query_available_devices()
                if not devices:
                    raise ArtifactLoadError(
                        "IREE Vulkan runtime found no devices. Compilation can run offline, "
                        "but execution requires a Vulkan 1.3 device and driver."
                    )
                config = runtime.Config("vulkan")
            vm_module = runtime.VmModule.mmap(config.vm_instance, str(self._path))
            bound_module = runtime.load_vm_module(vm_module, config)
            function = getattr(bound_module, self._function_name)
        except ArtifactLoadError:
            raise
        except Exception as exc:
            raise ArtifactLoadError(
                f"Failed to load IREE Vulkan VMFB {self._path}: {exc}."
            ) from exc
        self._config = config
        self._vm_module = vm_module
        self._bound_module = bound_module
        self._function = function
        return function

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        function = self._ensure_loaded()
        inputs = [
            tensor.detach().cpu().contiguous().numpy()
            for tensor in _flatten_tensors((args, kwargs))
        ]
        try:
            return _to_torch(function(*inputs))
        except Exception as exc:
            raise RuntimeError(f"IREE Vulkan execution failed for {self._path}: {exc}.") from exc


def query_vulkan_devices() -> tuple[Mapping[str, Any], ...]:
    if not _has_module("iree.runtime"):
        return ()
    try:
        runtime = _import_module("iree.runtime")
        return tuple(runtime.get_driver("vulkan").query_available_devices())
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError):
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
            raise TypeError(f"IREE Vulkan accepts tensor inputs only; got {type(item).__name__}.")

    walk(value)
    return tensors


def _to_torch(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_to_torch(item) for item in value)
    if isinstance(value, list):
        return [_to_torch(item) for item in value]
    to_host = getattr(value, "to_host", None)
    if to_host is None:
        raise TypeError(f"Unexpected IREE output type {type(value).__name__}.")
    return torch.from_numpy(to_host()).clone()


def _import_module(name: str) -> Any:
    return importlib.import_module(name)


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
