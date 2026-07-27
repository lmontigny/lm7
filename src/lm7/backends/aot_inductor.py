from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from ..cache import cache_dir
from ..detection import torch_device
from ..errors import ArtifactLoadError, CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support


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


class AOTInductorBackend:
    name = "aot_inductor"

    def probe(self) -> BackendInfo:
        compile_api = getattr(getattr(torch, "_inductor", None), "aoti_compile_and_package", None)
        load_api = getattr(getattr(torch, "_inductor", None), "aoti_load_package", None)
        available = callable(compile_api) and callable(load_api)
        reason = (
            "PyTorch AOTInductor package APIs are available."
            if available
            else "This PyTorch build has no AOTInductor package APIs."
        )
        return BackendInfo(self.name, torch.__version__, available, reason)

    def supports(self, request: CompileRequest) -> Support:
        probe = self.probe()
        if not probe.available:
            return Support(False, probe.reason)
        if request.target.vendor not in {"cpu", "apple"}:
            return Support(
                False,
                "LM7 v0.1 only validates packaged AOTInductor execution for CPU and "
                "Apple Silicon targets.",
            )
        return Support(
            True,
            "AOTInductor can package an ExportedProgram for CPU or Apple execution.",
            priority=90,
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
            export_args = _map_tensors(example_args, lambda tensor: tensor.to(device))
            export_kwargs = _map_tensors(dict(example_kwargs), lambda tensor: tensor.to(device))
            exported_program = torch.export.export(
                request.model,
                export_args,
                export_kwargs,
                strict=False,
            )
            artifact_root = cache_dir() / "aot_inductor"
            artifact_root.mkdir(parents=True, exist_ok=True)
            handle, package_name = tempfile.mkstemp(suffix=".pt2", dir=artifact_root)
            os.close(handle)
            Path(package_name).unlink()
            try:
                package_path = Path(package_name)
                self.compile_exported(exported_program, package_path, request.options)
                return Artifact(
                    self.name,
                    request.target,
                    path=package_path,
                    metadata={"compiled": True, "format": "pt2"},
                )
            except Exception:
                Path(package_name).unlink(missing_ok=True)
                raise
        except CompilationError:
            raise
        except Exception as exc:
            raise CompilationError(
                f"Compilation stage failed for target {request.target} with backend "
                f"aot_inductor: {exc}. Install a supported C++ compiler toolchain or "
                "use backend='eager'."
            ) from exc

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        if artifact.path is None:
            raise ArtifactLoadError("AOTInductor artifact has no package path.")
        return self.load_package(artifact.path)

    def compile_exported(
        self,
        exported_program: torch.export.ExportedProgram,
        package_path: Path,
        options: Mapping[str, Any] | None = None,
    ) -> Path:
        probe = self.probe()
        if not probe.available:
            raise CompilationError(probe.reason)
        configs = dict(options or {})
        try:
            result = torch._inductor.aoti_compile_and_package(
                exported_program,
                package_path=str(package_path),
                inductor_configs=configs or None,
            )
        except Exception as exc:
            raise CompilationError(
                f"AOTInductor packaging failed for {package_path}: {exc}. "
                "Verify the platform C++ compiler and PyTorch installation."
            ) from exc
        return Path(result)

    def load_package(self, package_path: Path) -> Callable[..., Any]:
        probe = self.probe()
        if not probe.available:
            raise ArtifactLoadError(probe.reason)
        try:
            return torch._inductor.aoti_load_package(str(package_path))
        except Exception as exc:
            raise ArtifactLoadError(
                f"AOTInductor package load failed for {package_path}: {exc}. "
                "Use a compatible PyTorch runtime and target architecture."
            ) from exc
