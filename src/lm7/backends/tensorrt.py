from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from ..detection import torch_device
from ..errors import ArtifactLoadError, CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support

SUPPORTED_VENDORS = frozenset({"nvidia"})


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
        if getattr(torch.version, "hip", None):
            return BackendInfo(
                self.name,
                version,
                False,
                "A ROCm PyTorch build is active; TensorRT requires NVIDIA CUDA.",
            )
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

    def compile_exported(
        self,
        exported_program: torch.export.ExportedProgram,
        output_path: Path,
        *,
        arg_inputs: Sequence[Any] = (),
        kwarg_inputs: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> Path:
        """Build a TensorRT engine from an ExportedProgram and serialize it.

        This is the AOT half of the backend. The engine build is the expensive
        part -- 54 s for SmolLM2-135M on an Ada GPU, against 4 s to load the
        result -- and the JIT path pays it again in every process. Writing the
        engine to ``output_path`` moves that cost to build time.

        The saved payload is a ``.pt2`` archive holding the engine plus its
        weights, and it is bound to the GPU architecture, TensorRT version, and
        Torch-TensorRT version that produced it.

        ``arg_inputs`` and ``kwarg_inputs`` are kept apart rather than flattened
        into one list: saving re-exports the module against them, so collapsing
        keyword inputs into positional ones would change how the *reloaded*
        artifact must be called.
        """
        probe = self.probe()
        if not probe.available:
            raise CompilationError(probe.reason)
        torch_tensorrt = importlib.import_module("torch_tensorrt")
        settings = dict(options or {})
        positional = list(arg_inputs)
        keyword = dict(kwarg_inputs or {})
        try:
            trt_module = torch_tensorrt.dynamo.compile(
                exported_program, arg_inputs=positional, kwarg_inputs=keyword, **settings
            )
            torch_tensorrt.save(
                trt_module,
                str(output_path),
                output_format="exported_program",
                arg_inputs=positional,
                kwarg_inputs=keyword,
            )
        except Exception as exc:
            hint = ""
            if "use_explicit_typing" in str(exc):
                # Torch-TensorRT 2.12 turns explicit typing on by default, and
                # then rejects enabled_precisions outright. The graph's own
                # dtypes select the precision, so the option is redundant.
                hint = (
                    " Drop options={'enabled_precisions': ...}: this Torch-TensorRT "
                    "takes the precision from the exported graph's dtypes instead."
                )
            raise CompilationError(
                f"TensorRT engine build failed for {output_path}: {exc}.{hint}"
            ) from exc
        return output_path

    def load_engine(self, path: Path) -> Callable[..., Any]:
        """Load a serialized TensorRT engine without rebuilding it."""
        probe = self.probe()
        if not probe.available:
            raise ArtifactLoadError(probe.reason)
        torch_tensorrt = importlib.import_module("torch_tensorrt")
        try:
            loaded = torch_tensorrt.load(str(path))
            module = loaded.module() if hasattr(loaded, "module") else loaded
        except Exception as exc:
            raise ArtifactLoadError(
                f"Loading the TensorRT artifact at {path} failed: {exc}. A serialized "
                "engine is tied to the GPU architecture, TensorRT version, and "
                "Torch-TensorRT version that built it -- re-export on this machine."
            ) from exc
        return cast(Callable[..., Any], module)

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        if artifact.callable is not None:
            return artifact.callable
        if artifact.path is None:
            raise ArtifactLoadError("TensorRT artifact has no engine path.")
        return self.load_engine(artifact.path)


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
