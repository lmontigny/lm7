from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import MISSING, asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .backends import registry
from .backends.aot_inductor import SUPPORTED_VENDORS as AOT_INDUCTOR_VENDORS
from .backends.aot_inductor import AOTInductorBackend
from .backends.iree_vulkan import (
    SUPPORTED_TARGET_VENDORS as IREE_VULKAN_VENDORS,
)
from .backends.iree_vulkan import IREEVulkanBackend
from .backends.openvino import OpenVINOBackend
from .cache import input_signature
from .detection import resolve_target, torch_device
from .errors import (
    ArtifactLoadError,
    BackendUnavailableError,
    UnsupportedModelError,
)
from .targets import TargetSpec, parse_target


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


FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
PROGRAM_NAME = "exported_program.pt2"
COMPILED_PROGRAM_NAME = "compiled_model.pt2"
COMPILED_IR_NAME = "compiled_model.xml"
COMPILED_IR_WEIGHTS_NAME = "compiled_model.bin"
COMPILED_VMFB_NAME = "compiled_model.vmfb"
DEBUG_DIR_NAME = "debug"
EXPORT_BACKENDS = frozenset({"export", "aot_inductor", "iree_vulkan", "openvino"})


@dataclass(frozen=True)
class DynamicDimension:
    """A named, bounded dynamic tensor dimension."""

    name: str
    min: int = 1
    max: int = 2**31 - 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Dynamic dimension name cannot be empty.")
        if self.min < 0:
            raise ValueError("Dynamic dimension minimum cannot be negative.")
        if self.max < self.min:
            raise ValueError("Dynamic dimension maximum must be at least its minimum.")


@dataclass(frozen=True)
class ShapeProfile:
    """Dynamic dimensions keyed by model argument name and tensor dimension."""

    inputs: Mapping[str, Mapping[int, DynamicDimension]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for input_name, dimensions in self.inputs.items():
            if not input_name:
                raise ValueError("Shape profile input names cannot be empty.")
            for dimension, constraint in dimensions.items():
                if dimension < 0:
                    raise ValueError("Shape profile dimension indexes cannot be negative.")
                if not isinstance(constraint, DynamicDimension):
                    raise TypeError("Shape profile constraints must be DynamicDimension instances.")


@dataclass(frozen=True)
class ArtifactManifest:
    format_version: int
    lm7_version: str
    torch_version: str
    created_at: str
    target: Mapping[str, Any]
    model_graph_hash: str
    cache_key: str
    input_signature: Any
    program_file: str
    program_sha256: str
    backend: str = "export"
    backend_version: str | None = None
    compiled_file: str | None = None
    compiled_sha256: str | None = None
    # OpenVINO IR is two files: the graph and its weight sibling.
    compiled_weights_file: str | None = None
    compiled_weights_sha256: str | None = None
    runtime_requirements: Mapping[str, Any] | None = None
    debug_requested: bool = False
    debug_artifacts: tuple[Mapping[str, str], ...] = ()
    shape_profile: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactManifest:
        required = {
            field.name
            for field in cls.__dataclass_fields__.values()
            if field.default is MISSING and field.default_factory is MISSING
        }
        missing = required - value.keys()
        if missing:
            raise ArtifactLoadError(
                f"Artifact manifest is missing required fields: {', '.join(sorted(missing))}."
            )
        known = {name: value[name] for name in cls.__dataclass_fields__ if name in value}
        return cls(**known)


@dataclass(frozen=True)
class ExportArtifact:
    path: Path
    manifest: ArtifactManifest
    exported_program: torch.export.ExportedProgram
    compiled_callable: Callable[..., Any] | None = None

    def module(self) -> Callable[..., Any]:
        if self.compiled_callable is not None:
            return self.compiled_callable
        return self.exported_program.module()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        _validate_shape_profile(self.manifest.shape_profile, args, kwargs)
        return self.module()(*args, **kwargs)

    def debug_files(self) -> tuple[Path, ...]:
        return tuple(self.path / artifact["path"] for artifact in self.manifest.debug_artifacts)


def export(
    model: torch.nn.Module | torch.export.ExportedProgram,
    *,
    args: tuple[Any, ...] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    target: str | TargetSpec = "auto",
    output: str | os.PathLike[str],
    backend: str = "export",
    options: Mapping[str, Any] | None = None,
    debug: bool = False,
    dynamic_shapes: Any = None,
    shape_profile: ShapeProfile | None = None,
    strict: bool = False,
) -> ExportArtifact:
    """Capture and persist a versioned LM7 source artifact."""
    kwargs = dict(kwargs or {})
    if dynamic_shapes is not None and shape_profile is not None:
        raise ValueError("dynamic_shapes and shape_profile cannot be supplied together.")
    if backend not in EXPORT_BACKENDS:
        choices = ", ".join(repr(name) for name in sorted(EXPORT_BACKENDS))
        raise BackendUnavailableError(
            f"Export backend {backend!r} is not supported; choose one of {choices}."
        )
    resolved_target = _artifact_target(target)
    if backend == "aot_inductor" and resolved_target.vendor not in AOT_INDUCTOR_VENDORS:
        raise BackendUnavailableError(
            "LM7 v0.1 only validates packaged AOTInductor artifacts for CPU, Apple "
            "Silicon, and NVIDIA targets."
        )
    if backend == "openvino" and resolved_target.vendor not in {"cpu", "intel"}:
        raise BackendUnavailableError(
            "OpenVINO artifacts are validated for Intel CPU targets only."
        )
    if backend == "iree_vulkan" and resolved_target.vendor not in IREE_VULKAN_VENDORS:
        raise BackendUnavailableError("IREE Vulkan artifacts target NVIDIA, AMD, or Intel GPUs.")
    if backend == "iree_vulkan" and (dynamic_shapes is not None or shape_profile is not None):
        raise BackendUnavailableError("IREE Vulkan artifacts currently require static shapes.")
    if isinstance(model, torch.export.ExportedProgram):
        if args is not None or kwargs:
            raise ValueError("args and kwargs cannot be supplied with an ExportedProgram.")
        exported_program = model
        signature: Any = None
    elif isinstance(model, torch.nn.Module):
        if args is None:
            raise ValueError("args must be supplied when exporting an nn.Module.")
        if backend == "iree_vulkan":
            # IREE owns device placement. Capture a host ExportedProgram even
            # when the VMFB targets a discrete GPU, so export does not require
            # CUDA, ROCm, or XPU to be available on the compiler host.
            model = model.to("cpu")
            args = _map_tensors(args, lambda tensor: tensor.detach().cpu())
            kwargs = _map_tensors(kwargs, lambda tensor: tensor.detach().cpu())
        elif resolved_target.vendor != "cpu":
            device = torch_device(resolved_target)
            model = model.to(device)
            args = _map_tensors(args, lambda tensor: tensor.to(device))
            kwargs = _map_tensors(kwargs, lambda tensor: tensor.to(device))
        profile_metadata = (
            _shape_profile_metadata(model, args, kwargs, shape_profile)
            if shape_profile is not None
            else None
        )
        torch_dynamic_shapes = (
            _torch_dynamic_shapes(profile_metadata)
            if profile_metadata is not None
            else dynamic_shapes
        )
        try:
            if backend == "iree_vulkan":
                with torch.no_grad():
                    exported_program = torch.export.export(
                        model,
                        args,
                        kwargs,
                        dynamic_shapes=torch_dynamic_shapes,
                        strict=strict,
                    )
            else:
                exported_program = torch.export.export(
                    model,
                    args,
                    kwargs,
                    dynamic_shapes=torch_dynamic_shapes,
                    strict=strict,
                )
        except Exception as exc:
            raise UnsupportedModelError(
                f"Model export stage failed for target {target}: {exc}. "
                "Check that the model is export-compatible and provide representative inputs."
            ) from exc
        signature = input_signature(args, kwargs)
    else:
        raise TypeError("model must be an nn.Module or torch.export.ExportedProgram.")
    if isinstance(model, torch.export.ExportedProgram):
        profile_metadata = None

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"Artifact output {destination} already exists; choose a new path or remove it explicitly."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    graph_hash = _graph_hash(exported_program)
    cache_key = artifact_cache_key(graph_hash, signature, resolved_target)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=str(destination.parent)))
    try:
        program_path = staging / PROGRAM_NAME
        torch.export.save(exported_program, program_path)
        program_sha256 = _file_sha256(program_path)
        compiled_file = None
        compiled_sha256 = None
        compiled_weights_file = None
        compiled_weights_sha256 = None
        iree_device_uri = None
        iree_vulkan_target = None
        debug_dir = staging / DEBUG_DIR_NAME
        if debug:
            _write_export_debug_files(exported_program, debug_dir)
        if backend == "openvino":
            openvino_backend = _openvino_backend()
            probe = openvino_backend.probe()
            if not probe.available:
                raise BackendUnavailableError(probe.reason)
            compiler_options = dict(options or {})
            openvino_backend.compile_exported(
                exported_program,
                staging / COMPILED_IR_NAME,
                static_shapes=_flat_tensors(args, kwargs)
                if compiler_options.get("static_shapes", True)
                else None,
                compress_to_fp16=bool(compiler_options.get("compress_to_fp16", False)),
            )
            compiled_file = COMPILED_IR_NAME
            compiled_sha256 = _file_sha256(staging / COMPILED_IR_NAME)
            compiled_weights_file = COMPILED_IR_WEIGHTS_NAME
            compiled_weights_sha256 = _file_sha256(staging / COMPILED_IR_WEIGHTS_NAME)
        if backend == "aot_inductor":
            selected_backend = registry.get("aot_inductor")
            if not isinstance(selected_backend, AOTInductorBackend):
                raise BackendUnavailableError("The registered aot_inductor backend is invalid.")
            support = selected_backend.probe()
            if not support.available:
                raise BackendUnavailableError(support.reason)
            compiled_path = staging / COMPILED_PROGRAM_NAME
            compiler_options = dict(options or {})
            if debug:
                compiler_options.update(_debug_inductor_options(debug_dir))
            selected_backend.compile_exported(
                exported_program, compiled_path, compiler_options, target=resolved_target
            )
            compiled_file = COMPILED_PROGRAM_NAME
            compiled_sha256 = _file_sha256(compiled_path)
            if debug:
                _extract_package_debug_files(compiled_path, debug_dir)
        if backend == "iree_vulkan":
            iree_backend = _iree_vulkan_backend()
            probe = iree_backend.probe()
            if not probe.available:
                raise BackendUnavailableError(probe.reason)
            compiler_options = dict(options or {})
            iree_device_uri = compiler_options.pop("device_uri", None)
            iree_vulkan_target = compiler_options.get("vulkan_target")
            compiled_path = staging / COMPILED_VMFB_NAME
            iree_backend.compile_exported(
                exported_program,
                compiled_path,
                options=compiler_options,
            )
            compiled_file = COMPILED_VMFB_NAME
            compiled_sha256 = _file_sha256(compiled_path)
        debug_artifacts = _index_debug_artifacts(staging, debug_dir) if debug else ()
        manifest = ArtifactManifest(
            format_version=FORMAT_VERSION,
            lm7_version=_lm7_version(),
            torch_version=torch.__version__,
            created_at=datetime.now(timezone.utc).isoformat(),
            target=asdict(resolved_target),
            model_graph_hash=graph_hash,
            cache_key=cache_key,
            input_signature=_json_value(signature),
            program_file=PROGRAM_NAME,
            program_sha256=program_sha256,
            backend=backend,
            backend_version=_backend_version(backend),
            compiled_file=compiled_file,
            compiled_sha256=compiled_sha256,
            compiled_weights_file=compiled_weights_file,
            compiled_weights_sha256=compiled_weights_sha256,
            runtime_requirements={
                "torch": torch.__version__,
                "device": resolved_target.vendor,
                "api_status": "stable" if backend == "export" else "beta",
                # The IR payload executes on the OpenVINO runtime alone; torch is
                # only needed to read the exported_program.pt2 alongside it.
                **({"openvino": _openvino_version()} if backend == "openvino" else {}),
                **(
                    {
                        "iree-base-runtime": _iree_runtime_version(),
                        "vulkan_device_uri": iree_device_uri,
                        "vulkan_target": iree_vulkan_target,
                    }
                    if backend == "iree_vulkan"
                    else {}
                ),
            },
            debug_requested=debug,
            debug_artifacts=debug_artifacts,
            shape_profile=profile_metadata,
        )
        (staging / MANIFEST_NAME).write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(destination)
    except Exception:
        _preserve_failure_debug(debug_dir if debug else None)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    compiled_callable = None
    if backend == "aot_inductor":
        selected_backend = registry.get("aot_inductor")
        assert isinstance(selected_backend, AOTInductorBackend)
        compiled_callable = selected_backend.load_package(destination / COMPILED_PROGRAM_NAME)
    elif backend == "openvino":
        compiled_callable = _openvino_backend().load_ir(destination / COMPILED_IR_NAME)
    elif backend == "iree_vulkan":
        compiled_callable = _iree_vulkan_backend().load_vmfb(
            destination / COMPILED_VMFB_NAME,
            device_uri=iree_device_uri,
        )
    return ExportArtifact(destination, manifest, exported_program, compiled_callable)


def load_artifact(path: str | os.PathLike[str]) -> ExportArtifact:
    """Load an LM7 source artifact after validating its metadata and payload."""
    artifact_path = Path(path).expanduser().resolve()
    manifest_path = artifact_path / MANIFEST_NAME
    if not artifact_path.is_dir() or not manifest_path.is_file():
        raise ArtifactLoadError(
            f"Artifact load stage failed for {artifact_path}: {MANIFEST_NAME} was not found."
        )
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = ArtifactManifest.from_dict(raw_manifest)
    except ArtifactLoadError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ArtifactLoadError(
            f"Artifact load stage failed for {artifact_path}: invalid manifest: {exc}."
        ) from exc
    if manifest.format_version != FORMAT_VERSION:
        raise ArtifactLoadError(
            f"Unsupported LM7 artifact format {manifest.format_version}; "
            f"this LM7 version supports format {FORMAT_VERSION}."
        )
    program_path = artifact_path / manifest.program_file
    if not program_path.is_file():
        raise ArtifactLoadError(
            f"Artifact load stage failed for {artifact_path}: {manifest.program_file} is missing."
        )
    if _file_sha256(program_path) != manifest.program_sha256:
        raise ArtifactLoadError(
            f"Artifact load stage failed for {artifact_path}: program checksum does not match "
            "the manifest. Re-export the model."
        )
    try:
        exported_program = torch.export.load(program_path)
    except Exception as exc:
        raise ArtifactLoadError(
            f"Artifact load stage failed for {artifact_path}: torch.export.load failed: {exc}."
        ) from exc
    compiled_callable = None
    if manifest.backend == "aot_inductor":
        if not manifest.compiled_file or not manifest.compiled_sha256:
            raise ArtifactLoadError(
                "AOTInductor artifact manifest is missing compiled payload metadata."
            )
        compiled_path = artifact_path / manifest.compiled_file
        if not compiled_path.is_file():
            raise ArtifactLoadError(
                f"Artifact load stage failed for {artifact_path}: "
                f"{manifest.compiled_file} is missing."
            )
        if _file_sha256(compiled_path) != manifest.compiled_sha256:
            raise ArtifactLoadError(
                f"Artifact load stage failed for {artifact_path}: compiled package checksum "
                "does not match the manifest. Re-export the model."
            )
        backend = registry.get("aot_inductor")
        if not isinstance(backend, AOTInductorBackend):
            raise ArtifactLoadError("The registered aot_inductor backend is invalid.")
        compiled_callable = backend.load_package(compiled_path)
    elif manifest.backend == "openvino":
        compiled_path = _verify_payload(
            artifact_path, manifest.compiled_file, manifest.compiled_sha256
        )
        # The weights sibling is verified too: OpenVINO reads it implicitly when
        # compiling the graph, so a corrupt .bin would otherwise pass unnoticed.
        _verify_payload(
            artifact_path, manifest.compiled_weights_file, manifest.compiled_weights_sha256
        )
        compiled_callable = _openvino_backend().load_ir(compiled_path)
    elif manifest.backend == "iree_vulkan":
        compiled_path = _verify_payload(
            artifact_path, manifest.compiled_file, manifest.compiled_sha256
        )
        requirements = manifest.runtime_requirements or {}
        compiled_callable = _iree_vulkan_backend().load_vmfb(
            compiled_path,
            device_uri=requirements.get("vulkan_device_uri"),
        )
    elif manifest.backend != "export":
        raise ArtifactLoadError(f"Unsupported artifact backend {manifest.backend!r}.")
    return ExportArtifact(artifact_path, manifest, exported_program, compiled_callable)


def _verify_payload(artifact_path: Path, name: str | None, expected_sha256: str | None) -> Path:
    if not name or not expected_sha256:
        raise ArtifactLoadError(
            f"Artifact manifest for {artifact_path} is missing compiled payload metadata."
        )
    payload_path = artifact_path / name
    if not payload_path.is_file():
        raise ArtifactLoadError(
            f"Artifact load stage failed for {artifact_path}: {name} is missing."
        )
    if _file_sha256(payload_path) != expected_sha256:
        raise ArtifactLoadError(
            f"Artifact load stage failed for {artifact_path}: {name} checksum does not match "
            "the manifest. Re-export the model."
        )
    return payload_path


def _openvino_backend() -> OpenVINOBackend:
    backend = registry.get("openvino")
    if not isinstance(backend, OpenVINOBackend):
        raise BackendUnavailableError("The registered openvino backend is invalid.")
    return backend


def _iree_vulkan_backend() -> IREEVulkanBackend:
    backend = registry.get("iree_vulkan")
    if not isinstance(backend, IREEVulkanBackend):
        raise BackendUnavailableError("The registered iree_vulkan backend is invalid.")
    return backend


def _openvino_version() -> str | None:
    try:
        return importlib.metadata.version("openvino")
    except importlib.metadata.PackageNotFoundError:
        return None


def _iree_runtime_version() -> str | None:
    try:
        return importlib.metadata.version("iree-base-runtime")
    except importlib.metadata.PackageNotFoundError:
        return None


def _backend_version(backend: str) -> str | None:
    if backend == "aot_inductor":
        return torch.__version__
    if backend == "openvino":
        return _openvino_version()
    if backend == "iree_vulkan":
        return _iree_runtime_version()
    return None


def _flat_tensors(args: tuple[Any, ...] | None, kwargs: Mapping[str, Any]) -> list[torch.Tensor]:
    found: list[torch.Tensor] = []

    def walk(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            found.append(item)
        elif isinstance(item, (tuple, list)):
            for entry in item:
                walk(entry)
        elif isinstance(item, dict):
            for entry in item.values():
                walk(entry)

    walk((args or (), dict(kwargs)))
    return found


def artifact_cache_key(model_graph_hash: str, signature: Any, target: TargetSpec) -> str:
    payload = {
        "format_version": FORMAT_VERSION,
        "lm7_version": _lm7_version(),
        "torch_version": torch.__version__,
        "model_graph_hash": model_graph_hash,
        "input_signature": _json_value(signature),
        "target": asdict(target),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact_target(target: str | TargetSpec) -> TargetSpec:
    parsed = parse_target(target)
    return resolve_target(parsed) if parsed.vendor == "auto" else parsed


def _graph_hash(exported_program: torch.export.ExportedProgram) -> str:
    graph = str(exported_program.graph_module.graph)
    state_metadata = sorted(
        (name, tuple(value.shape), str(value.dtype))
        for name, value in exported_program.state_dict.items()
    )
    value = json.dumps({"graph": graph, "state": state_metadata}, sort_keys=True)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _lm7_version() -> str:
    from . import __version__

    return __version__


def _shape_profile_metadata(
    model: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    profile: ShapeProfile,
) -> Mapping[str, Any]:
    try:
        bound = inspect.signature(model.forward).bind(*args, **kwargs)
    except TypeError as exc:
        raise ValueError(f"Cannot bind shape profile to model inputs: {exc}.") from exc
    unknown = set(profile.inputs) - set(bound.arguments)
    if unknown:
        raise ValueError(
            "Shape profile references unknown or unbound model inputs: "
            f"{', '.join(sorted(unknown))}."
        )
    inputs: dict[str, dict[str, Mapping[str, Any]]] = {}
    for input_name, dimensions in profile.inputs.items():
        value = bound.arguments[input_name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Shape profile input {input_name!r} is not a tensor.")
        serialized_dimensions: dict[str, Mapping[str, Any]] = {}
        for dimension, constraint in dimensions.items():
            if dimension >= value.dim():
                raise ValueError(
                    f"Shape profile dimension {dimension} is out of range for input "
                    f"{input_name!r} with {value.dim()} dimensions."
                )
            size = value.shape[dimension]
            if not constraint.min <= size <= constraint.max:
                raise ValueError(
                    f"Example input {input_name!r} dimension {dimension} has size {size}, "
                    f"outside [{constraint.min}, {constraint.max}]."
                )
            serialized_dimensions[str(dimension)] = asdict(constraint)
        inputs[input_name] = serialized_dimensions
    return {"argument_order": list(bound.arguments), "inputs": inputs}


def _torch_dynamic_shapes(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    inputs = profile["inputs"]
    return {
        input_name: {
            int(dimension): torch.export.Dim(
                constraint["name"],
                min=constraint["min"],
                max=constraint["max"],
            )
            for dimension, constraint in inputs.get(input_name, {}).items()
        }
        or None
        for input_name in profile["argument_order"]
    }


def _validate_shape_profile(
    profile: Mapping[str, Any] | None,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> None:
    if profile is None:
        return
    argument_order = profile["argument_order"]
    values = dict(kwargs)
    values.update(zip(argument_order, args, strict=False))
    for input_name, dimensions in profile["inputs"].items():
        if input_name not in values:
            raise ValueError(f"Shape-profiled input {input_name!r} was not supplied.")
        value = values[input_name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Shape-profiled input {input_name!r} must be a tensor.")
        for dimension_text, constraint in dimensions.items():
            dimension = int(dimension_text)
            if dimension >= value.dim():
                raise ValueError(
                    f"Input {input_name!r} has no dimension {dimension} required by its "
                    "shape profile."
                )
            size = value.shape[dimension]
            if not constraint["min"] <= size <= constraint["max"]:
                raise ValueError(
                    f"Input {input_name!r} dimension {dimension} has size {size}; "
                    f"expected [{constraint['min']}, {constraint['max']}]."
                )


def _debug_inductor_options(debug_dir: Path) -> dict[str, Any]:
    return {
        "trace.enabled": True,
        "trace.debug_dir": str(debug_dir),
        "trace.fx_graph": True,
        "trace.fx_graph_transformed": True,
        "trace.ir_pre_fusion": True,
        "trace.ir_post_fusion": True,
        "trace.output_code": True,
    }


def _write_export_debug_files(
    exported_program: torch.export.ExportedProgram, debug_dir: Path
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "exported_program.txt").write_text(str(exported_program) + "\n", encoding="utf-8")
    (debug_dir / "exported_graph.py").write_text(
        exported_program.graph_module.code, encoding="utf-8"
    )
    (debug_dir / "graph_signature.txt").write_text(
        str(exported_program.graph_signature) + "\n", encoding="utf-8"
    )


def _index_debug_artifacts(staging: Path, debug_dir: Path) -> tuple[Mapping[str, str], ...]:
    if not debug_dir.is_dir():
        return ()
    artifacts = []
    for path in sorted(item for item in debug_dir.rglob("*") if item.is_file()):
        level, kind = _debug_artifact_kind(path)
        artifacts.append(
            {
                "level": level,
                "kind": kind,
                "path": path.relative_to(staging).as_posix(),
                "sha256": _file_sha256(path),
            }
        )
    return tuple(artifacts)


def _extract_package_debug_files(package_path: Path, debug_dir: Path) -> None:
    if not zipfile.is_zipfile(package_path):
        return
    selected_suffixes = {
        ".c",
        ".cpp",
        ".cu",
        ".py",
        ".ptx",
        ".s",
        ".asm",
        ".cubin",
        ".hsaco",
    }
    package_debug_dir = debug_dir / "package"
    with zipfile.ZipFile(package_path) as archive:
        for entry in sorted(archive.infolist(), key=lambda item: item.filename):
            if entry.is_dir() or Path(entry.filename).suffix.lower() not in selected_suffixes:
                continue
            package_debug_dir.mkdir(parents=True, exist_ok=True)
            safe_name = entry.filename.replace("\\", "__").replace("/", "__")
            (package_debug_dir / safe_name).write_bytes(archive.read(entry))


def _preserve_failure_debug(debug_dir: Path | None) -> None:
    configured = os.environ.get("LM7_DEBUG_FAILURE_DIR")
    if debug_dir is None or not configured or not debug_dir.is_dir():
        return
    destination = Path(configured).expanduser().resolve()
    if destination.exists():
        suffix = 1
        while destination.with_name(f"{destination.name}-{suffix}").exists():
            suffix += 1
        destination = destination.with_name(f"{destination.name}-{suffix}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(debug_dir, destination)


def _debug_artifact_kind(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name.startswith(("exported_", "graph_signature")):
        return "export", "graph"
    if name.startswith("fx_graph"):
        return "fx", "graph"
    if "ir_pre_fusion" in name:
        return "inductor_ir_pre_fusion", "ir"
    if "ir_post_fusion" in name:
        return "inductor_ir_post_fusion", "ir"
    if suffix == ".ptx":
        return "device_code", "ptx"
    if suffix in {".s", ".asm"}:
        return "machine_code", "assembly"
    if suffix in {".cubin", ".hsaco"}:
        return "device_binary", suffix.removeprefix(".")
    if name.startswith("output_code"):
        return "generated_code", "source"
    if suffix in {".cpp", ".c", ".cu", ".py"}:
        return "generated_code", "source"
    return "compiler_debug", "diagnostic"
