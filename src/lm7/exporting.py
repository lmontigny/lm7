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
from .backends.coreml import DELEGATE as COREML_DELEGATE
from .backends.coreml import ExecuTorchCoreMLBackend
from .backends.executorch import DELEGATE as EXECUTORCH_DELEGATE
from .backends.executorch import ExecuTorchBackend
from .backends.iree_vulkan import (
    SUPPORTED_TARGET_VENDORS as IREE_VULKAN_VENDORS,
)
from .backends.iree_vulkan import IREEVulkanBackend
from .backends.litert import LiteRTBackend
from .backends.litert import parse_options as parse_litert_options
from .backends.onnxruntime import ONNXRuntimeBackend
from .backends.onnxruntime import parse_options as parse_onnxruntime_options
from .backends.openvino import OpenVINOBackend
from .backends.openvino import device_for_target as openvino_device_for_target
from .backends.qnn import DELEGATE as QNN_DELEGATE
from .backends.qnn import HTP_ARCH as QNN_HTP_ARCH
from .backends.qnn import RUNTIME_LIBRARIES as QNN_RUNTIME_LIBRARIES
from .backends.qnn import SOC_MODEL as QNN_SOC_MODEL
from .backends.qnn import VTCM_MB as QNN_VTCM_MB
from .backends.qnn import ExecuTorchQNNBackend
from .backends.stablehlo import StableHLOBackend
from .backends.tensorrt import SUPPORTED_VENDORS as TENSORRT_VENDORS
from .backends.tensorrt import TensorRTBackend
from .backends.tvm import TVMBackend
from .cache import input_signature
from .detection import resolve_target, torch_device
from .errors import (
    ArtifactLoadError,
    BackendUnavailableError,
    TargetNotFoundError,
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
COMPILED_STABLEHLO_NAME = "compiled_model.stablehlo.zip"
COMPILED_ONNX_NAME = "compiled_model.onnx"
COMPILED_PTE_NAME = "compiled_model.pte"
COMPILED_TFLITE_NAME = "compiled_model.tflite"
# Torch-TensorRT serializes the engine into a .pt2 archive; the infix keeps
# it distinct from the AOTInductor package, which uses the same extension.
COMPILED_TRT_NAME = "compiled_model.trt.pt2"
COMPILED_TVM_NAME = "compiled_model.tvm.so"
DEBUG_DIR_NAME = "debug"
EXPORT_BACKENDS = frozenset(
    {
        "export",
        "aot_inductor",
        "coreml",
        "executorch",
        "iree_vulkan",
        "litert",
        "onnxruntime",
        "openvino",
        "qnn",
        "stablehlo",
        "tensorrt",
        "tvm",
    }
)


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
            "OpenVINO artifacts are validated for Intel CPU and NPU targets only."
        )
    if (
        backend == "openvino"
        and openvino_device_for_target(resolved_target) == "NPU"
        and (dynamic_shapes is not None or shape_profile is not None)
    ):
        raise BackendUnavailableError(
            "The OpenVINO NPU plugin compiles static shapes only, so an intel:npu "
            "artifact cannot carry dynamic_shapes or a shape profile. Export one "
            "artifact per shape, or target='cpu'."
        )
    if backend == "tensorrt" and resolved_target.vendor not in TENSORRT_VENDORS:
        raise BackendUnavailableError("TensorRT artifacts target NVIDIA GPUs only.")
    if backend == "tensorrt" and (dynamic_shapes is not None or shape_profile is not None):
        # A dynamically shaped engine needs min/opt/max Input specs rather than
        # example tensors, and picking those well is its own evaluation.
        raise BackendUnavailableError(
            "TensorRT artifacts currently require static shapes. Export one artifact "
            "per shape, or use the JIT path with lm7.compile(backend='tensorrt')."
        )
    if backend == "executorch" and resolved_target.vendor != "cpu":
        raise BackendUnavailableError(
            "ExecuTorch artifacts use the XNNPACK delegate, which is a CPU target. "
            "Export with target='cpu'; the .pte then runs on Android and iOS CPUs too."
        )
    if backend == "qnn" and (
        resolved_target.vendor != "qualcomm" or resolved_target.model != "sm8750"
    ):
        raise BackendUnavailableError("The initial QNN backend requires target='qualcomm:sm8750'.")
    if backend == "qnn" and (dynamic_shapes is not None or shape_profile is not None):
        raise BackendUnavailableError(
            "QNN artifacts currently require static shapes; export one artifact per shape."
        )
    if backend == "coreml" and resolved_target.vendor != "apple":
        raise BackendUnavailableError(
            "Core ML artifacts require target='apple'; the delegate compiles and executes "
            "only on macOS. See docs/coreml.md."
        )
    if backend == "coreml" and (dynamic_shapes is not None or shape_profile is not None):
        raise BackendUnavailableError(
            "Core ML artifacts currently require static shapes; export one artifact per shape."
        )
    if backend == "iree_vulkan" and resolved_target.vendor not in IREE_VULKAN_VENDORS:
        raise BackendUnavailableError("IREE Vulkan artifacts target NVIDIA, AMD, or Intel GPUs.")
    if backend == "iree_vulkan" and (dynamic_shapes is not None or shape_profile is not None):
        raise BackendUnavailableError("IREE Vulkan artifacts currently require static shapes.")
    if backend == "onnxruntime" and resolved_target.vendor not in {"cpu", "nvidia"}:
        raise BackendUnavailableError(
            "ONNX Runtime artifacts are initially validated for CPU and NVIDIA targets only."
        )
    if backend == "litert" and resolved_target.vendor != "cpu":
        raise BackendUnavailableError(
            "LiteRT artifacts are initially validated for CPU/XNNPACK execution only."
        )
    if backend == "litert" and (dynamic_shapes is not None or shape_profile is not None):
        raise BackendUnavailableError("LiteRT artifacts currently require static shapes.")
    if backend == "tvm" and resolved_target.vendor != "cpu":
        raise BackendUnavailableError(
            "TVM artifacts are wired up for CPU (LLVM) targets only; see docs/tvm.md."
        )
    if backend == "tvm" and (dynamic_shapes is not None or shape_profile is not None):
        raise BackendUnavailableError(
            "TVM artifacts currently require static shapes; export one artifact per shape."
        )
    if backend == "litert" and isinstance(model, torch.export.ExportedProgram):
        raise BackendUnavailableError(
            "LiteRT conversion requires the source nn.Module and representative args; "
            "an ExportedProgram alone cannot be converted by the public LiteRT Torch API."
        )
    if isinstance(model, torch.export.ExportedProgram):
        if args is not None or kwargs:
            raise ValueError("args and kwargs cannot be supplied with an ExportedProgram.")
        exported_program = model
        signature: Any = None
    elif isinstance(model, torch.nn.Module):
        if args is None:
            raise ValueError("args must be supplied when exporting an nn.Module.")
        if backend in {"iree_vulkan", "litert", "onnxruntime", "qnn", "coreml"}:
            # These runtimes own device placement. Capture a host ExportedProgram
            # even when the artifact targets a GPU or NPU, so the accelerator
            # does not need to be attached to the compiler host.
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
            if backend in {"iree_vulkan", "litert", "onnxruntime", "qnn", "coreml"}:
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
        signature = input_signature(args, dict(kwargs))
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
        openvino_device = None
        executorch_delegated = None
        executorch_total = None
        executorch_quantization = None
        executorch_quantized_ops = None
        qnn_lowered = None
        coreml_lowered = None
        debug_dir = staging / DEBUG_DIR_NAME
        onnxruntime_settings = None
        litert_settings = None
        tvm_target = None
        if debug:
            _write_export_debug_files(exported_program, debug_dir)
        if backend == "openvino":
            openvino_backend = _openvino_backend()
            probe = openvino_backend.probe()
            if not probe.available:
                raise BackendUnavailableError(probe.reason)
            compiler_options = dict(options or {})
            openvino_device = openvino_device_for_target(resolved_target)
            if openvino_device == "NPU" and not compiler_options.get("static_shapes", True):
                raise BackendUnavailableError(
                    "The OpenVINO NPU plugin compiles static shapes only, so "
                    "static_shapes=False cannot be used with an intel:npu artifact."
                )
            openvino_backend.compile_exported(
                exported_program,
                staging / COMPILED_IR_NAME,
                static_shapes=_flat_tensors(args, kwargs)
                if compiler_options.get("static_shapes", True)
                else None,
                compress_to_fp16=bool(compiler_options.get("compress_to_fp16", False)),
                quantization=str(compiler_options.get("quantization", "none")),
            )
            compiled_file = COMPILED_IR_NAME
            compiled_sha256 = _file_sha256(staging / COMPILED_IR_NAME)
            compiled_weights_file = COMPILED_IR_WEIGHTS_NAME
            compiled_weights_sha256 = _file_sha256(staging / COMPILED_IR_WEIGHTS_NAME)
        if backend == "tensorrt":
            tensorrt_backend = _tensorrt_backend()
            probe = tensorrt_backend.probe()
            if not probe.available:
                # Unlike the other export backends, this one cannot be produced
                # on a build host without the GPU: TensorRT profiles real kernels
                # on the device to pick tactics.
                raise BackendUnavailableError(probe.reason)
            trt_args, trt_kwargs = _tensorrt_example_inputs(exported_program, args, kwargs)
            tensorrt_backend.compile_exported(
                exported_program,
                staging / COMPILED_TRT_NAME,
                arg_inputs=trt_args,
                kwarg_inputs=trt_kwargs,
                options=options,
            )
            compiled_file = COMPILED_TRT_NAME
            compiled_sha256 = _file_sha256(staging / COMPILED_TRT_NAME)
        if backend == "stablehlo":
            stablehlo_backend = _stablehlo_backend()
            probe = stablehlo_backend.probe()
            if not probe.available:
                raise BackendUnavailableError(probe.reason)
            compiled_path = staging / COMPILED_STABLEHLO_NAME
            stablehlo_backend.compile_exported(exported_program, compiled_path)
            compiled_file = COMPILED_STABLEHLO_NAME
            compiled_sha256 = _file_sha256(compiled_path)
        if backend == "executorch":
            executorch_backend = _executorch_backend()
            probe = executorch_backend.probe()
            if not probe.available:
                raise BackendUnavailableError(probe.reason)
            lowered = executorch_backend.compile_exported(
                exported_program, staging / COMPILED_PTE_NAME, options=options
            )
            compiled_file = COMPILED_PTE_NAME
            compiled_sha256 = _file_sha256(lowered.path)
            executorch_delegated = lowered.delegated_calls
            executorch_total = lowered.total_calls
            executorch_quantization = lowered.quantization
            executorch_quantized_ops = lowered.quantized_ops
        if backend == "qnn":
            qnn_backend = _qnn_backend()
            probe = qnn_backend.probe()
            if not probe.available:
                raise BackendUnavailableError(probe.reason)
            qnn_lowered = qnn_backend.compile_exported(
                exported_program,
                staging / COMPILED_PTE_NAME,
                target=resolved_target,
                options=options,
            )
            compiled_file = COMPILED_PTE_NAME
            compiled_sha256 = _file_sha256(qnn_lowered.path)
        if backend == "coreml":
            coreml_backend = _coreml_backend()
            probe = coreml_backend.probe()
            if not probe.available:
                raise BackendUnavailableError(probe.reason)
            coreml_lowered = coreml_backend.compile_exported(
                exported_program, staging / COMPILED_PTE_NAME, options=options
            )
            compiled_file = COMPILED_PTE_NAME
            compiled_sha256 = _file_sha256(coreml_lowered.path)
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
        if backend == "onnxruntime":
            onnxruntime_backend = _onnxruntime_backend()
            probe = onnxruntime_backend.probe()
            if not probe.available:
                raise BackendUnavailableError(probe.reason)
            onnxruntime_settings = parse_onnxruntime_options(resolved_target, options)
            compiled_path = staging / COMPILED_ONNX_NAME
            onnxruntime_backend.compile_exported(
                exported_program,
                compiled_path,
                options=onnxruntime_settings.compiler_options,
            )
            compiled_file = COMPILED_ONNX_NAME
            compiled_sha256 = _file_sha256(compiled_path)
        if backend == "litert":
            litert_backend = _litert_backend()
            probe = litert_backend.probe()
            if not probe.available:
                raise BackendUnavailableError(probe.reason)
            litert_settings = parse_litert_options(options)
            compiled_path = staging / COMPILED_TFLITE_NAME
            assert isinstance(model, torch.nn.Module)
            assert args is not None
            litert_backend.convert_module(
                model,
                args,
                kwargs,
                compiled_path,
                options=litert_settings.converter_options,
            )
            compiled_file = COMPILED_TFLITE_NAME
            compiled_sha256 = _file_sha256(compiled_path)
        if backend == "tvm":
            tvm_backend = _tvm_backend()
            probe = tvm_backend.probe()
            if not probe.available:
                raise BackendUnavailableError(probe.reason)
            tvm_target = dict(options or {}).get("target", "llvm")
            compiled_path = staging / COMPILED_TVM_NAME
            tvm_backend.compile_exported(exported_program, compiled_path, options=options)
            compiled_file = COMPILED_TVM_NAME
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
                # An AOTInductor package for a GPU is as architecture-bound as a
                # TensorRT engine, and until this it recorded neither the CUDA it
                # linked against nor the card it was built on -- so a load that
                # failed on another machine had nothing in the artifact to
                # explain itself with.
                **(
                    _aot_inductor_requirements(resolved_target) if backend == "aot_inductor" else {}
                ),
                # The IR payload executes on the OpenVINO runtime alone; torch is
                # only needed to read the exported_program.pt2 alongside it. The
                # IR itself is device-neutral, but which plugin compiles it is
                # not, so the device travels with the artifact.
                **(
                    {"openvino": _openvino_version(), "openvino_device": openvino_device}
                    if backend == "openvino"
                    else {}
                ),
                # A TensorRT engine is the least portable payload LM7 writes: it
                # is tuned for one GPU architecture by profiling real kernels,
                # and its format is tied to the TensorRT and Torch-TensorRT that
                # built it. Record all three so a failed load is diagnosable.
                **(
                    {
                        "torch-tensorrt": _torch_tensorrt_version(),
                        "tensorrt": _tensorrt_runtime_version(),
                        "device_bound": True,
                        **_cuda_device_requirements(),
                    }
                    if backend == "tensorrt"
                    else {}
                ),
                **(
                    {
                        "iree-base-runtime": _iree_runtime_version(),
                        "vulkan_device_uri": iree_device_uri,
                        "vulkan_target": iree_vulkan_target,
                    }
                    if backend == "iree_vulkan"
                    else {}
                ),
                **(
                    {
                        "onnxruntime": _onnxruntime_version(),
                        "execution_provider": onnxruntime_settings.provider,
                        "provider_options": dict(onnxruntime_settings.provider_options),
                        "disable_cpu_fallback": onnxruntime_settings.disable_cpu_fallback,
                        "opset_version": onnxruntime_settings.opset_version,
                    }
                    if backend == "onnxruntime" and onnxruntime_settings is not None
                    else {}
                ),
                **(
                    {
                        "litert-torch": _litert_version(),
                        "runtime": "LiteRT Interpreter/XNNPACK",
                        "static_shapes": True,
                        "strict_export": litert_settings.strict_export,
                        "lightweight_conversion": litert_settings.lightweight_conversion,
                        "enable_x64": litert_settings.enable_x64,
                        "runtime_constant_folding": litert_settings.runtime_constant_folding,
                    }
                    if backend == "litert" and litert_settings is not None
                    else {}
                ),
                # The StableHLO payload needs a PJRT plugin, not PyTorch, and the
                # plugin is chosen at load time -- so unlike every other compiled
                # payload this one is not pinned to the export-time device.
                **({"pjrt_plugin": "any", "device_bound": False} if backend == "stablehlo" else {}),
                # A .pte is executed by the ExecuTorch C++ runtime with no PyTorch
                # present, and the XNNPACK delegate covers ARM64 and x86-64 alike --
                # so this payload is not bound to the CPU that built it. The
                # delegate ratio records how much of the graph XNNPACK took; the
                # remainder runs on ExecuTorch's portable kernels.
                **(
                    {
                        "executorch": _executorch_backend().probe().version,
                        "delegate": EXECUTORCH_DELEGATE,
                        "delegated_calls": executorch_delegated,
                        "total_calls": executorch_total,
                        "quantization": executorch_quantization,
                        "quantized_ops": executorch_quantized_ops,
                        "calibration_samples": 1 if executorch_quantization == "int8" else 0,
                        "device_bound": False,
                    }
                    if backend == "executorch"
                    else {}
                ),
                **(
                    {
                        "executorch": _qnn_backend().probe().version,
                        "delegate": QNN_DELEGATE,
                        "backend": "htp",
                        "soc_model": QNN_SOC_MODEL,
                        "htp_arch": QNN_HTP_ARCH,
                        "vtcm_mb": QNN_VTCM_MB,
                        "precision": qnn_lowered.precision,
                        "delegated_calls": qnn_lowered.delegated_calls,
                        "total_calls": qnn_lowered.total_calls,
                        "qnn_sdk": _qnn_backend().sdk_version(),
                        "runtime_libraries": list(QNN_RUNTIME_LIBRARIES),
                        "device_bound": True,
                    }
                    if backend == "qnn" and qnn_lowered is not None
                    else {}
                ),
                # Unlike the QNN payload above, this one is not device-bound: the
                # .pte embeds an uncompiled Core ML model spec, and whichever Mac
                # loads it compiles it locally with Apple's own compiler -- see
                # docs/coreml.md.
                **(
                    {
                        "executorch": _coreml_backend().probe().version,
                        "delegate": COREML_DELEGATE,
                        "compute_unit": coreml_lowered.compute_unit,
                        "compute_precision": coreml_lowered.compute_precision,
                        "delegated_calls": coreml_lowered.delegated_calls,
                        "total_calls": coreml_lowered.total_calls,
                        "device_bound": False,
                    }
                    if backend == "coreml" and coreml_lowered is not None
                    else {}
                ),
                # The library embeds the exporting host's target triple (arm64
                # vs x86-64, plus any mcpu given in options), so unlike the
                # portable payloads above it only reloads on a matching
                # architecture -- see docs/tvm.md.
                **(
                    {
                        "tvm": _tvm_backend().probe().version,
                        "tvm_target": tvm_target,
                        "frontend": "relax.from_exported_program",
                        "device_bound": True,
                    }
                    if backend == "tvm"
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
        compiled_callable = _openvino_backend().load_ir(
            destination / COMPILED_IR_NAME,
            device=openvino_device or "CPU",
        )
    elif backend == "tensorrt":
        compiled_callable = _tensorrt_backend().load_engine(destination / COMPILED_TRT_NAME)
    elif backend == "iree_vulkan":
        compiled_callable = _iree_vulkan_backend().load_vmfb(
            destination / COMPILED_VMFB_NAME,
            device_uri=iree_device_uri,
        )
    elif backend == "onnxruntime":
        assert onnxruntime_settings is not None
        compiled_callable = _onnxruntime_backend().load_onnx(
            destination / COMPILED_ONNX_NAME,
            provider=onnxruntime_settings.provider,
            provider_options=onnxruntime_settings.provider_options,
            disable_cpu_fallback=onnxruntime_settings.disable_cpu_fallback,
        )
    elif backend == "litert":
        compiled_callable = _litert_backend().load_tflite(destination / COMPILED_TFLITE_NAME)
    elif backend == "stablehlo":
        compiled_callable = _stablehlo_backend().load_package(destination / COMPILED_STABLEHLO_NAME)
    elif backend == "executorch":
        compiled_callable = _executorch_backend().load_pte(destination / COMPILED_PTE_NAME)
    elif backend == "qnn":
        compiled_callable = _qnn_backend().load_pte(destination / COMPILED_PTE_NAME)
    elif backend == "coreml":
        compiled_callable = _coreml_backend().load_pte(destination / COMPILED_PTE_NAME)
    elif backend == "tvm":
        compiled_callable = _tvm_backend().load_library(destination / COMPILED_TVM_NAME)
    return ExportArtifact(destination, manifest, exported_program, compiled_callable)


# Backends whose compiled payload only runs on the architecture it was built
# for: AOTInductor emits kernels for one GPU compute capability, Torch-TensorRT
# tunes an engine for one GPU architecture, and TVM's LLVM codegen bakes in the
# exporting host's CPU target triple (arm64 vs x86-64). Everything else either
# carries a portable program or does not vary by architecture at load time.
_ARCHITECTURE_BOUND_BACKENDS = frozenset({"aot_inductor", "tensorrt", "tvm"})
# aot_inductor/tensorrt are bound to a GPU compute capability (vendor nvidia or
# amd); tvm is bound to the CPU instruction set (vendor cpu) instead.
_ARCHITECTURE_BOUND_VENDORS = {
    "aot_inductor": {"nvidia", "amd"},
    "tensorrt": {"nvidia", "amd"},
    "tvm": {"cpu"},
}


def _validate_target_architecture(manifest: ArtifactManifest, artifact_path: Path) -> None:
    """Refuse an architecture-bound artifact built for a different chip.

    Without this, a GPU artifact fails with "no kernel image is available for
    execution on the device" -- naming neither the artifact nor the fix -- and
    a TVM CPU artifact fails with an opaque dlopen/exec-format error instead.
    Verified by loading an ``sm89`` AOTInductor artifact on a Tesla T4 (``sm75``).

    Silent when the answer is not knowable: an artifact with no recorded
    architecture, or a host where the vendor's architecture cannot be resolved,
    must still load so that inspection and CPU-side use keep working.
    """
    if manifest.backend not in _ARCHITECTURE_BOUND_BACKENDS:
        return
    recorded = manifest.target.get("architecture")
    vendor = manifest.target.get("vendor")
    if not recorded or vendor not in _ARCHITECTURE_BOUND_VENDORS[manifest.backend]:
        return
    try:
        local = resolve_target(str(vendor)).architecture
    except TargetNotFoundError:
        # The vendor's hardware is absent. Loading is still allowed: the caller may
        # only want the manifest or the ExportedProgram, and asking for the compiled
        # payload will fail on its own terms.
        return
    if local is None or local == recorded:
        return
    raise ArtifactLoadError(
        f"Artifact load stage failed for {artifact_path}: its {manifest.backend} payload was "
        f"built for {vendor}:{recorded}, but this machine is {vendor}:{local}. Kernels are "
        f"compiled per architecture, so it cannot run here. Re-export on a matching machine, or ship a "
        f"bundle containing both architectures and load it with load_bundle(...).load()."
    )


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
    _validate_target_architecture(manifest, artifact_path)
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
        compiled_callable = backend.load_package(
            compiled_path, built_with=manifest.runtime_requirements
        )
    elif manifest.backend == "openvino":
        compiled_path = _verify_payload(
            artifact_path, manifest.compiled_file, manifest.compiled_sha256
        )
        # The weights sibling is verified too: OpenVINO reads it implicitly when
        # compiling the graph, so a corrupt .bin would otherwise pass unnoticed.
        _verify_payload(
            artifact_path, manifest.compiled_weights_file, manifest.compiled_weights_sha256
        )
        requirements = manifest.runtime_requirements or {}
        compiled_callable = _openvino_backend().load_ir(
            compiled_path,
            device=str(requirements.get("openvino_device") or "CPU"),
        )
    elif manifest.backend == "tensorrt":
        compiled_path = _verify_payload(
            artifact_path, manifest.compiled_file, manifest.compiled_sha256
        )
        compiled_callable = _tensorrt_backend().load_engine(compiled_path)
    elif manifest.backend == "iree_vulkan":
        compiled_path = _verify_payload(
            artifact_path, manifest.compiled_file, manifest.compiled_sha256
        )
        requirements = manifest.runtime_requirements or {}
        compiled_callable = _iree_vulkan_backend().load_vmfb(
            compiled_path,
            device_uri=requirements.get("vulkan_device_uri"),
        )
    elif manifest.backend == "onnxruntime":
        compiled_path = _verify_payload(
            artifact_path, manifest.compiled_file, manifest.compiled_sha256
        )
        requirements = manifest.runtime_requirements or {}
        compiled_callable = _onnxruntime_backend().load_onnx(
            compiled_path,
            provider=str(requirements.get("execution_provider", "CPUExecutionProvider")),
            provider_options=requirements.get("provider_options"),
            disable_cpu_fallback=bool(requirements.get("disable_cpu_fallback", True)),
        )
    elif manifest.backend == "litert":
        compiled_path = _verify_payload(
            artifact_path, manifest.compiled_file, manifest.compiled_sha256
        )
        compiled_callable = _litert_backend().load_tflite(compiled_path)
    elif manifest.backend == "stablehlo":
        compiled_path = _verify_payload(
            artifact_path, manifest.compiled_file, manifest.compiled_sha256
        )
        compiled_callable = _stablehlo_backend().load_package(compiled_path)
    elif manifest.backend == "executorch":
        compiled_path = _verify_payload(
            artifact_path, manifest.compiled_file, manifest.compiled_sha256
        )
        compiled_callable = _executorch_backend().load_pte(compiled_path)
    elif manifest.backend == "qnn":
        compiled_path = _verify_payload(
            artifact_path, manifest.compiled_file, manifest.compiled_sha256
        )
        compiled_callable = _qnn_backend().load_pte(compiled_path)
    elif manifest.backend == "coreml":
        compiled_path = _verify_payload(
            artifact_path, manifest.compiled_file, manifest.compiled_sha256
        )
        compiled_callable = _coreml_backend().load_pte(compiled_path)
    elif manifest.backend == "tvm":
        compiled_path = _verify_payload(
            artifact_path, manifest.compiled_file, manifest.compiled_sha256
        )
        compiled_callable = _tvm_backend().load_library(compiled_path)
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


def _tensorrt_example_inputs(
    exported_program: torch.export.ExportedProgram,
    args: tuple[Any, ...] | None,
    kwargs: Mapping[str, Any] | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Example inputs a TensorRT engine is built against.

    TensorRT profiles real kernels, so unlike the other export backends it needs
    the inputs themselves, and it needs positional and keyword ones kept apart
    so the saved artifact is called the way the source module was.
    """
    if args is not None:
        return tuple(args), dict(kwargs or {})
    captured = getattr(exported_program, "example_inputs", None)
    if captured:
        return tuple(captured[0]), dict(captured[1])
    raise BackendUnavailableError(
        "A TensorRT engine is built against example inputs, and this ExportedProgram "
        "carries none. Export from an nn.Module with args=... instead."
    )


def _tensorrt_backend() -> TensorRTBackend:
    backend = registry.get("tensorrt")
    if not isinstance(backend, TensorRTBackend):
        raise BackendUnavailableError("The registered tensorrt backend is invalid.")
    return backend


def _executorch_backend() -> ExecuTorchBackend:
    backend = registry.get("executorch")
    if not isinstance(backend, ExecuTorchBackend):
        raise BackendUnavailableError("The registered executorch backend is invalid.")
    return backend


def _qnn_backend() -> ExecuTorchQNNBackend:
    backend = registry.get("qnn")
    if not isinstance(backend, ExecuTorchQNNBackend):
        raise BackendUnavailableError("The registered qnn backend is invalid.")
    return backend


def _coreml_backend() -> ExecuTorchCoreMLBackend:
    backend = registry.get("coreml")
    if not isinstance(backend, ExecuTorchCoreMLBackend):
        raise BackendUnavailableError("The registered coreml backend is invalid.")
    return backend


def _tvm_backend() -> TVMBackend:
    backend = registry.get("tvm")
    if not isinstance(backend, TVMBackend):
        raise BackendUnavailableError("The registered tvm backend is invalid.")
    return backend


def _iree_vulkan_backend() -> IREEVulkanBackend:
    backend = registry.get("iree_vulkan")
    if not isinstance(backend, IREEVulkanBackend):
        raise BackendUnavailableError("The registered iree_vulkan backend is invalid.")
    return backend


def _onnxruntime_backend() -> ONNXRuntimeBackend:
    backend = registry.get("onnxruntime")
    if not isinstance(backend, ONNXRuntimeBackend):
        raise BackendUnavailableError("The registered onnxruntime backend is invalid.")
    return backend


def _litert_backend() -> LiteRTBackend:
    backend = registry.get("litert")
    if not isinstance(backend, LiteRTBackend):
        raise BackendUnavailableError("The registered litert backend is invalid.")
    return backend


def _stablehlo_backend() -> StableHLOBackend:
    backend = registry.get("stablehlo")
    if not isinstance(backend, StableHLOBackend):
        raise BackendUnavailableError("The registered stablehlo backend is invalid.")
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


def _onnxruntime_version() -> str | None:
    for distribution in ("onnxruntime", "onnxruntime-gpu"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def _torch_tensorrt_version() -> str | None:
    try:
        return importlib.metadata.version("torch-tensorrt")
    except importlib.metadata.PackageNotFoundError:
        return None


def _tensorrt_runtime_version() -> str | None:
    # The TensorRT wheel is CUDA-major-versioned, so the plain name is not
    # always the one that is installed.
    for distribution in ("tensorrt", "tensorrt-cu13", "tensorrt-cu12"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def _aot_inductor_requirements(target: TargetSpec) -> dict[str, Any]:
    """What an AOTInductor payload was built against, for a CUDA target.

    The package holds kernels compiled for one compute capability and a wrapper
    linked against one CUDA runtime, which is why `_validate_target_architecture`
    refuses to load it on a different GPU. Recording the pair is what makes both
    outcomes diagnosable from the artifact alone: a refusal can name what the
    artifact wanted, and a failure that gets past the guard can be told apart
    from a merely broken package.

    CPU and Apple targets record nothing here. Their payload is bound to a host
    toolchain too, but LM7 has not characterized how, and a guess in a manifest
    is worse than a gap.
    """
    if target.vendor != "nvidia":
        return {}
    return {"device_bound": True, **_cuda_device_requirements()}


def _cuda_device_requirements() -> dict[str, Any]:
    """The GPU an engine was tuned for, as recorded in the manifest."""
    try:
        major, minor = torch.cuda.get_device_capability()
        return {
            "cuda": torch.version.cuda,
            "compute_capability": f"sm{major}{minor}",
            "device_name": torch.cuda.get_device_name(),
        }
    except (AssertionError, RuntimeError):
        return {}


def _litert_version() -> str | None:
    try:
        return importlib.metadata.version("litert-torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def _backend_version(backend: str) -> str | None:
    if backend == "aot_inductor":
        return torch.__version__
    if backend == "openvino":
        return _openvino_version()
    if backend == "iree_vulkan":
        return _iree_runtime_version()
    if backend == "onnxruntime":
        return _onnxruntime_version()
    if backend == "litert":
        return _litert_version()
    if backend == "tensorrt":
        return _torch_tensorrt_version()
    if backend == "stablehlo":
        return _stablehlo_backend().probe().version
    if backend == "executorch":
        return _executorch_backend().probe().version
    if backend == "qnn":
        return _qnn_backend().probe().version
    if backend == "coreml":
        return _coreml_backend().probe().version
    if backend == "tvm":
        return _tvm_backend().probe().version
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


def _torch_dynamic_shapes(profile: Mapping[str, Any]) -> dict[str, Any]:
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
