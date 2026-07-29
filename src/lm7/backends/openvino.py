from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from ..cache import cache_dir
from ..errors import ArtifactLoadError, CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support

# OpenVINO's CPU plugin does not default to FP32: it is FP16 on ARM hosts and
# BF16 on x86 hosts with AMX. An FP32 model would then run in reduced precision
# and silently disagree with eager. LM7 pins FP32 and lets the caller opt out
# through options={"inference_precision": ...}.
_DEFAULT_INFERENCE_PRECISION = "f32"

# torch.export leaves batch and spatial dimensions symbolic, so convert_model
# produces IR shaped like [?,3,?,?]. LM7 reshapes to the example inputs so the
# compiled model matches the shapes it was captured for.
_DEFAULT_STATIC_SHAPES = True


class OpenVINOBackend:
    """Intel OpenVINO backend built on the IR artifact path.

    LM7 compiles through ``torch.export`` -> ``openvino.convert_model`` ->
    ``openvino.save_model``, rather than through OpenVINO's ``torch.compile``
    backend. The evaluation in ``docs/openvino-evaluation.md`` measured both:
    the IR path was faster than eager on every workload on Intel CPU, while the
    dynamo path lost to eager on two of them and never beat TorchInductor. The
    IR path is also the one that produces a portable artifact, which is what
    LM7's export and bundle story wants.
    """

    name = "openvino"

    def probe(self) -> BackendInfo:
        try:
            installed = importlib.util.find_spec("openvino") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            return BackendInfo(
                self.name,
                None,
                False,
                'OpenVINO is not installed; install LM7 with ".[openvino]".',
            )
        try:
            version = importlib.metadata.version("openvino")
        except importlib.metadata.PackageNotFoundError:
            version = None
        return BackendInfo(self.name, version, True, "OpenVINO is available.")

    def supports(self, request: CompileRequest) -> Support:
        probe = self.probe()
        if not probe.available:
            return Support(False, probe.reason)
        if request.target.vendor not in {"cpu", "intel"}:
            return Support(False, "OpenVINO supports Intel CPU targets only in LM7 v0.1.")
        # The OpenVINO runtime exchanges tensors through NumPy, which has no
        # bfloat16 dtype, so a bfloat16 model cannot round-trip.
        dtype = _model_dtype(request.model)
        if dtype is torch.bfloat16:
            return Support(
                False,
                "OpenVINO cannot execute bfloat16 models: its runtime exchanges tensors "
                "through NumPy, which has no bfloat16 dtype. Use float32 or float16.",
            )
        # Deliberately below inductor (100) and aot_inductor (90). The evaluation
        # shows a latency win but has not yet established operator coverage across
        # a wide model set, so automatic planning should still prefer Inductor.
        return Support(
            True,
            "OpenVINO can compile an ExportedProgram to IR for Intel CPU execution.",
            priority=80,
        )

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        try:
            openvino = importlib.import_module("openvino")
            model = request.model
            dtype = _model_dtype(model)
            if dtype is torch.bfloat16:
                raise CompilationError(
                    "OpenVINO cannot execute bfloat16 models: its runtime exchanges "
                    "tensors through NumPy, which has no bfloat16 dtype. Use float32 "
                    "or float16."
                )

            options = dict(request.options)
            device = str(options.pop("device", "CPU"))
            inference_precision = options.pop("inference_precision", _DEFAULT_INFERENCE_PRECISION)
            static_shapes = bool(options.pop("static_shapes", _DEFAULT_STATIC_SHAPES))
            # openvino.save_model compresses weights to FP16 by default, which shows
            # up as FP16-level error on an otherwise FP32 model.
            compress_to_fp16 = bool(options.pop("compress_to_fp16", False))
            plugin_config = dict(options.pop("config", {}))
            if inference_precision is not None:
                plugin_config.setdefault("INFERENCE_PRECISION_HINT", str(inference_precision))

            # OpenVINO executes on its own device; inputs are exchanged as host
            # tensors regardless of the torch device the caller used.
            export_args = _map_tensors(example_args, lambda tensor: tensor.detach().cpu())
            export_kwargs = _map_tensors(dict(example_kwargs), lambda tensor: tensor.detach().cpu())
            if request.transfers == "automatic":
                model.to("cpu")

            exported_program = torch.export.export(
                model,
                export_args,
                export_kwargs,
                strict=False,
            )
            flat_inputs = _flatten_tensors((export_args, export_kwargs))

            artifact_root = cache_dir() / "openvino"
            artifact_root.mkdir(parents=True, exist_ok=True)
            handle, stem = tempfile.mkstemp(suffix=".xml", dir=artifact_root)
            os.close(handle)
            model_path = Path(stem)
            model_path.unlink(missing_ok=True)
            try:
                self.compile_exported(
                    exported_program,
                    model_path,
                    static_shapes=flat_inputs if static_shapes else None,
                    compress_to_fp16=compress_to_fp16,
                )
                compiled = _compile_ir(
                    openvino,
                    model_path,
                    device=device,
                    config=plugin_config,
                )
                return Artifact(
                    self.name,
                    request.target,
                    callable=_OpenVINOCallable(compiled, len(flat_inputs)),
                    path=model_path,
                    metadata={
                        "compiled": True,
                        "format": "openvino_ir",
                        "device": device,
                        "openvino_version": getattr(openvino, "__version__", None),
                        "compress_to_fp16": compress_to_fp16,
                        "static_shapes": static_shapes,
                        "inference_precision": plugin_config.get("INFERENCE_PRECISION_HINT"),
                    },
                )
            except Exception:
                model_path.unlink(missing_ok=True)
                model_path.with_suffix(".bin").unlink(missing_ok=True)
                raise
        except CompilationError:
            raise
        except Exception as exc:
            raise CompilationError(
                f"Compilation stage failed for target {request.target} with backend openvino: "
                f"{exc}. Try backend='inductor', backend='eager', or fallback='warn'."
            ) from exc

    def compile_exported(
        self,
        exported_program: torch.export.ExportedProgram,
        model_path: Path,
        *,
        static_shapes: Sequence[torch.Tensor] | None = None,
        compress_to_fp16: bool = False,
    ) -> Path:
        """Convert an ExportedProgram to OpenVINO IR and save it to ``model_path``.

        Writes ``model_path`` and its ``.bin`` weight sibling. ``static_shapes``
        pins the IR to those tensor shapes; leaving it ``None`` keeps the symbolic
        dimensions ``torch.export`` produces.
        """
        probe = self.probe()
        if not probe.available:
            raise CompilationError(probe.reason)
        openvino = importlib.import_module("openvino")
        try:
            ov_model = openvino.convert_model(exported_program)
            if static_shapes is not None:
                _reshape_to_static(ov_model, static_shapes)
            openvino.save_model(ov_model, str(model_path), compress_to_fp16=compress_to_fp16)
        except Exception as exc:
            raise CompilationError(
                f"OpenVINO IR conversion failed for {model_path}: {exc}. "
                "Check that the model's operators are supported by the PyTorch frontend."
            ) from exc
        return model_path

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        if artifact.callable is not None:
            return artifact.callable
        if artifact.path is None:
            raise ArtifactLoadError("OpenVINO artifact has no IR path.")
        probe = self.probe()
        if not probe.available:
            raise ArtifactLoadError(probe.reason)
        openvino = importlib.import_module("openvino")
        return self.load_ir(
            artifact.path,
            device=str(artifact.metadata.get("device", "CPU")),
            inference_precision=artifact.metadata.get("inference_precision"),
            openvino=openvino,
        )

    def load_ir(
        self,
        model_path: Path,
        *,
        device: str = "CPU",
        inference_precision: str | None = _DEFAULT_INFERENCE_PRECISION,
        openvino: Any = None,
    ) -> Callable[..., Any]:
        """Load saved OpenVINO IR into a tensor-in/tensor-out callable."""
        probe = self.probe()
        if not probe.available:
            raise ArtifactLoadError(probe.reason)
        if openvino is None:
            openvino = importlib.import_module("openvino")
        config = (
            {"INFERENCE_PRECISION_HINT": str(inference_precision)} if inference_precision else {}
        )
        compiled = _compile_ir(openvino, model_path, device=device, config=config)
        return _OpenVINOCallable(compiled, len(compiled.inputs))


class _OpenVINOCallable:
    """Adapts an OpenVINO compiled model to LM7's tensor-in/tensor-out contract."""

    def __init__(self, compiled_model: Any, expected_inputs: int) -> None:
        self._compiled_model = compiled_model
        self._expected_inputs = expected_inputs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        flat = _flatten_tensors((args, kwargs))
        if len(flat) != self._expected_inputs:
            raise ValueError(
                f"OpenVINO artifact expects {self._expected_inputs} tensor inputs, got {len(flat)}."
            )
        inputs = [tensor.detach().cpu().numpy() for tensor in flat]
        results = self._compiled_model(inputs)
        outputs = [
            # The executor infers with share_outputs=True, so a returned array can
            # alias OpenVINO's output buffer and be overwritten by the next call.
            torch.from_numpy(results[port]).clone()
            for port in self._compiled_model.outputs
        ]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def _compile_ir(
    openvino: Any,
    model_path: Path,
    *,
    device: str,
    config: Mapping[str, Any],
) -> Any:
    core = openvino.Core()
    available = list(core.available_devices)
    base_device = device.split(".", 1)[0]
    if base_device not in available:
        # OpenVINO otherwise falls back silently, so a run labelled GPU or NPU can
        # quietly be the CPU plugin.
        raise CompilationError(
            f"OpenVINO device {device!r} is not available; the runtime reports "
            f"{available}. Install the matching plugin or use device='CPU'."
        )
    return core.compile_model(str(model_path), device, dict(config))


def _reshape_to_static(ov_model: Any, flat_inputs: Sequence[torch.Tensor]) -> None:
    if len(ov_model.inputs) != len(flat_inputs):
        return
    shapes = {
        port: list(tensor.shape) for port, tensor in zip(ov_model.inputs, flat_inputs, strict=True)
    }
    ov_model.reshape(shapes)


def _model_dtype(model: torch.nn.Module) -> torch.dtype | None:
    for parameter in model.parameters():
        return parameter.dtype
    return None


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
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

    walk(value)
    return found


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
