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
# Where a provider puts its tensors, which is not the same string as its name.
# Both CUDA-family providers allocate on the CUDA device, so both map to "cuda".
_PROVIDER_DEVICES = {
    "CPUExecutionProvider": "cpu",
    "CUDAExecutionProvider": "cuda",
    "TensorrtExecutionProvider": "cuda",
}
# ONNX's enum value for bfloat16, which is the one dtype `bind_input` cannot be
# handed as a numpy type because numpy has no equivalent.
_BFLOAT16_ONNX_TYPE = 16
# protobuf refuses to serialize a message of 2 GiB or more, and the graph shares
# that budget with the weights, so "auto" switches to a sidecar below the ceiling
# rather than at it.
EMBEDDED_WEIGHT_LIMIT = 1_800_000_000
# The exporter's own name for the sidecar. The graph references it relatively,
# so it has to sit beside the .onnx -- which is what a .lm7 directory gives it.
EXTERNAL_DATA_SUFFIX = ".data"


@dataclass(frozen=True)
class ONNXRuntimeOptions:
    provider: str
    provider_options: Mapping[str, Any]
    disable_cpu_fallback: bool
    opset_version: int | None
    optimize: bool
    external_data: bool | str = "auto"

    @property
    def compiler_options(self) -> Mapping[str, Any]:
        return {
            "opset_version": self.opset_version,
            "optimize": self.optimize,
            "external_data": self.external_data,
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
                    # What the export actually did, not what was asked for:
                    # `external_data="auto"` is a question, and the metadata has
                    # to answer it for anyone reading the artifact back.
                    "external_data": external_data_path(model_path).is_file(),
                },
            )
        except (ArtifactLoadError, CompilationError):
            _remove_model(model_path)
            raise
        except Exception as exc:
            _remove_model(model_path)
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
        external_data = compiler_options.pop("external_data", "auto")
        if compiler_options:
            raise CompilationError(
                f"Unsupported ONNX compiler options: {', '.join(sorted(compiler_options))}."
            )
        if opset_version is not None:
            try:
                opset_version = int(opset_version)
            except (TypeError, ValueError) as exc:
                raise CompilationError("ONNX opset_version must be an integer.") from exc
        # "auto" is a question about the weights rather than a preference: a model
        # whose initializers approach protobuf's ceiling has to put them in a
        # sidecar or fail to serialize at all.
        if external_data == "auto":
            external_data = _weight_bytes(exported_program) > EMBEDDED_WEIGHT_LIMIT
        external_data = bool(external_data)

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            onnx_program = torch.onnx.export(
                exported_program,
                args=(),
                f=None,
                dynamo=True,
                external_data=external_data,
                opset_version=opset_version,
                optimize=optimize,
            )
            if onnx_program is None or not hasattr(onnx_program, "save"):
                raise RuntimeError("torch.onnx.export did not return an ONNXProgram")
            onnx_program.save(model_path, external_data=external_data)
            if not model_path.is_file() or model_path.stat().st_size == 0:
                raise RuntimeError("the ONNX exporter did not write a non-empty model")
            # Unconditional, because the request is only a preference: above
            # protobuf's ceiling the exporter writes a sidecar whatever it was
            # asked for, and that file has to be found and packaged.
            _reconcile_external_data(model_path)
            onnx = _import_module("onnx")
            # Checked by path, not by loaded proto: the checker resolves the
            # sidecar itself, and a model over 2 GiB cannot be handed over as one
            # in-memory message at all.
            onnx.checker.check_model(str(model_path))
            return model_path
        except Exception as exc:
            _remove_model(model_path)
            raise CompilationError(
                f"ONNX conversion failed for {model_path}: {exc}. Check that the "
                "model's operators are supported by the torch.export-based ONNX exporter."
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
            return _ONNXRuntimeCallable(
                session,
                device_type=_PROVIDER_DEVICES.get(provider, "cpu"),
                device_id=int(dict(provider_options or {}).get("device_id", 0)),
            )
        except Exception as exc:
            raise ArtifactLoadError(
                f"Failed to load ONNX model {model_path} with provider {provider}: {exc}."
            ) from exc


class _ONNXRuntimeCallable:
    """A session driven through ORT's I/O binding rather than NumPy feeds.

    Binding torch storage directly means a CUDA session never copies its inputs
    down to the host or its outputs back up, and returns tensors on the device
    that produced them. The NumPy path this replaces did both copies on every
    call: affordable for a one-shot forward, and not affordable for anything
    that runs the session in a loop.

    The same path serves the CPU provider. There is no transfer to remove there,
    but one code path that is exercised by every target beats a second one that
    only the CPU tests ever reach.
    """

    def __init__(self, session: Any, *, device_type: str = "cpu", device_id: int = 0) -> None:
        self._session = session
        self._input_names = tuple(value.name for value in session.get_inputs())
        self._output_names = tuple(value.name for value in session.get_outputs())
        self._device_type = device_type
        self._device_id = device_id
        self._device = torch.device(
            device_type if device_type == "cpu" else f"{device_type}:{device_id}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        feeds = order_feeds(self._input_names, args, kwargs)
        binding = self._session.io_binding()
        # bind_input takes a raw address, so each tensor has to stay referenced
        # until the run is over -- and has to be the contiguous, on-device tensor
        # whose address was bound, not the one the caller passed.
        bound: list[torch.Tensor] = []
        try:
            for name in self._input_names:
                tensor = feeds[name].detach().to(self._device).contiguous()
                bound.append(tensor)
                binding.bind_input(
                    name,
                    self._device_type,
                    self._device_id,
                    _element_type(tensor.dtype),
                    tuple(tensor.shape),
                    tensor.data_ptr(),
                )
            for name in self._output_names:
                binding.bind_output(name, self._device_type, self._device_id)
            binding.synchronize_inputs()
            self._session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            # from_dlpack aliases ORT's buffer; the clone is what lets the binding
            # and its OrtValues be released when this call returns.
            outputs = tuple(torch.from_dlpack(value).clone() for value in binding.get_outputs())
        except Exception as exc:
            raise RuntimeError(f"ONNX Runtime execution failed: {exc}.") from exc
        return outputs[0] if len(outputs) == 1 else outputs


def order_feeds(
    input_names: tuple[str, ...],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Match call arguments to session inputs, by name before position.

    A keyword tensor binds to the input of the same name wherever that input sits
    in the capture order, because the exporter's order need not be the caller's.
    Everything left over fills the remaining inputs left to right.

    Kept separate from :class:`_ONNXRuntimeCallable` so the fast test suite can
    exercise it without onnxruntime installed: the binding itself needs real
    tensor addresses and cannot be driven by a fake session.
    """
    named = {
        name: value
        for name, value in kwargs.items()
        if name in input_names and isinstance(value, torch.Tensor)
    }
    positional = {name: value for name, value in kwargs.items() if name not in named}
    tensors = _flatten_tensors((args, positional))
    remaining = tuple(name for name in input_names if name not in named)
    if len(tensors) != len(remaining):
        raise ValueError(
            f"ONNX artifact expects {len(input_names)} tensor inputs, "
            f"got {len(named) + len(tensors)}."
        )
    named.update(dict(zip(remaining, tensors, strict=True)))
    return named


def external_data_path(model_path: Path) -> Path:
    """The weights sidecar beside ``model_path``, whether or not it exists."""
    model_path = Path(model_path)
    return model_path.with_name(model_path.name + EXTERNAL_DATA_SUFFIX)


def _reconcile_external_data(model_path: Path) -> None:
    """Make the sidecar on disk agree with what the graph actually references.

    The exporter decides this for itself in both directions. It keeps tensors
    below roughly a kilobyte inline whatever it was asked for, leaving a small
    model with a zero-byte sidecar -- deleted rather than packaged, since
    checksumming a payload the graph never reads would put a meaningless entry
    in the manifest. Above protobuf's 2 GiB ceiling it does the opposite and
    writes a sidecar even for ``external_data=False``, which is why this runs on
    every export rather than only the ones that asked for one.

    The reverse case is the one worth raising on. The sidecar's name is the
    exporter's convention rather than a promise, and the graph refers to it
    relatively -- so if it ever changes, the export still validates while the
    artifact ships a graph pointing at a file nobody packaged.
    """
    weights_path = external_data_path(model_path)
    locations = _external_locations(model_path)
    if not locations:
        weights_path.unlink(missing_ok=True)
        return
    if locations != {weights_path.name}:
        raise RuntimeError(
            f"the exporter wrote external data to {sorted(locations)} rather than "
            f"{weights_path.name}, which is the only sidecar the artifact packages"
        )


def _external_locations(model_path: Path) -> set[str]:
    """Sidecar filenames the graph references, read without loading the weights."""
    onnx = _import_module("onnx")
    model = onnx.load(str(model_path), load_external_data=False)
    return {
        entry.value
        for initializer in model.graph.initializer
        for entry in initializer.external_data
        if entry.key == "location"
    }


def _element_type(dtype: torch.dtype) -> Any:
    """The element type ``bind_input`` wants for a torch dtype.

    numpy is imported here rather than at module scope because this module is
    imported on every ``import lm7`` -- through ``exporting`` -- while numpy
    belongs to the optional onnxruntime extra.
    """
    if dtype == torch.bfloat16:
        return _BFLOAT16_ONNX_TYPE
    numpy = _import_module("numpy")
    try:
        return numpy.dtype(str(dtype).removeprefix("torch.")).type
    except TypeError as exc:
        raise RuntimeError(f"ONNX Runtime has no input binding for {dtype}.") from exc


def _weight_bytes(exported_program: torch.export.ExportedProgram) -> int:
    """What the initializers will occupy, which is what decides embedded vs sidecar.

    Tied weights are one allocation and are counted once, matching how the rest
    of LM7 reports weight size.
    """
    seen: set[int] = set()
    total = 0
    values = (
        *exported_program.state_dict.values(),
        *getattr(exported_program, "constants", {}).values(),
    )
    for tensor in values:
        if not isinstance(tensor, torch.Tensor):
            continue
        address = tensor.untyped_storage().data_ptr()
        if address in seen:
            continue
        seen.add(address)
        total += tensor.numel() * tensor.element_size()
    return total


def _remove_model(model_path: Path | None) -> None:
    """Delete a half-written export, sidecar included."""
    if model_path is None:
        return
    model_path.unlink(missing_ok=True)
    external_data_path(model_path).unlink(missing_ok=True)


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
    external_data = values.pop("external_data", "auto")
    if external_data != "auto" and not isinstance(external_data, bool):
        raise CompilationError("ONNX external_data must be True, False, or 'auto'.")
    if values:
        raise CompilationError(f"Unsupported ONNX Runtime options: {', '.join(sorted(values))}.")
    return ONNXRuntimeOptions(
        provider,
        provider_options,
        disable_cpu_fallback,
        opset_version,
        optimize,
        external_data,
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
