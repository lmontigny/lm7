from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import importlib.resources
import importlib.util
import os
import shutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..errors import ArtifactLoadError, CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support

DELEGATE = "xnnpack"


@dataclass(frozen=True)
class LoweredProgram:
    """A written `.pte` plus how much of the graph the delegate actually took."""

    path: Path
    delegated_calls: int
    total_calls: int
    quantization: str
    quantized_ops: int


@dataclass(frozen=True)
class ExecuTorchOptions:
    quantization: str


def parse_options(options: Mapping[str, Any] | None) -> ExecuTorchOptions:
    values = dict(options or {})
    quantization = values.pop("quantization", "none")
    if quantization not in {"none", "int8"}:
        raise CompilationError(
            f"ExecuTorch quantization must be 'none' or 'int8'; got {quantization!r}."
        )
    if values:
        raise CompilationError(f"Unsupported ExecuTorch options: {', '.join(sorted(values))}.")
    return ExecuTorchOptions(quantization)


_INSTALL_HINT = (
    'install LM7 with ".[executorch]" in an environment whose PyTorch matches the '
    "ExecuTorch release; the prebuilt runtime extension is ABI-tied to libtorch"
)


class ExecuTorchBackend:
    """Export-only backend packaging an ExportedProgram as an ExecuTorch `.pte`.

    This is LM7's edge path. A `.pte` is the artifact Android and iOS actually
    load: it carries the program plus its weights, and the on-device C++ runtime
    executes it with no PyTorch present. Unlike every other compiled payload
    here, it is not pinned to the machine that produced it -- the XNNPACK
    delegate covers ARM64 and x86-64 alike, so the same bytes run on a phone and
    on the host that built them.

    That last property is why this backend can be tested at all. Core ML, QNN,
    and the other ExecuTorch delegates need a Mac or a vendor SDK; XNNPACK needs
    neither, so export *and* execution are validated on ordinary CI hardware.
    See ``docs/executorch.md``.
    """

    name = "executorch"

    def probe(self) -> BackendInfo:
        try:
            installed = importlib.util.find_spec("executorch") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            return BackendInfo(
                self.name,
                None,
                False,
                f"ExecuTorch is not installed; {_INSTALL_HINT}.",
            )
        version = _executorch_version()
        try:
            importlib.import_module("executorch.exir")
            importlib.import_module("executorch.backends.xnnpack.partition.xnnpack_partitioner")
        except ImportError as exc:
            return BackendInfo(
                self.name,
                version,
                False,
                f"ExecuTorch is installed but its export path could not be imported: {exc}.",
            )
        if _flatc_path() is None:
            return BackendInfo(
                self.name,
                version,
                False,
                "ExecuTorch is installed but its flatc serializer could not be located; "
                "activate the environment so the wheel's bin directory is on PATH.",
            )
        return BackendInfo(
            self.name,
            version,
            True,
            f"ExecuTorch can lower an ExportedProgram to a .pte via the {DELEGATE} delegate.",
        )

    def supports(self, request: CompileRequest) -> Support:
        # Registered so `lm7 backends` and `lm7 explain` report it, but there is
        # no JIT path: a .pte is loaded by the ExecuTorch runtime, not compiled
        # into the calling process the way `inductor` is.
        del request
        return Support(
            False,
            "executorch is an export-only backend; use "
            "lm7.export(..., backend='executorch') to write a .pte artifact.",
        )

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        del request, example_args, example_kwargs
        raise CompilationError(
            "The executorch backend does not compile in-process. Use "
            "lm7.export(..., backend='executorch') to write a .pte for on-device use."
        )

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        if artifact.path is None:
            raise ArtifactLoadError("ExecuTorch artifact has no .pte path.")
        return self.load_pte(artifact.path)

    def compile_exported(
        self,
        exported_program: torch.export.ExportedProgram,
        program_path: Path,
        options: Mapping[str, Any] | None = None,
    ) -> LoweredProgram:
        """Lower to the XNNPACK delegate and write a single self-contained `.pte`.

        Returns partition coverage alongside the path. Unlike LM7's other
        compiled payloads this one is only *partly* handed to the accelerator --
        operators XNNPACK does not implement stay on ExecuTorch's portable
        kernels -- so the ratio is the honest measure of what the delegate took.
        """
        probe = self.probe()
        if not probe.available:
            raise CompilationError(probe.reason)
        try:
            exir = importlib.import_module("executorch.exir")
            partitioner_module = importlib.import_module(
                "executorch.backends.xnnpack.partition.xnnpack_partitioner"
            )
        except ImportError as exc:  # pragma: no cover - probe already guards this
            raise CompilationError(f"ExecuTorch could not be imported: {exc}.") from exc

        settings = parse_options(options)
        quantized_ops = 0
        if settings.quantization == "int8":
            if exported_program.range_constraints:
                raise CompilationError(
                    "ExecuTorch INT8 quantization requires a fixed-shape exported program."
                )
            exported_program, quantized_ops = _quantize_int8(exported_program)
        try:
            with _flatc_on_path():
                lowered = exir.to_edge_transform_and_lower(
                    exported_program,
                    partitioner=[partitioner_module.XnnpackPartitioner()],
                ).to_executorch()
                program_path.parent.mkdir(parents=True, exist_ok=True)
                program_path.write_bytes(lowered.buffer)
            delegated, total = _delegate_counts(lowered)
        except Exception as exc:
            raise CompilationError(
                f"ExecuTorch lowering failed for {program_path}: {exc}. Verify the "
                "ExecuTorch installation matches this PyTorch build."
            ) from exc
        return LoweredProgram(
            program_path,
            delegated,
            total,
            settings.quantization,
            quantized_ops,
        )

    def load_pte(self, program_path: Path) -> Callable[..., Any]:
        """Return a torch-callable backed by the ExecuTorch runtime.

        This uses the Python bindings, which exist to validate a `.pte` on the
        build host. On a phone the same file is loaded by the C++ runtime with
        no Python and no PyTorch involved.
        """
        probe = self.probe()
        if not probe.available:
            raise ArtifactLoadError(probe.reason)
        try:
            runtime_module = importlib.import_module("executorch.runtime")
            runtime = runtime_module.Runtime.get()
            program = runtime.load_program(Path(program_path))
            method = program.load_method("forward")
        except Exception as exc:
            raise ArtifactLoadError(
                f"ExecuTorch program load failed for {program_path}: {exc}. "
                "Use an ExecuTorch build compatible with this PyTorch runtime."
            ) from exc
        return _ExecuTorchMethod(method, Path(program_path))


class _ExecuTorchMethod:
    """Adapts an ExecuTorch method to LM7's positional torch-callable convention."""

    def __init__(self, method: Any, program_path: Path) -> None:
        self._method = method
        self._program_path = program_path

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise ArtifactLoadError(
                "An ExecuTorch method takes positional tensors only; "
                f"got keyword inputs {', '.join(sorted(kwargs))}. Capture the model "
                "with positional args, for example args=(input_ids, attention_mask)."
            )
        outputs = self._method.execute(list(args))
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def __repr__(self) -> str:
        return f"<ExecuTorchMethod forward from {self._program_path.name}>"


def _delegate_counts(lowered: Any) -> tuple[int, int]:
    """Count delegate calls against total call_function nodes in a lowered program."""
    try:
        graph = lowered.exported_program().graph_module.graph
    except (AttributeError, TypeError):
        return 0, 0
    calls = [node for node in graph.nodes if node.op == "call_function"]
    delegated = [node for node in calls if "executorch_call_delegate" in str(node.target)]
    return len(delegated), len(calls)


def _quantize_int8(
    exported_program: torch.export.ExportedProgram,
) -> tuple[torch.export.ExportedProgram, int]:
    """Apply calibrated XNNPACK PT2E INT8 quantization to a captured graph."""
    try:
        quantizer_module = importlib.import_module(
            "executorch.backends.xnnpack.quantizer.xnnpack_quantizer"
        )
        quantize_module = importlib.import_module("torchao.quantization.pt2e.quantize_pt2e")
    except ImportError as exc:
        raise CompilationError(
            "ExecuTorch INT8 quantization requires the XNNPACK quantizer and "
            "TorchAO PT2E APIs shipped with ExecuTorch."
        ) from exc

    try:
        example_args, example_kwargs = exported_program.example_inputs
        quantizer = quantizer_module.XNNPACKQuantizer().set_global(
            quantizer_module.get_symmetric_quantization_config(is_per_channel=True)
        )
        prepared = quantize_module.prepare_pt2e(exported_program.module(), quantizer)
        with torch.no_grad():
            prepared(*example_args, **example_kwargs)
        quantized = quantize_module.convert_pt2e(prepared)
        quantized_ops = sum(
            node.op == "call_function" and "quantized_decomposed.quantize" in str(node.target)
            for node in quantized.graph.nodes
        )
        if quantized_ops == 0:
            raise CompilationError(
                "XNNPACK INT8 quantization matched no operators in the exported graph."
            )
        return (
            torch.export.export(quantized, example_args, example_kwargs),
            quantized_ops,
        )
    except CompilationError:
        raise
    except Exception as exc:
        raise CompilationError(
            "ExecuTorch INT8 prepare/calibrate/convert failed: "
            f"{exc}. The captured example inputs are used as the calibration sample."
        ) from exc


def _executorch_version() -> str | None:
    try:
        return importlib.metadata.version("executorch")
    except importlib.metadata.PackageNotFoundError:
        return None


def _flatc_path() -> Path | None:
    """Locate the flatbuffer compiler ExecuTorch shells out to during serialization.

    ExecuTorch looks for `flatc` as a resource beside its serializer and then
    falls back to bare `flatc` on PATH. The wheel actually ships it under
    `executorch/data/bin/`, so running the interpreter by absolute path -- rather
    than activating the environment -- leaves it unresolvable.
    """
    found = shutil.which("flatc")
    if found:
        return Path(found)
    try:
        bundled = importlib.resources.files("executorch.data.bin").joinpath("flatc")
        if bundled.is_file():
            return Path(str(bundled))
    except (ImportError, ModuleNotFoundError, TypeError, AttributeError):
        return None
    return None


@contextlib.contextmanager
def _flatc_on_path() -> Iterator[None]:
    """Put the wheel's bundled flatc on PATH for the duration of a lowering.

    A no-op when `flatc` already resolves, so an activated environment or a
    system flatc is left exactly as it is.
    """
    if shutil.which("flatc"):
        yield
        return
    bundled = _flatc_path()
    if bundled is None:  # pragma: no cover - probe already guards this
        yield
        return
    previous = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(filter(None, (str(bundled.parent), previous)))
    try:
        yield
    finally:
        os.environ["PATH"] = previous
