from __future__ import annotations

import importlib
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..errors import ArtifactLoadError, CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support
from .executorch import (
    _delegate_counts,
    _executorch_version,
    _ExecuTorchMethod,
    _flatc_on_path,
    _flatc_path,
)

DELEGATE = "coreml"

_COMPUTE_UNITS = {"all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"}
_COMPUTE_PRECISIONS = {"float16", "float32"}

_INSTALL_HINT = (
    'install LM7 with ".[executorch]" in an environment whose PyTorch matches the '
    "ExecuTorch release, on macOS"
)


@dataclass(frozen=True)
class CoreMLLoweredProgram:
    """A `.pte` lowered through the Core ML delegate plus its delegate coverage."""

    path: Path
    delegated_calls: int
    total_calls: int
    compute_unit: str
    compute_precision: str


@dataclass(frozen=True)
class CoreMLOptions:
    compute_unit: str = "all"
    compute_precision: str = "float16"


def parse_options(options: Mapping[str, Any] | None) -> CoreMLOptions:
    values = dict(options or {})
    compute_unit = values.pop("compute_unit", "all")
    if compute_unit not in _COMPUTE_UNITS:
        raise CompilationError(
            f"Unsupported Core ML compute_unit {compute_unit!r}; expected one of "
            f"{', '.join(sorted(_COMPUTE_UNITS))}."
        )
    compute_precision = values.pop("compute_precision", "float16")
    if compute_precision not in _COMPUTE_PRECISIONS:
        raise CompilationError(
            f"Unsupported Core ML compute_precision {compute_precision!r}; expected one of "
            f"{', '.join(sorted(_COMPUTE_PRECISIONS))}."
        )
    if values:
        raise CompilationError(f"Unsupported Core ML options: {', '.join(sorted(values))}.")
    return CoreMLOptions(compute_unit, compute_precision)


class ExecuTorchCoreMLBackend:
    """Export-only ExecuTorch Core ML backend for Apple Silicon (and Intel Macs).

    Unlike the ExecuTorch QNN backend, a Core ML `.pte` is not deployment-only:
    ExecuTorch's Objective-C++ Core ML runtime is part of the same package that
    does the lowering, so the artifact loads and executes through
    ``executorch.runtime.Runtime`` on the same host that built it -- the ANE,
    GPU, or CPU compute units Core ML picks are all local to this Mac. That is
    also the reason this backend requires macOS: the runtime bridge is
    Objective-C++, unlike XNNPACK's portable C++.
    """

    name = "coreml"

    def probe(self) -> BackendInfo:
        try:
            installed = importlib.util.find_spec("executorch") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            return BackendInfo(
                self.name, None, False, f"ExecuTorch is not installed; {_INSTALL_HINT}."
            )
        version = _executorch_version()
        # platform.system(), not sys.platform: mypy special-cases sys.platform
        # literal comparisons and statically narrows them to whichever OS mypy
        # itself is running on, which makes the code below "unreachable" on a
        # Linux CI runner even though it is genuinely reachable on macOS.
        if platform.system() != "Darwin":
            return BackendInfo(
                self.name,
                version,
                False,
                "Core ML compiles and executes through Apple's own framework, so it is "
                f"macOS-only; this host reports platform.system()={platform.system()!r}.",
            )
        try:
            importlib.import_module("executorch.backends.apple.coreml.partition.coreml_partitioner")
            importlib.import_module("executorch.backends.apple.coreml.compiler")
            importlib.import_module("coremltools")
        except ImportError as exc:
            return BackendInfo(
                self.name,
                version,
                False,
                f"ExecuTorch is installed but its Core ML backend could not be imported: {exc}.",
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
        del request
        return Support(
            False,
            "coreml is an export-only backend; use "
            "lm7.export(..., target='apple', backend='coreml') to write a .pte artifact.",
        )

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        del request, example_args, example_kwargs
        raise CompilationError(
            "The coreml backend does not compile in-process. Use "
            "lm7.export(..., backend='coreml') to write a .pte for Core ML execution."
        )

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        if artifact.path is None:
            raise ArtifactLoadError("Core ML artifact has no .pte path.")
        return self.load_pte(artifact.path)

    def compile_exported(
        self,
        exported_program: torch.export.ExportedProgram,
        program_path: Path,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> CoreMLLoweredProgram:
        """Lower to the Core ML delegate and write a single self-contained `.pte`.

        Like the ExecuTorch XNNPACK path, the return value's coverage ratio is
        the honest measure of what Core ML actually took -- operators it does
        not implement stay on ExecuTorch's portable kernels.
        """
        probe = self.probe()
        if not probe.available:
            raise CompilationError(probe.reason)
        settings = parse_options(options)
        if exported_program.range_constraints:
            raise CompilationError(
                "The coreml backend requires static shapes; export one artifact per input shape."
            )
        try:
            exir = importlib.import_module("executorch.exir")
            partitioner_module = importlib.import_module(
                "executorch.backends.apple.coreml.partition.coreml_partitioner"
            )
            compiler_module = importlib.import_module("executorch.backends.apple.coreml.compiler")
            ct = importlib.import_module("coremltools")
            precision_module = importlib.import_module(
                "coremltools.converters.mil.mil.passes.defs.quantization"
            )
        except ImportError as exc:  # pragma: no cover - probe already guards this
            raise CompilationError(
                f"ExecuTorch's Core ML backend could not be imported: {exc}."
            ) from exc

        try:
            compile_specs = compiler_module.CoreMLBackend.generate_compile_specs(
                compute_unit=getattr(ct.ComputeUnit, settings.compute_unit.upper()),
                compute_precision=getattr(
                    precision_module.ComputePrecision, settings.compute_precision.upper()
                ),
            )
            partitioner = partitioner_module.CoreMLPartitioner(compile_specs=compile_specs)
            with _flatc_on_path():
                lowered = exir.to_edge_transform_and_lower(
                    exported_program, partitioner=[partitioner]
                ).to_executorch()
                delegated, total = _delegate_counts(lowered)
                if delegated == 0:
                    raise CompilationError(
                        "Core ML lowering delegated zero call sites; refusing to emit an "
                        "artifact that would silently run only portable kernels."
                    )
                program_path.parent.mkdir(parents=True, exist_ok=True)
                program_path.write_bytes(lowered.buffer)
        except CompilationError:
            raise
        except Exception as exc:
            raise CompilationError(
                f"ExecuTorch Core ML lowering failed for {program_path}: {exc}. Verify the "
                "coremltools/ExecuTorch installation and operator support for the captured graph."
            ) from exc
        return CoreMLLoweredProgram(
            program_path, delegated, total, settings.compute_unit, settings.compute_precision
        )

    def load_pte(self, program_path: Path) -> Callable[..., Any]:
        """Return a torch-callable backed by Core ML, executing on this Mac.

        The first load compiles the embedded Core ML model with Apple's own
        compiler and caches it; that step can take a while for a large model.
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
                f"ExecuTorch Core ML program load failed for {program_path}: {exc}. Verify the "
                "coremltools/ExecuTorch installation on macOS."
            ) from exc
        return _ExecuTorchMethod(method, Path(program_path))
