from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..errors import ArtifactLoadError, CompilationError
from ..targets import TargetSpec
from .base import Artifact, BackendInfo, CompileRequest, Support
from .executorch import (
    _delegate_counts,
    _executorch_version,
    _flatc_on_path,
    _flatc_path,
)

DELEGATE = "qnn"
SOC_MODEL = "SM8750"
HTP_ARCH = "v79"
VTCM_MB = 8
RUNTIME_LIBRARIES = (
    "libqnn_executorch_backend.so",
    "libQnnSystem.so",
    "libQnnHtp.so",
    "libQnnHtpV79Stub.so",
    "libQnnHtpV79Skel.so",
    "libQnnHtpPrepare.so",
)


@dataclass(frozen=True)
class QNNLoweredProgram:
    """A device-bound QNN PTE plus its delegate coverage."""

    path: Path
    delegated_calls: int
    total_calls: int
    soc_model: str
    htp_arch: str
    precision: str


@dataclass(frozen=True)
class QNNOptions:
    precision: str = "fp16"


def parse_options(options: Mapping[str, Any] | None) -> QNNOptions:
    values = dict(options or {})
    precision = values.pop("precision", "fp16")
    if precision != "fp16":
        raise CompilationError(
            f"QNN precision must be 'fp16' in this initial backend; got {precision!r}. "
            "Quantized QNN modes are a separate integration."
        )
    if values:
        raise CompilationError(f"Unsupported QNN options: {', '.join(sorted(values))}.")
    return QNNOptions(precision=precision)


class ExecuTorchQNNBackend:
    """Export-only ExecuTorch QNN backend for Snapdragon 8 Elite HTP."""

    name = "qnn"

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
                "ExecuTorch is not installed; install the version-matched executorch "
                "environment before configuring Qualcomm QNN.",
            )

        version = _executorch_version()
        sdk_value = os.environ.get("QNN_SDK_ROOT")
        if not sdk_value:
            return BackendInfo(
                self.name,
                version,
                False,
                "QNN_SDK_ROOT is not set. Install Qualcomm AI Engine Direct SDK and "
                "source its bin/envsetup.sh before exporting.",
            )
        sdk_root = Path(sdk_value).expanduser()
        if not sdk_root.is_dir() or not (sdk_root / "QNN_README.txt").is_file():
            return BackendInfo(
                self.name,
                version,
                False,
                f"QNN_SDK_ROOT={sdk_root} is not an AI Engine Direct SDK root; "
                "QNN_README.txt is missing.",
            )

        try:
            _qnn_modules()
        except (ImportError, OSError) as exc:
            return BackendInfo(
                self.name,
                version,
                False,
                "ExecuTorch's Qualcomm export path could not be imported: "
                f"{exc}. Source QNN_SDK_ROOT/bin/envsetup.sh and use the matching "
                "ExecuTorch source checkout.",
            )
        if _flatc_path() is None:
            return BackendInfo(
                self.name,
                version,
                False,
                "ExecuTorch is installed but its flatc serializer could not be located.",
            )
        return BackendInfo(
            self.name,
            version,
            True,
            f"ExecuTorch QNN can lower static FP16 graphs for {SOC_MODEL} HTP {HTP_ARCH}.",
        )

    def supports(self, request: CompileRequest) -> Support:
        del request
        return Support(
            False,
            "qnn is an export-only backend; use lm7.export(..., "
            "target='qualcomm:sm8750', backend='qnn').",
        )

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        del request, example_args, example_kwargs
        raise CompilationError(
            "The qnn backend does not compile in-process. Export a device-bound "
            "ExecuTorch .pte for a Qualcomm Android runtime."
        )

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        if artifact.path is None:
            raise ArtifactLoadError("QNN artifact has no .pte path.")
        return self.load_pte(artifact.path)

    def compile_exported(
        self,
        exported_program: torch.export.ExportedProgram,
        program_path: Path,
        *,
        target: TargetSpec,
        options: Mapping[str, Any] | None = None,
    ) -> QNNLoweredProgram:
        """Lower one fixed-shape FP16 graph to QNN HTP and write its PTE."""
        probe = self.probe()
        if not probe.available:
            raise CompilationError(probe.reason)
        if target.vendor != "qualcomm" or target.model != "sm8750":
            raise CompilationError("The initial QNN backend requires target='qualcomm:sm8750'.")
        settings = parse_options(options)
        if exported_program.range_constraints:
            raise CompilationError(
                "The initial QNN backend requires static shapes; export one artifact "
                "per input shape."
            )
        try:
            example_args, example_kwargs = exported_program.example_inputs
        except (AttributeError, TypeError) as exc:
            raise CompilationError(
                "QNN lowering requires example inputs stored in the ExportedProgram."
            ) from exc
        if example_kwargs:
            raise CompilationError(
                "QNN lowering currently requires positional tensor inputs; capture with "
                "args=(...) rather than kwargs."
            )
        if not example_args or any(not isinstance(item, torch.Tensor) for item in example_args):
            raise CompilationError(
                "QNN lowering currently requires one or more positional tensor inputs."
            )

        try:
            utils, schema = _qnn_modules()
            backend_options = utils.generate_htp_compiler_spec(use_fp16=True)
            compiler_specs = utils.generate_qnn_executorch_compiler_spec(
                soc_model=schema.QcomChipset.SM8750,
                backend_options=backend_options,
            )
            with _flatc_on_path():
                edge = utils.to_edge_transform_and_lower_to_qnn(
                    exported_program.module(),
                    tuple(example_args),
                    compiler_specs,
                )
                lowered = edge.to_executorch()
                delegated, total = _delegate_counts(lowered)
                if delegated == 0:
                    raise CompilationError(
                        "QNN lowering delegated zero call sites; refusing to emit an "
                        "artifact that would silently run only portable kernels."
                    )
                program_path.parent.mkdir(parents=True, exist_ok=True)
                program_path.write_bytes(lowered.buffer)
        except CompilationError:
            raise
        except Exception as exc:
            raise CompilationError(
                f"ExecuTorch QNN lowering failed for {program_path}: {exc}. Verify "
                "QNN_SDK_ROOT, LD_LIBRARY_PATH, ExecuTorch/QNN versions, and operator "
                "support for the captured graph."
            ) from exc

        return QNNLoweredProgram(
            path=program_path,
            delegated_calls=delegated,
            total_calls=total,
            soc_model=SOC_MODEL,
            htp_arch=HTP_ARCH,
            precision=settings.precision,
        )

    def load_pte(self, program_path: Path) -> Callable[..., Any]:
        path = Path(program_path)
        if not path.is_file():
            raise ArtifactLoadError(f"QNN program does not exist: {path}.")
        return _QNNDeploymentOnly(path)

    def sdk_version(self) -> str | None:
        value = os.environ.get("QNN_SDK_ROOT")
        if not value:
            return None
        root = Path(value).expanduser()
        return root.name if (root / "QNN_README.txt").is_file() else None


class _QNNDeploymentOnly:
    """Callable guard preventing a QNN artifact from masquerading as host execution."""

    def __init__(self, program_path: Path) -> None:
        self._program_path = program_path

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ArtifactLoadError(
            f"{self._program_path.name} is bound to {SOC_MODEL} HTP {HTP_ARCH} and "
            "cannot execute through LM7's host Python process. Use an Android "
            "ExecuTorch runtime built with the QNN backend and matching QNN libraries."
        )

    def __repr__(self) -> str:
        return f"<QNNDeploymentOnly {self._program_path.name} for {SOC_MODEL}>"


def _qnn_modules() -> tuple[Any, Any]:
    importlib.import_module("executorch.backends.qualcomm.partition.qnn_partitioner")
    utils = importlib.import_module("executorch.backends.qualcomm.utils.utils")
    schema = importlib.import_module("executorch.backends.qualcomm.serialization.qc_schema")
    return utils, schema
