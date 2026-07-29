from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from ..cache import cache_dir
from ..errors import ArtifactLoadError, CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support

# The three files a PJRT consumer actually needs, relative to the unpacked tree.
# `forward.meta` is the load-bearing one: it labels every input position as a
# baked parameter, a baked constant, or a runtime argument, so a loader can
# rebuild the call signature with no model definition present.
PROGRAM_ENTRY = "functions/forward.bytecode"
PROGRAM_TEXT_ENTRY = "functions/forward.mlir"
PROGRAM_META_ENTRY = "functions/forward.meta"


def _scratch_dir(prefix: str) -> Path:
    """Return a private directory under the LM7 cache, creating the cache if new.

    ``cache_dir()`` reports where the cache belongs; it does not guarantee the
    directory exists, and ``mkdtemp`` will not create parents.
    """
    root = cache_dir() / "stablehlo"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


def _has_keyword_inputs(exported_program: torch.export.ExportedProgram) -> bool:
    """Report whether a captured program takes any keyword input.

    ``in_spec`` is a two-child pytree: positional args, then kwargs. A program
    captured positionally has zero leaves in the second child.
    """
    call_spec = getattr(exported_program, "call_spec", None)
    in_spec = getattr(call_spec, "in_spec", None)
    if in_spec is None:
        return False
    children = in_spec.children() if hasattr(in_spec, "children") else None
    if children is None:
        children = getattr(in_spec, "children_specs", None)
    if not children or len(children) < 2:
        return False
    return bool(getattr(children[1], "num_leaves", 0))


class StableHLOBackend:
    """Export-only backend lowering an ExportedProgram to StableHLO.

    This is the one LM7 payload that is both PyTorch-free and vendor-neutral.
    An AOTInductor package needs PyTorch to load, and OpenVINO IR runs without
    PyTorch but only on Intel. StableHLO is consumed by a PJRT plugin, and the
    plugin is chosen by whoever loads the artifact -- so the *same bytes* run on
    a CPU or an NVIDIA GPU. See ``docs/stablehlo-pjrt-evaluation.md`` for the
    measurements behind that claim.

    Because the target is a load-time choice, this backend deliberately does not
    gate on ``request.target.vendor`` the way ``aot_inductor`` and ``openvino``
    do. The captured payload is target-independent.
    """

    name = "stablehlo"

    def probe(self) -> BackendInfo:
        try:
            installed = importlib.util.find_spec("torch_xla") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            return BackendInfo(
                self.name,
                None,
                False,
                "PyTorch/XLA is not installed; it provides the StableHLO lowering. "
                'Install LM7 with ".[stablehlo]" in an environment whose PyTorch '
                "matches the torch_xla release.",
            )
        try:
            version = importlib.metadata.version("torch-xla")
        except importlib.metadata.PackageNotFoundError:
            version = None
        return BackendInfo(self.name, version, True, "PyTorch/XLA can lower to StableHLO.")

    def supports(self, request: CompileRequest) -> Support:
        # Registered so `lm7 backends` and `lm7 explain` can report it, but there
        # is no JIT path here: compiling a module in-process through PyTorch/XLA
        # is what the `openxla` backend already does.
        del request
        return Support(
            False,
            "stablehlo is an export-only backend; use "
            "lm7.export(..., backend='stablehlo') to write a portable artifact.",
        )

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        del request, example_args, example_kwargs
        raise CompilationError(
            "The stablehlo backend does not compile in-process. Use "
            "lm7.export(..., backend='stablehlo'), or backend='openxla' for JIT execution."
        )

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        if artifact.path is None:
            raise ArtifactLoadError("StableHLO artifact has no package path.")
        return self.load_package(artifact.path)

    def compile_exported(
        self,
        exported_program: torch.export.ExportedProgram,
        package_path: Path,
    ) -> Path:
        """Lower to StableHLO and pack the tree into a single zip.

        ``save_as_stablehlo`` writes a directory -- one ``.npy`` per weight, so a
        135M-parameter model lands as roughly 280 files. LM7's manifest records
        one payload name and one checksum, so the tree is zipped rather than
        copied in loose. Stored uncompressed: the payload is mostly weights,
        which do not compress usefully, and stored entries can be read without
        inflating the whole archive.
        """
        probe = self.probe()
        if not probe.available:
            raise CompilationError(probe.reason)
        try:
            stablehlo = importlib.import_module("torch_xla.stablehlo")
        except ImportError as exc:  # pragma: no cover - probe already guards this
            raise CompilationError(f"PyTorch/XLA could not be imported: {exc}.") from exc

        # torch_xla's lowering fails on a program with keyword inputs, and the
        # message it raises ("Export to stablehlo doesnt support kwargs yet.")
        # surfaces far from the call that chose them. Say so up front instead.
        if _has_keyword_inputs(exported_program):
            raise CompilationError(
                "PyTorch/XLA cannot lower an ExportedProgram captured with keyword "
                "inputs to StableHLO. Capture the model with positional args, for "
                "example args=(input_ids, attention_mask) rather than kwargs."
            )

        staging = _scratch_dir("package-")
        try:
            tree = staging / "stablehlo"
            stablehlo.save_as_stablehlo(exported_program, str(tree))
            missing = [
                entry
                for entry in (PROGRAM_ENTRY, PROGRAM_META_ENTRY)
                if not (tree / entry).is_file()
            ]
            if missing:
                raise CompilationError(
                    f"PyTorch/XLA wrote a StableHLO tree without {', '.join(missing)}; "
                    "LM7 cannot package an artifact a PJRT client could not load."
                )
            package_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_STORED) as archive:
                for item in sorted(tree.rglob("*")):
                    if item.is_file():
                        archive.write(item, item.relative_to(tree).as_posix())
        except CompilationError:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise CompilationError(
                f"StableHLO lowering failed for {package_path}: {exc}. Verify the "
                "PyTorch/XLA installation matches this PyTorch build."
            ) from exc
        shutil.rmtree(staging, ignore_errors=True)
        return package_path

    def load_package(self, package_path: Path) -> Callable[..., Any]:
        """Return a torch-callable for the packaged StableHLO.

        This path needs PyTorch/XLA, because turning StableHLO back into
        something that accepts torch tensors is what PyTorch/XLA does. The
        PyTorch-free route is to unpack the archive and hand
        ``functions/forward.bytecode`` to a PJRT client directly -- see
        ``benchmarks/stablehlo_pjrt.py``.
        """
        probe = self.probe()
        if not probe.available:
            raise ArtifactLoadError(probe.reason)
        try:
            stablehlo = importlib.import_module("torch_xla.stablehlo")
            unpacked = _scratch_dir("load-")
            with zipfile.ZipFile(package_path) as archive:
                archive.extractall(unpacked)
            # A zip stores no empty directories, and a model with no baked
            # constants (or no weights) leaves one behind. The loader stats both
            # unconditionally, so recreate them rather than ship empty entries.
            for required in ("constants", "data"):
                (unpacked / required).mkdir(exist_ok=True)
            return stablehlo.StableHLOGraphModule.load(str(unpacked))
        except Exception as exc:
            raise ArtifactLoadError(
                f"StableHLO package load failed for {package_path}: {exc}. "
                "Use a PyTorch/XLA build compatible with this PyTorch runtime."
            ) from exc

    def program_entries(self, package_path: Path) -> tuple[str, ...]:
        """List the archive members, so callers can confirm what shipped."""
        try:
            with zipfile.ZipFile(package_path) as archive:
                return tuple(archive.namelist())
        except (OSError, zipfile.BadZipFile) as exc:
            raise ArtifactLoadError(f"StableHLO package {package_path} is unreadable: {exc}.")
