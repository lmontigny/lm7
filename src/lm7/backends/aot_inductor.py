from __future__ import annotations

import contextlib
import os
import shutil
import site
import sys
import sysconfig
import tempfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch

from ..cache import cache_dir
from ..detection import torch_device
from ..errors import ArtifactLoadError, CompilationError
from ..targets import TargetSpec
from .base import Artifact, BackendInfo, CompileRequest, Support

# Vendors LM7 will package AOTInductor output for. CPU, Apple and NVIDIA are
# validated on physical hardware; AMD is not -- no AMD GPU has run LM7 at all --
# and is here because `exporting._ARCHITECTURE_BOUND_VENDORS` already lists
# `amd` for this backend, so the artifact gate was wired for a payload nothing
# could produce. See docs/limitations.md#hardware-validation.
SUPPORTED_VENDORS = frozenset({"cpu", "apple", "nvidia", "amd"})

# Vendors that reach the GPU through CUDA, and therefore need a CUDA toolkit at
# package time. JIT Inductor does not: it generates Triton kernels and compiles
# them through Triton's own bundled PTX path. AOTInductor additionally compiles
# and links a C++ wrapper against the CUDA headers, so a CUDA target needs
# headers the PyTorch wheel does not ship.
#
# AMD is deliberately not in here. ROCm reaches the GPU through `torch.cuda`,
# but the packaging problem is not the same one: the CUDA case is a *partial*
# toolkit, where the PyTorch wheel bundles the runtime headers and omits the
# compiler front end, so LM7 has to find and splice in the missing half. ROCm
# ships as one tree under `/opt/rocm` and either is or is not installed, which
# `_rocm_home` answers directly.
_CUDA_VENDORS = frozenset({"nvidia"})

# Vendors whose wrapper build needs a ROCm installation on the host.
_ROCM_VENDORS = frozenset({"amd"})

# Where ROCm installs, in the order the ROCm build tooling itself looks. A
# PyTorch ROCm wheel links against this tree rather than bundling it, so the
# wrapper build needs it present.
_ROCM_ENVIRONMENT_VARIABLES = ("ROCM_HOME", "ROCM_PATH")
_ROCM_DEFAULT_ROOT = Path("/opt/rocm")

_ROCM_HINT = (
    "install ROCm (the PyTorch ROCm wheel links against it rather than bundling it), "
    "or set ROCM_HOME to an existing installation"
)

# The PyTorch CUDA wheel bundles the runtime headers but not the compiler front
# end, so `crt/host_defines.h` (nvidia-cuda-crt) and `nv/target` (nvidia-cuda-cccl)
# are missing and the wrapper build fails deep inside g++. Probe for them by name
# so the failure is reported before compilation starts.
_CUDA_TOOLKIT_HEADERS = ("include/crt/host_defines.h", "include/nv/target")

# WSL keeps the CUDA driver library outside the default linker search path, so an
# otherwise complete setup fails to link the wrapper with `cannot find -lcuda`.
_WSL_DRIVER_DIR = Path("/usr/lib/wsl/lib")

_CUDA_TOOLKIT_HINT = (
    'install the CUDA toolkit wheels with `uv pip install -e ".[cuda-aot]"`, or set '
    "CUDA_HOME to a CUDA toolkit installation"
)


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


def _site_package_dirs() -> list[Path]:
    candidates = [sysconfig.get_paths().get("purelib"), *sys.path]
    with contextlib.suppress(AttributeError):
        candidates.extend(site.getsitepackages())
    seen: dict[str, Path] = {}
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        seen.setdefault(str(path), path)
    return list(seen.values())


def _cuda_home_candidates() -> list[Path]:
    """Return every plausible CUDA toolkit root, most explicit first."""
    candidates: list[Path] = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value))
    # PyTorch's CUDA wheels unpack into `nvidia/cu13`, and the toolkit wheels
    # unpack into that same tree, so one directory holds runtime and compiler.
    cuda_version = getattr(torch.version, "cuda", None)
    if cuda_version:
        major = cuda_version.split(".")[0]
        for directory in _site_package_dirs():
            candidates.append(directory / "nvidia" / f"cu{major}")
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc).resolve().parent.parent)
    candidates.append(Path("/usr/local/cuda"))
    return candidates


def _cuda_toolkit_home() -> Path | None:
    """Return the first CUDA root holding the headers AOTInductor compiles against."""
    for candidate in _cuda_home_candidates():
        if all((candidate / header).is_file() for header in _CUDA_TOOLKIT_HEADERS):
            return candidate
    return None


def _cuda_driver_library_dirs(cuda_home: Path | None) -> list[Path]:
    """Return directories holding a linkable `libcuda.so`.

    The wrapper links against the driver library, and the toolkit wheels ship no
    stub for it. Distributions that install the driver leave `libcuda.so` on the
    default linker path; the candidates here cover the hosts that do not.
    """
    candidates = [_WSL_DRIVER_DIR]
    if cuda_home is not None:
        candidates.extend([cuda_home / "lib64" / "stubs", cuda_home / "lib" / "stubs"])
    return [directory for directory in candidates if (directory / "libcuda.so").is_file()]


@contextlib.contextmanager
def _cuda_build_environment(target: TargetSpec) -> Iterator[None]:
    """Point the AOTInductor wrapper build at the CUDA toolkit LM7 found.

    Only fills in what the caller has not set, so an explicit CUDA_HOME still
    wins. `torch.utils.cpp_extension` resolves CUDA_HOME once at import time and
    inductor has already imported it by now, so the module attribute has to be
    overridden alongside the environment variable.
    """
    if target.vendor not in _CUDA_VENDORS:
        yield
        return

    cuda_home = _cuda_toolkit_home()
    overrides: dict[str, str] = {}
    if cuda_home is not None and not any(
        os.environ.get(name) for name in ("CUDA_HOME", "CUDA_PATH")
    ):
        overrides["CUDA_HOME"] = str(cuda_home)
    link_dirs = _cuda_driver_library_dirs(cuda_home)
    if link_dirs:
        existing = os.environ.get("LIBRARY_PATH")
        entries = [str(directory) for directory in link_dirs]
        if existing:
            entries.append(existing)
        overrides["LIBRARY_PATH"] = os.pathsep.join(entries)

    cpp_extension = getattr(torch.utils, "cpp_extension", None)
    previous_env = {name: os.environ.get(name) for name in overrides}
    patch_module = cuda_home is not None and cpp_extension is not None
    previous_module_home = getattr(cpp_extension, "CUDA_HOME", None) if patch_module else None
    try:
        os.environ.update(overrides)
        if patch_module and not previous_module_home:
            assert cpp_extension is not None
            cpp_extension.CUDA_HOME = str(cuda_home)
        yield
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if patch_module and not previous_module_home:
            assert cpp_extension is not None
            cpp_extension.CUDA_HOME = previous_module_home


def _rocm_home() -> Path | None:
    """The ROCm installation the wrapper build will link against, if there is one."""
    for name in _ROCM_ENVIRONMENT_VARIABLES:
        value = os.environ.get(name)
        if value and Path(value).is_dir():
            return Path(value)
    return _ROCM_DEFAULT_ROOT if _ROCM_DEFAULT_ROOT.is_dir() else None


def _current_compute_capability() -> str | None:
    """The `smXX` of the GPU in this process, or None when there is not one.

    NVIDIA only. `torch.cuda.get_device_capability` answers on ROCm too -- it
    returns (9, 4) on a gfx942 -- and formatting that as `sm94` would invent an
    NVIDIA architecture that does not exist. `_current_gcn_architecture` is the
    AMD counterpart.
    """
    if getattr(torch.version, "hip", None):
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
    except (AssertionError, RuntimeError):
        return None
    return f"sm{major}{minor}"


def _current_gcn_architecture() -> str | None:
    """The `gfxNNN` of the AMD GPU in this process, or None when there is not one.

    Normalized the same way `detection.detect_targets` normalizes it: ROCm
    reports `gfx942:sramecc+:xnack-` and the feature suffixes are not part of the
    architecture an artifact is bound to.
    """
    if not getattr(torch.version, "hip", None):
        return None
    try:
        name = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    except (AssertionError, RuntimeError):
        return None
    return str(name).split(":", 1)[0] if name else None


# Manifest keys that describe what a package was built against, and the words to
# use for them when a load fails. A manifest carries the CUDA pair or the ROCm
# pair, never both, so a hint names whichever the artifact actually recorded.
_ENVIRONMENT_LABELS = {
    "torch": "PyTorch",
    "cuda": "CUDA runtime",
    "hip": "ROCm runtime",
    "compute_capability": "GPU architecture",
    "gcn_architecture": "GPU architecture",
}


def _environment_mismatch_hint(built_with: Mapping[str, Any] | None) -> str:
    """Explain a failed load by comparing the build environment to this one.

    A package that will not load is usually a package built somewhere else, and
    the manifest already knows where. Naming the field that moved turns an
    `undefined symbol` or `no kernel image is available` into an instruction.
    Saying "re-export" when nothing differs would send the reader to the wrong
    place, so a matching environment is reported as such instead.
    """
    if not built_with:
        return " Use a compatible PyTorch runtime and target architecture."
    current: Mapping[str, str | None] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "compute_capability": _current_compute_capability(),
        "gcn_architecture": _current_gcn_architecture(),
    }
    built = ", ".join(
        f"{label} {built_with[key]}"
        for key, label in _ENVIRONMENT_LABELS.items()
        if built_with.get(key)
    )
    if not built:
        return " Use a compatible PyTorch runtime and target architecture."
    differences = [
        f"{label} {built_with[key]} -> {current[key] or 'absent'}"
        for key, label in _ENVIRONMENT_LABELS.items()
        if built_with.get(key) and built_with[key] != current[key]
    ]
    if not differences:
        return (
            f" The artifact was built with {built}, which is what this process has, so the "
            "package or its dependencies are at fault rather than the environment."
        )
    runtime = "ROCm" if built_with.get("hip") else "CUDA"
    return (
        f" The artifact was built with {built}, and this process differs: "
        f"{'; '.join(differences)}. An AOTInductor package holds kernels compiled for one "
        f"architecture and a wrapper linked against one {runtime} runtime, so re-export the "
        "model on this machine."
    )


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
        if request.target.vendor not in SUPPORTED_VENDORS:
            return Support(
                False,
                "LM7 packages AOTInductor output for CPU, Apple Silicon, NVIDIA, and AMD "
                f"targets only; {request.target} is none of those.",
            )
        if request.target.vendor in _CUDA_VENDORS and _cuda_toolkit_home() is None:
            return Support(
                False,
                "AOTInductor needs a CUDA toolkit to build its wrapper for a CUDA "
                f"target, and LM7 found none: {_CUDA_TOOLKIT_HINT}.",
            )
        if request.target.vendor in _ROCM_VENDORS and _rocm_home() is None:
            return Support(
                False,
                "AOTInductor needs a ROCm installation to build its wrapper for an AMD "
                f"target, and LM7 found none: {_ROCM_HINT}.",
            )
        return Support(
            True,
            "AOTInductor can package an ExportedProgram for CPU, Apple, NVIDIA, or AMD execution.",
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
                self.compile_exported(
                    exported_program, package_path, request.options, target=request.target
                )
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
        *,
        target: TargetSpec | None = None,
    ) -> Path:
        probe = self.probe()
        if not probe.available:
            raise CompilationError(probe.reason)
        if target is not None and target.vendor in _CUDA_VENDORS and _cuda_toolkit_home() is None:
            raise CompilationError(
                f"AOTInductor packaging failed for {package_path}: no CUDA toolkit was "
                f"found, and the wrapper for a {target.vendor} target cannot be built "
                f"without one. To fix this, {_CUDA_TOOLKIT_HINT}."
            )
        if target is not None and target.vendor in _ROCM_VENDORS and _rocm_home() is None:
            raise CompilationError(
                f"AOTInductor packaging failed for {package_path}: no ROCm installation "
                f"was found, and the wrapper for a {target.vendor} target cannot be built "
                f"without one. To fix this, {_ROCM_HINT}."
            )
        configs = dict(options or {})
        try:
            with _cuda_build_environment(target) if target else contextlib.nullcontext():
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

    def load_package(
        self,
        package_path: Path,
        *,
        built_with: Mapping[str, Any] | None = None,
    ) -> Callable[..., Any]:
        probe = self.probe()
        if not probe.available:
            raise ArtifactLoadError(probe.reason)
        try:
            return torch._inductor.aoti_load_package(str(package_path))
        except Exception as exc:
            raise ArtifactLoadError(
                f"AOTInductor package load failed for {package_path}: {exc}."
                f"{_environment_mismatch_hint(built_with)}"
            ) from exc
