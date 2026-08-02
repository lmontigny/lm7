from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SUPPORTED_ARCHITECTURES = frozenset({"73", "75", "79", "81"})
SUPPORTED_PYTHON_VERSIONS = frozenset({(3, 10), (3, 11)})


@dataclass(frozen=True)
class HexagonDiagnosticCheck:
    """One non-invasive Hexagon-MLIR toolchain readiness check."""

    name: str
    category: str
    status: str
    required_for: tuple[str, ...]
    value: str | None = None
    detail: str = ""
    remediation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HexagonToolchainDiagnostics:
    """Host, compiler, simulator, and device readiness for Hexagon-MLIR."""

    host: Mapping[str, str]
    checks: tuple[HexagonDiagnosticCheck, ...]
    compile_ready: bool
    simulator_ready: bool
    device_ready: bool

    def ready_for(self, mode: str) -> bool:
        if mode == "compile":
            return self.compile_ready
        if mode == "simulator":
            return self.simulator_ready
        if mode == "device":
            return self.device_ready
        raise ValueError(f"Unknown Hexagon diagnostic mode {mode!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": dict(self.host),
            "compile_ready": self.compile_ready,
            "simulator_ready": self.simulator_ready,
            "device_ready": self.device_ready,
            "checks": [check.to_dict() for check in self.checks],
        }


def diagnose_hexagon_toolchain(
    *, environ: Mapping[str, str] | None = None
) -> HexagonToolchainDiagnostics:
    """Inspect Hexagon-MLIR prerequisites without importing or running the toolchain."""
    env = dict(os.environ if environ is None else environ)
    os_release = _os_release()
    host = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "pytorch": _package_version("torch") or "unknown",
        "distribution": os_release.get("PRETTY_NAME", "unknown"),
    }
    checks: list[HexagonDiagnosticCheck] = []
    checks.extend(_host_checks(host, os_release))
    checks.append(_architecture_check(env))
    checks.extend(_root_checks(env))
    checks.extend(_module_checks())
    checks.extend(_executable_checks(env))
    checks.extend(_device_environment_checks(env))
    checks.extend(_advisory_environment_checks(env))

    compile_ready = _mode_ready(checks, "compile")
    simulator_ready = compile_ready and _mode_ready(checks, "simulator")
    device_ready = compile_ready and _mode_ready(checks, "device")
    return HexagonToolchainDiagnostics(
        host=host,
        checks=tuple(checks),
        compile_ready=compile_ready,
        simulator_ready=simulator_ready,
        device_ready=device_ready,
    )


def _mode_ready(checks: list[HexagonDiagnosticCheck], mode: str) -> bool:
    relevant = [check for check in checks if mode in check.required_for]
    return bool(relevant) and all(check.status == "ok" for check in relevant)


def _host_checks(
    host: Mapping[str, str], os_release: Mapping[str, str]
) -> tuple[HexagonDiagnosticCheck, ...]:
    system_ok = host["system"] == "Linux" and host["machine"] in {"x86_64", "AMD64"}
    host_check = HexagonDiagnosticCheck(
        "host_platform",
        "host",
        "ok" if system_ok else "unsupported",
        ("compile",),
        f"{host['system']} {host['machine']}",
        "Hexagon-MLIR cross-compilation requires an x86-64 Linux host.",
        "Use an x86-64 Linux environment.",
    )
    python_version = platform.python_version_tuple()
    python_pair = (int(python_version[0]), int(python_version[1]))
    python_ok = python_pair in SUPPORTED_PYTHON_VERSIONS
    python_check = HexagonDiagnosticCheck(
        "python",
        "host",
        "ok" if python_ok else "unsupported",
        ("compile",),
        host["python"],
        "Current upstream environments use Python 3.10 or recommend Python 3.11.",
        "Create a dedicated Python 3.10 or 3.11 Hexagon-MLIR environment.",
    )
    ubuntu_version = os_release.get("VERSION_ID")
    ubuntu_status = (
        "ok" if os_release.get("ID") == "ubuntu" and ubuntu_version == "22.04" else "warning"
    )
    ubuntu_check = HexagonDiagnosticCheck(
        "host_distribution",
        "host",
        ubuntu_status,
        (),
        host["distribution"],
        "Ubuntu 22.04 is the upstream recommended build host.",
        "Prefer Ubuntu 22.04 when creating the dedicated toolchain environment.",
    )
    return host_check, python_check, ubuntu_check


def _architecture_check(env: Mapping[str, str]) -> HexagonDiagnosticCheck:
    raw = env.get("HEXAGON_ARCH_VERSION", "").strip()
    architecture = raw.lower().removeprefix("v")
    if not architecture:
        return HexagonDiagnosticCheck(
            "HEXAGON_ARCH_VERSION",
            "target",
            "missing",
            ("compile",),
            remediation="Set HEXAGON_ARCH_VERSION to 73, 75, 79, or 81.",
        )
    if architecture not in SUPPORTED_ARCHITECTURES:
        return HexagonDiagnosticCheck(
            "HEXAGON_ARCH_VERSION",
            "target",
            "unsupported",
            ("compile",),
            raw,
            f"Supported architectures: {', '.join(sorted(SUPPORTED_ARCHITECTURES))}.",
            "Select the architecture matching the target Hexagon NPU.",
        )
    detail = "HexKL is not supported for v81." if architecture == "81" else ""
    return HexagonDiagnosticCheck(
        "HEXAGON_ARCH_VERSION", "target", "ok", ("compile",), f"v{architecture}", detail
    )


def _root_checks(env: Mapping[str, str]) -> tuple[HexagonDiagnosticCheck, ...]:
    return (
        _root_check(
            env,
            "HEXAGON_MLIR_ROOT",
            ("qcom_hexagon_backend", "scripts/set_local_env.sh"),
            "Clone qualcomm/hexagon-mlir and source scripts/set_local_env.sh.",
        ),
        _root_check(
            env,
            "HEXAGON_SDK_ROOT",
            ("incs", "libs/run_main_on_hexagon"),
            "Install Hexagon SDK 6.4.0.2 and set HEXAGON_SDK_ROOT.",
        ),
        _root_check(
            env,
            "HEXAGON_TOOLS",
            ("bin/hexagon-clang++",),
            "Install Hexagon Tools 19.0.02 and set HEXAGON_TOOLS.",
        ),
        _root_check(
            env,
            "HEXKL_ROOT",
            (),
            "Install Hexagon Kernel Library 1.0.0 and set HEXKL_ROOT.",
        ),
    )


def _root_check(
    env: Mapping[str, str],
    variable: str,
    markers: tuple[str, ...],
    remediation: str,
) -> HexagonDiagnosticCheck:
    value = env.get(variable, "").strip()
    if not value:
        return HexagonDiagnosticCheck(
            variable, "environment", "missing", ("compile",), remediation=remediation
        )
    root = Path(value).expanduser()
    if not root.is_dir():
        return HexagonDiagnosticCheck(
            variable,
            "environment",
            "invalid",
            ("compile",),
            str(root),
            "Directory does not exist.",
            remediation,
        )
    missing = [marker for marker in markers if not (root / marker).exists()]
    if missing:
        return HexagonDiagnosticCheck(
            variable,
            "environment",
            "invalid",
            ("compile",),
            str(root.resolve()),
            f"Missing expected paths: {', '.join(missing)}.",
            remediation,
        )
    return HexagonDiagnosticCheck(variable, "environment", "ok", ("compile",), str(root.resolve()))


def _module_checks() -> tuple[HexagonDiagnosticCheck, ...]:
    modules = (
        ("torch_mlir", "torch-mlir", "Install the torch-mlir snapshot pinned by Hexagon-MLIR."),
        (
            "triton.backends.qcom_hexagon_backend.compiler",
            "qcom_hexagon_backend",
            "Build patched Triton with qcom_hexagon_backend enabled.",
        ),
        (
            "triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher",
            "hexagon_launcher",
            "Source Hexagon-MLIR's scripts/set_local_env.sh so patched Triton is importable.",
        ),
    )
    return tuple(_module_check(module, name, remediation) for module, name, remediation in modules)


def _module_check(module: str, name: str, remediation: str) -> HexagonDiagnosticCheck:
    try:
        available = importlib.util.find_spec(module) is not None
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        available = False
    version = _package_version("torch-mlir" if module == "torch_mlir" else "triton")
    return HexagonDiagnosticCheck(
        name,
        "python",
        "ok" if available else "missing",
        ("compile",),
        version if available else None,
        module,
        "" if available else remediation,
    )


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _executable_checks(env: Mapping[str, str]) -> tuple[HexagonDiagnosticCheck, ...]:
    tools_root = Path(env["HEXAGON_TOOLS"]).expanduser() if env.get("HEXAGON_TOOLS") else None
    specifications = (
        ("cmake", None, None),
        ("ninja", None, None),
        ("git", None, None),
        ("linalg-hexagon-opt", "compile", None),
        ("linalg-hexagon-translate", "compile", None),
        ("hexagon-clang++", "compile", tools_root),
        ("hexagon-sim", "simulator", tools_root),
        ("adb", "device", None),
    )
    return tuple(_executable_check(name, mode, root) for name, mode, root in specifications)


def _executable_check(name: str, mode: str | None, root: Path | None) -> HexagonDiagnosticCheck:
    candidate = shutil.which(name)
    if candidate is None and root is not None:
        rooted = root / "bin" / name
        candidate = str(rooted) if rooted.is_file() else None
    if candidate is None:
        return HexagonDiagnosticCheck(
            name,
            "executable",
            "missing" if mode else "warning",
            (mode,) if mode else (),
            remediation=_executable_remediation(name),
        )
    path = Path(candidate).resolve()
    return HexagonDiagnosticCheck(
        name,
        "executable",
        "ok",
        (mode,) if mode else (),
        str(path),
        _command_version(path),
    )


def _executable_remediation(name: str) -> str:
    if name == "adb":
        return "Install Android platform-tools and put adb on PATH."
    if name == "hexagon-sim":
        return "Install Hexagon Tools and verify HEXAGON_TOOLS/bin/hexagon-sim."
    if name == "hexagon-clang++":
        return "Install Hexagon Tools and verify HEXAGON_TOOLS/bin/hexagon-clang++."
    if name in {"cmake", "ninja", "git"}:
        return f"Install {name}; it is needed when building Hexagon-MLIR from source."
    return "Build Hexagon-MLIR and source scripts/set_local_env.sh to update PATH."


def _command_version(path: Path) -> str:
    try:
        result = subprocess.run(
            [str(path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return "version unavailable"
    output = result.stdout.strip() or result.stderr.strip()
    return output.splitlines()[0][:200] if output else "version unavailable"


def _device_environment_checks(
    env: Mapping[str, str],
) -> tuple[HexagonDiagnosticCheck, ...]:
    return tuple(
        HexagonDiagnosticCheck(
            name,
            "device",
            "ok" if env.get(name) else "missing",
            ("device",),
            "set" if env.get(name) else None,
            "Connectivity is not tested by this command.",
            "Set the value for an adb-reachable device; the command will not open a tunnel."
            if not env.get(name)
            else "",
        )
        for name in ("ANDROID_HOST", "ANDROID_SERIAL")
    )


def _advisory_environment_checks(
    env: Mapping[str, str],
) -> tuple[HexagonDiagnosticCheck, ...]:
    variables = (
        "LLVM_PROJECT_BUILD_DIR",
        "TRITON_SHARED_OPT_PATH",
        "HEXAGON_RUNTIME_LIBS_DIR",
    )
    return tuple(
        HexagonDiagnosticCheck(
            name,
            "environment",
            "ok" if env.get(name) else "warning",
            (),
            env.get(name),
            "Optional for some build or runtime paths; set by the upstream environment scripts.",
            "Source scripts/set_local_env.sh if the selected workflow needs this value."
            if not env.get(name)
            else "",
        )
        for name in variables
    )


def _os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value.strip().strip('"')
    except OSError:
        return {}
    return values
