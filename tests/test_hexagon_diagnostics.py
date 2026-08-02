from __future__ import annotations

import json
from pathlib import Path

from lm7 import cli
from lm7.hexagon import (
    HexagonDiagnosticCheck,
    HexagonToolchainDiagnostics,
    diagnose_hexagon_toolchain,
)


def _host(monkeypatch) -> None:
    monkeypatch.setattr("lm7.hexagon.platform.system", lambda: "Linux")
    monkeypatch.setattr("lm7.hexagon.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("lm7.hexagon.platform.python_version", lambda: "3.11.9")
    monkeypatch.setattr("lm7.hexagon.platform.python_version_tuple", lambda: ("3", "11", "9"))
    monkeypatch.setattr(
        "lm7.hexagon._os_release",
        lambda: {"ID": "ubuntu", "VERSION_ID": "22.04", "PRETTY_NAME": "Ubuntu 22.04"},
    )


def _file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_missing_toolchain_reports_actionable_failures(monkeypatch):
    _host(monkeypatch)
    monkeypatch.setattr("lm7.hexagon.shutil.which", lambda name: None)
    monkeypatch.setattr("lm7.hexagon.importlib.util.find_spec", lambda name: None)

    result = diagnose_hexagon_toolchain(environ={})

    assert result.compile_ready is False
    assert result.simulator_ready is False
    assert result.device_ready is False
    checks = {check.name: check for check in result.checks}
    assert checks["HEXAGON_SDK_ROOT"].status == "missing"
    assert "SDK 6.4.0.2" in checks["HEXAGON_SDK_ROOT"].remediation
    assert checks["torch-mlir"].status == "missing"
    assert checks["ANDROID_HOST"].detail == "Connectivity is not tested by this command."


def test_complete_toolchain_is_ready_for_all_modes(tmp_path, monkeypatch):
    _host(monkeypatch)
    mlir = tmp_path / "hexagon-mlir"
    sdk = tmp_path / "Hexagon_SDK" / "6.4.0.2"
    tools = tmp_path / "Tools"
    hexkl = tmp_path / "hexkl_addon"
    (mlir / "qcom_hexagon_backend").mkdir(parents=True)
    _file(mlir / "scripts" / "set_local_env.sh")
    (sdk / "incs").mkdir(parents=True)
    (sdk / "libs" / "run_main_on_hexagon").mkdir(parents=True)
    hexkl.mkdir()
    clang = _file(tools / "bin" / "hexagon-clang++")
    simulator = _file(tools / "bin" / "hexagon-sim")
    executables = {
        "linalg-hexagon-opt": str(_file(tmp_path / "bin" / "linalg-hexagon-opt")),
        "linalg-hexagon-translate": str(_file(tmp_path / "bin" / "linalg-hexagon-translate")),
        "hexagon-clang++": str(clang),
        "hexagon-sim": str(simulator),
        "adb": str(_file(tmp_path / "bin" / "adb")),
    }
    monkeypatch.setattr("lm7.hexagon.shutil.which", executables.get)
    monkeypatch.setattr("lm7.hexagon.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("lm7.hexagon._command_version", lambda path: "test version")
    env = {
        "HEXAGON_MLIR_ROOT": str(mlir),
        "HEXAGON_SDK_ROOT": str(sdk),
        "HEXAGON_TOOLS": str(tools),
        "HEXKL_ROOT": str(hexkl),
        "HEXAGON_ARCH_VERSION": "v79",
        "ANDROID_HOST": "localhost",
        "ANDROID_SERIAL": "device-1",
    }

    result = diagnose_hexagon_toolchain(environ=env)

    assert result.compile_ready is True
    assert result.simulator_ready is True
    assert result.device_ready is True
    assert all(result.ready_for(mode) for mode in ("compile", "simulator", "device"))
    assert {check.status for check in result.checks if check.required_for} == {"ok"}


def test_v81_reports_hexkl_limitation(monkeypatch):
    _host(monkeypatch)
    monkeypatch.setattr("lm7.hexagon.shutil.which", lambda name: None)
    monkeypatch.setattr("lm7.hexagon.importlib.util.find_spec", lambda name: None)

    result = diagnose_hexagon_toolchain(environ={"HEXAGON_ARCH_VERSION": "81"})

    architecture = next(check for check in result.checks if check.name == "HEXAGON_ARCH_VERSION")
    assert architecture.status == "ok"
    assert "HexKL is not supported" in architecture.detail


def test_hexagon_doctor_cli_json(monkeypatch, capsys):
    result = HexagonToolchainDiagnostics(
        host={
            "system": "Linux",
            "machine": "x86_64",
            "python": "3.11.9",
            "distribution": "Ubuntu 22.04",
        },
        checks=(
            HexagonDiagnosticCheck(
                "host_platform", "host", "ok", ("compile", "device"), "Linux x86_64"
            ),
        ),
        compile_ready=True,
        simulator_ready=False,
        device_ready=True,
    )
    monkeypatch.setattr(cli, "diagnose_hexagon_toolchain", lambda: result)

    assert cli.main(["hexagon", "doctor", "--mode", "device", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["requested_mode"] == "device"
    assert output["ready"] is True
    assert output["checks"][0]["status"] == "ok"


def test_hexagon_doctor_cli_returns_one_when_mode_is_not_ready(monkeypatch, capsys):
    result = HexagonToolchainDiagnostics(
        host={
            "system": "Linux",
            "machine": "x86_64",
            "python": "3.12.3",
            "distribution": "Ubuntu 24.04",
        },
        checks=(
            HexagonDiagnosticCheck(
                "python",
                "host",
                "unsupported",
                ("compile",),
                "3.12.3",
                remediation="Use Python 3.10 or 3.11.",
            ),
        ),
        compile_ready=False,
        simulator_ready=False,
        device_ready=False,
    )
    monkeypatch.setattr(cli, "diagnose_hexagon_toolchain", lambda: result)

    assert cli.main(["hexagon", "doctor"]) == 1

    output = capsys.readouterr().out
    assert "Compilation: not ready" in output
    assert "Fix: Use Python 3.10 or 3.11." in output
