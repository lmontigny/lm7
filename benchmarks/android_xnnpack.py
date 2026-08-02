"""Prepare and validate an LM7 ExecuTorch XNNPACK artifact on Android ARM64.

The default workload is a deterministic, small float32 MLP. The host exports it
through LM7, verifies the resulting ``.pte`` against eager PyTorch, and wraps it
as an ExecuTorch BundledProgram containing its input and expected output. With
``--prepare-only`` this needs no Android device. Without that flag, the harness
pushes the bundle and an ARM64 ``example_runner`` to an adb-reachable device and
uses the runner's strict output verification as the on-device correctness gate.

An optional externally managed adb endpoint can be supplied with ``--adb-host``
and ``--adb-port``. The script does not provision devices or connectivity.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

import lm7
from lm7.backends.executorch import _flatc_on_path
from lm7.exporting import COMPILED_PTE_NAME

DEFAULT_DEVICE_DIR = "/data/local/tmp/lm7-xnnpack"
_READY_LINE = re.compile(r"^(?P<serial>\S+)\s+device(?:\s|$)")
_SAFE_DEVICE_DIR = re.compile(r"^/data/local/tmp/[A-Za-z0-9._/-]+$")


@dataclass(frozen=True)
class PreparedArtifact:
    artifact_dir: Path
    pte: Path
    bundled_program: Path
    max_abs_diff_host: float
    pte_bytes: int
    bundled_program_bytes: int
    delegated_calls: int
    total_calls: int


@dataclass(frozen=True)
class DeviceValidation:
    serial: str
    properties: dict[str, str]
    passed: bool


class AdbClient:
    """Small adb transport supporting local and externally managed adb servers."""

    def __init__(
        self,
        *,
        executable: str = "adb",
        host: str | None = None,
        port: int | None = None,
        serial: str | None = None,
    ) -> None:
        if (host is None) != (port is None):
            raise ValueError("adb host and port must be provided together")
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("adb port must be between 1 and 65535")
        self.executable = executable
        self.host = host
        self.port = port
        self.serial = serial

    def command(self, *arguments: str, include_serial: bool = True) -> list[str]:
        command = [self.executable]
        if self.host is not None and self.port is not None:
            command.extend(["-H", self.host, "-P", str(self.port)])
        if include_serial and self.serial is not None:
            command.extend(["-s", self.serial])
        command.extend(arguments)
        return command

    def run(
        self,
        *arguments: str,
        include_serial: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.command(*arguments, include_serial=include_serial),
            capture_output=True,
            text=True,
            check=check,
        )

    def ready_serial(self) -> str:
        completed = self.run("devices", "-l", include_serial=False)
        ready = []
        for line in completed.stdout.splitlines():
            match = _READY_LINE.match(line.strip())
            if match:
                ready.append(match.group("serial"))
        if self.serial is not None:
            if self.serial not in ready:
                raise RuntimeError(
                    f"adb device {self.serial!r} is not ready; ready devices: "
                    f"{', '.join(ready) or 'none'}"
                )
            return self.serial
        if len(ready) != 1:
            raise RuntimeError(
                f"expected one ready adb device, found {len(ready)}: "
                f"{', '.join(ready) or 'none'}; pass --serial"
            )
        self.serial = ready[0]
        return ready[0]

    def shell(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self.run("shell", *arguments, check=check)


def workload() -> tuple[torch.nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()
    return model, torch.randn(8, 16)


def _serialize_bundle(pte: Path, example: torch.Tensor, expected: torch.Tensor) -> bytes:
    try:
        from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
        from executorch.devtools.bundled_program.core import BundledProgram
        from executorch.devtools.bundled_program.serialize import (
            serialize_from_bundled_program_to_flatbuffer,
        )
    except ImportError as exc:
        raise RuntimeError(
            'BundledProgram support is unavailable; install LM7 with ".[executorch]" '
            "in the version-matched ExecuTorch environment"
        ) from exc

    suite = MethodTestSuite(
        method_name="forward",
        test_cases=[MethodTestCase(inputs=[example], expected_outputs=[expected])],
    )
    program = BundledProgram(None, [suite], pte_file_path=str(pte))
    with _flatc_on_path():
        return serialize_from_bundled_program_to_flatbuffer(program)


def prepare(output_dir: Path) -> PreparedArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, example = workload()
    with torch.no_grad():
        expected = model(example)

    artifact_dir = output_dir / "model.lm7"
    artifact = lm7.export(
        model,
        args=(example,),
        target="cpu",
        backend="executorch",
        output=artifact_dir,
    )
    with torch.no_grad():
        host_output = artifact(example)
    max_abs_diff = float((host_output - expected).abs().max())
    torch.testing.assert_close(host_output, expected, rtol=1e-4, atol=1e-4)

    pte = artifact_dir / COMPILED_PTE_NAME
    bundled_program = output_dir / "model.bpte"
    bundled_program.write_bytes(_serialize_bundle(pte, example, expected))
    requirements = artifact.manifest.runtime_requirements
    return PreparedArtifact(
        artifact_dir=artifact_dir,
        pte=pte,
        bundled_program=bundled_program,
        max_abs_diff_host=max_abs_diff,
        pte_bytes=pte.stat().st_size,
        bundled_program_bytes=bundled_program.stat().st_size,
        delegated_calls=int(requirements.get("delegated_calls", 0)),
        total_calls=int(requirements.get("total_calls", 0)),
    )


def _device_properties(client: AdbClient) -> dict[str, str]:
    wanted = {
        "model": "ro.product.model",
        "soc": "ro.soc.model",
        "platform": "ro.board.platform",
        "android_release": "ro.build.version.release",
        "android_sdk": "ro.build.version.sdk",
        "abi": "ro.product.cpu.abi",
    }
    return {
        name: client.shell("getprop", prop).stdout.strip() or "unknown"
        for name, prop in wanted.items()
    }


def validate_device(
    prepared: PreparedArtifact,
    *,
    runner: Path,
    client: AdbClient,
    device_dir: str = DEFAULT_DEVICE_DIR,
) -> DeviceValidation:
    if (
        not _SAFE_DEVICE_DIR.fullmatch(device_dir)
        or "/../" in device_dir
        or device_dir.endswith("/..")
    ):
        raise ValueError("device directory must be an absolute, simple path below /data/local/tmp")
    serial = client.ready_serial()
    client.shell("mkdir", "-p", device_dir)
    client.run("push", str(prepared.bundled_program), f"{device_dir}/model.bpte")
    client.run("push", str(runner), f"{device_dir}/example_runner")
    client.shell("chmod", "755", f"{device_dir}/example_runner")
    completed = client.shell(
        f"{device_dir}/example_runner",
        f"--bundled_program_path={device_dir}/model.bpte",
        "--output_verification",
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).strip()
        raise RuntimeError(
            f"on-device ExecuTorch verification failed with exit code "
            f"{completed.returncode}: {detail[-600:]}"
        )
    return DeviceValidation(serial, _device_properties(client), True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/android-xnnpack"),
        help="host directory for the LM7 artifact, bundle, and report",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="create and host-validate deployable files without contacting a device",
    )
    parser.add_argument("--runner", type=Path, help="ARM64 ExecuTorch example_runner")
    parser.add_argument("--adb", default="adb", help="adb executable (default: adb)")
    parser.add_argument("--adb-host", help="forwarded adb server host")
    parser.add_argument("--adb-port", type=int, help="forwarded adb server port")
    parser.add_argument("--serial", help="device serial when several devices are attached")
    parser.add_argument("--device-dir", default=DEFAULT_DEVICE_DIR)
    return parser


def _jsonable_prepared(prepared: PreparedArtifact) -> dict[str, Any]:
    data = asdict(prepared)
    for key in ("artifact_dir", "pte", "bundled_program"):
        data[key] = str(data[key])
    return data


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if (arguments.adb_host is None) != (arguments.adb_port is None):
        parser.error("--adb-host and --adb-port must be provided together")
    if not arguments.prepare_only:
        if arguments.runner is None:
            parser.error("--runner is required unless --prepare-only is used")
        if not arguments.runner.is_file():
            parser.error(f"runner not found: {arguments.runner}")
        if shutil.which(arguments.adb) is None and not Path(arguments.adb).is_file():
            parser.error(f"adb executable not found: {arguments.adb}")

    output_dir = arguments.output_dir.expanduser().resolve()
    prepared = prepare(output_dir)
    device = None
    if not arguments.prepare_only:
        client = AdbClient(
            executable=arguments.adb,
            host=arguments.adb_host,
            port=arguments.adb_port,
            serial=arguments.serial,
        )
        device = validate_device(
            prepared,
            runner=arguments.runner,
            client=client,
            device_dir=arguments.device_dir,
        )

    report = {
        "schema_version": 1,
        "workload": "deterministic-mlp-float32",
        "backend": "executorch",
        "delegate": "xnnpack",
        "prepared": _jsonable_prepared(prepared),
        "device": asdict(device) if device is not None else None,
        "device_validation": "passed" if device is not None else "not-run",
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"host validation passed (max abs diff {prepared.max_abs_diff_host:.3g})")
    print(f"bundle: {prepared.bundled_program}")
    print(f"device validation: {report['device_validation']}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
