from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "android_xnnpack.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("benchmarks_android_xnnpack", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


def test_adb_command_supports_a_forwarded_server(harness):
    client = harness.AdbClient(
        executable="/opt/platform-tools/adb",
        host="127.0.0.1",
        port=5039,
        serial="device-1",
    )

    assert client.command("shell", "id") == [
        "/opt/platform-tools/adb",
        "-H",
        "127.0.0.1",
        "-P",
        "5039",
        "-s",
        "device-1",
        "shell",
        "id",
    ]
    assert client.command("devices", include_serial=False)[-1] == "devices"
    assert "-s" not in client.command("devices", include_serial=False)


@pytest.mark.parametrize(
    ("host", "port"),
    [("127.0.0.1", None), (None, 5039), ("127.0.0.1", 0), ("127.0.0.1", 65536)],
)
def test_adb_client_rejects_incomplete_or_invalid_endpoints(harness, host, port):
    with pytest.raises(ValueError):
        harness.AdbClient(host=host, port=port)


def test_ready_serial_selects_the_only_ready_device(harness, monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "List of devices attached\n"
                "offline-1 offline transport_id:1\n"
                "device-1 device product:sun model:Sun_for_arm64 transport_id:2\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    client = harness.AdbClient(host="127.0.0.1", port=5039)

    assert client.ready_serial() == "device-1"
    assert client.serial == "device-1"


def test_ready_serial_requires_an_explicit_choice_for_multiple_devices(harness, monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="List of devices attached\na device\nb device\n",
            stderr="",
        )

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="pass --serial"):
        harness.AdbClient().ready_serial()


def test_validate_device_pushes_bundle_and_runs_strict_verification(harness, tmp_path):
    bundle = tmp_path / "model.bpte"
    runner = tmp_path / "example_runner"
    bundle.write_bytes(b"bundle")
    runner.write_bytes(b"runner")
    prepared = harness.PreparedArtifact(
        artifact_dir=tmp_path / "model.lm7",
        pte=tmp_path / "model.pte",
        bundled_program=bundle,
        max_abs_diff_host=0.0,
        pte_bytes=1,
        bundled_program_bytes=6,
        delegated_calls=1,
        total_calls=1,
    )

    class FakeClient:
        def __init__(self):
            self.calls = []

        def ready_serial(self):
            return "device-1"

        def run(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        def shell(self, *args, **kwargs):
            self.calls.append((("shell", *args), kwargs))
            stdout = "SM8750\n" if args[:1] == ("getprop",) else ""
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    client = FakeClient()
    result = harness.validate_device(prepared, runner=runner, client=client)

    assert result.passed is True
    assert result.serial == "device-1"
    flattened = [call[0] for call in client.calls]
    assert (
        "push",
        str(bundle),
        f"{harness.DEFAULT_DEVICE_DIR}/model.bpte",
    ) in flattened
    assert any("--output_verification" in call for call in flattened)


def test_validate_device_rejects_a_path_outside_android_tmp(harness, tmp_path):
    prepared = harness.PreparedArtifact(
        artifact_dir=tmp_path,
        pte=tmp_path / "model.pte",
        bundled_program=tmp_path / "model.bpte",
        max_abs_diff_host=0.0,
        pte_bytes=0,
        bundled_program_bytes=0,
        delegated_calls=0,
        total_calls=0,
    )

    with pytest.raises(ValueError, match="below /data/local/tmp"):
        harness.validate_device(
            prepared,
            runner=tmp_path / "runner",
            client=object(),
            device_dir="/data/local/tmp/../system",
        )


def test_workload_is_deterministic(harness):
    model_a, input_a = harness.workload()
    model_b, input_b = harness.workload()

    torch.testing.assert_close(input_a, input_b)
    for first, second in zip(model_a.parameters(), model_b.parameters(), strict=True):
        torch.testing.assert_close(first, second)


def test_prepare_only_writes_a_not_run_report(harness, monkeypatch, tmp_path):
    prepared = harness.PreparedArtifact(
        artifact_dir=tmp_path / "model.lm7",
        pte=tmp_path / "model.lm7" / "model.pte",
        bundled_program=tmp_path / "model.bpte",
        max_abs_diff_host=1e-7,
        pte_bytes=100,
        bundled_program_bytes=200,
        delegated_calls=1,
        total_calls=1,
    )
    monkeypatch.setattr(harness, "prepare", lambda output_dir: prepared)

    assert harness.main(["--prepare-only", "--output-dir", str(tmp_path)]) == 0

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["delegate"] == "xnnpack"
    assert report["device"] is None
    assert report["device_validation"] == "not-run"
    assert report["prepared"]["max_abs_diff_host"] == pytest.approx(1e-7)
