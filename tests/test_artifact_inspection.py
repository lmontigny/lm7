from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from lm7 import ArtifactManifest, cli, inspect_artifact


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_artifact(tmp_path, *, delegated_calls=5, total_calls=7):
    artifact = tmp_path / "model.lm7"
    artifact.mkdir()
    program = b"portable exported program"
    compiled = b"device-bound pte"
    (artifact / "exported_program.pt2").write_bytes(program)
    (artifact / "compiled_model.pte").write_bytes(compiled)
    manifest = ArtifactManifest(
        format_version=1,
        lm7_version="0.1.0",
        torch_version="2.10.0",
        created_at="2026-08-02T00:00:00+00:00",
        target={
            "vendor": "qualcomm",
            "kind": "npu",
            "architecture": "v79",
            "model": "sm8750",
            "ordinal": None,
            "remote": True,
        },
        model_graph_hash="graph",
        cache_key="cache",
        input_signature=None,
        program_file="exported_program.pt2",
        program_sha256=_sha256(program),
        backend="qnn",
        backend_version="1.3.1",
        compiled_file="compiled_model.pte",
        compiled_sha256=_sha256(compiled),
        runtime_requirements={
            "executorch": "1.3.1",
            "delegate": "qnn",
            "backend": "htp",
            "soc_model": "SM8750",
            "htp_arch": "v79",
            "precision": "fp16",
            "delegated_calls": delegated_calls,
            "total_calls": total_calls,
            "qnn_sdk": "2.37.0",
            "runtime_libraries": ["libQnnHtp.so", "libqnn_executorch_backend.so"],
            "device_bound": True,
        },
    )
    (artifact / "manifest.json").write_text(json.dumps(asdict(manifest)), encoding="utf-8")
    return artifact


def test_inspect_qnn_artifact_without_loading_runtime(tmp_path):
    artifact = _write_artifact(tmp_path)

    result = inspect_artifact(artifact)

    assert result.backend == "qnn"
    assert result.target == "qualcomm:sm8750"
    assert result.payload == "compiled_model.pte"
    assert result.device_bound is True
    assert result.host_executable is False
    assert result.checksum_status == "valid"
    assert result.valid is True
    assert result.delegation_ratio == 5 / 7
    assert result.runtime_requirements["qnn_sdk"] == "2.37.0"
    assert "SM8750" in result.deployment


def test_inspection_reports_checksum_mismatch(tmp_path):
    artifact = _write_artifact(tmp_path)
    (artifact / "compiled_model.pte").write_bytes(b"corrupt")

    result = inspect_artifact(artifact)

    assert result.checksum_status == "invalid"
    assert result.valid is False
    assert result.checksums[1].status == "mismatch"
    assert "compiled_model.pte: mismatch" in result.errors


def test_inspection_warns_on_low_delegate_coverage(tmp_path):
    artifact = _write_artifact(tmp_path, delegated_calls=2, total_calls=7)

    result = inspect_artifact(artifact)

    assert len(result.warnings) == 1
    assert "Low delegate coverage" in result.warnings[0]


def test_artifact_inspect_cli_json(tmp_path, capsys):
    artifact = _write_artifact(tmp_path)

    assert cli.main(["artifact", "inspect", str(artifact), "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["backend"] == "qnn"
    assert output["target"] == "qualcomm:sm8750"
    assert output["runtime_requirements"]["precision"] == "fp16"
    assert output["checksums"][1]["status"] == "valid"


def test_artifact_inspect_cli_returns_one_for_corrupt_payload(tmp_path, capsys):
    artifact = _write_artifact(tmp_path)
    (artifact / "compiled_model.pte").unlink()

    assert cli.main(["artifact", "inspect", str(artifact)]) == 1

    output = capsys.readouterr().out
    assert "Checksums:        invalid" in output
    assert "compiled_model.pte: missing" in output
