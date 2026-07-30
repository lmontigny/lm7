from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lm7 import cli
from lm7.huggingface import (
    HuggingFaceExportResult,
    HuggingFaceGenerateResult,
    HuggingFaceRunResult,
)
from lm7.targets import DeviceInfo, TargetSpec


@pytest.fixture
def detected_devices() -> list[DeviceInfo]:
    return [
        DeviceInfo(
            TargetSpec("nvidia", "gpu", architecture="sm89", ordinal=0),
            "Test GPU",
            12 * 1024**3,
            {"compute_capability": (8, 9)},
        ),
        DeviceInfo(TargetSpec("cpu", "cpu", architecture="x86_64"), "Test CPU"),
    ]


def test_targets_json(monkeypatch, capsys, detected_devices):
    monkeypatch.setattr(cli, "detect_targets", lambda: detected_devices)

    assert cli.main(["targets", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert [target["target"] for target in output["targets"]] == [
        "nvidia:sm89",
        "cpu:x86_64",
    ]
    assert output["targets"][0]["total_memory_bytes"] == 12 * 1024**3


def test_backends_text(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "inspect_backends",
        lambda: (
            {"name": "eager", "available": True, "version": None, "reason": "PyTorch"},
            {
                "name": "optional",
                "available": False,
                "version": None,
                "reason": "dependency missing",
            },
        ),
    )

    assert cli.main(["backends"]) == 0

    output = capsys.readouterr().out
    assert "eager: available" in output
    assert "optional: unavailable" in output
    assert "dependency missing" in output


def test_doctor_json(monkeypatch, capsys, detected_devices, tmp_path):
    monkeypatch.setattr(cli, "detect_targets", lambda: detected_devices)
    monkeypatch.setattr(
        cli,
        "inspect_backends",
        lambda: ({"name": "eager", "available": True, "version": None, "reason": ""},),
    )
    monkeypatch.setattr(cli, "cache_dir", lambda: tmp_path)

    assert cli.main(["doctor", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ok"
    assert output["lm7_version"]
    assert output["python_version"]
    assert output["pytorch_version"]
    assert output["cache_dir"] == str(tmp_path)
    assert output["targets"][0]["name"] == "Test GPU"
    assert output["backends"][0]["name"] == "eager"


def test_explain_json(capsys):
    assert cli.main(["explain", "--target", "cpu", "--backend", "eager", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["resolved_target"]["vendor"] == "cpu"
    assert output["selected_backend"] == "eager"
    assert any(candidate["backend"] == "eager" for candidate in output["candidates"])


def test_explain_invalid_target_returns_structured_error(capsys):
    assert cli.main(["explain", "--target", "invalid", "--json"]) == 2

    output = json.loads(capsys.readouterr().out)
    assert output["error"]["type"] == "TargetNotFoundError"
    assert "Invalid target" in output["error"]["message"]


def test_model_run_json(monkeypatch, capsys):
    calls = {}
    result = HuggingFaceRunResult(
        model_uri="hf://example/tiny",
        model_id="example/tiny",
        prompt="Hello",
        target="nvidia:sm89",
        backend="inductor",
        dtype="bfloat16",
        quantization="fp8",
        parameter_count=10,
        baseline_model_storage_bytes=100,
        model_storage_bytes=75,
        quantized_modules=3,
        input_tokens=2,
        output_shape=(1, 2, 8),
        quantization_ms=0,
        first_call_ms=12.5,
        latency_ms=1.5,
        peak_memory_bytes=1024,
        next_token_id=5,
        next_token=" world",
    )

    def run_model(model_uri, **kwargs):
        calls["model_uri"] = model_uri
        calls.update(kwargs)
        return result

    monkeypatch.setattr(cli, "run_hf_model", run_model)

    assert (
        cli.main(
            [
                "model",
                "run",
                "hf://example/tiny",
                "--prompt",
                "Hello",
                "--target",
                "nvidia",
                "--backend",
                "inductor",
                "--dtype",
                "bfloat16",
                "--quantize",
                "fp8",
                "--json",
            ]
        )
        == 0
    )

    assert calls == {
        "model_uri": "hf://example/tiny",
        "prompt": "Hello",
        "target": "nvidia",
        "backend": "inductor",
        "dtype": "bfloat16",
        "quantization": "fp8",
    }
    output = json.loads(capsys.readouterr().out)
    assert output["model_uri"] == "hf://example/tiny"
    assert output["target"] == "nvidia:sm89"
    assert output["latency_ms"] == 1.5
    assert output["model_storage_bytes"] == 75
    assert output["peak_memory_bytes"] == 1024
    assert output["next_token"] == " world"


def test_model_run_accepts_the_deprecated_quantization_flag(monkeypatch, capsys):
    """`--quantization int8-weight-only` predates `--quantize int8`. It still
    parses, and run_hf_model normalizes the long name."""
    calls = {}

    def run_model(model_uri, **kwargs):
        calls.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(cli, "run_hf_model", run_model)

    with pytest.raises(SystemExit):
        cli.main(
            ["model", "run", "hf://example/tiny", "--quantization", "int8-weight-only", "--json"]
        )

    assert calls["quantization"] == "int8-weight-only"


def test_model_generate_json(monkeypatch, capsys):
    calls = {}
    result = HuggingFaceGenerateResult(
        model_uri="hf://example/tiny",
        model_id="example/tiny",
        prompt="Hello",
        target="nvidia:sm89",
        backend="inductor",
        dtype="float16",
        parameter_count=10,
        input_tokens=2,
        generated_tokens=4,
        max_new_tokens=4,
        cache_implementation="static",
        first_call_ms=12000.0,
        latency_ms=20.0,
        peak_memory_bytes=1024,
        generated_token_ids=(4, 5, 6, 7),
        generated_text=" world",
    )

    def generate_model(model_uri, **kwargs):
        calls["model_uri"] = model_uri
        calls.update(kwargs)
        return result

    monkeypatch.setattr(cli, "generate_hf_model", generate_model)

    assert (
        cli.main(
            [
                "model",
                "generate",
                "hf://example/tiny",
                "--prompt",
                "Hello",
                "--max-new-tokens",
                "4",
                "--target",
                "nvidia",
                "--backend",
                "inductor",
                "--dtype",
                "float16",
                "--json",
            ]
        )
        == 0
    )

    assert calls == {
        "model_uri": "hf://example/tiny",
        "prompt": "Hello",
        "max_new_tokens": 4,
        "target": "nvidia",
        "backend": "inductor",
        "dtype": "float16",
    }
    output = json.loads(capsys.readouterr().out)
    assert output["backend"] == "inductor"
    assert output["cache_implementation"] == "static"
    assert output["generated_token_ids"] == [4, 5, 6, 7]
    assert output["generated_text"] == " world"


def _fake_bundle(path):
    return SimpleNamespace(
        path=path,
        manifest=SimpleNamespace(
            model_graph_hash="graph123",
            entries=(
                {
                    "key": "cpu-x86_64--aot_inductor",
                    "target": {
                        "vendor": "cpu",
                        "kind": "cpu",
                        "architecture": "x86_64",
                        "model": None,
                        "ordinal": None,
                        "remote": False,
                    },
                    "backend": "aot_inductor",
                    "path": "targets/cpu-x86_64--aot_inductor",
                },
            ),
        ),
    )


def test_bundle_inspect_json(monkeypatch, capsys, tmp_path):
    bundle_path = tmp_path / "model.bundle.lm7"
    monkeypatch.setattr(cli, "load_bundle", lambda path: _fake_bundle(bundle_path))

    assert cli.main(["bundle", "inspect", str(bundle_path), "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["path"] == str(bundle_path)
    assert output["model_graph_hash"] == "graph123"
    assert output["entries"][0]["target"]["target"] == "cpu:x86_64"
    assert output["entries"][0]["backend"] == "aot_inductor"


def test_bundle_create_text(monkeypatch, capsys, tmp_path):
    calls = {}
    bundle_path = tmp_path / "model.bundle.lm7"

    def create_bundle(artifacts, *, output):
        calls["artifacts"] = artifacts
        calls["output"] = output
        return _fake_bundle(bundle_path)

    monkeypatch.setattr(cli, "create_bundle", create_bundle)
    monkeypatch.setattr(cli, "load_bundle", lambda path: _fake_bundle(bundle_path))

    assert (
        cli.main(
            [
                "bundle",
                "create",
                str(bundle_path),
                "build/cpu.lm7",
                "build/apple.lm7",
            ]
        )
        == 0
    )

    assert calls == {
        "artifacts": ["build/cpu.lm7", "build/apple.lm7"],
        "output": str(bundle_path),
    }
    output = capsys.readouterr().out
    assert "Bundle:" in output
    assert "cpu-x86_64--aot_inductor: cpu:x86_64 / aot_inductor" in output


def test_model_export_json(monkeypatch, capsys):
    calls = {}
    result = HuggingFaceExportResult(
        model_uri="hf://example/tiny",
        model_id="example/tiny",
        target="cpu:x86_64",
        backend="aot_inductor",
        dtype="float32",
        output="/tmp/model.lm7",
        prompt="Hello",
        input_tokens=2,
        parameter_count=10,
        export_ms=42.0,
        artifact_bytes=2048,
        files=("compiled_model.pt2", "exported_program.pt2", "manifest.json"),
    )

    def export_model(model_uri, **kwargs):
        calls["model_uri"] = model_uri
        calls.update(kwargs)
        return result

    monkeypatch.setattr(cli, "export_hf_model", export_model)

    assert (
        cli.main(
            [
                "model",
                "export",
                "hf://example/tiny",
                "/tmp/model.lm7",
                "--target",
                "cpu",
                "--backend",
                "aot_inductor",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["model_id"] == "example/tiny"
    assert payload["backend"] == "aot_inductor"
    assert calls["model_uri"] == "hf://example/tiny"
    assert calls["output"] == "/tmp/model.lm7"
    assert calls["backend"] == "aot_inductor"


def test_model_export_passes_int8_quantization(monkeypatch):
    calls = {}

    def export_model(model_uri, **kwargs):
        calls.update(kwargs)
        return HuggingFaceExportResult(
            model_uri=model_uri,
            model_id="example/tiny",
            target="cpu:x86_64",
            backend="executorch",
            dtype="float32",
            output="/tmp/model.lm7",
            prompt="Hello",
            input_tokens=2,
            parameter_count=10,
            export_ms=42.0,
            artifact_bytes=2048,
            files=("model.pte", "manifest.json"),
            quantization="int8",
        )

    monkeypatch.setattr(cli, "export_hf_model", export_model)
    assert (
        cli.main(
            [
                "model",
                "export",
                "hf://example/tiny",
                "/tmp/m.lm7",
                "--backend",
                "executorch",
                "--quantize",
                "int8",
            ]
        )
        == 0
    )
    assert calls["quantization"] == "int8"


def test_model_export_rejects_an_unknown_backend(capsys):
    with pytest.raises(SystemExit):
        cli.main(["model", "export", "hf://example/tiny", "/tmp/m.lm7", "--backend", "nosuchthing"])


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ([], False),
        (["--dynamic-seq"], True),
        (["--dynamic-seq", "4:512"], (4, 512)),
    ],
)
def test_model_export_passes_dynamic_sequence(monkeypatch, argument, expected):
    calls = {}

    def export_model(model_uri, **kwargs):
        calls.update(kwargs)
        return HuggingFaceExportResult(
            model_uri=model_uri,
            model_id="example/tiny",
            target="cpu:x86_64",
            backend="export",
            dtype="float32",
            output="/tmp/model.lm7",
            prompt="Hello",
            input_tokens=2,
            parameter_count=10,
            export_ms=42.0,
            artifact_bytes=2048,
            files=("exported_program.pt2", "manifest.json"),
            sequence_bounds=expected if isinstance(expected, tuple) else None,
        )

    monkeypatch.setattr(cli, "export_hf_model", export_model)
    assert cli.main(["model", "export", "hf://example/tiny", "/tmp/m.lm7", *argument]) == 0
    assert calls["dynamic_sequence"] == expected


@pytest.mark.parametrize("value", ["2048", "a:b"])
def test_model_export_rejects_a_malformed_sequence_range(value, capsys):
    assert cli.main(["model", "export", "hf://example/tiny", "/tmp/m.lm7", "--dynamic-seq", value])
    assert "--dynamic-seq" in capsys.readouterr().err


def test_model_export_reports_dynamic_bounds(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "export_hf_model",
        lambda model_uri, **kwargs: HuggingFaceExportResult(
            model_uri=model_uri,
            model_id="example/tiny",
            target="cpu:x86_64",
            backend="export",
            dtype="float32",
            output="/tmp/model.lm7",
            prompt="Hello",
            input_tokens=2,
            parameter_count=10,
            export_ms=42.0,
            artifact_bytes=2048,
            files=("exported_program.pt2", "manifest.json"),
            sequence_bounds=(1, 2048),
        ),
    )

    assert cli.main(["model", "export", "hf://example/tiny", "/tmp/m.lm7", "--dynamic-seq"]) == 0
    assert "sequence dynamic in [1, 2048]" in capsys.readouterr().out
