from __future__ import annotations

import json

import pytest

from lm7 import cli
from lm7.huggingface import HuggingFaceRunResult
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
        dtype="float16",
        parameter_count=10,
        input_tokens=2,
        output_shape=(1, 2, 8),
        first_call_ms=12.5,
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
                "float16",
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
        "dtype": "float16",
    }
    output = json.loads(capsys.readouterr().out)
    assert output["model_uri"] == "hf://example/tiny"
    assert output["target"] == "nvidia:sm89"
    assert output["next_token"] == " world"
