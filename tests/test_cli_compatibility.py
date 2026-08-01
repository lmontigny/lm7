from __future__ import annotations

import json

from lm7 import cli
from lm7.compatibility import (
    BackendCompatibility,
    CompatibilityCheck,
    ModelCompatibilityResult,
)


def result() -> ModelCompatibilityResult:
    return ModelCompatibilityResult(
        model_uri="hf://example/tiny",
        model_id="example/tiny",
        status="compatible",
        model_type="tiny",
        architectures=("TinyForCausalLM",),
        task="causal-lm",
        config_class="TinyConfig",
        dtype="float16",
        context_length=4096,
        vocab_size=32000,
        hidden_size=512,
        num_hidden_layers=8,
        is_encoder_decoder=False,
        is_multimodal=False,
        requires_remote_code=False,
        target="nvidia:sm89",
        requested_backend="auto",
        selected_backend="inductor",
        workflows=(CompatibilityCheck("run", "compatible", "registered"),),
        quantization=(CompatibilityCheck("int8", "compatible", "validated"),),
        backend_candidates=(BackendCompatibility("inductor", True, 100, "available"),),
        notes=("config only",),
    )


def test_model_compatibility_json(monkeypatch, capsys):
    calls = {}

    def inspect(model_uri, **kwargs):
        calls["model_uri"] = model_uri
        calls.update(kwargs)
        return result()

    monkeypatch.setattr(cli, "inspect_hf_model", inspect)

    assert (
        cli.main(
            [
                "model",
                "compatibility",
                "hf://example/tiny",
                "--target",
                "nvidia",
                "--backend",
                "auto",
                "--json",
            ]
        )
        == 0
    )

    assert calls == {
        "model_uri": "hf://example/tiny",
        "target": "nvidia",
        "backend": "auto",
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "compatible"
    assert payload["context_length"] == 4096
    assert payload["workflows"][0]["name"] == "run"
    assert payload["config_only"] is True


def test_model_compatibility_text(monkeypatch, capsys):
    monkeypatch.setattr(cli, "inspect_hf_model", lambda *args, **kwargs: result())

    assert cli.main(["model", "compatibility", "hf://example/tiny"]) == 0

    output = capsys.readouterr().out
    assert "Status: compatible" in output
    assert "Architecture: TinyForCausalLM (tiny)" in output
    assert "run: compatible" in output
    assert "int8: compatible" in output
