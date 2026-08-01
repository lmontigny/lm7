from __future__ import annotations

from types import SimpleNamespace

import pytest

from lm7 import compatibility
from lm7.errors import BackendUnavailableError, UnsupportedModelError
from lm7.targets import TargetSpec


class CausalConfig:
    def __init__(self):
        self.model_type = "tiny"
        self.architectures = ["TinyForCausalLM"]
        self.is_encoder_decoder = False
        self.auto_map = {}
        self.max_position_embeddings = 4096
        self.vocab_size = 32000
        self.hidden_size = 512
        self.num_hidden_layers = 8
        self.dtype = "float16"
        self.vision_config = None
        self.audio_config = None


def fake_transformers(config=None, *, error=None, registered=True):
    class AutoConfig:
        @staticmethod
        def from_pretrained(model_id, *, trust_remote_code):
            assert model_id == "example/tiny"
            assert trust_remote_code is False
            if error is not None:
                raise error
            return config

    mapping = {type(config): object} if registered and config is not None else {}
    return SimpleNamespace(
        AutoConfig=AutoConfig,
        AutoModelForCausalLM=SimpleNamespace(_model_mapping=mapping),
    )


def patch_environment(monkeypatch, transformers, *, selected="inductor"):
    monkeypatch.setattr(compatibility, "_load_transformers", lambda: transformers)
    monkeypatch.setattr(
        compatibility,
        "resolve_target",
        lambda target: TargetSpec("nvidia", "gpu", architecture="sm89"),
    )
    candidates = (
        compatibility.BackendCompatibility("eager", True, 0, "available"),
        compatibility.BackendCompatibility("inductor", True, 100, "available"),
    )
    monkeypatch.setattr(
        compatibility,
        "_backend_compatibility",
        lambda target, backend: (selected, candidates),
    )


def test_causal_lm_config_reports_workflows_without_loading_weights(monkeypatch):
    patch_environment(monkeypatch, fake_transformers(CausalConfig()))
    monkeypatch.setattr(compatibility, "_validate_quantization", lambda *args: None)

    result = compatibility.inspect_hf_model("hf://example/tiny", target="nvidia", backend="auto")

    assert result.status == "compatible"
    assert result.task == "causal-lm"
    assert result.architectures == ("TinyForCausalLM",)
    assert result.context_length == 4096
    assert result.vocab_size == 32000
    assert result.hidden_size == 512
    assert result.num_hidden_layers == 8
    assert result.selected_backend == "inductor"
    assert {item.name: item.status for item in result.workflows} == {
        "run": "compatible",
        "generate": "conditional",
        "export": "conditional",
    }
    assert all(item.status == "compatible" for item in result.quantization)
    assert result.config_only is True


def test_encoder_decoder_is_reported_as_unsupported(monkeypatch):
    config = CausalConfig()
    config.is_encoder_decoder = True
    patch_environment(monkeypatch, fake_transformers(config))

    result = compatibility.inspect_hf_model("hf://example/tiny")

    assert result.status == "incompatible"
    assert result.task == "seq2seq"
    assert all(item.status == "unsupported" for item in result.workflows)


def test_multimodal_config_is_reported_as_unsupported(monkeypatch):
    config = CausalConfig()
    config.vision_config = SimpleNamespace()
    patch_environment(monkeypatch, fake_transformers(config))

    result = compatibility.inspect_hf_model("hf://example/tiny")

    assert result.status == "incompatible"
    assert result.task == "multimodal"
    assert "text tensors only" in result.workflows[0].reason


def test_unregistered_causal_architecture_is_unknown(monkeypatch):
    patch_environment(monkeypatch, fake_transformers(CausalConfig(), registered=False))

    result = compatibility.inspect_hf_model("hf://example/tiny")

    assert result.status == "unknown"
    assert all(item.status == "unknown" for item in result.workflows)


def test_custom_config_code_returns_an_incompatible_report(monkeypatch):
    error = ValueError("Set trust_remote_code=True to execute the configuration file")
    patch_environment(monkeypatch, fake_transformers(error=error))

    result = compatibility.inspect_hf_model("hf://example/tiny")

    assert result.status == "incompatible"
    assert result.requires_remote_code is True
    assert result.config_class is None
    assert all(item.status == "unsupported" for item in result.quantization)


def test_config_download_failure_is_actionable(monkeypatch):
    patch_environment(monkeypatch, fake_transformers(error=OSError("repository missing")))

    with pytest.raises(UnsupportedModelError, match="config inspection failed"):
        compatibility.inspect_hf_model("hf://example/tiny")


def test_unknown_backend_is_rejected():
    with pytest.raises(BackendUnavailableError, match="not registered"):
        compatibility._backend_compatibility(TargetSpec("cpu", "cpu"), "missing")
