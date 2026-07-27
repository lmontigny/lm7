from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lm7 import huggingface
from lm7.errors import UnsupportedModelError
from lm7.targets import TargetSpec


class FakeCausalLM(torch.nn.Module):
    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
        logits[:, -1, 5] = 1
        return SimpleNamespace(logits=logits)


class FakeTokenizer:
    def __call__(self, prompt, return_tensors):
        assert prompt
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens is False
        return f"token-{token_ids[0]}"


def _fake_transformers(calls):
    class TokenizerFactory:
        @staticmethod
        def from_pretrained(model_id):
            calls["tokenizer_model_id"] = model_id
            return FakeTokenizer()

    class ModelFactory:
        @staticmethod
        def from_pretrained(model_id, *, dtype):
            calls["model_id"] = model_id
            calls["dtype"] = dtype
            return FakeCausalLM()

    return SimpleNamespace(
        AutoTokenizer=TokenizerFactory,
        AutoModelForCausalLM=ModelFactory,
    )


def test_run_hf_model_uses_lm7_and_reports_next_token(monkeypatch):
    calls = {}
    monkeypatch.setattr(huggingface, "_load_transformers", lambda: _fake_transformers(calls))

    result = huggingface.run_hf_model(
        "hf://example/tiny-model",
        prompt="Hello",
        target="cpu",
        backend="eager",
    )

    assert calls == {
        "tokenizer_model_id": "example/tiny-model",
        "model_id": "example/tiny-model",
        "dtype": torch.float32,
    }
    assert result.model_id == "example/tiny-model"
    assert result.target.startswith("cpu")
    assert result.backend == "eager"
    assert result.dtype == "float32"
    assert result.quantization == "none"
    assert result.parameter_count == 0
    assert result.input_tokens == 3
    assert result.output_shape == (1, 3, 8)
    assert result.first_call_ms >= 0
    assert result.latency_ms >= 0
    assert result.quantization_ms == 0
    assert result.peak_memory_bytes is None
    assert result.next_token_id == 5
    assert result.next_token == "token-5"


@pytest.mark.parametrize("value", ["model", "hf://", "hf://model"])
def test_run_hf_model_rejects_invalid_uri(value):
    with pytest.raises(UnsupportedModelError, match="Hugging Face"):
        huggingface.run_hf_model(value, prompt="Hello")


def test_auto_dtype_depends_on_target():
    assert huggingface._resolve_dtype("auto", TargetSpec("cpu", "cpu")) == torch.float32
    assert huggingface._resolve_dtype("auto", TargetSpec("nvidia", "gpu")) == torch.float16
    assert huggingface._resolve_dtype("auto", TargetSpec("tpu", "accelerator")) == torch.bfloat16
    assert (
        huggingface._resolve_dtype(
            "auto",
            TargetSpec("nvidia", "gpu"),
            huggingface.INT8_WEIGHT_ONLY,
        )
        == torch.bfloat16
    )


def test_int8_weight_only_uses_torchao_version_two(monkeypatch):
    calls = {}

    class Config:
        def __init__(self, *, version):
            calls["version"] = version

    def quantize(model, config, *, filter_fn, device):
        calls["model"] = model
        calls["config"] = config
        calls["filter_fn"] = filter_fn
        calls["device"] = device

    fake_torchao = SimpleNamespace(
        Int8WeightOnlyConfig=Config,
        quantize_=quantize,
    )
    monkeypatch.setattr(huggingface, "_load_torchao_quantization", lambda: fake_torchao)

    model = FakeCausalLM()
    elapsed = huggingface._apply_quantization(
        model,
        TargetSpec("cpu", "cpu"),
        huggingface.INT8_WEIGHT_ONLY,
    )

    assert calls["version"] == 2
    assert calls["model"] is model
    assert calls["filter_fn"] is huggingface._is_quantizable_linear
    assert calls["device"] == torch.device("cpu")
    assert elapsed >= 0


def test_int8_weight_only_excludes_lm_head():
    assert huggingface._is_quantizable_linear(torch.nn.Linear(4, 4), "model.q_proj")
    assert not huggingface._is_quantizable_linear(torch.nn.Linear(4, 4), "lm_head")
    assert not huggingface._is_quantizable_linear(torch.nn.Linear(4, 4), "decoder.lm_head")
    assert not huggingface._is_quantizable_linear(torch.nn.ReLU(), "model.activation")


@pytest.mark.parametrize(
    ("target", "backend", "dtype", "message"),
    [
        (TargetSpec("cpu", "cpu"), "inductor", "bfloat16", "NVIDIA"),
        (TargetSpec("nvidia", "gpu"), "eager", "bfloat16", "backend"),
        (TargetSpec("nvidia", "gpu"), "inductor", "float16", "dtype"),
    ],
)
def test_int8_weight_only_rejects_unsupported_combinations(target, backend, dtype, message):
    with pytest.raises(UnsupportedModelError, match=message):
        huggingface._validate_quantization(
            huggingface.INT8_WEIGHT_ONLY,
            target,
            backend,
            dtype,
        )


def test_int8_weight_only_rejects_unvalidated_model():
    with pytest.raises(UnsupportedModelError, match="not validated"):
        huggingface._validate_quantization(
            huggingface.INT8_WEIGHT_ONLY,
            TargetSpec("nvidia", "gpu"),
            "inductor",
            "bfloat16",
            "LiquidAI/LFM2.5-230M",
        )


def test_missing_transformers_has_install_hint(monkeypatch):
    def missing(_name):
        raise ImportError("missing")

    monkeypatch.setattr(huggingface.importlib, "import_module", missing)

    with pytest.raises(UnsupportedModelError, match=r'pip install "lm7\[hf\]"'):
        huggingface._load_transformers()


def test_missing_torchao_has_install_hint(monkeypatch):
    def missing(_name):
        raise ImportError("missing")

    monkeypatch.setattr(huggingface.importlib, "import_module", missing)

    with pytest.raises(UnsupportedModelError, match=r"lm7\[hf,torchao\]"):
        huggingface._load_torchao_quantization()
