from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lm7 import huggingface
from lm7.detection import torch_device
from lm7.errors import UnsupportedModelError
from lm7.targets import TargetSpec


class FakeDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Named so the fully qualified path contains ".mlp.", which is what the
        # FP8 filter selects on. Both dimensions are multiples of 16 so the
        # NVFP4 filter matches it too.
        self.mlp = torch.nn.Sequential(torch.nn.Linear(16, 16))


class FakeQuantizableLM(torch.nn.Module):
    """A fake with layers every weight-only filter actually matches."""

    def __init__(self):
        super().__init__()
        self.layer = FakeDecoderLayer()


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


class _FakeCompiled:
    """Stands in for CompiledModule so a backend's artifact can be faked.

    run_hf_model reads `target`, `selected_backend` and `artifact` off whatever
    lm7.compile returns, so a test that needs a specific artifact path supplies
    this instead of compiling for real.
    """

    def __init__(self, model: torch.nn.Module, backend: str) -> None:
        self.model = model
        self.target = TargetSpec("cpu", "cpu")
        self.selected_backend = backend
        self.artifact = None

    def __call__(self, *args, **kwargs):
        # The export-backed path calls positionally through _LogitsOnly, the
        # torch-level path calls with keywords; both have to reach the fake model.
        return self.model(*args, **kwargs)


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
    assert result.baseline_model_storage_bytes == 0
    assert result.model_storage_bytes == 0
    assert result.input_tokens == 3
    assert result.output_shape == (1, 3, 8)
    assert result.first_call_ms >= 0
    assert result.latency_ms >= 0
    assert result.quantization_ms == 0
    assert result.peak_memory_bytes is None
    assert result.next_token_id == 5
    assert result.next_token == "token-5"
    assert result.compiled_weight_bytes is None


def test_openvino_quantization_goes_to_the_backend_not_torchao(monkeypatch, tmp_path):
    """`--backend openvino --quantize int8` must compress the IR, not the module.

    The two mechanisms are unrelated: TorchAO swaps torch Linear weights before
    compiling, NNCF compresses the IR during it. Routing this request through
    TorchAO would quantize twice and report the wrong saving, so the torch module
    has to come out untouched and the request has to arrive as a backend option.
    """
    calls: dict = {}
    monkeypatch.setattr(huggingface, "_load_transformers", lambda: _fake_transformers(calls))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("TorchAO must not run for the openvino backend")

    monkeypatch.setattr(huggingface, "_apply_quantization", fail_if_called)

    # A stand-in for the IR the backend would have written, so the reported saving
    # comes from a real file rather than a hard-coded number.
    weights = tmp_path / "compiled_model.bin"
    weights.write_bytes(b"x" * 4096)

    recorded: dict = {}

    def fake_compile(model, **kwargs):
        recorded.update(kwargs)
        wrapped = _FakeCompiled(model, "openvino")
        wrapped.artifact = SimpleNamespace(path=tmp_path / "compiled_model.xml")
        return wrapped

    monkeypatch.setattr(huggingface, "compile", fake_compile)
    monkeypatch.setattr(huggingface, "VALIDATED_OPENVINO_INT8", frozenset({"example/tiny-model"}))

    result = huggingface.run_hf_model(
        "hf://example/tiny-model",
        prompt="Hello",
        target="cpu",
        backend="openvino",
        quantization="int8",
    )

    assert recorded["options"] == {"quantization": "int8"}
    assert result.quantization == "int8"
    assert result.quantized_modules == 0
    assert result.quantization_ms == 0.0
    # The torch module is untouched, so the saving is only visible in the artifact.
    assert result.baseline_model_storage_bytes == result.model_storage_bytes
    assert result.compiled_weight_bytes == 4096


def test_openvino_quantization_rejects_unvalidated_models():
    with pytest.raises(UnsupportedModelError, match="not validated"):
        huggingface.run_hf_model(
            "hf://example/never-measured",
            prompt="Hello",
            target="cpu",
            backend="openvino",
            quantization="int8",
        )


@pytest.mark.parametrize("quantization", [huggingface.FP8, huggingface.NVFP4])
def test_openvino_quantization_rejects_torchao_only_formats(quantization):
    """NNCF implements INT8 here; the narrower formats belong to TorchAO."""
    with pytest.raises(UnsupportedModelError, match="implements 'int8' only"):
        huggingface.run_hf_model(
            "hf://HuggingFaceTB/SmolLM2-135M-Instruct",
            prompt="Hello",
            target="cpu",
            backend="openvino",
            quantization=quantization,
        )


def test_openvino_quantization_rejects_non_intel_targets():
    """Checked against the validator directly, not through resolve_target.

    Asking run_hf_model for target="nvidia" fails for lack of a GPU long before it
    reaches this rule, so routing through it would make the test pass or fail on
    host hardware rather than on the rule.
    """
    with pytest.raises(UnsupportedModelError, match="Intel CPU and NPU"):
        huggingface._validate_quantization(
            "int8",
            TargetSpec("nvidia", "gpu"),
            "openvino",
            "auto",
            "HuggingFaceTB/SmolLM2-135M-Instruct",
        )


class FakeGenerationTokenizer:
    def __call__(self, prompt, return_tensors):
        assert prompt == "Hello"
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens is True
        return " ".join(f"token-{token_id}" for token_id in token_ids)


class FakeGeneratingCausalLM(torch.nn.Module):
    def __init__(self, calls):
        super().__init__()
        self.calls = calls

    def generate(self, **kwargs):
        self.calls.setdefault("generate", []).append(
            {
                "max_new_tokens": kwargs["max_new_tokens"],
                "do_sample": kwargs["do_sample"],
                "cache_implementation": kwargs["cache_implementation"],
                "compile_config": kwargs["compile_config"],
            }
        )
        suffix = torch.tensor([[4, 5, 6, 7]], device=kwargs["input_ids"].device)
        return torch.cat((kwargs["input_ids"], suffix), dim=1)


def _fake_generation_transformers(calls):
    class CompileConfig:
        def __init__(self, **kwargs):
            calls["compile_config"] = kwargs

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(model_id):
            calls["tokenizer_model_id"] = model_id
            return FakeGenerationTokenizer()

    class ModelFactory:
        @staticmethod
        def from_pretrained(model_id, *, dtype):
            calls["model_id"] = model_id
            calls["dtype"] = dtype
            return FakeGeneratingCausalLM(calls)

    return SimpleNamespace(
        AutoTokenizer=TokenizerFactory,
        AutoModelForCausalLM=ModelFactory,
        CompileConfig=CompileConfig,
    )


def test_generate_hf_model_uses_static_cache_and_compiled_decode(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        huggingface, "_load_transformers", lambda: _fake_generation_transformers(calls)
    )

    result = huggingface.generate_hf_model(
        "hf://example/tiny-model",
        prompt="Hello",
        max_new_tokens=4,
        target="cpu",
    )

    assert calls["model_id"] == "example/tiny-model"
    assert calls["dtype"] == torch.float32
    assert calls["compile_config"] == {
        "backend": "inductor",
        "mode": "reduce-overhead",
        "fullgraph": False,
        "dynamic": None,
    }
    assert len(calls["generate"]) == 2
    assert all(call["cache_implementation"] == "static" for call in calls["generate"])
    assert all(call["do_sample"] is False for call in calls["generate"])
    # A CPU target does not meet Transformers' compilation criteria, so the decode
    # loop runs eagerly however the compile_config is spelled.
    assert result.backend == "eager"
    assert result.cache_implementation == "static"
    assert result.input_tokens == 3
    assert result.generated_tokens == 4
    assert result.generated_token_ids == (4, 5, 6, 7)
    assert result.generated_text == "token-4 token-5 token-6 token-7"
    assert result.first_call_ms >= 0
    assert result.latency_ms >= 0


@pytest.mark.parametrize(
    ("target", "compiled"),
    [
        (TargetSpec("nvidia", "gpu"), True),
        (TargetSpec("amd", "gpu"), True),
        (TargetSpec("intel", "gpu"), True),
        (TargetSpec("intel", "npu"), False),
        (TargetSpec("apple", "gpu"), False),
        (TargetSpec("tpu", "accelerator"), False),
        (TargetSpec("tenstorrent", "accelerator"), False),
        (TargetSpec("cpu", "cpu"), False),
    ],
)
def test_compiled_decode_follows_the_target_device_type(target, compiled):
    """Which targets Transformers will compile a decode step on.

    Transformers gates compiled generation on the torch device type, so LM7's own
    target mapping decides this: `apple` reaches `mps` and `tpu` reaches `xla`,
    neither of which is on the upstream list, and generation there decodes eagerly
    no matter what `compile_config` says. Constructing the device is enough to
    check it, which keeps this runnable on a CPU-only build.
    """
    assert huggingface.compiles_decode(torch_device(target)) is compiled


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"max_new_tokens": 1}, ">= 2"), ({"backend": "eager"}, "inductor")]
)
def test_generate_hf_model_rejects_non_compiled_requests(kwargs, message):
    with pytest.raises(UnsupportedModelError, match=message):
        huggingface.generate_hf_model("hf://example/tiny-model", prompt="Hello", **kwargs)


@pytest.mark.parametrize("value", ["model", "hf://", "hf://model"])
def test_run_hf_model_rejects_invalid_uri(value):
    with pytest.raises(UnsupportedModelError, match="Hugging Face"):
        huggingface.run_hf_model(value, prompt="Hello")


@pytest.mark.parametrize(
    "quantization",
    [huggingface.INT8, huggingface.FP8],
)
def test_auto_dtype_depends_on_target(quantization):
    assert huggingface._resolve_dtype("auto", TargetSpec("cpu", "cpu")) == torch.float32
    assert huggingface._resolve_dtype("auto", TargetSpec("nvidia", "gpu")) == torch.float16
    assert huggingface._resolve_dtype("auto", TargetSpec("tpu", "accelerator")) == torch.bfloat16
    assert (
        huggingface._resolve_dtype(
            "auto",
            TargetSpec("nvidia", "gpu"),
            quantization,
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

    model = FakeQuantizableLM()
    elapsed, converted = huggingface._apply_quantization(
        model,
        TargetSpec("cpu", "cpu"),
        huggingface.INT8,
    )

    assert calls["version"] == 2
    assert calls["model"] is model
    assert calls["filter_fn"] is huggingface._is_quantizable_linear
    assert calls["device"] == torch.device("cpu")
    assert elapsed >= 0
    assert converted > 0


def test_fp8_weight_only_uses_torchao_version_two(monkeypatch):
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
        Float8WeightOnlyConfig=Config,
        quantize_=quantize,
    )
    monkeypatch.setattr(huggingface, "_load_torchao_quantization", lambda: fake_torchao)
    monkeypatch.setattr(huggingface, "_synchronize", lambda _target: None)

    model = FakeQuantizableLM()
    elapsed, converted = huggingface._apply_quantization(
        model,
        TargetSpec("nvidia", "gpu"),
        huggingface.FP8,
    )

    assert calls["version"] == 2
    assert calls["model"] is model
    assert calls["filter_fn"] is huggingface._is_fp8_quantizable_linear
    assert calls["device"] == torch.device("cuda", 0)
    assert elapsed >= 0
    assert converted > 0


def test_nvfp4_uses_the_prototype_mx_formats_config(monkeypatch):
    """NVFP4 lives in torchao.prototype, not torchao.quantization, so it is
    loaded separately and takes no version argument."""
    calls = {}

    class Config:
        def __init__(self):
            calls["config_built"] = True

    def quantize(model, config, *, filter_fn, device):
        calls["filter_fn"] = filter_fn
        calls["device"] = device

    monkeypatch.setattr(
        huggingface,
        "_load_torchao_quantization",
        lambda: SimpleNamespace(quantize_=quantize),
    )
    monkeypatch.setattr(
        huggingface,
        "_load_torchao_nvfp4",
        lambda: SimpleNamespace(NVFP4WeightOnlyConfig=Config),
    )
    monkeypatch.setattr(huggingface, "_synchronize", lambda _target: None)

    model = FakeQuantizableLM()
    elapsed, converted = huggingface._apply_quantization(
        model,
        TargetSpec("nvidia", "gpu"),
        huggingface.NVFP4,
    )

    assert calls["config_built"] is True
    assert calls["filter_fn"] is huggingface._is_nvfp4_quantizable_linear
    assert elapsed >= 0
    assert converted > 0


def test_int8_weight_only_excludes_lm_head():
    assert huggingface._is_quantizable_linear(torch.nn.Linear(4, 4), "model.q_proj")
    assert not huggingface._is_quantizable_linear(torch.nn.Linear(4, 4), "lm_head")
    assert not huggingface._is_quantizable_linear(torch.nn.Linear(4, 4), "decoder.lm_head")
    assert not huggingface._is_quantizable_linear(torch.nn.ReLU(), "model.activation")


def test_fp8_weight_only_selects_mlp_linears():
    assert huggingface._is_fp8_quantizable_linear(
        torch.nn.Linear(4, 4), "model.layers.0.mlp.up_proj"
    )
    assert not huggingface._is_fp8_quantizable_linear(
        torch.nn.Linear(4, 4), "model.layers.0.self_attn.q_proj"
    )
    assert not huggingface._is_fp8_quantizable_linear(torch.nn.Linear(4, 4), "lm_head")


@pytest.mark.parametrize(
    ("target", "backend", "dtype", "message"),
    [
        # CPU is allowed for INT8 now, but its compute dtype is FP32, not BF16.
        (TargetSpec("cpu", "cpu"), "inductor", "bfloat16", "dtype"),
        (TargetSpec("nvidia", "gpu"), "eager", "bfloat16", "backend"),
        (TargetSpec("nvidia", "gpu"), "inductor", "float16", "dtype"),
        (TargetSpec("amd", "gpu"), "inductor", "bfloat16", "NVIDIA"),
    ],
)
def test_int8_weight_only_rejects_unsupported_combinations(target, backend, dtype, message):
    with pytest.raises(UnsupportedModelError, match=message):
        huggingface._validate_quantization(
            huggingface.INT8,
            target,
            backend,
            dtype,
        )


def test_int8_weight_only_accepts_cpu():
    """INT8 is the one mode measured off NVIDIA: on CPU it kept 4/4 top-1 tokens
    at a 1.36 max logit difference, so the target gate admits it."""
    huggingface._validate_quantization(
        huggingface.INT8,
        TargetSpec("cpu", "cpu"),
        "inductor",
        "auto",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
    )


@pytest.mark.parametrize("quantization", [huggingface.FP8, huggingface.NVFP4])
def test_narrow_formats_stay_nvidia_only(quantization):
    """FP8 needs Ada tensor cores; NVFP4 on CPU kept 2/4 top-1 and ran 8.5x
    slower than compiled FP32, so neither is admitted off NVIDIA."""
    with pytest.raises(UnsupportedModelError, match="NVIDIA"):
        huggingface._validate_quantization(
            quantization,
            TargetSpec("cpu", "cpu"),
            "inductor",
            "auto",
        )


def test_auto_dtype_under_quantization_is_target_specific():
    """BF16 on NVIDIA, FP32 on CPU — x86 without AVX-512 has no native BF16."""
    assert (
        huggingface._resolve_dtype("auto", TargetSpec("nvidia", "gpu"), huggingface.INT8)
        == torch.bfloat16
    )
    assert (
        huggingface._resolve_dtype("auto", TargetSpec("cpu", "cpu"), huggingface.INT8)
        == torch.float32
    )


def test_int8_weight_only_rejects_unvalidated_model():
    with pytest.raises(UnsupportedModelError, match="not validated"):
        huggingface._validate_quantization(
            huggingface.INT8,
            TargetSpec("nvidia", "gpu"),
            "inductor",
            "bfloat16",
            "LiquidAI/LFM2.5-230M",
        )


def test_fp8_weight_only_rejects_ampere():
    with pytest.raises(UnsupportedModelError, match="Ada"):
        huggingface._validate_quantization(
            huggingface.FP8,
            TargetSpec("nvidia", "gpu", architecture="sm80"),
            "inductor",
            "bfloat16",
        )


def test_model_storage_bytes_counts_parameters_and_buffers_once():
    model = torch.nn.Linear(4, 2, bias=False)
    model.register_buffer("scale", torch.ones(2, dtype=torch.float16))

    assert huggingface._model_storage_bytes(model) == 4 * 2 * 4 + 2 * 2


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


class ExportableCausalLM(torch.nn.Module):
    """A fake that torch.export can trace, returning the HF output dataclass."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def _exportable_transformers(calls):
    class TokenizerFactory:
        @staticmethod
        def from_pretrained(model_id):
            calls["tokenizer_model_id"] = model_id
            return FakeTokenizer()

    class ModelFactory:
        @staticmethod
        def from_pretrained(model_id, *, dtype, attn_implementation=None):
            calls["model_id"] = model_id
            calls["attn_implementation"] = attn_implementation
            return ExportableCausalLM()

    return SimpleNamespace(AutoTokenizer=TokenizerFactory, AutoModelForCausalLM=ModelFactory)


def test_export_hf_model_writes_a_loadable_artifact(monkeypatch, tmp_path):
    """A Hugging Face causal LM returns CausalLMOutputWithPast, which torch.export
    puts in the output pytree and torch.export.load then cannot deserialize. LM7
    captures a logits-only graph so the artifact actually round-trips."""
    import lm7

    monkeypatch.setattr(huggingface, "_load_transformers", lambda: _exportable_transformers({}))
    output = tmp_path / "model.lm7"

    result = huggingface.export_hf_model(
        "hf://example/tiny-model",
        output=str(output),
        prompt="Hello",
        target="cpu",
    )

    assert result.model_id == "example/tiny-model"
    assert result.backend == "export"
    assert result.input_tokens == 3
    assert "exported_program.pt2" in result.files
    assert result.artifact_bytes > 0

    reloaded = lm7.load_artifact(output)
    logits = reloaded(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
    )
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 3, 16)


def test_export_hf_model_passes_int8_to_executorch(monkeypatch, tmp_path):
    import lm7.exporting

    calls = {}
    monkeypatch.setattr(huggingface, "_load_transformers", lambda: _exportable_transformers({}))
    output = tmp_path / "model.lm7"
    output.mkdir()
    (output / "manifest.json").write_text("{}", encoding="utf-8")

    def export_artifact(*args, **kwargs):
        calls.update(kwargs)
        return SimpleNamespace(path=output)

    monkeypatch.setattr(lm7.exporting, "export", export_artifact)
    result = huggingface.export_hf_model(
        "hf://example/tiny-model",
        output=str(output),
        target="cpu",
        backend="executorch",
        quantization="int8",
    )

    assert calls["options"] == {"quantization": "int8"}
    assert result.quantization == "int8"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "export", "quantization": "int8"}, "executorch, openvino"),
        (
            {"backend": "executorch", "quantization": "int8", "dynamic_sequence": True},
            "requires a fixed input shape",
        ),
        ({"backend": "executorch", "quantization": "int4"}, "expected 'none' or 'int8'"),
        # NNCF compresses the vocabulary projection too, so the gate is per model.
        ({"backend": "openvino", "quantization": "int8"}, "not validated"),
    ],
)
def test_export_hf_model_rejects_invalid_quantization(tmp_path, kwargs, message):
    with pytest.raises(UnsupportedModelError, match=message):
        huggingface.export_hf_model(
            "hf://example/tiny-model", output=str(tmp_path / "m.lm7"), **kwargs
        )


def test_openvino_int8_export_passes_the_option_for_a_validated_model(monkeypatch, tmp_path):
    """The dynamic-sequence restriction is ExecuTorch's, because its calibration
    sample is the captured example. NNCF needs no calibration, so OpenVINO keeps
    dynamic sequences."""
    calls = {}
    monkeypatch.setattr(huggingface, "_load_transformers", lambda: _exportable_transformers(calls))
    recorded = {}

    def fake_export(model, **kwargs):
        recorded.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr("lm7.exporting.export", fake_export)

    model_id = next(iter(huggingface.VALIDATED_OPENVINO_INT8))
    with pytest.raises(SystemExit):
        huggingface.export_hf_model(
            f"hf://{model_id}",
            output=str(tmp_path / "m.lm7"),
            backend="openvino",
            quantization="int8",
        )

    assert recorded["options"] == {"quantization": "int8"}


def test_export_hf_model_rejects_a_non_hf_uri(tmp_path):
    with pytest.raises(UnsupportedModelError, match="expected a Hugging Face URI"):
        huggingface.export_hf_model("./local/model", output=str(tmp_path / "m.lm7"))


def test_export_hf_model_is_fixed_shape_by_default(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(huggingface, "_load_transformers", lambda: _exportable_transformers(calls))

    result = huggingface.export_hf_model(
        "hf://example/tiny-model",
        output=str(tmp_path / "model.lm7"),
        prompt="Hello",
        target="cpu",
    )

    assert result.sequence_bounds is None
    # A fixed capture keeps the model's faster default attention.
    assert calls["attn_implementation"] is None


def test_export_hf_model_captures_a_dynamic_sequence(monkeypatch, tmp_path):
    import lm7

    calls = {}
    monkeypatch.setattr(huggingface, "_load_transformers", lambda: _exportable_transformers(calls))
    output = tmp_path / "model.lm7"

    result = huggingface.export_hf_model(
        "hf://example/tiny-model",
        output=str(output),
        prompt="Hello",
        target="cpu",
        dynamic_sequence=(1, 12),
    )

    assert result.sequence_bounds == (1, 12)
    assert calls["attn_implementation"] == "eager"

    reloaded = lm7.load_artifact(output)
    # One artifact, captured at 3 tokens, serving other lengths.
    for length in (1, 3, 7):
        logits = reloaded(
            input_ids=torch.ones((1, length), dtype=torch.long),
            attention_mask=torch.ones((1, length), dtype=torch.long),
        )
        assert logits.shape == (1, length, 16)


def test_dynamic_artifact_rejects_lengths_outside_its_bounds(monkeypatch, tmp_path):
    import lm7

    monkeypatch.setattr(huggingface, "_load_transformers", lambda: _exportable_transformers({}))
    output = tmp_path / "model.lm7"
    huggingface.export_hf_model(
        "hf://example/tiny-model",
        output=str(output),
        prompt="Hello",
        target="cpu",
        dynamic_sequence=(1, 8),
    )

    reloaded = lm7.load_artifact(output)
    with pytest.raises(ValueError, match=r"size 9; expected \[1, 8\]"):
        reloaded(
            input_ids=torch.ones((1, 9), dtype=torch.long),
            attention_mask=torch.ones((1, 9), dtype=torch.long),
        )


def test_dynamic_export_rejects_a_prompt_outside_its_bounds(monkeypatch, tmp_path):
    monkeypatch.setattr(huggingface, "_load_transformers", lambda: _exportable_transformers({}))

    with pytest.raises(UnsupportedModelError, match="outside the requested sequence bounds"):
        huggingface.export_hf_model(
            "hf://example/tiny-model",
            output=str(tmp_path / "model.lm7"),
            prompt="Hello",
            target="cpu",
            dynamic_sequence=(8, 64),
        )


def test_sequence_bounds_default_to_the_model_config():
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=64))
    assert huggingface._sequence_bounds(model, None) == (1, 64)

    # A model that advertises a huge context is capped, and one that advertises
    # nothing still gets usable bounds.
    large = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=131072))
    assert huggingface._sequence_bounds(large, None) == (1, huggingface._DEFAULT_MAX_SEQUENCE)
    assert huggingface._sequence_bounds(SimpleNamespace(), None) == (
        1,
        huggingface._DEFAULT_MAX_SEQUENCE,
    )


@pytest.mark.parametrize("bounds", [(0, 8), (8, 4)])
def test_sequence_bounds_reject_impossible_ranges(bounds):
    with pytest.raises(ValueError):
        huggingface._sequence_bounds(SimpleNamespace(), bounds)
