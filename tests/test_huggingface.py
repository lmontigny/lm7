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
                "compile_config": kwargs.get("compile_config"),
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


def test_generate_hf_model_sends_compile_config_where_decode_compiles(monkeypatch):
    """The other half: where Transformers will compile, LM7 must still ask."""
    calls = {}
    monkeypatch.setattr(
        huggingface, "_load_transformers", lambda: _fake_generation_transformers(calls)
    )
    monkeypatch.setattr(huggingface, "compiles_decode", lambda device: True)

    result = huggingface.generate_hf_model(
        "hf://example/tiny-model",
        prompt="Hello",
        max_new_tokens=4,
        target="cpu",
    )

    assert all(call["compile_config"] is not None for call in calls["generate"])
    assert result.backend == "inductor"


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
    # A CPU target does not meet Transformers' compilation criteria, so the
    # decode loop runs eagerly however the compile_config is spelled -- and
    # sending one anyway only produces a "Compilation will be skipped" warning
    # that reads like an LM7 fault. LM7 asks only when the answer is yes.
    assert all(call["compile_config"] is None for call in calls["generate"])
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


@pytest.mark.parametrize(
    ("architecture", "native"),
    [
        ("sm70", False),  # Volta
        ("sm75", False),  # Turing, e.g. Tesla T4
        ("sm80", True),  # Ampere onwards
        ("sm89", True),
        ("sm90", True),
        (None, True),  # unqualified `nvidia`, architecture not yet resolved
        ("gfx942", True),  # not an sm number, so not gated
    ],
)
def test_native_bf16_follows_nvidia_architecture(architecture, native):
    """torch.cuda.is_bf16_supported() cannot answer this.

    It reports True on a Tesla T4, where BF16 is emulated: measured there, BF16
    prefill ran 3.4x slower than FP16 and took 2.8x longer to compile. So the check
    is on the capability number, not on torch.
    """
    target = TargetSpec("nvidia", "gpu", architecture=architecture)
    assert huggingface.supports_native_bf16(target) is native


def test_native_bf16_is_not_gated_off_nvidia():
    assert huggingface.supports_native_bf16(TargetSpec("cpu", "cpu")) is True


@pytest.mark.parametrize("quantization", ["int8", "fp8", "nvfp4"])
def test_weight_only_quantization_is_rejected_below_ampere(quantization):
    """Turing has no usable INT8 path, so it is refused rather than offered.

    Measured on a Tesla T4 (sm75) with SmolLM2-135M: BF16 compute is emulated and ran
    3.3x slower than unquantized FP16 while dropping to 3/4 top-1, and FP16 compute
    produced NaN logits at 0/4. A mode whose best case is a regression should raise.
    """
    turing = TargetSpec("nvidia", "gpu", architecture="sm75")
    with pytest.raises(UnsupportedModelError, match="Ampere"):
        huggingface._validate_quantization(
            quantization, turing, "inductor", "auto", "unsloth/Llama-3.2-1B-Instruct"
        )


def test_ampere_and_newer_still_accept_int8():
    for architecture in ("sm80", "sm89"):
        huggingface._validate_quantization(
            "int8",
            TargetSpec("nvidia", "gpu", architecture=architecture),
            "inductor",
            "bfloat16",
            "HuggingFaceTB/SmolLM2-135M-Instruct",
        )


def test_fp8_capability_gate_uses_the_shared_parser():
    """sm75 is the case a T4 exercises for real; sm89 is the local GPU."""
    assert not huggingface._supports_fp8(TargetSpec("nvidia", "gpu", architecture="sm75"))
    assert huggingface._supports_fp8(TargetSpec("nvidia", "gpu", architecture="sm89"))
    # An unparseable or absent architecture must not gate.
    assert huggingface._supports_fp8(TargetSpec("nvidia", "gpu", architecture=None))
    assert huggingface._supports_fp8(TargetSpec("nvidia", "gpu", architecture="gfx942"))


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
    monkeypatch.setattr(huggingface, "synchronize", lambda _target: None)

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
    monkeypatch.setattr(huggingface, "synchronize", lambda _target: None)

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


@pytest.mark.parametrize(
    "quantization",
    [huggingface.FP8, huggingface.FP8_DYNAMIC, huggingface.FP8_DYNAMIC_ROWWISE],
)
def test_fp8_modes_are_admitted_on_cdna3(quantization):
    """Measured on an MI300X (gfx942), and on the kernel rather than the API.

    `benchmarks/fp8_kernel_check.py` shows both dynamic modes emitting
    `_scaled_mm` with no plain `mm` there -- the same generated code the sm90 and
    sm120 rows show -- so CDNA 3 computes in FP8 rather than dequantizing into
    BF16. Both hold 4/4 top-1. See docs/amd-mi300x.md.
    """
    huggingface._validate_quantization(
        quantization,
        TargetSpec("amd", "gpu", architecture="gfx942"),
        "inductor",
        "auto",
        "unsloth/Llama-3.2-1B-Instruct",
    )


@pytest.mark.parametrize(
    "quantization",
    [huggingface.FP8, huggingface.FP8_DYNAMIC, huggingface.FP8_DYNAMIC_ROWWISE],
)
def test_fp8_is_refused_on_an_amd_part_without_it(quantization):
    """The gate the NVIDIA capability comparison cannot provide.

    `compute_capability` is None for every `gfx`, and every capability check
    reads None as "do not gate" -- correct for an unresolved NVIDIA target, and
    wrong here, where it would hand FP8 to a CDNA 2 part that has none.
    """
    with pytest.raises(UnsupportedModelError, match="native FP8"):
        huggingface._validate_quantization(
            quantization,
            TargetSpec("amd", "gpu", architecture="gfx90a"),
            "inductor",
            "auto",
            "unsloth/Llama-3.2-1B-Instruct",
        )


def test_int8_stays_off_amd_despite_holding_its_tokens():
    """INT8 keeps 4/4 top-1 on gfx942 and is still refused.

    It runs ~10x slower than BF16 there, measured twice -- on torch 2.10 and
    again on 2.13, where torchao's compiled extensions load and the number moved
    under 3%. That is the shape LM7 already refuses on Turing: a mode whose best
    case is a regression.
    """
    with pytest.raises(UnsupportedModelError, match="NVIDIA GPUs and CPU"):
        huggingface._validate_quantization(
            huggingface.INT8,
            TargetSpec("amd", "gpu", architecture="gfx942"),
            "inductor",
            "auto",
            "unsloth/Llama-3.2-1B-Instruct",
        )


@pytest.mark.parametrize("quantization", [huggingface.NVFP4, huggingface.NVFP4_DYNAMIC])
def test_nvfp4_stays_off_amd(quantization):
    """No shipping CDNA part computes FP4. Weight-only NVFP4 scored 3/4 on
    gfx942 and torchao refuses the dynamic mode with its own `sm100+`
    assertion."""
    with pytest.raises(UnsupportedModelError, match="NVIDIA"):
        huggingface._validate_quantization(
            quantization,
            TargetSpec("amd", "gpu", architecture="gfx942"),
            "inductor",
            "auto",
            "unsloth/Llama-3.2-1B-Instruct",
        )


def test_amd_quantization_pins_bfloat16():
    """`_QUANTIZED_COMPUTE_DTYPE` needed an `amd` entry or this raised KeyError
    rather than an LM7Error -- the subscript is bare."""
    assert huggingface._QUANTIZED_COMPUTE_DTYPE["amd"] == "bfloat16"
    assert (
        huggingface._resolve_dtype(
            "auto", TargetSpec("amd", "gpu", architecture="gfx942"), huggingface.FP8
        )
        == torch.bfloat16
    )
    with pytest.raises(UnsupportedModelError, match="dtype"):
        huggingface._validate_quantization(
            huggingface.FP8,
            TargetSpec("amd", "gpu", architecture="gfx942"),
            "inductor",
            "float16",
            "unsloth/Llama-3.2-1B-Instruct",
        )


@pytest.mark.parametrize(
    "target",
    [
        TargetSpec("nvidia", "gpu", architecture="sm90"),
        TargetSpec("amd", "gpu", architecture="gfx942"),
    ],
)
@pytest.mark.parametrize(
    "quantization",
    [huggingface.FP8, huggingface.FP8_DYNAMIC, huggingface.FP8_DYNAMIC_ROWWISE],
)
def test_auto_dtype_covers_dynamic_activation_modes(target, quantization):
    """`--dtype auto` used to resolve the dynamic modes to float16.

    The branch keyed on `WEIGHT_ONLY_QUANTIZATIONS`, so `fp8-dynamic` and
    `fp8-dynamic-rowwise` fell through to the GPU default and torchao raised
    "PerRow quantization only works for bfloat16 precision input weight" from
    inside `quantize_`. `_validate_quantization` cannot catch that: the caller
    passes the "auto" it accepts, and the wrong dtype is produced afterwards.

    Vendor-independent -- it reproduced on `sm90` in the same check that found it
    on `gfx942`. It survived because every recorded measurement of these modes
    came from `benchmarks/quantization.py`, which passes `--dtype bfloat16`.
    """
    assert huggingface._resolve_dtype("auto", target, quantization) == torch.bfloat16


def test_native_bf16_is_read_from_the_gfx_table():
    """AMD's BF16 answer comes from the same table `lm7 targets` prints, not from
    a blanket True for every non-NVIDIA vendor."""
    assert huggingface.supports_native_bf16(TargetSpec("amd", "gpu", architecture="gfx942"))
    assert huggingface.supports_native_bf16(TargetSpec("amd", "gpu", architecture="gfx90a"))
    # Vega 20 predates CDNA's BF16 matrix instructions.
    assert not huggingface.supports_native_bf16(TargetSpec("amd", "gpu", architecture="gfx906"))
    # An unresolved or unrecognized part keeps the old "do not refuse" meaning.
    assert huggingface.supports_native_bf16(TargetSpec("amd", "gpu"))
    assert huggingface.supports_native_bf16(TargetSpec("amd", "gpu", architecture="gfx1250"))


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
