from __future__ import annotations

import os

import pytest
import torch

import lm7
from lm7.detection import resolve_target, torch_device
from lm7.huggingface import (
    FP8_WEIGHT_ONLY,
    INT8_WEIGHT_ONLY,
    _apply_quantization,
    _model_storage_bytes,
    export_hf_model,
    run_hf_model,
)

MODEL_IDS = (
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "LiquidAI/LFM2.5-230M",
    "unsloth/Llama-3.2-1B-Instruct",
    "Qwen/Qwen3.5-0.8B",
)
RUN_HF_TESTS = os.environ.get("LM7_RUN_HF_TESTS") == "1"
HAS_ACCELERATOR = torch.cuda.is_available() or torch.backends.mps.is_available()

pytestmark = [
    pytest.mark.hf,
    pytest.mark.skipif(not RUN_HF_TESTS, reason="set LM7_RUN_HF_TESTS=1"),
]


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="CUDA or MPS GPU is unavailable")
@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_causal_lm_inductor_logits_and_generation(model_id):
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
    ).eval()
    inputs = tokenizer("The capital of France is", return_tensors="pt")

    target = resolve_target("auto")
    device = torch_device(target)
    model.to(device)
    device_inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        expected = model(**device_inputs, use_cache=False).logits

    compiled = lm7.compile(
        model,
        target=target,
        backend="inductor",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(**inputs, use_cache=False).logits

    assert compiled.selected_backend == "inductor"
    assert compiled.target is not None
    assert compiled.target.vendor == target.vendor
    # MPS float16 reductions accumulate in a different order than CUDA and land a
    # wider tail of outlier logits; measured max abs diff was 0.195 on SmolLM2.
    atol = 0.25 if target.vendor == "apple" else 0.075
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=atol)
    assert actual[:, -1].argmax().equal(expected[:, -1].argmax())

    with torch.inference_mode():
        generated = model.generate(
            **device_inputs,
            do_sample=False,
            max_new_tokens=4,
        )
    prompt_length = inputs["input_ids"].shape[1]
    assert prompt_length < generated.shape[1] <= prompt_length + 4


@pytest.mark.skipif(not HAS_ACCELERATOR, reason="CUDA or MPS GPU is unavailable")
def test_hf_model_runner_on_accelerator():
    result = run_hf_model(
        "hf://HuggingFaceTB/SmolLM2-135M-Instruct",
        prompt="The capital of France is",
        target="auto",
        backend="inductor",
    )

    assert result.target.split(":", 1)[0] in {"nvidia", "amd", "apple"}
    assert result.backend == "inductor"
    assert result.dtype == "float16"
    assert result.parameter_count > 100_000_000
    assert result.baseline_model_storage_bytes == result.model_storage_bytes
    assert result.model_storage_bytes > 0
    assert result.input_tokens > 0
    assert result.first_call_ms > 0
    assert result.next_token


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable")
@pytest.mark.parametrize(
    ("quantization", "minimum_cosine", "maximum_p99_error", "maximum_storage_ratio"),
    [
        (INT8_WEIGHT_ONLY, 0.99, 1.0, 0.7),
        (FP8_WEIGHT_ONLY, 0.995, 2.0, 0.75),
    ],
)
def test_weight_only_quantization_matches_bfloat16_logits(
    quantization,
    minimum_cosine,
    maximum_p99_error,
    maximum_storage_ratio,
):
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torchao")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    ).eval()
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    target = resolve_target("nvidia")
    model.cuda()
    cuda_inputs = {name: value.cuda() for name, value in inputs.items()}

    with torch.inference_mode():
        expected = model(**cuda_inputs, use_cache=False).logits

    baseline_storage = _model_storage_bytes(model)
    _apply_quantization(model, target, quantization)
    quantized_storage = _model_storage_bytes(model)
    compiled = lm7.compile(
        model,
        target=target,
        backend="inductor",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(**inputs, use_cache=False).logits

    expected_float = expected.float()
    actual_float = actual.float()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        actual_float.flatten(),
        expected_float.flatten(),
        dim=0,
    )
    p99_absolute_error = torch.quantile((actual_float - expected_float).abs(), 0.99)

    assert cosine_similarity.item() >= minimum_cosine
    # Weight-only quantization can move low-probability logits while preserving
    # the output distribution and selected token.
    assert p99_absolute_error.item() <= maximum_p99_error
    assert actual[:, -1].argmax().equal(expected[:, -1].argmax())
    assert quantized_storage < baseline_storage * maximum_storage_ratio


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable")
def test_dynamic_sequence_artifact_serves_many_prompt_lengths(tmp_path):
    """One artifact, many prompt lengths, no compiler in the loading process."""
    transformers = pytest.importorskip("transformers")
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    major, minor = torch.cuda.get_device_capability()

    result = export_hf_model(
        f"hf://{model_id}",
        output=str(tmp_path / "dynamic.lm7"),
        prompt="The capital of France is",
        target=f"nvidia:sm{major}{minor}",
        backend="aot_inductor",
        dtype="float16",
        dynamic_sequence=(1, 256),
    )
    assert result.sequence_bounds == (1, 256)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    reference = (
        transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            attn_implementation="eager",
        )
        .eval()
        .cuda()
    )
    artifact = lm7.load_artifact(result.output)

    # The capture used a 5-token prompt; none of these are 5 tokens.
    prompts = ("Hi", "Write a short poem about the sea and the sky on a summer evening")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        logits = artifact(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        with torch.no_grad():
            expected = reference(**inputs, use_cache=False).logits
        assert logits.shape == expected.shape
        assert logits[0, -1].argmax() == expected[0, -1].argmax()

    too_long = torch.ones((1, 257), dtype=torch.long, device="cuda")
    with pytest.raises(ValueError, match=r"expected \[1, 256\]"):
        artifact(input_ids=too_long, attention_mask=too_long)
