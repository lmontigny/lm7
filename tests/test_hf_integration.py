from __future__ import annotations

import os

import pytest
import torch

import lm7
from lm7.huggingface import run_hf_model

MODEL_IDS = (
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "LiquidAI/LFM2.5-230M",
)
RUN_HF_TESTS = os.environ.get("LM7_RUN_HF_TESTS") == "1"

pytestmark = [
    pytest.mark.hf,
    pytest.mark.skipif(not RUN_HF_TESTS, reason="set LM7_RUN_HF_TESTS=1"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable"),
]


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_causal_lm_inductor_logits_and_generation(model_id):
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
    ).eval()
    inputs = tokenizer("The capital of France is", return_tensors="pt")

    model.cuda()
    cuda_inputs = {name: value.cuda() for name, value in inputs.items()}
    with torch.inference_mode():
        expected = model(**cuda_inputs, use_cache=False).logits

    compiled = lm7.compile(
        model,
        target="nvidia",
        backend="inductor",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(**inputs, use_cache=False).logits

    assert compiled.selected_backend == "inductor"
    assert compiled.target is not None
    assert compiled.target.vendor == "nvidia"
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.075)
    assert actual[:, -1].argmax().equal(expected[:, -1].argmax())

    with torch.inference_mode():
        generated = model.generate(
            **cuda_inputs,
            do_sample=False,
            max_new_tokens=4,
        )
    prompt_length = inputs["input_ids"].shape[1]
    assert prompt_length < generated.shape[1] <= prompt_length + 4


def test_hf_model_runner_on_cuda():
    result = run_hf_model(
        "hf://HuggingFaceTB/SmolLM2-135M-Instruct",
        prompt="The capital of France is",
        target="nvidia",
        backend="inductor",
    )

    assert result.target.startswith("nvidia:")
    assert result.backend == "inductor"
    assert result.dtype == "float16"
    assert result.parameter_count > 100_000_000
    assert result.input_tokens > 0
    assert result.first_call_ms > 0
    assert result.next_token
