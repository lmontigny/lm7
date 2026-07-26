from __future__ import annotations

import os

import pytest
import torch

import lm7

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
RUN_HF_TESTS = os.environ.get("LM7_RUN_HF_TESTS") == "1"

pytestmark = [
    pytest.mark.hf,
    pytest.mark.skipif(not RUN_HF_TESTS, reason="set LM7_RUN_HF_TESTS=1"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable"),
]


def test_smollm2_inductor_logits_and_generation():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
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
    assert generated.shape[1] == inputs["input_ids"].shape[1] + 4
