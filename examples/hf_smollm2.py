from __future__ import annotations

import argparse

import torch

import lm7

DEFAULT_MODEL = "hf://HuggingFaceTB/SmolLM2-135M-Instruct"


def _hf_model_id(value: str) -> str:
    return value.removeprefix("hf://")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a Hugging Face causal language model through LM7."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    arguments = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit('Install Hugging Face support with: pip install -e ".[hf]"') from None

    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-capable NVIDIA GPU.")

    model_id = _hf_model_id(arguments.model)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).eval()
    inputs = tokenizer(arguments.prompt, return_tensors="pt")

    model.cuda()
    cuda_inputs = {name: value.cuda() for name, value in inputs.items()}
    with torch.inference_mode():
        eager_logits = model(**cuda_inputs, use_cache=False).logits

    compiled = lm7.compile(
        model,
        target="nvidia",
        backend="inductor",
        transfers="automatic",
        fallback="error",
    )
    compiled_logits = compiled(**inputs, use_cache=False).logits
    torch.testing.assert_close(compiled_logits, eager_logits, rtol=0.02, atol=0.075)
    assert compiled_logits[:, -1].argmax().equal(eager_logits[:, -1].argmax())

    with torch.inference_mode():
        generated = model.generate(
            **cuda_inputs,
            do_sample=False,
            max_new_tokens=arguments.max_new_tokens,
        )

    print(f"Model: {arguments.model}")
    print(f"Target: {compiled.target}")
    print(f"Backend: {compiled.selected_backend}")
    print("Compiled logits match eager CUDA.")
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
