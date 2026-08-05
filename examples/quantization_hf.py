from __future__ import annotations

import argparse

from lm7.errors import UnsupportedModelError
from lm7.huggingface import run_hf_model

DEFAULT_MODEL = "hf://HuggingFaceTB/SmolLM2-135M-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Hugging Face causal LM with weight quantization, "
        "and compare its footprint against an unquantized baseline."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target", default="cpu", help="target selector (default: cpu)")
    parser.add_argument(
        "--quantize",
        default="int8",
        help="none, int8, fp8, nvfp4, fp8-dynamic, or nvfp4-dynamic -- see docs/quantization.md",
    )
    parser.add_argument("--prompt", default="The capital of France is")
    arguments = parser.parse_args()

    baseline = run_hf_model(arguments.model, prompt=arguments.prompt, target=arguments.target)
    try:
        quantized = run_hf_model(
            arguments.model,
            prompt=arguments.prompt,
            target=arguments.target,
            quantization=arguments.quantize,
        )
    except UnsupportedModelError as error:
        raise SystemExit(
            f"{error} See docs/quantization.md for which (target, quantize) pairs are validated."
        ) from None

    baseline_mib = baseline.model_storage_bytes / (1024 * 1024)
    quantized_mib = quantized.model_storage_bytes / (1024 * 1024)
    print(f"Model: {arguments.model}")
    print(f"Target: {quantized.target}, backend: {quantized.backend}")
    print(f"Baseline ({baseline.quantization}) storage: {baseline_mib:.1f} MiB")
    print(f"Quantized ({quantized.quantization}) storage: {quantized_mib:.1f} MiB")
    print(f"Shrink: {baseline_mib / quantized_mib:.2f}x")
    print(f"Baseline next token:  {baseline.next_token!r}")
    print(f"Quantized next token: {quantized.next_token!r}")
    if baseline.next_token == quantized.next_token:
        print("Quantized output agrees with the unquantized baseline on the next token.")
    else:
        print("Quantized output disagrees with the baseline -- expected for some modes.")


if __name__ == "__main__":
    main()
