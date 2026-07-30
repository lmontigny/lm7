"""Validate a tiny sparse Mixture-of-Experts model (Mixtral architecture) through
LM7's JIT compile backends.

MoE routing (``for expert_idx in expert_hit: ...`` in the reference transformers
implementation) is data-dependent: the number of expert dispatches is only known
at runtime. torch.compile tolerates that -- Dynamo graph-breaks around the
routing loop and runs it eagerly -- but torch.export cannot capture it at all.
That means this model works through LM7's JIT backends (inductor, tensorrt) but
not through any export-based backend (aot_inductor, openvino, onnxruntime,
executorch, iree_vulkan, litert, stablehlo); see docs/limitations.md.
"""

from __future__ import annotations

import argparse
import copy

import torch

import lm7

# Small enough to compile quickly on a CPU CI runner: 2 layers, 4 experts,
# hidden_size=32. Random init, no pretrained weights -- this is an
# architecture smoke test, not a quality benchmark.
_VOCAB_SIZE = 256


def model():
    from transformers import MixtralConfig, MixtralForCausalLM

    config = MixtralConfig(
        vocab_size=_VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    built = MixtralForCausalLM(config).eval()
    built.config.use_cache = False
    return built


def _validate(
    source: torch.nn.Module,
    example_inputs: dict[str, torch.Tensor],
    expected: torch.Tensor,
    *,
    target: str,
    backend: str,
) -> None:
    compiled = lm7.compile(
        copy.deepcopy(source),
        target=target,
        backend=backend,
        transfers="automatic",
        fallback="error",
        cache=False,
    )
    actual = compiled(**example_inputs).logits.cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    print(
        f"{target:>7}: backend={compiled.selected_backend}, "
        f"resolved={compiled.target}, output={tuple(actual.shape)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a tiny sparse Mixture-of-Experts model through LM7's JIT compile backends."
    )
    parser.add_argument("--backend", default="inductor")
    parser.add_argument(
        "--require-nvidia",
        action="store_true",
        help="Fail instead of skipping when an NVIDIA CUDA GPU is unavailable.",
    )
    arguments = parser.parse_args()

    try:
        import transformers  # noqa: F401
    except ImportError:
        raise SystemExit('Install Hugging Face support with: pip install -e ".[hf]"') from None

    torch.manual_seed(0)
    source = model()
    example_inputs = {"input_ids": torch.randint(0, _VOCAB_SIZE, (1, 16))}
    with torch.inference_mode():
        expected = source(**example_inputs).logits

    _validate(source, example_inputs, expected, target="cpu", backend=arguments.backend)

    nvidia_available = torch.cuda.is_available() and not getattr(torch.version, "hip", None)
    if nvidia_available:
        _validate(source, example_inputs, expected, target="nvidia", backend=arguments.backend)
    elif arguments.require_nvidia:
        raise SystemExit("An NVIDIA CUDA GPU is required but unavailable.")
    else:
        print(" nvidia: skipped (NVIDIA CUDA GPU unavailable)")


if __name__ == "__main__":
    main()
