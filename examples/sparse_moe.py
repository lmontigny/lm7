"""Validate tiny sparse Mixture-of-Experts models through LM7's JIT backends.

Two architectures, because they do not behave the same. Mixtral's pre-5.x
transformers implementation routes tokens with a data-dependent Python loop
(``for expert_idx in expert_hit: ...``) whose iteration count is only known at
runtime. torch.compile tolerates that; torch.export must capture one static graph
and fails, which takes every export-based backend down with it. OLMoE's routing
never used that loop, and transformers 5.x removed it from Mixtral as well.

That loop is an *export* problem only. What breaks Dynamo on the pinned
transformers is a different thing -- ``aten.nonzero`` in the router, whose output
shape depends on tensor data -- and it affects both architectures equally, eight
or nine breaks each. On transformers 5.x neither breaks at all. See
``benchmarks/moe.py``, which measures this rather than inferring it, and
docs/limitations.md.

So JIT works everywhere here, and exportability depends on the pair:

    transformers 4.57.3   mixtral: JIT only      olmoe: JIT and export
    transformers 5.14.1   mixtral: JIT + export  olmoe: JIT and export

Run with ``--architecture olmoe`` for the second one. See docs/limitations.md.
"""

from __future__ import annotations

import argparse
import copy

import torch

import lm7

# Small enough to compile quickly on a CPU CI runner: 2 layers, a handful of
# experts. Random init, no pretrained weights -- this is an architecture smoke
# test, not a quality benchmark.
#
# Dimensions are multiples of 16 on purpose. The transformers 5.x MoE path goes
# through `grouped_mm`, which requires strides that are multiples of 16 bytes and
# raises in *eager* otherwise -- an earlier revision used intermediate_size=37
# and stopped working when that landed.
_VOCAB_SIZE = 256
_ARCHITECTURES = ("mixtral", "olmoe")


def model(architecture: str = "mixtral"):
    common = {
        "vocab_size": _VOCAB_SIZE,
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
    }
    if architecture == "olmoe":
        from transformers import OlmoeConfig, OlmoeForCausalLM

        built = OlmoeForCausalLM(OlmoeConfig(num_experts=8, num_experts_per_tok=2, **common)).eval()
    else:
        from transformers import MixtralConfig, MixtralForCausalLM

        built = MixtralForCausalLM(
            MixtralConfig(num_local_experts=4, num_experts_per_tok=2, **common)
        ).eval()
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
        "--architecture",
        choices=_ARCHITECTURES,
        default="mixtral",
        help="Which MoE routing implementation to exercise (default: mixtral).",
    )
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
    source = model(arguments.architecture)
    print(f"architecture: {arguments.architecture}")
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
