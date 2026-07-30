"""Run selected TorchBench models, plus a few models TorchBench doesn't ship,
through LM7's torch.compile backend.

TorchBench is intentionally an external CI checkout rather than an LM7
dependency. The workflow adds that checkout to ``PYTHONPATH`` before invoking
this script.
"""

from __future__ import annotations

import argparse
import importlib

import torch

import lm7

DEFAULT_MODELS = ("resnet18", "mobilenet_v2")


def _build_mixtral_moe_tiny() -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """A deliberately tiny sparse Mixture-of-Experts model (Mixtral architecture).

    TorchBench's model zoo has no MoE workload (``hf_mixtral`` is listed as a
    config in its HF harness but ships no runnable model directory), so this
    is built directly from transformers instead. Random init, no pretrained
    weights, and few enough parameters/experts to compile quickly on a CPU CI
    runner -- unlike the dense hf_Bert/timm_vision_transformer workloads, its
    expert-routing loop is data-dependent, so torch.compile graph-breaks
    around it rather than tracing it whole; LM7's inductor backend allows
    that (it does not pass fullgraph=True), and it still matches eager.
    """
    from transformers import MixtralConfig, MixtralForCausalLM

    config = MixtralConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    model = MixtralForCausalLM(config)
    model.eval()
    model.config.use_cache = False
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    return model, {"input_ids": input_ids}


LOCAL_MODELS = {"mixtral_moe_tiny": _build_mixtral_moe_tiny}


def run(model_name: str) -> None:
    """Compile one eval model and compare its output with eager."""
    if model_name in LOCAL_MODELS:
        model, example_inputs = LOCAL_MODELS[model_name]()
    else:
        try:
            module = importlib.import_module(f"torchbenchmark.models.{model_name}")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TorchBench is unavailable. Check out pytorch/benchmark and add it "
                "to PYTHONPATH before running this smoke test."
            ) from exc

        benchmark = module.Model(test="eval", device="cpu", batch_size=1)
        model, example_inputs = benchmark.get_module()

    # Vision models return a tuple of positional args; HF transformer models
    # (e.g. hf_Bert) return a dict of keyword args instead.
    if isinstance(example_inputs, dict):

        def call(m: torch.nn.Module):
            return m(**example_inputs)

    else:

        def call(m: torch.nn.Module):
            return m(*example_inputs)

    with torch.inference_mode():
        expected = call(model)

    compiled = lm7.compile(
        model,
        target="cpu",
        backend="inductor",
        fallback="error",
        cache=False,
    )
    actual = call(compiled)

    if compiled.selected_backend != "inductor" or compiled.state != "compiled":
        raise RuntimeError(
            f"{model_name}: expected compiled Inductor execution, got "
            f"backend={compiled.selected_backend!r}, state={compiled.state!r}"
        )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    print(f"{model_name}: eager and LM7 torch.compile outputs match")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()
    for model_name in args.models:
        run(model_name)


if __name__ == "__main__":
    main()
