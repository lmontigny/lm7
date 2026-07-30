"""Run selected TorchBench models through LM7's torch.compile backend.

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


def run(model_name: str) -> None:
    """Compile one TorchBench eval model and compare its output with eager."""
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
