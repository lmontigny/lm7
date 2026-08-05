from __future__ import annotations

import tempfile
from pathlib import Path

import torch

import lm7
from lm7.errors import BackendUnavailableError


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    ).eval()


def main() -> None:
    torch.manual_seed(0)
    source = model()
    example_input = torch.randn(8, 16)
    expected = source(example_input)

    with tempfile.TemporaryDirectory(prefix="lm7-litert-") as temporary_directory:
        output = Path(temporary_directory) / "model.lm7"
        try:
            artifact = lm7.export(
                source,
                args=(example_input,),
                target="cpu",
                backend="litert",
                output=output,
            )
        except BackendUnavailableError as error:
            print(f"LiteRT unavailable: {error}")
            raise SystemExit(
                'Install it in a dedicated env: pip install -e ".[litert]" '
                "(LiteRT Torch pins PyTorch <2.13 -- see docs/litert.md)"
            ) from None

        actual = artifact(example_input)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        reloaded = lm7.load_artifact(output)
        reloaded_output = reloaded(example_input)
        torch.testing.assert_close(reloaded_output, expected, rtol=1e-5, atol=1e-6)

        print(f"Artifact: {artifact.path}")
        print(f"Backend: {artifact.manifest.backend}")
        print(f"Output: shape={tuple(actual.shape)}")
        print("LiteRT output matches eager, immediately and after reload.")
        print("This is the generic tensor-model path, not LiteRT-LM -- see docs/litert.md.")


if __name__ == "__main__":
    main()
