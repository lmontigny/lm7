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

    # Export-only, and portable in a way LM7's other AOT formats are not: the
    # payload is StableHLO plus raw weight files, loadable by any PJRT client
    # -- CPU, NVIDIA, AMD, TPU -- with no PyTorch installed at all. This
    # example still round-trips through lm7.load_artifact(), which does have
    # PyTorch; see docs/stablehlo-pjrt-evaluation.md for the PyTorch-free load.
    with tempfile.TemporaryDirectory(prefix="lm7-stablehlo-") as temporary_directory:
        output = Path(temporary_directory) / "model.lm7"
        try:
            artifact = lm7.export(
                source,
                args=(example_input,),
                target="cpu",
                backend="stablehlo",
                output=output,
            )
        except BackendUnavailableError as error:
            print(f"StableHLO/PJRT unavailable: {error}")
            raise SystemExit('Install it with: pip install -e ".[stablehlo]"') from None

        actual = artifact(example_input)
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)

        reloaded = lm7.load_artifact(output)
        reloaded_output = reloaded(example_input)
        torch.testing.assert_close(reloaded_output.cpu(), expected, rtol=1e-5, atol=1e-6)

        print(f"Artifact: {artifact.path}")
        print(f"Backend: {artifact.manifest.backend}")
        print(f"Output: shape={tuple(actual.shape)}")
        print("StableHLO output matches eager, immediately and after reload.")


if __name__ == "__main__":
    main()
