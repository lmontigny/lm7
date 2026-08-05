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

    with tempfile.TemporaryDirectory(prefix="lm7-executorch-") as temporary_directory:
        output = Path(temporary_directory) / "model.lm7"
        try:
            artifact = lm7.export(
                source,
                args=(example_input,),
                target="cpu",
                backend="executorch",
                output=output,
            )
        except BackendUnavailableError as error:
            print(f"ExecuTorch unavailable: {error}")
            raise SystemExit('Install it with: pip install -e ".[executorch]"') from None

        # The XNNPACK delegate is portable across ARM64 and x86-64, so the
        # same .pte this produces for a phone also runs right here.
        actual = artifact(example_input)
        torch.testing.assert_close(actual, expected)

        reloaded = lm7.load_artifact(output)
        reloaded_output = reloaded(example_input)
        torch.testing.assert_close(reloaded_output, expected)

        print(f"Artifact: {artifact.path}")
        print(f"Backend: {artifact.manifest.backend}")
        print(f"Output: shape={tuple(actual.shape)}")
        print("ExecuTorch (XNNPACK) output matches eager, immediately and after reload.")
        print("This .pte also runs on Android/iOS -- see docs/executorch.md.")


if __name__ == "__main__":
    main()
