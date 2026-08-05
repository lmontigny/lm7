from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

import lm7
from lm7.errors import BackendUnavailableError


def model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()


def main() -> None:
    if sys.platform != "darwin":
        raise SystemExit("The Core ML backend compiles and executes on macOS only.")

    source = model()
    example_input = torch.randn(8, 16)
    with torch.no_grad():
        expected = source(example_input)

    with tempfile.TemporaryDirectory(prefix="lm7-coreml-") as temporary_directory:
        output = Path(temporary_directory) / "model.lm7"
        try:
            artifact = lm7.export(
                source,
                args=(example_input,),
                target="apple",
                backend="coreml",
                output=output,
            )
        except BackendUnavailableError as error:
            print(f"Core ML export unavailable: {error}", file=sys.stderr)
            print('Install LM7 with ".[executorch]" on macOS to run this example.', file=sys.stderr)
            raise SystemExit(2) from None

        # Unlike QNN, Core ML is part of macOS itself: the artifact runs
        # immediately, on the machine that built it, no separate device.
        actual = artifact(example_input)
        # Default compute_precision is float16, so this is looser than the
        # float32-exact tolerances elsewhere in examples/ -- see docs/coreml.md.
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.01)

        reloaded = lm7.load_artifact(output)
        reloaded_output = reloaded(example_input)
        torch.testing.assert_close(reloaded_output, expected, rtol=0.05, atol=0.01)

        print(f"Artifact: {artifact.path}")
        print(f"Backend: {artifact.manifest.backend}")
        print(f"Device bound: {artifact.manifest.runtime_requirements['device_bound']}")
        print(f"Output: shape={tuple(actual.shape)}")
        print("Core ML output matches eager, immediately and after reload.")


if __name__ == "__main__":
    main()
