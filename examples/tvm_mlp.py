from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Compile or export through LM7's TVM backend.")
    parser.add_argument(
        "--mode",
        choices=("jit", "export"),
        default="jit",
        help="lm7.compile() (JIT, nothing outlives the process) or lm7.export() (AOT artifact)",
    )
    arguments = parser.parse_args()

    torch.manual_seed(0)
    source = model()
    example_input = torch.randn(8, 16)
    expected = source(example_input)

    try:
        if arguments.mode == "jit":
            compiled = lm7.compile(source, target="cpu", backend="tvm", fallback="error")
            actual = compiled(example_input)
            backend = compiled.selected_backend
        else:
            with tempfile.TemporaryDirectory(prefix="lm7-tvm-") as temporary_directory:
                output = Path(temporary_directory) / "model.lm7"
                lm7.export(
                    source, args=(example_input,), target="cpu", backend="tvm", output=output
                )
                reloaded = lm7.load_artifact(output)
                actual = reloaded(example_input)
                backend = reloaded.manifest.backend
    except BackendUnavailableError as error:
        print(f"TVM unavailable: {error}")
        raise SystemExit('Install it with: pip install -e ".[tvm]" -- see docs/tvm.md') from None

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    print(f"Mode: {arguments.mode}")
    print(f"Backend: {backend}")
    print(f"Output: shape={tuple(actual.shape)}")
    print("TVM output matches eager CPU.")
    print("TVM is explicit-only and far slower than Inductor -- see docs/tvm.md.")


if __name__ == "__main__":
    main()
