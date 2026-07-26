from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

import lm7
from lm7.errors import CompilationError


def main() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    ).eval()
    example_input = torch.randn(2, 16)
    expected = model(example_input)

    with tempfile.TemporaryDirectory(prefix="lm7-aot-") as temporary_directory:
        output = Path(temporary_directory) / "model.lm7"
        try:
            artifact = lm7.export(
                model,
                args=(example_input,),
                target="cpu",
                backend="aot_inductor",
                output=output,
                debug=True,
            )
        except CompilationError as error:
            print(f"AOTInductor compilation failed: {error}", file=sys.stderr)
            if os.name == "nt" and shutil.which("cl") is None:
                print(
                    "Open a Visual Studio Developer PowerShell with the "
                    "'Desktop development with C++' workload installed.",
                    file=sys.stderr,
                )
            raise SystemExit(2) from None
        loaded = lm7.load_artifact(output)
        actual = loaded(example_input)
        torch.testing.assert_close(actual, expected)

        print(f"AOT artifact: {artifact.path}")
        print(f"Backend: {artifact.manifest.backend}")
        print("Debug artifacts:")
        for path in artifact.debug_files():
            print(f"- {path.relative_to(artifact.path)}")
        print("AOTInductor output matches eager PyTorch.")


if __name__ == "__main__":
    main()
