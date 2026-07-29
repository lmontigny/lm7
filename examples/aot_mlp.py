from __future__ import annotations

import argparse
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import torch

import lm7
from lm7.detection import torch_device
from lm7.errors import CompilationError
from lm7.targets import parse_target


def _model_and_input() -> tuple[torch.nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    ).eval()
    example_input = torch.randn(2, 16)
    return model, example_input


def _device_input(example_input: torch.Tensor, target: str) -> torch.Tensor:
    """Move the example input to the device the artifact was captured for."""
    return example_input.to(torch_device(parse_target(target)))


def _verify_loaded(path: Path) -> None:
    model, example_input = _model_and_input()
    expected = model(example_input)
    loaded = lm7.load_artifact(path)
    target = loaded.manifest.target["vendor"]
    actual = loaded(_device_input(example_input, target))
    torch.testing.assert_close(actual.cpu(), expected)
    print(f"Reloaded artifact matches eager PyTorch: {path}")


def _compile(output: Path, target: str) -> None:
    model, example_input = _model_and_input()
    expected = model(example_input)
    try:
        artifact = lm7.export(
            model,
            args=(example_input,),
            target=target,
            backend="aot_inductor",
            output=output,
            debug=True,
        )
    except CompilationError as error:
        print(f"AOTInductor compilation failed: {error}", file=sys.stderr)
        print("Run this test in Linux or WSL with a working g++ toolchain.", file=sys.stderr)
        raise SystemExit(2) from None
    loaded = lm7.load_artifact(output)
    actual = loaded(_device_input(example_input, target))
    torch.testing.assert_close(actual.cpu(), expected)

    print(f"AOT artifact: {artifact.path}")
    print(f"Backend: {artifact.manifest.backend}")
    print("Debug artifacts:")
    for path in artifact.debug_files():
        print(f"- {path.relative_to(artifact.path)}")
    print("AOTInductor output matches eager PyTorch.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile or reload the LM7 AOT example.")
    parser.add_argument(
        "--target",
        default="cpu",
        help="Target to package for; aot_inductor validates cpu, apple, and nvidia.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output",
        type=Path,
        help="Keep the compiled artifact at this path instead of using a temporary directory.",
    )
    output_group.add_argument(
        "--load",
        type=Path,
        help="Load and validate an artifact created by a previous process.",
    )
    arguments = parser.parse_args()

    if arguments.load is not None:
        _verify_loaded(arguments.load.resolve())
        return

    temporary_context = (
        nullcontext(None)
        if arguments.output is not None
        else tempfile.TemporaryDirectory(prefix="lm7-aot-")
    )
    with temporary_context as temporary_directory:
        output = (
            arguments.output.resolve()
            if arguments.output is not None
            else Path(temporary_directory) / "model.lm7"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        _compile(output, arguments.target)


if __name__ == "__main__":
    main()
