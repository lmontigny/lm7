from __future__ import annotations

import argparse
import copy

import torch

import lm7


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()


def _validate(
    source: torch.nn.Module,
    example_input: torch.Tensor,
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
    )
    actual = compiled(example_input).cpu()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    print(
        f"{target:>7}: backend={compiled.selected_backend}, "
        f"resolved={compiled.target}, output={tuple(actual.shape)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the same LM7 model on local CPU, NVIDIA, and Apple GPUs."
    )
    parser.add_argument("--backend", default="inductor")
    parser.add_argument(
        "--require-nvidia",
        action="store_true",
        help="Fail instead of skipping when an NVIDIA CUDA GPU is unavailable.",
    )
    parser.add_argument(
        "--require-apple",
        action="store_true",
        help="Fail instead of skipping when an Apple Silicon GPU is unavailable.",
    )
    arguments = parser.parse_args()

    torch.manual_seed(0)
    source = model()
    example_input = torch.randn(8, 16)
    expected = source(example_input)

    _validate(
        source,
        example_input,
        expected,
        target="cpu",
        backend=arguments.backend,
    )

    nvidia_available = torch.cuda.is_available() and not getattr(torch.version, "hip", None)
    if nvidia_available:
        _validate(
            source,
            example_input,
            expected,
            target="nvidia",
            backend=arguments.backend,
        )
    elif arguments.require_nvidia:
        raise SystemExit("An NVIDIA CUDA GPU is required but unavailable.")
    else:
        print(" nvidia: skipped (NVIDIA CUDA GPU unavailable)")

    apple_available = torch.backends.mps.is_available()
    if apple_available:
        _validate(
            source,
            example_input,
            expected,
            target="apple",
            backend=arguments.backend,
        )
    elif arguments.require_apple:
        raise SystemExit("An Apple Silicon GPU is required but unavailable.")
    else:
        print("  apple: skipped (Apple Silicon GPU unavailable)")


if __name__ == "__main__":
    main()
