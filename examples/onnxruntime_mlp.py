from __future__ import annotations

import copy

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
    reference = copy.deepcopy(source)
    example_input = torch.randn(8, 16)
    expected = reference(example_input)

    try:
        compiled = lm7.compile(
            source,
            target="cpu",
            backend="onnxruntime",
            fallback="error",
        )
        actual = compiled(example_input)
    except BackendUnavailableError as error:
        print(f"ONNX Runtime unavailable: {error}")
        raise SystemExit('Install it with: pip install -e ".[onnxruntime]"') from None

    torch.testing.assert_close(actual, expected)

    print(f"Target: {compiled.target}")
    print(f"Backend: {compiled.selected_backend}")
    print(f"Output: shape={tuple(actual.shape)}, device={actual.device}")
    print("ONNX Runtime output matches eager CPU.")
    print("For NVIDIA, pass target='nvidia:sm89' with the '.[onnxruntime-gpu]' extra instead.")


if __name__ == "__main__":
    main()
