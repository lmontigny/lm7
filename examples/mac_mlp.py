from __future__ import annotations

import copy

import torch

import lm7


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("An Apple Silicon GPU with Metal (MPS) support is required.")

    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source).to("mps")
    example_input = torch.randn(8, 16)
    expected = reference(example_input.to("mps"))

    compiled = lm7.compile(
        source,
        target="apple",
        backend="inductor",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(example_input)
    # See tests/test_mac_integration.py for why this isn't the tight float32
    # default: it fails on GitHub's macos-26 CI runner's Apple GPU generation,
    # most likely Inductor's Metal GELU codegen against eager MPS's kernel.
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.25)

    print(f"Target: {compiled.target}")
    print(f"Backend: {compiled.selected_backend}")
    print(f"Output: shape={tuple(actual.shape)}, device={actual.device}")
    print("TorchInductor output matches eager MPS.")


if __name__ == "__main__":
    main()
