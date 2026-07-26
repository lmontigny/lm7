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
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        raise SystemExit("A ROCm-enabled PyTorch build and supported AMD GPU are required.")

    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source).cuda()
    example_input = torch.randn(8, 16)
    expected = reference(example_input.cuda())

    compiled = lm7.compile(
        source,
        target="amd",
        backend="inductor",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(example_input)
    torch.testing.assert_close(actual, expected)

    assert compiled.target is not None
    print(f"GPU: {torch.cuda.get_device_name(compiled.target.ordinal or 0)}")
    print(f"ROCm: {torch.version.hip}")
    print(f"Target: {compiled.target}")
    print(f"Backend: {compiled.selected_backend}")
    print(f"Output: shape={tuple(actual.shape)}, device={actual.device}")
    print("TorchInductor output matches eager ROCm.")


if __name__ == "__main__":
    main()
