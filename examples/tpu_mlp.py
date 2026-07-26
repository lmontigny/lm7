from __future__ import annotations

import torch

import lm7


def main() -> None:
    try:
        import torch_xla.runtime as xr
    except ImportError:
        raise SystemExit('Install TPU support with: pip install -e ".[openxla]"') from None
    if xr.device_type() != "TPU":
        raise SystemExit(f"Expected a TPU PJRT runtime, found {xr.device_type() or 'none'}.")

    torch.manual_seed(0)
    source = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()
    example_input = torch.randn(8, 16)
    expected = source(example_input)

    compiled = lm7.compile(
        source,
        target="tpu",
        backend="openxla",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(example_input)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=2e-3)

    print(f"Target: {compiled.target}")
    print(f"Backend: {compiled.selected_backend}")
    print(f"Addressable TPU devices: {xr.addressable_device_count()}")
    print(f"Output: shape={tuple(actual.shape)}, device={actual.device}")
    print("OpenXLA output matches eager CPU.")


if __name__ == "__main__":
    main()
