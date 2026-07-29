from __future__ import annotations

import torch

import lm7


def main() -> None:
    try:
        import torch_xla.runtime as xr
    except ImportError:
        raise SystemExit(
            "Install the Tenstorrent PJRT plugin with: "
            "uv pip install pjrt-plugin-tt "
            "--extra-index-url https://pypi.eng.aws.tenstorrent.com/"
        ) from None
    if xr.device_type() != "TT":
        xr.set_device_type("TT")
    if xr.device_type() != "TT":
        raise SystemExit(f"Expected a TT PJRT runtime, found {xr.device_type() or 'none'}.")

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
        target="tenstorrent",
        backend="tenstorrent",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(example_input)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)

    print(f"Target: {compiled.target}")
    print(f"Backend: {compiled.selected_backend}")
    print(f"Addressable Tenstorrent devices: {xr.addressable_device_count()}")
    print(f"Output: shape={tuple(actual.shape)}, device={actual.device}")
    print("tt-xla output matches eager CPU.")


if __name__ == "__main__":
    main()
