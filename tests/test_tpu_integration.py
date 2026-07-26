from __future__ import annotations

import importlib.util

import pytest
import torch

import lm7


def _tpu_available() -> bool:
    try:
        if importlib.util.find_spec("torch_xla") is None:
            return False
        import torch_xla.runtime as xr

        return xr.device_type() == "TPU" and xr.addressable_device_count() > 0
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError):
        return False


pytestmark = [
    pytest.mark.tpu,
    pytest.mark.skipif(not _tpu_available(), reason="Google TPU runtime is unavailable"),
]


def test_openxla_tpu_matches_cpu_eager():
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

    assert compiled.selected_backend == "openxla"
    assert compiled.target is not None
    assert compiled.target.vendor == "tpu"
    assert actual.device.type == "xla"
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=2e-3)
