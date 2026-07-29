from __future__ import annotations

import importlib.util

import pytest
import torch

import lm7


def _tenstorrent_available() -> bool:
    try:
        if importlib.util.find_spec("torch_plugin_tt") is None:
            return False
        import torch_xla.runtime as xr

        if xr.device_type() != "TT":
            xr.set_device_type("TT")
        return xr.device_type() == "TT" and xr.addressable_device_count() > 0
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError):
        return False


pytestmark = [
    pytest.mark.tenstorrent,
    pytest.mark.skipif(
        not _tenstorrent_available(), reason="Tenstorrent PJRT runtime is unavailable"
    ),
]


def test_tenstorrent_matches_cpu_eager():
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

    assert compiled.selected_backend == "tenstorrent"
    assert compiled.target is not None
    assert compiled.target.vendor == "tenstorrent"
    assert actual.device.type == "xla"
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)


def test_tenstorrent_is_the_automatic_backend():
    compiled = lm7.compile(
        torch.nn.Linear(16, 4).eval(),
        target="tenstorrent",
        transfers="automatic",
        fallback="error",
    )
    compiled(torch.randn(8, 16))

    assert compiled.selected_backend == "tenstorrent"
