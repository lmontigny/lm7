from __future__ import annotations

import importlib.util

import pytest
import torch

import lm7


def _zentorch_available() -> bool:
    try:
        if importlib.util.find_spec("zentorch") is None:
            return False
        importlib.import_module("zentorch")
        return True
    except (ImportError, AttributeError, ValueError, OSError):
        return False


pytestmark = [
    pytest.mark.zentorch,
    pytest.mark.skipif(not _zentorch_available(), reason="zentorch is unavailable"),
]


def model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()


def test_compiled_output_matches_eager():
    source = model()
    example = torch.randn(8, 16)
    with torch.no_grad():
        expected = source(example)

    compiled = lm7.compile(source, target="cpu", backend="zentorch", fallback="error")

    torch.testing.assert_close(compiled(example), expected, rtol=1e-4, atol=1e-4)
    assert compiled.selected_backend == "zentorch"


def test_backend_reports_available():
    info = next(item for item in lm7.backends() if item["name"] == "zentorch")

    assert info["available"]
    assert info["version"]
