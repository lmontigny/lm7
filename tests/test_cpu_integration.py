from __future__ import annotations

import copy

import pytest
import torch

import lm7

pytestmark = pytest.mark.cpu


def test_cpu_inductor_matches_eager():
    torch.manual_seed(0)
    source = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()
    reference = copy.deepcopy(source)
    example_input = torch.randn(8, 16)
    expected = reference(example_input)

    compiled = lm7.compile(
        source,
        target="cpu",
        backend="inductor",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(example_input)

    assert compiled.selected_backend == "inductor"
    assert compiled.target is not None
    assert compiled.target.vendor == "cpu"
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected)
