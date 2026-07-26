from __future__ import annotations

import copy

import pytest
import torch

import lm7

pytestmark = [
    pytest.mark.rocm,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not getattr(torch.version, "hip", None),
        reason="ROCm GPU is unavailable",
    ),
]


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()


def test_amd_inductor_matches_eager_with_automatic_transfers():
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source).cuda()
    example_input = torch.randn(8, 16)
    expected = reference(example_input.cuda())
    architecture = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]

    compiled = lm7.compile(
        source,
        target=f"amd:{architecture}",
        backend="inductor",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(example_input)

    assert compiled.selected_backend == "inductor"
    assert compiled.target is not None
    assert compiled.target.vendor == "amd"
    assert compiled.target.architecture == architecture
    assert next(compiled.model.parameters()).device.type == "cuda"
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, expected)
