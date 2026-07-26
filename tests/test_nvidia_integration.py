from __future__ import annotations

import copy

import pytest
import torch

import lm7
from lm7.errors import InputDeviceError

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable"),
]


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    ).eval()


def test_nvidia_inductor_matches_eager_with_automatic_transfers():
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source).cuda()
    example_input = torch.randn(8, 16)
    expected = reference(example_input.cuda())
    major, minor = torch.cuda.get_device_capability()

    compiled = lm7.compile(
        source,
        target=f"nvidia:sm{major}{minor}",
        backend="inductor",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(example_input)

    assert compiled.selected_backend == "inductor"
    assert compiled.target is not None
    assert compiled.target.vendor == "nvidia"
    assert compiled.target.architecture == f"sm{major}{minor}"
    assert next(compiled.model.parameters()).device.type == "cuda"
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, expected)


def test_nvidia_explicit_transfers_reject_cpu_inputs():
    compiled = lm7.compile(
        model().cuda(),
        target="nvidia",
        backend="eager",
        transfers="explicit",
        fallback="error",
    )

    with pytest.raises(InputDeviceError, match="Move inputs explicitly"):
        compiled(torch.randn(8, 16))
