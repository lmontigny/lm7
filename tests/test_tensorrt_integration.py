from __future__ import annotations

import copy
import importlib.util

import pytest
import torch

import lm7

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.tensorrt,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable"),
    pytest.mark.skipif(
        importlib.util.find_spec("torch_tensorrt") is None,
        reason="Torch-TensorRT is not installed",
    ),
]


def test_nvidia_tensorrt_matches_eager():
    torch.manual_seed(0)
    source = torch.nn.Sequential(
        torch.nn.Linear(32, 128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 8),
    ).eval()
    reference = copy.deepcopy(source).cuda().half()
    source.half()
    example_input = torch.randn(4, 32, dtype=torch.float16)
    expected = reference(example_input.cuda())

    compiled = lm7.compile(
        source,
        target="nvidia",
        backend="tensorrt",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(example_input)

    assert compiled.selected_backend == "tensorrt"
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
