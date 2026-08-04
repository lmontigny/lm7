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


def _capability() -> int:
    major, minor = torch.cuda.get_device_capability()
    return int(f"{major}{minor}")


blackwell_only = pytest.mark.skipif(
    _capability() < 100 if torch.cuda.is_available() else True,
    reason="requires a Blackwell GPU (sm100 or newer)",
)


def test_detected_nvidia_gpu_reports_generation_and_precision():
    """Every CUDA host should name its silicon and say what it computes natively.

    Runs on any NVIDIA GPU, not only Blackwell, because the report is only
    trustworthy if it is right about the card the developer already has.
    """
    device = next(item for item in lm7.detection.detect_targets() if item.target.vendor == "nvidia")

    assert device.capabilities["generation"]
    precision = device.capabilities["precision"]
    assert precision["fp32"] == "native"
    # Whatever the card, every format has an answer and none of them is a guess.
    assert set(precision) == {"fp32", "fp16", "bf16", "int8", "fp8", "fp4"}
    assert set(precision.values()) <= {"native", "emulated", "absent"}


@blackwell_only
def test_blackwell_reports_native_fp4_and_names_itself():
    device = next(item for item in lm7.detection.detect_targets() if item.target.vendor == "nvidia")

    assert device.capabilities["generation"] == "Blackwell"
    assert device.capabilities["precision"]["fp4"] == "native"
    assert device.capabilities["precision"]["fp8"] == "native"
    assert device.capabilities["precision"]["bf16"] == "native"


@blackwell_only
def test_blackwell_resolves_and_compiles_without_a_special_case():
    """sm120 is above sm89 by the integer comparison every gate uses, so nothing
    in LM7 needed a Blackwell branch. This is the test that would fail if some
    future gate started matching architectures by name instead."""
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source).cuda()
    example_input = torch.randn(8, 16)
    expected = reference(example_input.cuda())

    compiled = lm7.compile(
        source,
        target="auto",
        backend="inductor",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(example_input)

    assert compiled.target is not None
    assert compiled.target.architecture is not None
    assert int(compiled.target.architecture.removeprefix("sm")) >= 100
    assert compiled.selected_backend == "inductor"
    torch.testing.assert_close(actual, expected)
