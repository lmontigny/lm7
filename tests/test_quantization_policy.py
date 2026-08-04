"""Quantization gate behaviour that needs neither a GPU nor a model download."""

from __future__ import annotations

import pytest
import torch

from lm7.errors import UnsupportedModelError
from lm7.huggingface import (
    FP8,
    INT8,
    NVFP4,
    VALIDATED_WEIGHT_ONLY,
    _apply_quantization,
    _validate_quantization,
    normalize_quantization,
)
from lm7.targets import parse_target


def validate(model_id: str, quantization: str) -> None:
    _validate_quantization(
        quantization,
        parse_target("nvidia:sm89"),
        "inductor",
        "bfloat16",
        model_id,
    )


def test_validated_models_are_accepted():
    for model_id, modes in VALIDATED_WEIGHT_ONLY.items():
        for quantization in modes:
            validate(model_id, quantization)


def test_unvalidated_model_is_rejected():
    with pytest.raises(UnsupportedModelError, match="not validated"):
        validate("some/unknown-model", INT8)


def test_validation_is_per_mode_not_per_model():
    """A model can be safe in one mode and not the other, so the gate is keyed on
    the pair. LFM2.5-230M is the case that motivated this: it holds its top-1
    token under FP8 but diverges completely under INT8."""
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    original = VALIDATED_WEIGHT_ONLY[model_id]
    VALIDATED_WEIGHT_ONLY[model_id] = frozenset({FP8})
    try:
        validate(model_id, FP8)
        with pytest.raises(UnsupportedModelError, match="not validated"):
            validate(model_id, INT8)
    finally:
        VALIDATED_WEIGHT_ONLY[model_id] = original


def test_llama_8b_is_admitted_for_int8_only():
    """Only INT8 is claimed for the 8B pair, now on evidence from both targets.

    This entry used to rest on CPU measurements alone, because no GPU here could
    hold the model. A Blackwell sm120 can, and INT8 passed there at 4/4 top-1
    (maximum logit difference 0.39), so the NVIDIA half is measured rather than
    assumed. The same run rejected FP8 (3/4) and NVFP4 (2/4) on that model, which
    is why widening this set would be wrong even though the hardware now exists
    to try. See docs/quantization.md.
    """
    admitted = VALIDATED_WEIGHT_ONLY["unsloth/Llama-3.1-8B-Instruct"]
    assert admitted == frozenset({INT8})
    assert FP8 not in admitted
    assert NVFP4 not in admitted


def test_long_form_names_normalize_to_short_names():
    """`--quantization int8-weight-only` predates `--quantize int8`; both work."""
    assert normalize_quantization("int8-weight-only") == INT8
    assert normalize_quantization("fp8-weight-only") == FP8
    assert normalize_quantization(INT8) == INT8
    assert normalize_quantization(NVFP4) == NVFP4
    assert normalize_quantization("none") == "none"


def test_unknown_quantization_lists_the_short_names():
    with pytest.raises(UnsupportedModelError, match="none, fp8, int8, nvfp4"):
        validate("HuggingFaceTB/SmolLM2-135M-Instruct", "int4")


def test_quantization_that_matches_no_layer_raises():
    """torchao quietly does nothing when the filter matches no module. The FP8
    filter selects only ``.mlp.`` linears, so a model without that naming would
    otherwise report a successful quantization that changed nothing — which is
    what LFM2.5-230M did, at 1.00x storage reduction and identical logits."""
    pytest.importorskip("torchao")
    model = torch.nn.Sequential(torch.nn.Linear(8, 8)).eval()

    with pytest.raises(UnsupportedModelError, match="matched no quantizable layers"):
        _apply_quantization(model, parse_target("cpu"), FP8)


def test_int8_filter_matches_plain_linears():
    """The INT8 filter takes every linear but lm_head, so the same model that has
    no FP8-eligible layers still quantizes under INT8."""
    pytest.importorskip("torchao")
    model = torch.nn.Sequential(torch.nn.Linear(8, 8)).eval()

    _, converted = _apply_quantization(model, parse_target("cpu"), INT8)
    assert converted == 1


def test_nvfp4_skips_layers_whose_dims_are_not_multiples_of_16():
    """NVFP4 stores a scale per 16-element block, so torchao raises on any other
    shape. LM7 filters those layers out rather than failing the whole model."""
    pytest.importorskip("torchao.prototype.mx_formats")
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 32),  # eligible
        torch.nn.Linear(30, 32),  # in_features not a multiple of 16
        torch.nn.Linear(32, 20),  # out_features not a multiple of 16
    ).eval()

    _, converted = _apply_quantization(model, parse_target("cpu"), NVFP4)
    assert converted == 1


def test_nvfp4_raises_when_every_layer_is_ineligible():
    pytest.importorskip("torchao.prototype.mx_formats")
    model = torch.nn.Sequential(torch.nn.Linear(30, 20)).eval()

    with pytest.raises(UnsupportedModelError, match="matched no quantizable layers"):
        _apply_quantization(model, parse_target("cpu"), NVFP4)


def test_nvfp4_is_validated_far_more_narrowly_than_int8():
    """4-bit weights cost more accuracy than 8-bit at these model sizes. Only
    Llama-3.2-1B held its top-1 token on all four prompts; SmolLM2-135M managed
    2/4 and LFM2.5-230M 3/4, so neither is admitted. See docs/quantization.md."""
    assert NVFP4 in VALIDATED_WEIGHT_ONLY["unsloth/Llama-3.2-1B-Instruct"]
    assert NVFP4 not in VALIDATED_WEIGHT_ONLY["HuggingFaceTB/SmolLM2-135M-Instruct"]

    with pytest.raises(UnsupportedModelError, match="not validated"):
        validate("HuggingFaceTB/SmolLM2-135M-Instruct", NVFP4)
