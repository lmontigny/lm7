"""Quantization gate behaviour that needs neither a GPU nor a model download."""

from __future__ import annotations

import pytest
import torch

from lm7.errors import UnsupportedModelError
from lm7.huggingface import (
    FP8_WEIGHT_ONLY,
    INT8_WEIGHT_ONLY,
    VALIDATED_WEIGHT_ONLY,
    _apply_quantization,
    _validate_quantization,
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
        validate("some/unknown-model", INT8_WEIGHT_ONLY)


def test_validation_is_per_mode_not_per_model():
    """A model can be safe in one mode and not the other, so the gate is keyed on
    the pair. LFM2.5-230M is the case that motivated this: it holds its top-1
    token under FP8 but diverges completely under INT8."""
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    original = VALIDATED_WEIGHT_ONLY[model_id]
    VALIDATED_WEIGHT_ONLY[model_id] = frozenset({FP8_WEIGHT_ONLY})
    try:
        validate(model_id, FP8_WEIGHT_ONLY)
        with pytest.raises(UnsupportedModelError, match="not validated"):
            validate(model_id, INT8_WEIGHT_ONLY)
    finally:
        VALIDATED_WEIGHT_ONLY[model_id] = original


def test_quantization_that_matches_no_layer_raises():
    """torchao quietly does nothing when the filter matches no module. The FP8
    filter selects only ``.mlp.`` linears, so a model without that naming would
    otherwise report a successful quantization that changed nothing — which is
    what LFM2.5-230M did, at 1.00x storage reduction and identical logits."""
    pytest.importorskip("torchao")
    model = torch.nn.Sequential(torch.nn.Linear(8, 8)).eval()

    with pytest.raises(UnsupportedModelError, match="matched no quantizable layers"):
        _apply_quantization(model, parse_target("cpu"), FP8_WEIGHT_ONLY)


def test_int8_filter_matches_plain_linears():
    """The INT8 filter takes every linear but lm_head, so the same model that has
    no FP8-eligible layers still quantizes under INT8."""
    pytest.importorskip("torchao")
    model = torch.nn.Sequential(torch.nn.Linear(8, 8)).eval()

    _apply_quantization(model, parse_target("cpu"), INT8_WEIGHT_ONLY)
