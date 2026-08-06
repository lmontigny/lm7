"""Quantization gate behaviour that needs neither a GPU nor a model download."""

from __future__ import annotations

import pytest
import torch

from lm7.errors import UnsupportedModelError
from lm7.huggingface import (
    DYNAMIC_ACTIVATION_QUANTIZATIONS,
    FP8,
    FP8_DYNAMIC,
    FP8_DYNAMIC_ROWWISE,
    INT8,
    NVFP4,
    NVFP4_DYNAMIC,
    VALIDATED_ACTIVATION,
    VALIDATED_WEIGHT_ONLY,
    WEIGHT_ONLY_QUANTIZATIONS,
    _apply_quantization,
    _validate_quantization,
    fp8_scale_granularity,
    normalize_quantization,
    nvfp4_dynamic_kernel,
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
    """The list names both families, so a reader sees that activation modes exist."""
    with pytest.raises(
        UnsupportedModelError,
        match="none, fp8, fp8-dynamic, fp8-dynamic-rowwise, int8, nvfp4, nvfp4-dynamic",
    ):
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


def test_dynamic_modes_are_separate_from_weight_only_ones():
    """`nvfp4` did not silently change meaning when `nvfp4-dynamic` was added.

    Adding activation quantization under the existing name would have changed what
    an existing command does, so the short names keep their weight-only meaning and
    the explicit long form resolves to the same thing.
    """
    assert normalize_quantization("nvfp4-weight-only") == NVFP4
    assert NVFP4 in WEIGHT_ONLY_QUANTIZATIONS
    assert NVFP4 not in DYNAMIC_ACTIVATION_QUANTIZATIONS
    assert DYNAMIC_ACTIVATION_QUANTIZATIONS == frozenset(
        {FP8_DYNAMIC, FP8_DYNAMIC_ROWWISE, NVFP4_DYNAMIC}
    )
    # The same rule applied again: adding per-row scaling under the existing
    # `fp8-dynamic` name would have changed what that command computes, so it is
    # a mode of its own and `fp8-dynamic` still means per-tensor.
    assert FP8 in WEIGHT_ONLY_QUANTIZATIONS
    assert FP8_DYNAMIC_ROWWISE not in WEIGHT_ONLY_QUANTIZATIONS


def test_nvfp4_dynamic_needs_blackwell_but_weight_only_does_not():
    """The two NVFP4 modes have different hardware floors, for a real reason.

    Weight-only NVFP4 unpacks to BF16 inside the kernel and never issues an FP4
    matmul, so it runs on anything Ampere or newer. The dynamic mode asks the
    tensor cores to multiply in FP4, which exists only on Blackwell.
    """
    ada = parse_target("nvidia:sm89")
    blackwell = parse_target("nvidia:sm120")

    _validate_quantization(NVFP4, ada, "inductor", "bfloat16", None)
    with pytest.raises(UnsupportedModelError, match="Blackwell"):
        _validate_quantization(NVFP4_DYNAMIC, ada, "inductor", "bfloat16", None)
    _validate_quantization(NVFP4_DYNAMIC, blackwell, "inductor", "bfloat16", None)


def test_fp8_dynamic_shares_the_ada_floor_with_fp8():
    with pytest.raises(UnsupportedModelError, match="Ada"):
        _validate_quantization(FP8_DYNAMIC, parse_target("nvidia:sm80"), "inductor", "auto", None)
    _validate_quantization(FP8_DYNAMIC, parse_target("nvidia:sm89"), "inductor", "auto", None)


def test_dynamic_modes_use_their_own_validated_table():
    """A model validated for weight-only FP8 is not thereby validated for FP8
    dynamic: the second changes what the matmul executes in, not just what is
    stored, so it has to earn its own entry."""
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert FP8 in VALIDATED_WEIGHT_ONLY[model_id]

    with pytest.raises(UnsupportedModelError, match="not validated"):
        _validate_quantization(
            FP8_DYNAMIC, parse_target("nvidia:sm120"), "inductor", "auto", model_id
        )


def test_dynamic_modes_convert_the_same_layers_as_their_weight_only_pair():
    """Otherwise the pair would not be comparable, and comparing them is the point."""
    from lm7.huggingface import _QUANTIZATION_FILTERS

    assert _QUANTIZATION_FILTERS[FP8_DYNAMIC] is _QUANTIZATION_FILTERS[FP8]
    assert _QUANTIZATION_FILTERS[NVFP4_DYNAMIC] is _QUANTIZATION_FILTERS[NVFP4]
    # Per-row scaling changes the scales, not the layer selection -- so the two
    # FP8 dynamic modes stay comparable to each other and to weight-only FP8.
    assert _QUANTIZATION_FILTERS[FP8_DYNAMIC_ROWWISE] is _QUANTIZATION_FILTERS[FP8]


def test_fp8_dynamic_rowwise_shares_the_ada_floor():
    """Per-row scaling is a scale layout, not new silicon: same sm89 floor as FP8."""
    with pytest.raises(UnsupportedModelError, match="Ada"):
        _validate_quantization(
            FP8_DYNAMIC_ROWWISE, parse_target("nvidia:sm80"), "inductor", "auto", None
        )
    _validate_quantization(
        FP8_DYNAMIC_ROWWISE, parse_target("nvidia:sm89"), "inductor", "auto", None
    )
    _validate_quantization(
        FP8_DYNAMIC_ROWWISE, parse_target("nvidia:sm90"), "inductor", "auto", None
    )


def test_fp8_granularity_is_reported_per_mode():
    """ "FP8 dynamic ran" and "it scaled per row" are different claims.

    The two modes differ only in scale granularity, so a benchmark that does not
    say which one it measured is not reproducible.
    """
    assert fp8_scale_granularity(FP8_DYNAMIC) == "per-tensor"
    assert fp8_scale_granularity(FP8_DYNAMIC_ROWWISE) == "per-row"
    assert fp8_scale_granularity(FP8) is None
    assert fp8_scale_granularity(INT8) is None


def test_fp8_dynamic_rowwise_requests_per_row_granularity():
    """The PerRow config reaches TorchAO, rather than silently defaulting.

    TorchAO's `granularity=None` resolves to PerTensor, so an omitted argument
    and an explicit per-tensor request are indistinguishable from the call site.
    This pins that rowwise actually passes PerRow.
    """
    from lm7.huggingface import _quantization_config

    class _Recorder:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        class PerRow:
            pass

        def Float8DynamicActivationFloat8WeightConfig(self, **kwargs: object) -> str:
            self.kwargs = kwargs
            return "config"

    recorder = _Recorder()
    assert _quantization_config(recorder, FP8_DYNAMIC) == "config"
    assert recorder.kwargs == {}

    recorder = _Recorder()
    assert _quantization_config(recorder, FP8_DYNAMIC_ROWWISE) == "config"
    assert isinstance(recorder.kwargs["granularity"], _Recorder.PerRow)


def test_nvfp4_kernel_is_reported_not_assumed():
    """ "NVFP4 dynamic ran" and "the fused Triton kernel ran" are different claims."""
    assert nvfp4_dynamic_kernel() in {"triton-mslk", "torch-fallback"}


def test_fp8_dynamic_is_admitted_for_llama_1b_and_nvfp4_dynamic_is_not():
    """Measured on a Blackwell sm120 against a BF16 baseline, Llama-3.2-1B.

    fp8-dynamic kept 4/4 top-1 at a maximum logit difference of 1.59 and came out
    at 0.97x baseline latency -- the first mode in this repo to be faster than not
    quantizing. nvfp4-dynamic scored 3/4 at 5.03 and ran 1.48x slower, so 4-bit
    activations on top of 4-bit weights is where this model stops holding its
    token. See docs/quantization.md.
    """
    admitted = VALIDATED_ACTIVATION["unsloth/Llama-3.2-1B-Instruct"]
    assert admitted == frozenset({FP8_DYNAMIC})
    assert NVFP4_DYNAMIC not in admitted

    blackwell = parse_target("nvidia:sm120")
    _validate_quantization(
        FP8_DYNAMIC, blackwell, "inductor", "auto", "unsloth/Llama-3.2-1B-Instruct"
    )
    with pytest.raises(UnsupportedModelError, match="not validated"):
        _validate_quantization(
            NVFP4_DYNAMIC, blackwell, "inductor", "auto", "unsloth/Llama-3.2-1B-Instruct"
        )
