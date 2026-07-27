from __future__ import annotations

import copy

import pytest
import torch

import lm7

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="Apple Silicon MPS GPU is unavailable"
    ),
]


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()


def test_apple_inductor_matches_eager_with_automatic_transfers():
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source).to("mps")
    example_input = torch.randn(8, 16)
    expected = reference(example_input.to("mps")).cpu()

    compiled = lm7.compile(
        source,
        target="apple",
        backend="inductor",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(example_input).cpu()

    assert compiled.selected_backend == "inductor"
    assert compiled.target is not None
    assert compiled.target.vendor == "apple"
    assert compiled.target.architecture == "metal"
    assert next(compiled.model.parameters()).device.type == "mps"
    torch.testing.assert_close(actual, expected)


def test_apple_aot_inductor_compile_matches_eager():
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source).to("mps")
    example_input = torch.randn(8, 16)
    expected = reference(example_input.to("mps")).cpu()

    compiled = lm7.compile(
        source,
        target="apple",
        backend="aot_inductor",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(example_input).cpu()

    assert compiled.selected_backend == "aot_inductor"
    assert compiled.target is not None
    assert compiled.target.vendor == "apple"
    torch.testing.assert_close(actual, expected)


def test_apple_aot_inductor_export_and_reload_round_trip(tmp_path):
    torch.manual_seed(0)
    source = model()
    example_input = torch.randn(8, 16)
    expected = source(example_input)

    output = tmp_path / "model.lm7"
    artifact = lm7.export(
        source,
        args=(example_input,),
        target="apple",
        backend="aot_inductor",
        output=output,
    )
    assert artifact.manifest.backend == "aot_inductor"
    assert artifact.manifest.target["vendor"] == "apple"

    loaded = lm7.load_artifact(output)
    actual = loaded(example_input.to("mps")).cpu()
    torch.testing.assert_close(actual, expected)
