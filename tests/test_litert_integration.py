from __future__ import annotations

import copy
import importlib.util

import pytest
import torch

import lm7


def _litert_stack_installed() -> bool:
    try:
        installed = all(
            importlib.util.find_spec(module) is not None
            for module in ("litert_torch", "ai_edge_litert")
        )
    except ModuleNotFoundError:
        return False
    release = torch.__version__.split("+", 1)[0]
    major, minor, *_ = release.split(".")
    return installed and (2, 4) <= (int(major), int(minor)) < (2, 13)


pytestmark = [
    pytest.mark.litert,
    pytest.mark.skipif(
        not _litert_stack_installed(),
        reason='install LM7 with ".[litert]" in a PyTorch >=2.4,<2.13 environment',
    ),
]


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    ).eval()


def test_cpu_artifact_round_trips(tmp_path):
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source)
    example = torch.randn(2, 4)
    expected = reference(example)

    artifact = lm7.export(
        source,
        args=(example,),
        target="cpu",
        backend="litert",
        output=tmp_path / "model.lm7",
    )
    reloaded = lm7.load_artifact(artifact.path)
    actual = reloaded(example)

    assert artifact.manifest.compiled_file == "compiled_model.tflite"
    assert (artifact.path / "compiled_model.tflite").stat().st_size > 0
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_static_kwargs_and_tuple_outputs_round_trip(tmp_path):
    class KeywordModel(torch.nn.Module):
        def forward(
            self, x: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return x * scale, x + scale

    source = KeywordModel().eval()
    x = torch.randn(2, 4)
    scale = torch.tensor(2.0)
    artifact = lm7.export(
        source,
        args=(),
        kwargs={"x": x, "scale": scale},
        target="cpu",
        backend="litert",
        output=tmp_path / "kwargs.lm7",
    )

    first, second = artifact(scale=scale, x=x)

    torch.testing.assert_close(first, x * scale)
    torch.testing.assert_close(second, x + scale)


def test_lightweight_conversion_option_round_trips(tmp_path):
    artifact = lm7.export(
        model(),
        args=(torch.randn(2, 4),),
        target="cpu",
        backend="litert",
        output=tmp_path / "lightweight.lm7",
        options={"lightweight_conversion": True},
    )

    assert artifact.manifest.runtime_requirements["lightweight_conversion"] is True
    assert artifact(torch.randn(2, 4)).shape == (2, 3)
