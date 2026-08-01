from __future__ import annotations

import pytest
import torch

from lm7.backends.base import CompileRequest
from lm7.backends.inductor import InductorBackend
from lm7.errors import CompilationError
from lm7.targets import TargetSpec


def request(options=None) -> CompileRequest:
    return CompileRequest(
        torch.nn.Identity().eval(),
        TargetSpec("cpu", "cpu"),
        "lazy",
        "automatic",
        "error",
        options or {},
    )


def install_fake_compile(monkeypatch):
    calls = {}

    def fake_compile(model, **kwargs):
        calls.update(kwargs)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    return calls


@pytest.mark.parametrize(
    "mode",
    ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
)
def test_compile_mode_is_forwarded_to_torch_compile(monkeypatch, mode):
    calls = install_fake_compile(monkeypatch)

    InductorBackend().compile(
        request({"compile_mode": mode, "dynamic": False, "fullgraph": True}),
        (torch.ones(1),),
        {},
    )

    assert calls == {
        "backend": "inductor",
        "mode": mode,
        "dynamic": False,
        "fullgraph": True,
        "options": None,
    }


def test_backend_options_are_forwarded_to_torch_compile(monkeypatch):
    calls = install_fake_compile(monkeypatch)

    InductorBackend().compile(
        request(
            {
                "max_autotune": True,
                "triton.cudagraphs": False,
                "shape_padding": True,
            }
        ),
        (torch.ones(1),),
        {},
    )

    assert calls["mode"] is None
    assert calls["options"] == {
        "max_autotune": True,
        "triton.cudagraphs": False,
        "shape_padding": True,
    }


def test_compile_mode_rejects_backend_options(monkeypatch):
    calls = install_fake_compile(monkeypatch)

    with pytest.raises(CompilationError, match="cannot be combined"):
        InductorBackend().compile(
            request({"compile_mode": "max-autotune", "shape_padding": True}),
            (torch.ones(1),),
            {},
        )

    assert calls == {}
