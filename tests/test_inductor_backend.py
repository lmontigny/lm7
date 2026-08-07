from __future__ import annotations

import pytest
import torch

from lm7.backends.base import CompileRequest
from lm7.backends.inductor import InductorBackend, cudagraph_skips, cudagraphs_requested
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


def test_the_warmup_call_is_on_by_default_and_can_be_declined(monkeypatch):
    """Compiling by executing is the default, and is not free for every model.

    A graph that writes into a KV cache advances it once per execution, so an
    unasked-for warmup consumes a cache slot the caller never asked for — see
    src/lm7/generation.py. `warmup: False` is how such a caller opts out, and it
    must not reach torch.compile as an Inductor config key.
    """
    calls = install_fake_compile(monkeypatch)
    executed = []

    class Counting(torch.nn.Module):
        def forward(self, x):
            executed.append(x)
            return x

    warm = InductorBackend().compile(
        CompileRequest(
            Counting().eval(), TargetSpec("cpu", "cpu"), "lazy", "automatic", "error", {}
        ),
        (torch.ones(1),),
        {},
    )
    assert len(executed) == 1
    assert warm.metadata["warmup"] is True
    assert warm.metadata["cudagraph_skips"] == 0

    cold = InductorBackend().compile(
        CompileRequest(
            Counting().eval(),
            TargetSpec("cpu", "cpu"),
            "lazy",
            "automatic",
            "error",
            {"warmup": False},
        ),
        (torch.ones(1),),
        {},
    )
    assert len(executed) == 1, "warmup=False still executed the model"
    assert cold.metadata["warmup"] is False
    # Nothing has run, so neither answer is known and neither is claimed.
    assert cold.metadata["cudagraph_skips"] is None
    assert cold.metadata["cudagraphs_active"] is None
    assert calls["options"] is None


def test_warmup_can_be_declined_alongside_a_preset(monkeypatch):
    """`warmup` is a backend control, not a config key, so a preset still works."""
    calls = install_fake_compile(monkeypatch)

    artifact = InductorBackend().compile(
        request({"compile_mode": "reduce-overhead", "warmup": False}),
        (torch.ones(1),),
        {},
    )

    assert calls["mode"] == "reduce-overhead"
    assert calls["options"] is None
    assert artifact.metadata["cudagraphs"] is True
    assert artifact.metadata["cudagraphs_active"] is None


def test_cudagraph_request_is_read_from_the_preset_not_the_name():
    """Two of the four preset names say nothing about CUDA Graphs, and one of
    those two enables them. `reduce-overhead` and `max-autotune` set
    triton.cudagraphs; `default` and `max-autotune-no-cudagraphs` do not."""
    assert cudagraphs_requested("reduce-overhead", {}) is True
    assert cudagraphs_requested("max-autotune", {}) is True
    assert cudagraphs_requested("default", {}) is False
    assert cudagraphs_requested("max-autotune-no-cudagraphs", {}) is False
    assert cudagraphs_requested(None, {}) is False


def test_explicit_option_overrides_the_preset():
    """torch.compile lets an explicit option win, so the report has to agree."""
    assert cudagraphs_requested("max-autotune-no-cudagraphs", {"triton.cudagraphs": True}) is True
    assert cudagraphs_requested("reduce-overhead", {"triton.cudagraphs": False}) is False


def test_cudagraph_skip_counter_is_readable():
    """Requesting CUDA Graphs and getting them are different things; this is the
    counter that separates them. It must never raise, only report."""
    assert isinstance(cudagraph_skips(), int)
    assert cudagraph_skips() >= 0
