import pytest
import torch

import lm7
from lm7.errors import CompilationError


def test_compile_is_lazy_and_matches_eager():
    torch.manual_seed(0)
    original = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
    wrapped = lm7.compile(original, target="cpu", backend="eager")
    assert wrapped.state == "uncompiled"
    x = torch.randn(2, 4)
    expected = original(x)
    actual = wrapped(x)
    assert wrapped.state == "compiled"
    assert wrapped.selected_backend == "eager"
    torch.testing.assert_close(actual, expected)


def test_different_signatures_create_variants():
    wrapped = lm7.compile(torch.nn.Identity().eval(), target="cpu", backend="eager")
    wrapped(torch.randn(1, 2))
    wrapped(torch.randn(2, 2))
    assert len(wrapped._variants) == 2


def test_model_exceptions_are_not_swallowed():
    class Broken(torch.nn.Module):
        def forward(self, value):
            raise RuntimeError("model exploded")

    wrapped = lm7.compile(Broken().eval(), target="cpu", backend="eager")
    try:
        wrapped(torch.tensor(1))
    except RuntimeError as exc:
        assert str(exc) == "model exploded"
    else:
        raise AssertionError("Expected model exception")


def test_nested_inputs():
    class Nested(torch.nn.Module):
        def forward(self, values):
            return values["x"][0] + 1

    wrapped = lm7.compile(Nested().eval(), target="cpu", backend="eager")
    assert wrapped({"x": [torch.tensor(1)]}).item() == 2


def test_inductor_failure_warns_and_falls_back(monkeypatch):
    def fake_compile(*args, **kwargs):
        def fail(*call_args, **call_kwargs):
            raise RuntimeError("compiler unavailable")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)
    wrapped = lm7.compile(
        torch.nn.Identity().eval(), target="cpu", backend="inductor", fallback="warn"
    )
    with pytest.warns(RuntimeWarning, match="falling back"):
        result = wrapped(torch.tensor(3))
    assert result.item() == 3
    assert wrapped.selected_backend == "eager"


def test_inductor_failure_is_strict_when_requested(monkeypatch):
    def fake_compile(*args, **kwargs):
        def fail(*call_args, **call_kwargs):
            raise RuntimeError("compiler unavailable")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)
    wrapped = lm7.compile(
        torch.nn.Identity().eval(), target="cpu", backend="inductor", fallback="error"
    )
    with pytest.raises(CompilationError, match="compiler unavailable"):
        wrapped(torch.tensor(3))


def test_an_indexed_device_satisfies_an_unindexed_target() -> None:
    """`torch.device("mps") != torch.device("mps", 0)`, and `.to()` yields the latter.

    A plain `!=` therefore made `transfers="explicit"` impossible to satisfy on
    every indexed device: placing the input exactly where LM7 asked still
    raised. Checked directly because reproducing it needs an accelerator, while
    the comparison it turns on does not.
    """
    from lm7.module import _same_device

    assert _same_device(torch.device("mps", 0), torch.device("mps"))
    assert _same_device(torch.device("cuda", 0), torch.device("cuda"))
    assert _same_device(torch.device("cpu"), torch.device("cpu"))
    assert not _same_device(torch.device("cuda", 1), torch.device("cuda"))
    assert not _same_device(torch.device("cpu"), torch.device("mps"))
