import torch

from lm7.backends import registry
from lm7.backends.base import CompileRequest
from lm7.planner import plan
from lm7.targets import TargetSpec


def request():
    return CompileRequest(
        torch.nn.Identity(), TargetSpec("cpu", "cpu"), "lazy", "automatic", "warn", {}
    )


def test_explicit_eager():
    backend, result = plan(request(), "eager", registry)
    assert backend.name == result.selected == "eager"


def test_auto_is_deterministic():
    first = plan(request(), "auto", registry)[1].selected
    second = plan(request(), "auto", registry)[1].selected
    assert first == second
