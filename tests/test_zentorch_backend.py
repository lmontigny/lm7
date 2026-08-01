from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from lm7.backends import registry
from lm7.backends.base import CompileRequest
from lm7.backends.zentorch import ZENTORCH_PRIORITY, ZenTorchBackend
from lm7.errors import CompilationError
from lm7.targets import parse_target

zentorch_module = importlib.import_module("lm7.backends.zentorch")


def model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()


def request_for(target: str = "cpu", options=None) -> CompileRequest:
    return CompileRequest(
        model=model(),
        target=parse_target(target),
        mode="lazy",
        transfers="automatic",
        fallback="error",
        options=options or {},
    )


def patch_find_spec(monkeypatch, *, present: bool) -> None:
    """Answer only for the zentorch namespace.

    Replacing importlib.util.find_spec outright lies to every lazy import in the
    process, torch's own included, which breaks unrelated machinery inside
    compile().
    """
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "zentorch" or name.startswith("zentorch."):
            return SimpleNamespace() if present else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(zentorch_module.importlib.util, "find_spec", fake_find_spec)


def test_backend_is_registered():
    assert "zentorch" in {backend.name for backend in registry.all()}


def test_probe_reports_the_install_extra_when_absent(monkeypatch):
    patch_find_spec(monkeypatch, present=False)

    info = ZenTorchBackend().probe()

    assert not info.available
    assert ".[zentorch]" in info.reason


def test_probe_requires_torch_compile(monkeypatch):
    patch_find_spec(monkeypatch, present=True)
    monkeypatch.setattr(torch, "compile", None)

    info = ZenTorchBackend().probe()

    assert not info.available
    assert "torch.compile" in info.reason


def test_gpu_targets_are_declined(monkeypatch):
    """`amd` in an LM7 target is the ROCm GPU, which shares nothing with the
    ZenDNN CPU extension. Claiming it would silently compile for the CPU."""
    patch_find_spec(monkeypatch, present=True)

    for target in ("amd", "nvidia", "apple"):
        support = ZenTorchBackend().supports(request_for(target))
        assert not support.supported
        assert "cpu targets only" in support.reason


def test_cpu_is_supported_below_inductor(monkeypatch):
    patch_find_spec(monkeypatch, present=True)

    support = ZenTorchBackend().supports(request_for("cpu"))

    assert support.supported
    assert support.priority == ZENTORCH_PRIORITY


def test_auto_never_selects_zentorch_over_inductor(monkeypatch):
    """The whole point of the priority: on the hardware LM7 has measured,
    Inductor was faster, so zentorch has to be asked for by name."""
    from lm7.backends.base import CompileRequest as Request
    from lm7.planner import plan

    patch_find_spec(monkeypatch, present=True)
    request = Request(
        model=model(),
        target=parse_target("cpu"),
        mode="lazy",
        transfers="automatic",
        fallback="error",
        options={},
    )

    _, selected = plan(request, "auto", registry)

    assert selected.selected == "inductor"
    assert any(
        candidate.backend == "zentorch" and candidate.support.supported
        for candidate in selected.candidates
    )


def test_inductor_options_are_refused_rather_than_ignored(monkeypatch):
    """torch.compile forwards `options` to the chosen backend, so Inductor's
    config keys would be silently dropped by another one."""
    patch_find_spec(monkeypatch, present=True)

    with pytest.raises(CompilationError, match="dynamic"):
        ZenTorchBackend().compile(
            request_for("cpu", options={"max_autotune": True}), (torch.randn(2, 4),), {}
        )
