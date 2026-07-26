from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from lm7.backends.base import CompileRequest
from lm7.backends.openxla import OpenXLABackend
from lm7.errors import CompilationError
from lm7.targets import TargetSpec

openxla_backend_module = importlib.import_module("lm7.backends.openxla")


def request(*, vendor: str = "tpu", options=None) -> CompileRequest:
    return CompileRequest(
        torch.nn.Identity(),
        TargetSpec(vendor, "accelerator" if vendor == "tpu" else "cpu"),
        "lazy",
        "automatic",
        "error",
        options or {},
    )


def test_probe_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setattr(openxla_backend_module.importlib.util, "find_spec", lambda name: None)

    info = OpenXLABackend().probe()

    assert not info.available
    assert ".[openxla]" in info.reason


def test_probe_requires_tpu_pjrt_runtime(monkeypatch):
    runtime = SimpleNamespace(device_type=lambda: "CPU")
    monkeypatch.setattr(
        openxla_backend_module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(),
    )
    monkeypatch.setattr(
        openxla_backend_module.importlib.metadata,
        "version",
        lambda name: "test-version",
    )
    monkeypatch.setattr(
        openxla_backend_module.importlib,
        "import_module",
        lambda name: runtime,
    )

    info = OpenXLABackend().probe()

    assert not info.available
    assert "CPU, not TPU" in info.reason


def test_support_is_tpu_only(monkeypatch):
    backend = OpenXLABackend()
    monkeypatch.setattr(
        backend,
        "probe",
        lambda: SimpleNamespace(available=True, reason="available"),
    )

    assert backend.supports(request()).supported
    assert backend.supports(request(vendor="cpu")).reason == (
        "OpenXLA supports Google TPU targets only in LM7."
    )


def test_compile_uses_registered_openxla_backend(monkeypatch):
    backend = OpenXLABackend()
    calls = {}
    torch_xla = SimpleNamespace(
        __version__="test-version",
        device=lambda index: torch.device("cpu"),
        sync=lambda **kwargs: calls.update(sync=kwargs),
    )
    monkeypatch.setattr(
        openxla_backend_module.importlib,
        "import_module",
        lambda name: torch_xla,
    )

    def fake_compile(model, **kwargs):
        calls.update(compile=kwargs)

        def compiled(value):
            calls["grad_enabled"] = torch.is_grad_enabled()
            calls["inference_mode"] = torch.is_inference_mode_enabled()
            return model(value)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    artifact = backend.compile(
        request(options={"dynamic": False, "partitioner": "test"}),
        (torch.ones(2),),
        {},
    )

    assert calls == {
        "compile": {
            "backend": "openxla",
            "dynamic": False,
            "options": {"partitioner": "test"},
        },
        "sync": {"wait": True},
        "grad_enabled": False,
        "inference_mode": False,
    }
    assert artifact.metadata["torch_xla_version"] == "test-version"


def test_compile_wraps_backend_failure(monkeypatch):
    backend = OpenXLABackend()
    torch_xla = SimpleNamespace(
        __version__="test-version",
        device=lambda index: torch.device("cpu"),
        sync=lambda **kwargs: None,
    )
    monkeypatch.setattr(
        openxla_backend_module.importlib,
        "import_module",
        lambda name: torch_xla,
    )
    monkeypatch.setattr(
        torch,
        "compile",
        lambda model, **kwargs: (
            lambda *args, **call_kwargs: (_ for _ in ()).throw(RuntimeError("OpenXLA build failed"))
        ),
    )

    with pytest.raises(CompilationError, match="OpenXLA build failed"):
        backend.compile(request(), (torch.ones(2),), {})
