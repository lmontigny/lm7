from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from lm7.backends.base import CompileRequest
from lm7.backends.tensorrt import TensorRTBackend
from lm7.errors import CompilationError
from lm7.targets import TargetSpec

tensorrt_backend_module = importlib.import_module("lm7.backends.tensorrt")


def request(*, vendor: str = "nvidia", options=None) -> CompileRequest:
    return CompileRequest(
        torch.nn.Identity(),
        TargetSpec(vendor, "gpu" if vendor != "cpu" else "cpu"),
        "lazy",
        "automatic",
        "error",
        options or {},
    )


def test_probe_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setattr(tensorrt_backend_module.importlib.util, "find_spec", lambda name: None)

    info = TensorRTBackend().probe()

    assert not info.available
    assert ".[tensorrt]" in info.reason


def test_probe_rejects_rocm_runtime(monkeypatch):
    monkeypatch.setattr(
        tensorrt_backend_module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(),
    )
    monkeypatch.setattr(
        tensorrt_backend_module.importlib.metadata,
        "version",
        lambda name: "test-version",
    )
    monkeypatch.setattr(torch.version, "hip", "7.0-test")

    info = TensorRTBackend().probe()

    assert not info.available
    assert "ROCm" in info.reason


def test_support_is_nvidia_only(monkeypatch):
    backend = TensorRTBackend()
    monkeypatch.setattr(
        backend,
        "probe",
        lambda: SimpleNamespace(available=True, reason="available"),
    )

    assert backend.supports(request()).supported
    assert backend.supports(request(vendor="cpu")).reason == (
        "TensorRT supports NVIDIA GPU targets only."
    )
    assert backend.supports(request()).priority < 100


def test_compile_uses_registered_tensorrt_backend(monkeypatch):
    backend = TensorRTBackend()
    calls = {}

    monkeypatch.setattr(
        tensorrt_backend_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(__version__="test-version"),
    )
    monkeypatch.setattr(tensorrt_backend_module, "torch_device", lambda target: torch.device("cpu"))

    def fake_compile(model, **kwargs):
        calls.update(kwargs)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    artifact = backend.compile(
        request(options={"dynamic": True, "min_block_size": 2}),
        (torch.ones(2),),
        {},
    )

    assert calls == {
        "backend": "tensorrt",
        "dynamic": True,
        "options": {"min_block_size": 2},
    }
    assert artifact.metadata["torch_tensorrt_version"] == "test-version"
    torch.testing.assert_close(artifact.callable(torch.ones(2)), torch.ones(2))


def test_compile_wraps_lazy_backend_failure(monkeypatch):
    backend = TensorRTBackend()
    monkeypatch.setattr(
        tensorrt_backend_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(__version__="test-version"),
    )
    monkeypatch.setattr(tensorrt_backend_module, "torch_device", lambda target: torch.device("cpu"))

    def fail_on_call(model, **kwargs):
        def compiled(*args, **call_kwargs):
            raise RuntimeError("TensorRT build failed")

        return compiled

    monkeypatch.setattr(torch, "compile", fail_on_call)

    with pytest.raises(CompilationError, match="TensorRT build failed"):
        backend.compile(request(), (torch.ones(2),), {})
