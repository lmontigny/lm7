from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from lm7.backends.base import CompileRequest
from lm7.backends.openxla import OpenXLABackend
from lm7.errors import CompilationError, ConfigurationError
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


def _precision_fakes(monkeypatch, *, executed: int | None, calls: dict):
    """Install name-aware torch_xla fakes covering the matmul-precision modules.

    ``executed`` is what the ExecuteComputation counter reports: None before any
    XLA computation has run, an int afterwards.
    """
    torch_xla = SimpleNamespace(
        __version__="test-version",
        device=lambda index: torch.device("cpu"),
        sync=lambda **kwargs: calls.update(sync=kwargs),
    )
    modules = {
        "torch_xla": torch_xla,
        "torch_xla.debug.metrics": SimpleNamespace(counter_value=lambda name: executed),
        "torch_xla.backends": SimpleNamespace(
            set_mat_mul_precision=lambda value: calls.update(precision=value)
        ),
    }
    monkeypatch.setattr(
        openxla_backend_module.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    return calls


def test_compile_applies_mat_mul_precision(monkeypatch):
    calls: dict = {}
    _precision_fakes(monkeypatch, executed=None, calls=calls)

    artifact = OpenXLABackend().compile(
        request(options={"mat_mul_precision": "highest"}), (torch.ones(2),), {}
    )

    assert calls["precision"] == "highest"
    assert artifact.metadata["mat_mul_precision"] == "highest"


def test_compile_does_not_forward_mat_mul_precision_to_torch_compile(monkeypatch):
    calls: dict = {}
    _precision_fakes(monkeypatch, executed=None, calls=calls)
    monkeypatch.setattr(
        torch, "compile", lambda model, **kwargs: calls.update(compile=kwargs) or model
    )

    OpenXLABackend().compile(request(options={"mat_mul_precision": "high"}), (torch.ones(2),), {})

    # It is an LM7-level control, not something torch.compile understands.
    assert calls["compile"] == {"backend": "openxla"}


def test_compile_rejects_unknown_mat_mul_precision(monkeypatch):
    # ConfigurationError, not CompilationError: fallback="warn" must not answer
    # a typo by silently dropping the compiler.
    _precision_fakes(monkeypatch, executed=None, calls={})

    with pytest.raises(ConfigurationError, match="Unsupported mat_mul_precision"):
        OpenXLABackend().compile(
            request(options={"mat_mul_precision": "float32"}), (torch.ones(2),), {}
        )


def test_compile_refuses_mat_mul_precision_after_a_computation_ran(monkeypatch):
    # The setter would silently do nothing here, so LM7 has to raise instead.
    calls: dict = {}
    _precision_fakes(monkeypatch, executed=1, calls=calls)

    with pytest.raises(CompilationError, match="already"):
        OpenXLABackend().compile(
            request(options={"mat_mul_precision": "highest"}), (torch.ones(2),), {}
        )
    assert "precision" not in calls


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
