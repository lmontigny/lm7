from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from lm7.backends import registry
from lm7.backends.base import CompileRequest
from lm7.backends.eager import EagerBackend
from lm7.backends.inductor import InductorBackend
from lm7.backends.openvino import OpenVINOBackend
from lm7.errors import BackendUnavailableError
from lm7.planner import plan
from lm7.targets import parse_target

inductor_module = importlib.import_module("lm7.backends.inductor")
openvino_module = importlib.import_module("lm7.backends.openvino")


def request_for(target: str = "intel:gpu") -> CompileRequest:
    return CompileRequest(
        torch.nn.Linear(4, 4).eval(),
        parse_target(target),
        "lazy",
        "automatic",
        "error",
        {},
    )


def _triton_registers(monkeypatch, names: frozenset[str] | None) -> None:
    """Pretend Triton is installed and generates for `names`."""
    monkeypatch.setattr(inductor_module.importlib.util, "find_spec", lambda name: SimpleNamespace())
    monkeypatch.setattr(inductor_module, "triton_backends", lambda: names)


def _openvino_installed(monkeypatch) -> None:
    monkeypatch.setattr(openvino_module.importlib.util, "find_spec", lambda name: SimpleNamespace())
    monkeypatch.setattr(openvino_module.importlib.metadata, "version", lambda name: "2026.2.1-test")


def test_inductor_declines_an_intel_gpu_without_the_xpu_triton_backend(monkeypatch):
    _triton_registers(monkeypatch, frozenset({"nvidia", "amd"}))

    support = InductorBackend().supports(request_for("intel:gpu"))

    assert not support.supported
    assert "pytorch-triton-xpu" in support.reason
    # The reason has to name what *is* installed, or the user cannot tell this
    # apart from having no Triton at all.
    assert "amd, nvidia" in support.reason


def test_inductor_supports_an_intel_gpu_once_the_xpu_backend_is_present(monkeypatch):
    _triton_registers(monkeypatch, frozenset({"intel", "nvidia"}))

    assert InductorBackend().supports(request_for("intel:gpu")).supported


def test_inductor_declines_an_intel_gpu_with_no_triton_at_all(monkeypatch):
    monkeypatch.setattr(inductor_module.importlib.util, "find_spec", lambda name: None)

    support = InductorBackend().supports(request_for("intel:gpu"))

    assert not support.supported
    assert "no triton package is installed" in support.reason


def test_an_unreadable_triton_registry_does_not_block_the_intel_gpu(monkeypatch):
    """Fail open: "cannot tell" must not be reported as a missing backend."""
    _triton_registers(monkeypatch, None)

    assert InductorBackend().supports(request_for("intel:gpu")).supported


def test_the_xpu_preflight_does_not_touch_other_targets(monkeypatch):
    _triton_registers(monkeypatch, frozenset({"nvidia"}))
    backend = InductorBackend()

    assert backend.supports(request_for("cpu")).supported
    assert backend.supports(request_for("nvidia")).supported
    assert backend.supports(request_for("apple")).supported


def test_openvino_declines_an_intel_gpu_rather_than_using_the_cpu_plugin(monkeypatch):
    _openvino_installed(monkeypatch)

    support = OpenVINOBackend().supports(request_for("intel:gpu"))

    assert not support.supported
    assert "CPU plugin" in support.reason
    # The CPU and NPU targets it does serve stay supported.
    assert OpenVINOBackend().supports(request_for("cpu")).supported
    assert OpenVINOBackend().supports(request_for("intel:npu")).supported


def test_an_intel_gpu_without_xpu_triton_plans_eager_not_a_cpu_backend(monkeypatch):
    """The regression the openvino guard exists for.

    Declining in `inductor` alone would hand the target to `openvino` at
    priority 80, which maps every non-NPU target to the CPU plugin -- turning a
    slow-but-correct eager XPU run into a silent CPU one.
    """
    _triton_registers(monkeypatch, frozenset({"nvidia"}))
    _openvino_installed(monkeypatch)

    _, selected = plan(request_for("intel:gpu"), "auto", registry)

    assert selected.selected == "eager"
    assert EagerBackend().supports(request_for("intel:gpu")).supported


def test_export_rejects_openvino_artifacts_for_an_intel_gpu(tmp_path):
    from lm7 import export

    with pytest.raises(BackendUnavailableError, match="OpenVINO GPU plugin"):
        export(
            torch.nn.Linear(4, 4).eval(),
            args=(torch.randn(2, 4),),
            target="intel:gpu",
            output=tmp_path / "gpu.lm7",
            backend="openvino",
        )


def test_doctor_reports_which_vendors_triton_generates_for(monkeypatch):
    _triton_registers(monkeypatch, frozenset({"nvidia"}))
    monkeypatch.setattr(inductor_module, "_triton_version", lambda: "3.7.1-test")

    assert "Triton 3.7.1-test generates for nvidia." in InductorBackend().probe().reason


def test_doctor_says_so_when_triton_is_missing(monkeypatch):
    monkeypatch.setattr(inductor_module, "_triton_version", lambda: None)

    assert "Triton is not installed" in InductorBackend().probe().reason
