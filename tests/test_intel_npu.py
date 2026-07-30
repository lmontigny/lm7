from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from lm7.backends import registry
from lm7.backends.base import CompileRequest
from lm7.backends.eager import EagerBackend
from lm7.backends.inductor import InductorBackend
from lm7.backends.openvino import (
    OpenVINOBackend,
    _resolve_inference_precision,
    device_for_target,
)
from lm7.errors import BackendUnavailableError, CompilationError, TargetNotFoundError
from lm7.planner import plan
from lm7.targets import TargetSpec, parse_target

openvino_module = importlib.import_module("lm7.backends.openvino")

requires_openvino = pytest.mark.skipif(
    importlib.util.find_spec("openvino") is None,
    reason="OpenVINO is not installed",
)


def request_for(target: str = "intel:npu", options=None) -> CompileRequest:
    return CompileRequest(
        torch.nn.Linear(4, 4).eval(),
        parse_target(target),
        "lazy",
        "automatic",
        "error",
        options or {},
    )


def _install_openvino(monkeypatch) -> None:
    """Make the backend probe report OpenVINO without requiring the package."""
    monkeypatch.setattr(openvino_module.importlib.util, "find_spec", lambda name: SimpleNamespace())
    monkeypatch.setattr(openvino_module.importlib.metadata, "version", lambda name: "2026.2.1-test")


def test_npu_target_round_trips_through_its_string():
    target = parse_target("intel:npu")

    assert (target.vendor, target.kind, target.architecture, target.model) == (
        "intel",
        "npu",
        None,
        None,
    )
    # "intel" alone means the GPU, so the NPU spec must not print as its vendor.
    assert str(target) == "intel:npu"
    assert parse_target(str(target)) == target


def test_device_for_target_maps_only_the_npu_off_the_cpu_plugin():
    assert device_for_target(parse_target("intel:npu")) == "NPU"
    assert device_for_target(parse_target("cpu")) == "CPU"
    # Unevaluated: the Intel GPU still runs on the CPU plugin.
    assert device_for_target(parse_target("intel:gpu")) == "CPU"


def test_inference_precision_defaults_to_fp32_on_cpu_but_is_unset_on_npu():
    assert _resolve_inference_precision("auto", "CPU") == "f32"
    # The NPU plugin executes in FP16 and has no FP32 mode to pin.
    assert _resolve_inference_precision("auto", "NPU") is None
    assert _resolve_inference_precision("auto", "NPU.1") is None
    # An explicit request always wins over the per-device default.
    assert _resolve_inference_precision("f16", "NPU") == "f16"
    assert _resolve_inference_precision(None, "CPU") is None


def test_openvino_supports_the_npu_target(monkeypatch):
    _install_openvino(monkeypatch)

    support = OpenVINOBackend().supports(request_for("intel:npu"))

    assert support.supported
    assert "NPU" in support.reason


def test_inductor_declines_the_npu_target():
    support = InductorBackend().supports(request_for("intel:npu"))

    assert not support.supported
    assert "openvino" in support.reason


def test_eager_declines_the_npu_target():
    support = EagerBackend().supports(request_for("intel:npu"))

    assert not support.supported
    assert "openvino" in support.reason


def test_eager_still_supports_the_intel_gpu_target():
    assert EagerBackend().supports(request_for("intel:gpu")).supported


@requires_openvino
def test_automatic_planning_selects_openvino_on_the_npu():
    _, selected = plan(request_for("intel:npu"), "auto", registry)

    assert selected.selected == "openvino"
    declining = {
        candidate.backend for candidate in selected.candidates if not candidate.support.supported
    }
    # Nothing may quietly claim an NPU target and run on the host instead.
    assert {"inductor", "aot_inductor", "eager"} <= declining


@requires_openvino
def test_npu_compile_rejects_dynamic_shapes():
    backend = OpenVINOBackend()
    example = torch.randn(2, 4)

    with pytest.raises(CompilationError, match="static shapes only"):
        backend.compile(request_for("intel:npu", {"static_shapes": False}), (example,), {})


@requires_openvino
def test_npu_compile_refuses_to_run_on_the_cpu_plugin_instead():
    """The whole IR path runs -- export, convert, reshape -- and then stops at
    device selection rather than compiling an NPU request onto the CPU."""
    core = pytest.importorskip("openvino").Core()
    if any(name.split(".", 1)[0] == "NPU" for name in core.available_devices):
        pytest.skip("this host exposes an Intel NPU")

    with pytest.raises(CompilationError, match="intel_vpu"):
        OpenVINOBackend().compile(request_for("intel:npu"), (torch.randn(2, 4),), {})


def test_npu_export_rejects_dynamic_shapes(tmp_path):
    from lm7 import export

    with pytest.raises(BackendUnavailableError, match="static shapes only"):
        export(
            torch.nn.Linear(4, 4).eval(),
            args=(torch.randn(2, 4),),
            target="intel:npu",
            output=tmp_path / "npu.lm7",
            backend="openvino",
            dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=8)},),
        )


def test_intel_npu_detection_reads_the_openvino_device_list(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    properties = {
        "FULL_DEVICE_NAME": "Intel(R) AI Boost",
        "DEVICE_ARCHITECTURE": "4000",
        "NPU_DRIVER_VERSION": "1234",
    }
    core = SimpleNamespace(
        available_devices=["CPU", "GPU", "NPU"],
        get_property=lambda device, key: properties[key],
    )
    fake_openvino = SimpleNamespace(__version__="2026.2.1-test", Core=lambda: core)
    monkeypatch.setattr(
        detection_module.importlib.util, "find_spec", lambda name: SimpleNamespace()
    )
    monkeypatch.setattr(detection_module.importlib, "import_module", lambda name: fake_openvino)
    monkeypatch.setattr(detection_module, "intel_npu_device_nodes", lambda: ["accel0"])

    devices = detection_module._detect_intel_npu_targets()

    assert [str(device.target) for device in devices] == ["intel:npu"]
    assert devices[0].target.kind == "npu"
    assert devices[0].name == "Intel(R) AI Boost"
    assert devices[0].capabilities["openvino_device"] == "NPU"
    assert devices[0].capabilities["device_architecture"] == "4000"
    assert devices[0].capabilities["device_nodes"] == ["accel0"]


def test_intel_npu_detection_is_quiet_without_the_plugin(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    core = SimpleNamespace(available_devices=["CPU"], get_property=lambda device, key: "")
    monkeypatch.setattr(
        detection_module.importlib.util, "find_spec", lambda name: SimpleNamespace()
    )
    monkeypatch.setattr(
        detection_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(__version__="2026.2.1-test", Core=lambda: core),
    )

    assert detection_module._detect_intel_npu_targets() == []


def test_intel_npu_detection_survives_a_broken_runtime(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")

    def explode() -> None:
        raise RuntimeError("cannot open the device")

    monkeypatch.setattr(
        detection_module.importlib.util, "find_spec", lambda name: SimpleNamespace()
    )
    monkeypatch.setattr(
        detection_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(Core=explode),
    )

    assert detection_module._detect_intel_npu_targets() == []
    # CPU detection must still work when the NPU probe fails.
    assert any(device.target.vendor == "cpu" for device in detection_module.detect_targets())


def test_a_missing_npu_says_whether_the_driver_or_the_plugin_is_absent(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    monkeypatch.setattr(detection_module, "_detect_intel_npu_targets", list)

    monkeypatch.setattr(detection_module, "intel_npu_device_nodes", list)
    with pytest.raises(TargetNotFoundError, match="Neither an OpenVINO NPU device"):
        detection_module.resolve_target("intel:npu")

    # Driver present, plugin absent: the fix is an install, not new hardware.
    monkeypatch.setattr(detection_module, "intel_npu_device_nodes", lambda: ["accel0"])
    with pytest.raises(TargetNotFoundError, match=r"\.\[openvino\]"):
        detection_module.resolve_target("intel:npu")


def test_npu_ordinals_come_from_the_openvino_device_suffix():
    detection_module = importlib.import_module("lm7.detection")

    assert detection_module._intel_npu_ordinal("NPU") is None
    assert detection_module._intel_npu_ordinal("NPU.1") == 1


def test_inputs_for_an_npu_target_stay_on_the_host():
    from lm7.detection import torch_device

    assert torch_device(TargetSpec("intel", "npu")) == torch.device("cpu")
    assert torch_device(TargetSpec("intel", "gpu")) == torch.device("xpu", 0)


def test_auto_never_resolves_to_the_npu(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    npu = detection_module.DeviceInfo(TargetSpec("intel", "npu"), "Intel(R) AI Boost")
    cpu = detection_module.DeviceInfo(TargetSpec("cpu", "cpu", architecture="x86_64"), "CPU")
    monkeypatch.setattr(detection_module, "detect_targets", lambda: [npu, cpu])

    assert detection_module.resolve_target("auto").vendor == "cpu"
    assert str(detection_module.resolve_target("intel:npu")) == "intel:npu"
