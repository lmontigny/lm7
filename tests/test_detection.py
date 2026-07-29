import importlib
from types import SimpleNamespace

import pytest
import torch

from lm7.detection import (
    _detect_tenstorrent_targets,
    _detect_tpu_targets,
    detect_targets,
    resolve_target,
    torch_device,
)
from lm7.targets import DeviceInfo, TargetSpec


def test_cpu_is_always_detected():
    assert any(device.target.vendor == "cpu" for device in detect_targets())


def test_explicit_cpu_resolves():
    assert resolve_target("cpu").vendor == "cpu"


def test_rocm_device_reports_normalized_gfx_architecture(monkeypatch):
    properties = SimpleNamespace(
        name="AMD Radeon Test GPU",
        total_memory=16 * 1024**3,
        gcnArchName="gfx1100:sramecc+:xnack-",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda ordinal: properties)
    monkeypatch.setattr(torch.version, "hip", "7.0-test")

    device = next(item for item in detect_targets() if item.target.vendor == "amd")

    assert device.target.architecture == "gfx1100"
    assert device.name == "AMD Radeon Test GPU"
    assert device.capabilities == {
        "hip": "7.0-test",
        "gcn_arch_name": "gfx1100:sramecc+:xnack-",
    }


def test_tpu_detection_uses_pjrt_runtime(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    torch_xla = SimpleNamespace(__version__="2.9-test")
    runtime = SimpleNamespace(
        device_type=lambda: "TPU",
        addressable_device_count=lambda: 2,
        global_runtime_device_attributes=lambda: [{"device_kind": "TPU v5e"}],
    )
    monkeypatch.setattr(
        detection_module.importlib.util, "find_spec", lambda name: SimpleNamespace()
    )
    monkeypatch.setattr(
        detection_module.importlib,
        "import_module",
        lambda name: runtime if name == "torch_xla.runtime" else torch_xla,
    )

    devices = _detect_tpu_targets()

    assert [device.target.ordinal for device in devices] == [0, 1]
    assert all(device.target.vendor == "tpu" for device in devices)
    assert all(device.target.model == "v5e" for device in devices)
    assert devices[0].name == "TPU v5e"
    assert devices[0].capabilities["pjrt_device"] == "TPU"


def test_tpu_detection_ignores_xla_cpu_runtime(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    runtime = SimpleNamespace(device_type=lambda: "CPU")
    monkeypatch.setattr(
        detection_module.importlib.util, "find_spec", lambda name: SimpleNamespace()
    )
    monkeypatch.setattr(
        detection_module.importlib,
        "import_module",
        lambda name: runtime,
    )

    assert _detect_tpu_targets() == []


def test_auto_prefers_detected_tpu_over_cpu(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    monkeypatch.setattr(
        detection_module,
        "detect_targets",
        lambda: [
            DeviceInfo(TargetSpec("tpu", "accelerator", model="v5e"), "TPU v5e"),
            DeviceInfo(TargetSpec("cpu", "cpu"), "CPU"),
        ],
    )

    assert resolve_target("auto").vendor == "tpu"
    assert torch_device(TargetSpec("tpu", "accelerator", ordinal=1)) == torch.device("xla:1")


def _patch_tenstorrent_runtime(monkeypatch, runtime, *, torch_xla=None) -> None:
    detection_module = importlib.import_module("lm7.detection")
    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    monkeypatch.setattr(
        detection_module.importlib.util, "find_spec", lambda name: SimpleNamespace()
    )
    monkeypatch.setattr(
        detection_module.importlib,
        "import_module",
        lambda name: runtime if name == "torch_xla.runtime" else (torch_xla or SimpleNamespace()),
    )


def test_tenstorrent_detection_selects_the_tt_pjrt_device(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    selected = {}
    runtime = SimpleNamespace(
        device_type=lambda: selected.get("device_type", "CPU"),
        set_device_type=lambda value: selected.update(device_type=value),
        addressable_device_count=lambda: 2,
        global_runtime_device_attributes=lambda: [{"device_kind": "Blackhole p150"}],
    )
    _patch_tenstorrent_runtime(monkeypatch, runtime, torch_xla=SimpleNamespace(__version__="2.9"))
    monkeypatch.setattr(detection_module, "tenstorrent_device_nodes", lambda: ["0", "1"])

    devices = _detect_tenstorrent_targets()

    assert [device.target.ordinal for device in devices] == [0, 1]
    assert all(device.target.vendor == "tenstorrent" for device in devices)
    assert all(device.target.kind == "accelerator" for device in devices)
    assert all(device.target.architecture == "blackhole" for device in devices)
    assert devices[0].name == "Blackhole p150"
    assert devices[0].capabilities["pjrt_device"] == "TT"
    assert devices[0].capabilities["device_nodes"] == ["0", "1"]


def test_tenstorrent_detection_requires_the_plugin(monkeypatch):
    detection_module = importlib.import_module("lm7.detection")
    monkeypatch.setattr(detection_module.importlib.util, "find_spec", lambda name: None)

    assert _detect_tenstorrent_targets() == []


def test_tenstorrent_detection_never_hijacks_a_tpu_runtime(monkeypatch):
    runtime = SimpleNamespace(
        device_type=lambda: "TPU",
        set_device_type=lambda value: pytest.fail("must not reassign a live PJRT runtime"),
        addressable_device_count=lambda: 4,
        global_runtime_device_attributes=lambda: [{"device_kind": "TPU v5e"}],
    )
    _patch_tenstorrent_runtime(monkeypatch, runtime)

    assert _detect_tenstorrent_targets() == []


def test_tenstorrent_detection_honours_an_explicit_pjrt_device(monkeypatch):
    runtime = SimpleNamespace(
        device_type=lambda: "CUDA",
        set_device_type=lambda value: pytest.fail("must not override an explicit PJRT_DEVICE"),
        addressable_device_count=lambda: 1,
        global_runtime_device_attributes=lambda: [{"device_kind": "Wormhole n300"}],
    )
    _patch_tenstorrent_runtime(monkeypatch, runtime)
    monkeypatch.setenv("PJRT_DEVICE", "CUDA")

    assert _detect_tenstorrent_targets() == []


def test_tenstorrent_detection_ignores_a_card_less_runtime(monkeypatch):
    runtime = SimpleNamespace(
        device_type=lambda: "TT",
        set_device_type=lambda value: None,
        addressable_device_count=lambda: 0,
        global_runtime_device_attributes=list,
    )
    _patch_tenstorrent_runtime(monkeypatch, runtime)

    assert _detect_tenstorrent_targets() == []


def test_tenstorrent_uses_the_xla_device(monkeypatch):
    target = TargetSpec("tenstorrent", "accelerator", architecture="wormhole", ordinal=1)

    assert torch_device(target) == torch.device("xla:1")
