from types import SimpleNamespace

import torch

from lm7.detection import detect_targets, resolve_target


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
