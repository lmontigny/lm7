from lm7.detection import detect_targets, resolve_target


def test_cpu_is_always_detected():
    assert any(device.target.vendor == "cpu" for device in detect_targets())


def test_explicit_cpu_resolves():
    assert resolve_target("cpu").vendor == "cpu"
