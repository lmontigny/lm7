import pytest

from lm7.errors import TargetNotFoundError
from lm7.targets import parse_target


@pytest.mark.parametrize(
    ("value", "vendor", "kind", "architecture", "model"),
    [
        ("auto", "auto", "auto", None, None),
        ("cpu", "cpu", "cpu", None, None),
        ("nvidia", "nvidia", "gpu", None, None),
        ("nvidia:h100", "nvidia", "gpu", None, "h100"),
        ("nvidia:sm90", "nvidia", "gpu", "sm90", None),
        ("amd:gfx942", "amd", "gpu", "gfx942", None),
        ("intel:gpu", "intel", "gpu", None, None),
        ("intel:npu", "intel", "npu", None, None),
        ("qualcomm:sm8750", "qualcomm", "npu", "v79", "sm8750"),
        ("tenstorrent", "tenstorrent", "accelerator", None, None),
        ("tenstorrent:blackhole", "tenstorrent", "accelerator", "blackhole", None),
        ("tenstorrent:wormhole", "tenstorrent", "accelerator", "wormhole", None),
        ("tenstorrent:n300", "tenstorrent", "accelerator", None, "n300"),
        ("arm", "arm", "gpu", None, None),
        ("arm:valhall", "arm", "gpu", "valhall", None),
        ("arm:valhall4", "arm", "gpu", "valhall4", None),
        ("arm:bifrost", "arm", "gpu", "bifrost", None),
        ("arm:mali-g715", "arm", "gpu", None, "mali-g715"),
        ("arm:immortalis-g925", "arm", "gpu", None, "immortalis-g925"),
    ],
)
def test_parse_target(value, vendor, kind, architecture, model):
    result = parse_target(value)
    assert (result.vendor, result.kind, result.architecture, result.model) == (
        vendor,
        kind,
        architecture,
        model,
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "wat",
        "nvidia:",
        "nvidia:h100:0",
        "intel:vpu",
        "qualcomm",
        "qualcomm:sm8650",
        "arm:g715",
        "arm:mali",
        "arm:adreno",
    ],
)
def test_parse_invalid_target(value):
    with pytest.raises(TargetNotFoundError):
        parse_target(value)


def test_qualcomm_target_round_trips_with_device_metadata():
    target = parse_target("qualcomm:sm8750")

    assert str(target) == "qualcomm:sm8750"
    assert target.remote is True


@pytest.mark.parametrize("value", ["arm", "arm:valhall4", "arm:mali-g715"])
def test_arm_gpu_targets_round_trip_and_are_remote(value):
    """Arm GPUs are export destinations, never local PyTorch devices, so they
    must skip local detection the way qualcomm:sm8750 does."""
    target = parse_target(value)

    assert str(target) == value
    assert target.remote is True
