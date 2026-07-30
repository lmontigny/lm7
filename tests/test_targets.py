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
        ("tenstorrent", "tenstorrent", "accelerator", None, None),
        ("tenstorrent:blackhole", "tenstorrent", "accelerator", "blackhole", None),
        ("tenstorrent:wormhole", "tenstorrent", "accelerator", "wormhole", None),
        ("tenstorrent:n300", "tenstorrent", "accelerator", None, "n300"),
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


@pytest.mark.parametrize("value", ["", "wat", "nvidia:", "nvidia:h100:0", "intel:vpu"])
def test_parse_invalid_target(value):
    with pytest.raises(TargetNotFoundError):
        parse_target(value)
