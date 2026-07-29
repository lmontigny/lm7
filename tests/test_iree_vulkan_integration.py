from __future__ import annotations

import copy
import importlib.util

import pytest
import torch

import lm7
from lm7.backends.iree_vulkan import query_vulkan_devices


def _iree_installed() -> bool:
    for module in ("iree.compiler", "iree.runtime", "iree.turbine.aot"):
        try:
            if importlib.util.find_spec(module) is None:
                return False
        except ModuleNotFoundError:
            return False
    return True


pytestmark = [
    pytest.mark.iree_vulkan,
    pytest.mark.skipif(not _iree_installed(), reason='install LM7 with ".[iree-vulkan]"'),
]


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    ).eval()


def test_real_vulkan_export_produces_a_vmfb_without_a_local_device(tmp_path):
    artifact = lm7.export(
        model(),
        args=(torch.randn(2, 4),),
        target="nvidia:sm89",
        backend="iree_vulkan",
        output=tmp_path / "model.lm7",
    )

    vmfb = artifact.path / "compiled_model.vmfb"
    assert artifact.manifest.backend == "iree_vulkan"
    assert artifact.manifest.target["vendor"] == "nvidia"
    assert artifact.manifest.compiled_file == vmfb.name
    assert artifact.manifest.compiled_sha256
    assert vmfb.stat().st_size > 0


def test_real_vulkan_runtime_matches_eager_when_a_device_is_visible(tmp_path):
    if not query_vulkan_devices():
        pytest.skip("IREE Vulkan runtime enumerates no device on this host")

    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source)
    example = torch.randn(2, 4)
    expected = reference(example)
    artifact = lm7.export(
        source,
        args=(example,),
        target="nvidia:sm89",
        backend="iree_vulkan",
        output=tmp_path / "model.lm7",
    )

    actual = artifact(example)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
