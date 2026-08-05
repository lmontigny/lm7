from __future__ import annotations

import importlib.util
import sys

import pytest
import torch

import lm7
from lm7.exporting import COMPILED_PTE_NAME


def _coreml_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        if importlib.util.find_spec("executorch") is None:
            return False
        importlib.import_module("executorch.backends.apple.coreml.partition.coreml_partitioner")
        importlib.import_module("coremltools")
        return True
    except (ImportError, AttributeError, ValueError):
        return False


pytestmark = [
    pytest.mark.coreml,
    pytest.mark.skipif(not _coreml_available(), reason="ExecuTorch Core ML is unavailable"),
]


def model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()


def test_export_and_reload_matches_eager(tmp_path):
    source = model()
    example = (torch.randn(8, 16),)
    with torch.no_grad():
        expected = source(*example)
    output = tmp_path / "model.lm7"

    artifact = lm7.export(source, args=example, target="apple", backend="coreml", output=output)

    assert artifact.manifest.backend == "coreml"
    assert artifact.manifest.runtime_requirements["device_bound"] is False
    assert (output / COMPILED_PTE_NAME).is_file()
    # Default compute_precision is float16, so this needs a looser tolerance
    # than the float32-exact paths elsewhere in the suite.
    torch.testing.assert_close(artifact(*example), expected, rtol=0.05, atol=0.01)

    reloaded = lm7.load_artifact(output)
    torch.testing.assert_close(reloaded(*example), expected, rtol=0.05, atol=0.01)


def test_cpu_only_float32_matches_eager_tightly(tmp_path):
    """No ANE/GPU float16 rounding in this path, so accuracy should be exact."""
    source = model()
    example = (torch.randn(8, 16),)
    with torch.no_grad():
        expected = source(*example)

    artifact = lm7.export(
        source,
        args=example,
        target="apple",
        backend="coreml",
        output=tmp_path / "model.lm7",
        options={"compute_unit": "cpu_only", "compute_precision": "float32"},
    )

    requirements = artifact.manifest.runtime_requirements
    assert requirements["compute_unit"] == "cpu_only"
    assert requirements["compute_precision"] == "float32"
    torch.testing.assert_close(artifact(*example), expected, rtol=1e-4, atol=1e-4)


def test_embedding_compiles(tmp_path):
    """Same reason as the equivalent TVM/ExecuTorch tests: proves the operator
    table this backend actually reaches, not just a Linear+GELU MLP."""

    class Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 8)
            self.out = torch.nn.Linear(8, 4)

        def forward(self, ids):
            return self.out(self.embedding(ids))

    source = Embed().eval()
    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        expected = source(ids)

    artifact = lm7.export(
        source,
        args=(ids,),
        target="apple",
        backend="coreml",
        output=tmp_path / "model.lm7",
        options={"compute_unit": "cpu_only", "compute_precision": "float32"},
    )

    torch.testing.assert_close(artifact(ids), expected, rtol=1e-4, atol=1e-4)


def test_non_apple_target_is_rejected(tmp_path):
    from lm7.errors import BackendUnavailableError

    with pytest.raises(BackendUnavailableError, match="target='apple'"):
        lm7.export(
            model(),
            args=(torch.randn(8, 16),),
            target="cpu",
            backend="coreml",
            output=tmp_path / "model.lm7",
        )
