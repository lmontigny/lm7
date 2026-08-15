"""The AMD half of `tests/test_nvidia_aot_integration.py`.

Nothing in this file has ever run. No AMD GPU has executed LM7 at all, so these
are the assertions a first `gfx942` session is meant to either confirm or
correct -- in particular whether the AOTInductor wrapper links against ROCm,
which is the one thing about the AMD packaging path that cannot be settled by
reading code. See docs/limitations.md#hardware-validation.
"""

from __future__ import annotations

import copy
import subprocess
import sys
import textwrap

import pytest
import torch

import lm7
from lm7.backends.aot_inductor import AOTInductorBackend, _rocm_home
from lm7.backends.base import CompileRequest
from lm7.targets import parse_target

pytestmark = [
    pytest.mark.rocm,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not getattr(torch.version, "hip", None),
        reason="ROCm GPU is unavailable",
    ),
    pytest.mark.skipif(
        _rocm_home() is None,
        reason="no ROCm installation; the AOTInductor wrapper cannot be built",
    ),
]


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    ).eval()


def architecture() -> str:
    """The normalized `gfx` string, matching what `detect_targets` records.

    ROCm reports `gfx942:sramecc+:xnack-`; the feature suffixes are not part of
    the architecture an artifact is bound to.
    """
    return torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]


def test_backend_reports_amd_support():
    support = AOTInductorBackend().supports(
        CompileRequest(
            model=model(),
            target=parse_target(f"amd:{architecture()}"),
            mode="lazy",
            transfers="automatic",
            fallback="error",
        )
    )
    assert support.supported is True
    assert support.priority == 90


def test_amd_aot_compile_matches_eager():
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source).cuda()
    example_input = torch.randn(8, 16)
    expected = reference(example_input.cuda())

    compiled = lm7.compile(
        source,
        target=f"amd:{architecture()}",
        backend="aot_inductor",
        fallback="error",
    )
    actual = compiled(example_input)

    assert compiled.selected_backend == "aot_inductor"
    # ROCm devices are `cuda` devices to torch; the vendor split is LM7's.
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, expected)


def test_amd_aot_artifact_records_the_rocm_pair(tmp_path):
    """The manifest half, which is what makes a failed load elsewhere diagnosable.

    `torch.version.cuda` is None on ROCm and `get_device_capability` answers
    (9, 4) on a gfx942, so a manifest built from the CUDA fields would claim no
    runtime and an `sm94` that no NVIDIA part has ever had.
    """
    artifact = lm7.export(
        model(),
        args=(torch.randn(8, 16),),
        target=f"amd:{architecture()}",
        backend="aot_inductor",
        output=tmp_path / "model.lm7",
    )

    assert artifact.manifest.backend == "aot_inductor"
    assert artifact.manifest.target["vendor"] == "amd"
    assert artifact.manifest.target["architecture"] == architecture()
    requirements = artifact.manifest.runtime_requirements
    assert requirements["device_bound"] is True
    assert requirements["gcn_architecture"] == architecture()
    assert requirements["hip"] == torch.version.hip
    assert "compute_capability" not in requirements
    assert "cuda" not in requirements


def test_amd_aot_artifact_reloads_without_compiling(tmp_path):
    """The point of the AOT path: a second process runs with no compiler."""
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source).cuda()
    example_input = torch.randn(8, 16)
    expected = reference(example_input.cuda())

    artifact = lm7.export(
        source,
        args=(example_input,),
        target=f"amd:{architecture()}",
        backend="aot_inductor",
        output=tmp_path / "model.lm7",
    )
    assert (artifact.path / artifact.manifest.compiled_file).is_file()
    torch.testing.assert_close(artifact(example_input.cuda()), expected)

    input_path = tmp_path / "input.pt"
    output_path = tmp_path / "output.pt"
    torch.save(example_input, input_path)
    script = textwrap.dedent(
        """
        import sys, torch, lm7
        artifact = lm7.load_artifact(sys.argv[1])
        result = artifact(torch.load(sys.argv[2]).cuda())
        assert result.device.type == "cuda", result.device
        torch.save(result.cpu(), sys.argv[3])
        """
    )
    subprocess.run(
        [sys.executable, "-c", script, str(artifact.path), str(input_path), str(output_path)],
        check=True,
    )

    torch.testing.assert_close(torch.load(output_path), expected.cpu())


def test_amd_aot_export_reports_a_missing_rocm_installation(tmp_path, monkeypatch):
    from lm7.backends import aot_inductor

    monkeypatch.setattr(aot_inductor, "_rocm_home", lambda: None)
    with pytest.raises(lm7.errors.CompilationError, match="no ROCm installation was found"):
        lm7.export(
            model(),
            args=(torch.randn(8, 16),),
            target=f"amd:{architecture()}",
            backend="aot_inductor",
            output=tmp_path / "model.lm7",
        )
    assert not (tmp_path / "model.lm7").exists()


def test_an_amd_artifact_is_refused_on_a_different_gfx(tmp_path):
    """The architecture gate has listed `amd` since it was written and, until
    the AOT path existed, guarded a payload nothing could produce."""
    artifact = lm7.export(
        model(),
        args=(torch.randn(8, 16),),
        target=f"amd:{architecture()}",
        backend="aot_inductor",
        output=tmp_path / "model.lm7",
    )
    manifest_path = artifact.path / "manifest.json"
    manifest = manifest_path.read_text(encoding="utf-8")
    # A gfx this host is not, so the guard has something to disagree with.
    other = "gfx90a" if architecture() != "gfx90a" else "gfx942"
    manifest_path.write_text(
        manifest.replace(f'"{architecture()}"', f'"{other}"'), encoding="utf-8"
    )

    with pytest.raises(lm7.errors.ArtifactLoadError, match=other):
        lm7.load_artifact(artifact.path)
