from __future__ import annotations

import copy
import subprocess
import sys
import textwrap

import pytest
import torch

import lm7
from lm7.backends.aot_inductor import AOTInductorBackend, _cuda_toolkit_home
from lm7.backends.base import CompileRequest
from lm7.targets import parse_target

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable"),
    pytest.mark.skipif(
        _cuda_toolkit_home() is None,
        reason='no CUDA toolkit; install LM7 with ".[cuda-aot]"',
    ),
]


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    ).eval()


def architecture() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def test_backend_reports_cuda_support():
    support = AOTInductorBackend().supports(
        CompileRequest(
            model=model(),
            target=parse_target(f"nvidia:{architecture()}"),
            mode="lazy",
            transfers="automatic",
            fallback="error",
        )
    )
    assert support.supported is True
    assert support.priority == 90


def test_nvidia_aot_compile_matches_eager():
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source).cuda()
    example_input = torch.randn(8, 16)
    expected = reference(example_input.cuda())

    compiled = lm7.compile(
        source,
        target=f"nvidia:{architecture()}",
        backend="aot_inductor",
        fallback="error",
    )
    actual = compiled(example_input)

    assert compiled.selected_backend == "aot_inductor"
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, expected)


def test_nvidia_aot_artifact_reloads_without_compiling(tmp_path):
    """The point of the CUDA AOT path: a second process runs with no compiler."""
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source).cuda()
    example_input = torch.randn(8, 16)
    expected = reference(example_input.cuda())

    artifact = lm7.export(
        source,
        args=(example_input,),
        target=f"nvidia:{architecture()}",
        backend="aot_inductor",
        output=tmp_path / "model.lm7",
    )
    assert artifact.manifest.backend == "aot_inductor"
    assert artifact.manifest.target["vendor"] == "nvidia"
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


def test_nvidia_aot_export_reports_a_missing_toolkit(tmp_path, monkeypatch):
    from lm7.backends import aot_inductor

    monkeypatch.setattr(aot_inductor, "_cuda_toolkit_home", lambda: None)
    with pytest.raises(lm7.errors.CompilationError, match="no CUDA toolkit was found"):
        lm7.export(
            model(),
            args=(torch.randn(8, 16),),
            target=f"nvidia:{architecture()}",
            backend="aot_inductor",
            output=tmp_path / "model.lm7",
        )
    assert not (tmp_path / "model.lm7").exists()
