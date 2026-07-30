from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest
import torch

import lm7
from lm7.exporting import COMPILED_PTE_NAME

pytestmark = [
    pytest.mark.executorch,
    pytest.mark.skipif(
        importlib.util.find_spec("executorch") is None,
        reason="ExecuTorch is not installed",
    ),
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

    artifact = lm7.export(source, args=example, target="cpu", backend="executorch", output=output)

    assert artifact.manifest.backend == "executorch"
    assert (output / COMPILED_PTE_NAME).is_file()
    torch.testing.assert_close(artifact(*example), expected, rtol=1e-4, atol=1e-4)

    reloaded = lm7.load_artifact(output)
    torch.testing.assert_close(reloaded(*example), expected, rtol=1e-4, atol=1e-4)


def test_int8_export_and_reload_matches_eager(tmp_path):
    source = model()
    example = (torch.randn(8, 16),)
    with torch.no_grad():
        expected = source(*example)

    artifact = lm7.export(
        source,
        args=example,
        target="cpu",
        backend="executorch",
        output=tmp_path / "model-int8.lm7",
        options={"quantization": "int8"},
    )

    requirements = artifact.manifest.runtime_requirements
    assert requirements["quantization"] == "int8"
    assert requirements["quantized_ops"] > 0
    assert requirements["calibration_samples"] == 1
    torch.testing.assert_close(artifact(*example), expected, rtol=2e-2, atol=2e-2)

    reloaded = lm7.load_artifact(artifact.path)
    torch.testing.assert_close(reloaded(*example), expected, rtol=2e-2, atol=2e-2)


def test_manifest_records_delegate_partition(tmp_path):
    artifact = lm7.export(
        model(),
        args=(torch.randn(8, 16),),
        target="cpu",
        backend="executorch",
        output=tmp_path / "model.lm7",
    )

    requirements = artifact.manifest.runtime_requirements
    assert requirements["delegate"] == "xnnpack"
    # A plain MLP is fully fusible, so XNNPACK must take at least one partition.
    assert requirements["delegated_calls"] >= 1
    assert requirements["total_calls"] >= requirements["delegated_calls"]
    assert requirements["quantization"] == "none"
    assert requirements["quantized_ops"] == 0
    assert requirements["calibration_samples"] == 0
    assert requirements["device_bound"] is False


def test_pte_runs_without_lm7_in_the_process(tmp_path):
    """The point of a .pte: the ExecuTorch runtime alone can execute it.

    On a phone this is the C++ runtime. Here it is the Python binding over the
    same program, in a fresh interpreter that never imports lm7.
    """
    source = model()
    example_input = torch.randn(8, 16)
    with torch.no_grad():
        expected = source(example_input)
    output = tmp_path / "model.lm7"
    lm7.export(source, args=(example_input,), target="cpu", backend="executorch", output=output)

    input_path = tmp_path / "input.pt"
    torch.save(example_input, input_path)
    script = textwrap.dedent(
        """
        import sys
        import torch
        from executorch.runtime import Runtime

        program_path, input_path, output_path = sys.argv[1:4]
        method = Runtime.get().load_program(program_path).load_method("forward")
        result = method.execute([torch.load(input_path)])[0]
        torch.save(result, output_path)
        assert "lm7" not in sys.modules
        """
    )
    result_path = tmp_path / "output.pt"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(output / COMPILED_PTE_NAME),
            str(input_path),
            str(result_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    torch.testing.assert_close(torch.load(result_path), expected, rtol=1e-4, atol=1e-4)


def test_non_cpu_target_is_rejected(tmp_path):
    from lm7.errors import BackendUnavailableError

    with pytest.raises(BackendUnavailableError, match="CPU target"):
        lm7.export(
            model(),
            args=(torch.randn(8, 16),),
            target="nvidia",
            backend="executorch",
            output=tmp_path / "model.lm7",
        )
