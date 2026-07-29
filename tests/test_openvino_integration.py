from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy
import pytest
import torch

import lm7
from lm7.backends.base import CompileRequest
from lm7.backends.openvino import OpenVINOBackend
from lm7.errors import BackendUnavailableError, CompilationError
from lm7.targets import parse_target

pytestmark = [
    pytest.mark.openvino,
    pytest.mark.skipif(
        importlib.util.find_spec("openvino") is None,
        reason="OpenVINO is not installed",
    ),
]


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    ).eval()


def request_for(module: torch.nn.Module) -> CompileRequest:
    return CompileRequest(
        model=module,
        target=parse_target("cpu"),
        mode="lazy",
        transfers="automatic",
        fallback="error",
    )


def test_openvino_matches_eager():
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source)
    example_input = torch.randn(8, 16)
    expected = reference(example_input)

    compiled = lm7.compile(source, target="cpu", backend="openvino", fallback="error")
    actual = compiled(example_input)

    assert compiled.selected_backend == "openvino"
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_openvino_ranks_below_inductor_for_automatic_planning():
    """The evaluation shows a latency win but not broad operator coverage, so
    ``backend="auto"`` must still choose Inductor on a CPU target."""
    backend = OpenVINOBackend()
    support = backend.supports(request_for(model()))

    assert support.supported
    assert support.priority == 80
    assert "inductor" in lm7.explain(target="cpu").splitlines()[0]


def test_openvino_rejects_bfloat16_models():
    """OpenVINO exchanges tensors through NumPy, which has no bfloat16 dtype, so a
    bfloat16 model would only ever measure a silent fallback."""
    bf16_model = model().to(torch.bfloat16)
    support = OpenVINOBackend().supports(request_for(bf16_model))

    assert not support.supported
    assert "bfloat16" in support.reason

    with pytest.raises(BackendUnavailableError, match="bfloat16"):
        lm7.compile(bf16_model, target="cpu", backend="openvino", fallback="error")(
            torch.randn(8, 16, dtype=torch.bfloat16)
        )


def test_openvino_rejects_unavailable_device_instead_of_falling_back():
    """OpenVINO compiles an unavailable device onto the CPU plugin silently, which
    would report a CPU run as a GPU or NPU result."""
    core = pytest.importorskip("openvino").Core()
    if "NPU" in core.available_devices:
        pytest.skip("this host exposes an NPU")

    compiled = lm7.compile(
        model(),
        target="cpu",
        backend="openvino",
        fallback="error",
        options={"device": "NPU"},
    )

    with pytest.raises(CompilationError, match="is not available"):
        compiled(torch.randn(8, 16))


def test_openvino_does_not_compress_weights_to_fp16_by_default():
    """openvino.save_model defaults to compress_to_fp16=True, which shows up as
    FP16-level error on an otherwise FP32 model."""
    torch.manual_seed(0)
    source = model()
    example_input = torch.randn(8, 16)
    reference = copy.deepcopy(source)(example_input)

    compiled = lm7.compile(source, target="cpu", backend="openvino", fallback="error")
    actual = compiled(example_input)

    assert compiled.artifact is not None
    assert compiled.artifact.metadata["compress_to_fp16"] is False
    # FP16 weight compression lands around 1e-3 on this model; FP32 stays near 1e-6.
    assert (actual - reference).abs().max().item() < 1e-5


def test_openvino_ir_artifact_loads_without_torch():
    """The IR is a portable artifact: it must run in a process that never imports
    torch, which no other LM7 backend offers."""
    torch.manual_seed(0)
    compiled = lm7.compile(model(), target="cpu", backend="openvino", fallback="error")
    example_input = torch.randn(8, 16)
    expected = compiled(example_input)
    artifact_path = compiled.artifact.path
    assert artifact_path is not None and artifact_path.exists()

    script = (
        "import sys, numpy, openvino\n"
        "model = openvino.Core().compile_model(sys.argv[1], 'CPU')\n"
        "result = model([numpy.load(sys.argv[2])])[model.outputs[0]]\n"
        "numpy.save(sys.argv[3], result)\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.npy"
        output_path = Path(tmp) / "output.npy"
        numpy.save(input_path, example_input.numpy())
        subprocess.run(
            [sys.executable, "-c", script, str(artifact_path), str(input_path), str(output_path)],
            check=True,
        )
        actual = torch.from_numpy(numpy.load(output_path))

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
