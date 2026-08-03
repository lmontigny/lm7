from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

import lm7
from lm7.backends.base import CompileRequest
from lm7.backends.openvino import _DEFAULT_INFERENCE_PRECISION, OpenVINOBackend
from lm7.errors import ArtifactLoadError, BackendUnavailableError, CompilationError
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


def requires_intel_npu() -> None:
    """Skip unless the OpenVINO NPU plugin reports a usable device.

    An Intel NPU needs both the plugin and the ``intel_vpu`` driver, so this is
    a runtime question rather than a package one.
    """
    core = pytest.importorskip("openvino").Core()
    if not any(name.split(".", 1)[0] == "NPU" for name in core.available_devices):
        pytest.skip("this host exposes no Intel NPU to OpenVINO")


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


def test_export_writes_openvino_ir_and_round_trips(tmp_path):
    torch.manual_seed(0)
    source = model()
    example = torch.randn(8, 16)
    expected = copy.deepcopy(source)(example)

    artifact = lm7.export(
        source,
        args=(example,),
        target="cpu",
        backend="openvino",
        output=tmp_path / "model.lm7",
    )

    assert artifact.manifest.backend == "openvino"
    assert artifact.manifest.runtime_requirements["openvino"] is not None
    assert (artifact.path / "compiled_model.xml").is_file()
    assert (artifact.path / "compiled_model.bin").is_file()
    torch.testing.assert_close(artifact(example), expected, rtol=1e-4, atol=1e-4)

    reloaded = lm7.load_artifact(artifact.path)
    torch.testing.assert_close(reloaded(example), expected, rtol=1e-4, atol=1e-4)


def test_openvino_matches_eager_on_the_npu():
    """The NPU plugin computes in FP16, so the tolerance here is FP16-level and
    deliberately looser than the CPU test above."""
    requires_intel_npu()
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source)
    example_input = torch.randn(8, 16)
    expected = reference(example_input)

    compiled = lm7.compile(source, target="intel:npu", backend="auto", fallback="error")
    actual = compiled(example_input)

    assert compiled.selected_backend == "openvino"
    assert compiled.artifact is not None
    assert compiled.artifact.metadata["device"] == "NPU"
    # No INFERENCE_PRECISION_HINT: the plugin has no FP32 mode to pin.
    assert compiled.artifact.metadata["inference_precision"] is None
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_npu_export_round_trips_through_the_npu_plugin(tmp_path):
    requires_intel_npu()
    torch.manual_seed(0)
    source = model()
    example = torch.randn(8, 16)
    expected = copy.deepcopy(source)(example)

    artifact = lm7.export(
        source,
        args=(example,),
        target="intel:npu",
        backend="openvino",
        output=tmp_path / "npu.lm7",
    )

    assert artifact.manifest.runtime_requirements["openvino_device"] == "NPU"
    torch.testing.assert_close(artifact(example), expected, rtol=2e-2, atol=2e-2)

    # The device travels with the artifact: a reload must not drop to the CPU.
    reloaded = lm7.load_artifact(artifact.path)
    torch.testing.assert_close(reloaded(example), expected, rtol=2e-2, atol=2e-2)


def test_export_rejects_openvino_for_non_intel_targets(tmp_path):
    with pytest.raises(BackendUnavailableError, match="Intel CPU"):
        lm7.export(
            model(),
            args=(torch.randn(8, 16),),
            target="nvidia",
            backend="openvino",
            output=tmp_path / "model.lm7",
        )


def test_corrupt_openvino_weights_fail_checksum_validation(tmp_path):
    artifact = lm7.export(
        model(),
        args=(torch.randn(8, 16),),
        target="cpu",
        backend="openvino",
        output=tmp_path / "model.lm7",
    )
    weights = artifact.path / "compiled_model.bin"
    weights.write_bytes(weights.read_bytes() + b"corrupt")

    with pytest.raises(ArtifactLoadError, match="checksum does not match"):
        lm7.load_artifact(artifact.path)


def test_openvino_ir_artifact_loads_without_torch():
    """The IR is a portable artifact: it must run in a process that never imports
    torch, which no other LM7 backend offers."""
    # Imported here rather than at module scope: CI's torch build ships without
    # NumPy, and a top-level import fails collection before the skip applies.
    numpy = pytest.importorskip("numpy")

    torch.manual_seed(0)
    compiled = lm7.compile(model(), target="cpu", backend="openvino", fallback="error")
    example_input = torch.randn(8, 16)
    expected = compiled(example_input)
    artifact_path = compiled.artifact.path
    assert artifact_path is not None and artifact_path.exists()

    # The IR carries no inference precision: it is a load-time choice, and the
    # CPU plugin defaults to BF16 on an x86 host with AMX and FP16 on ARM. LM7
    # pins FP32 when it compiles, so a consumer comparing against LM7's own
    # output has to pin it too, or the comparison measures the host's default
    # precision rather than the artifact. Left unpinned this passed on most
    # GitHub runners and failed by 2.5e-3 on the AMX-capable ones -- which is
    # bf16 rounding of this model, to the digit.
    script = (
        "import sys, numpy, openvino\n"
        "config = {'INFERENCE_PRECISION_HINT': sys.argv[4]}\n"
        "model = openvino.Core().compile_model(sys.argv[1], 'CPU', config)\n"
        "result = model([numpy.load(sys.argv[2])])[model.outputs[0]]\n"
        "numpy.save(sys.argv[3], result)\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.npy"
        output_path = Path(tmp) / "output.npy"
        numpy.save(input_path, example_input.numpy())
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(artifact_path),
                str(input_path),
                str(output_path),
                _DEFAULT_INFERENCE_PRECISION,
            ],
            check=True,
        )
        actual = torch.from_numpy(numpy.load(output_path))

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_int8_export_shrinks_the_ir_and_still_matches_eager(tmp_path):
    """NNCF weight compression is applied between convert_model and save_model,
    so it shrinks the IR the artifact ships rather than the graph shape."""
    pytest.importorskip("nncf")
    torch.manual_seed(0)
    source = model()
    example = torch.randn(8, 16)
    expected = copy.deepcopy(source)(example)

    baseline = lm7.export(
        copy.deepcopy(source),
        args=(example,),
        target="cpu",
        backend="openvino",
        output=tmp_path / "fp32.lm7",
    )
    quantized = lm7.export(
        source,
        args=(example,),
        target="cpu",
        backend="openvino",
        output=tmp_path / "int8.lm7",
        options={"quantization": "int8"},
    )

    baseline_bytes = (baseline.path / "compiled_model.bin").stat().st_size
    quantized_bytes = (quantized.path / "compiled_model.bin").stat().st_size
    assert quantized_bytes < baseline_bytes

    # INT8 weights move the outputs well beyond the FP32 path's 1e-6, so this
    # only asserts the artifact still computes the same function approximately.
    torch.testing.assert_close(quantized(example), expected, rtol=0.1, atol=0.1)
    reloaded = lm7.load_artifact(quantized.path)
    torch.testing.assert_close(reloaded(example), expected, rtol=0.1, atol=0.1)


def test_openvino_rejects_an_unknown_quantization():
    with pytest.raises(CompilationError, match="Unsupported OpenVINO quantization"):
        lm7.compile(
            model(),
            target="cpu",
            backend="openvino",
            fallback="error",
            options={"quantization": "int4"},
        )(torch.randn(8, 16))
