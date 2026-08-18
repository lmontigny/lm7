from __future__ import annotations

import copy
import importlib.util
import os

import pytest
import torch

import lm7
from lm7.backends.onnxruntime import ONNXRuntimeBackend, available_providers
from lm7.errors import ArtifactLoadError
from lm7.exporting import DynamicDimension, ShapeProfile
from lm7.huggingface import _LogitsOnly


def _onnx_stack_installed() -> bool:
    for module in ("onnx", "onnxscript", "onnxruntime"):
        try:
            if importlib.util.find_spec(module) is None:
                return False
        except ModuleNotFoundError:
            return False
    return True


pytestmark = [
    pytest.mark.onnxruntime,
    pytest.mark.skipif(not _onnx_stack_installed(), reason='install LM7 with ".[onnxruntime]"'),
]


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    ).eval()


def wide_model() -> torch.nn.Module:
    """Wide enough that the exporter actually moves weights out of the graph.

    Tensors below roughly a kilobyte stay inline whatever the export was asked
    for, so a narrower model would produce an empty sidecar and prove nothing.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 8),
    ).eval()


def test_cpu_compile_matches_eager():
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source)
    example = torch.randn(2, 4)
    expected = reference(example)

    compiled = lm7.compile(
        source,
        target="cpu",
        backend="onnxruntime",
        fallback="error",
    )
    actual = compiled(example)

    assert compiled.selected_backend == "onnxruntime"
    assert compiled.artifact.metadata["provider"] == "CPUExecutionProvider"
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_cpu_artifact_round_trips(tmp_path):
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source)
    example = torch.randn(2, 4)
    expected = reference(example)

    artifact = lm7.export(
        source,
        args=(example,),
        target="cpu",
        backend="onnxruntime",
        output=tmp_path / "model.lm7",
    )
    reloaded = lm7.load_artifact(artifact.path)

    assert artifact.manifest.compiled_file == "compiled_model.onnx"
    assert (artifact.path / "compiled_model.onnx").stat().st_size > 0
    # Small enough to stay in one payload, which is what "auto" should pick.
    assert artifact.manifest.compiled_weights_file is None
    assert not (artifact.path / "compiled_model.onnx.data").exists()
    torch.testing.assert_close(reloaded(example), expected, rtol=1e-5, atol=1e-6)


def test_cpu_artifact_retains_bounded_dynamic_batch(tmp_path):
    torch.manual_seed(0)
    source = model()
    profile = ShapeProfile(inputs={"input": {0: DynamicDimension("batch", min=1, max=8)}})
    artifact = lm7.export(
        source,
        args=(torch.randn(2, 4),),
        target="cpu",
        backend="onnxruntime",
        output=tmp_path / "dynamic.lm7",
        shape_profile=profile,
    )

    for batch in (1, 5, 8):
        value = torch.randn(batch, 4)
        actual = artifact(value)
        expected = source(value)
        assert actual.shape == (batch, 3)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
def test_cuda_provider_matches_eager_without_cpu_fallback():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is unavailable")
    if "CUDAExecutionProvider" not in available_providers():
        pytest.skip("install onnxruntime-gpu with the matching CUDA major version")

    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source)
    example = torch.randn(2, 4)
    expected = reference(example)

    compiled = lm7.compile(
        source,
        target="nvidia:sm89",
        backend="onnxruntime",
        fallback="error",
    )
    actual = compiled(example)

    assert compiled.artifact.metadata["provider"] == "CUDAExecutionProvider"
    assert compiled.artifact.metadata["disable_cpu_fallback"] is True
    # I/O binding leaves the result where the provider produced it. A CPU tensor
    # here would mean the session round-tripped through host memory again.
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.cuda
def test_cuda_accepts_an_input_already_on_the_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is unavailable")
    if "CUDAExecutionProvider" not in available_providers():
        pytest.skip("install onnxruntime-gpu with the matching CUDA major version")

    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source)
    example = torch.randn(2, 4)
    expected = reference(example)

    compiled = lm7.compile(
        source,
        target="nvidia:sm89",
        backend="onnxruntime",
        fallback="error",
    )
    # The NumPy path took .cpu() on everything, so a device-resident input was
    # copied down and straight back up again.
    actual = compiled(example.cuda())

    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.hf
@pytest.mark.skipif(
    os.environ.get("LM7_RUN_HF_TESTS") != "1",
    reason="set LM7_RUN_HF_TESTS=1",
)
def test_smollm2_fixed_logits_export_matches_eager(tmp_path):
    transformers = pytest.importorskip("transformers")
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    source = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
    ).eval()
    wrapped = _LogitsOnly(source).eval()
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    args = (inputs["input_ids"], inputs["attention_mask"])
    with torch.no_grad():
        expected = wrapped(*args)
        exported = torch.export.export(wrapped, args, strict=False)

    path = tmp_path / "smollm2.onnx"
    backend = ONNXRuntimeBackend()
    backend.compile_exported(exported, path)
    actual = backend.load_onnx(
        path,
        provider="CPUExecutionProvider",
        disable_cpu_fallback=False,
    )(*args)

    assert path.stat().st_size > 500_000_000
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert torch.equal(actual[:, -1].argmax(-1), expected[:, -1].argmax(-1))


def test_cpu_artifact_packages_external_data(tmp_path):
    torch.manual_seed(0)
    source = wide_model()
    reference = copy.deepcopy(source)
    example = torch.randn(2, 256)
    expected = reference(example)

    artifact = lm7.export(
        source,
        args=(example,),
        target="cpu",
        backend="onnxruntime",
        output=tmp_path / "external.lm7",
        options={"external_data": True},
    )
    weights = artifact.path / "compiled_model.onnx.data"

    assert artifact.manifest.compiled_weights_file == "compiled_model.onnx.data"
    assert artifact.manifest.compiled_weights_sha256
    assert weights.stat().st_size > 0
    # Once the weights move out, what is left of the graph is only its structure.
    assert (artifact.path / "compiled_model.onnx").stat().st_size < weights.stat().st_size

    reloaded = lm7.load_artifact(artifact.path)
    torch.testing.assert_close(reloaded(example), expected, rtol=1e-5, atol=1e-6)


def test_corrupt_external_data_fails_checksum_validation(tmp_path):
    torch.manual_seed(0)
    artifact = lm7.export(
        wide_model(),
        args=(torch.randn(2, 256),),
        target="cpu",
        backend="onnxruntime",
        output=tmp_path / "external.lm7",
        options={"external_data": True},
    )
    weights = artifact.path / "compiled_model.onnx.data"
    weights.write_bytes(b"\x00" * weights.stat().st_size)

    # ORT reads the sidecar implicitly while building the session, so without an
    # explicit check the corruption would surface as wrong numbers, not an error.
    with pytest.raises(ArtifactLoadError):
        lm7.load_artifact(artifact.path)
