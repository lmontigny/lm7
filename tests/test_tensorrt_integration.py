from __future__ import annotations

import copy
import importlib.util
import os

import pytest
import torch

import lm7
from lm7.errors import ArtifactLoadError

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.tensorrt,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable"),
    pytest.mark.skipif(
        importlib.util.find_spec("torch_tensorrt") is None,
        reason="Torch-TensorRT is not installed",
    ),
]

RUN_HF_TESTS = os.environ.get("LM7_RUN_HF_TESTS") == "1"


def test_nvidia_tensorrt_matches_eager():
    torch.manual_seed(0)
    source = torch.nn.Sequential(
        torch.nn.Linear(32, 128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 8),
    ).eval()
    reference = copy.deepcopy(source).cuda().half()
    source.half()
    example_input = torch.randn(4, 32, dtype=torch.float16)
    expected = reference(example_input.cuda())

    compiled = lm7.compile(
        source,
        target="nvidia",
        backend="tensorrt",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(example_input)

    assert compiled.selected_backend == "tensorrt"
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.hf
@pytest.mark.skipif(not RUN_HF_TESTS, reason="set LM7_RUN_HF_TESTS=1")
def test_smollm2_tensorrt_matches_eager_logits():
    transformers = pytest.importorskip("transformers")
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
    ).eval()
    inputs = tokenizer("The capital of France is", return_tensors="pt")

    reference = copy.deepcopy(model).cuda()
    device_inputs = {name: value.cuda() for name, value in inputs.items()}
    with torch.inference_mode():
        expected = reference(**device_inputs, use_cache=False).logits

    compiled = lm7.compile(
        model,
        target="nvidia:sm89",
        backend="tensorrt",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(**inputs, use_cache=False).logits

    assert compiled.selected_backend == "tensorrt"
    actual_last = actual[:, -1].float()
    expected_last = expected[:, -1].float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_last.flatten(),
        expected_last.flatten(),
        dim=0,
    )
    p99_error = torch.quantile((actual_last - expected_last).abs(), 0.99)

    assert cosine.item() >= 0.9999
    assert p99_error.item() <= 0.15
    assert actual[:, -1].argmax().equal(expected[:, -1].argmax())


def test_tensorrt_engine_survives_export_and_reload(tmp_path):
    """The point of the AOT path: a second process must not rebuild the engine.

    Building SmolLM2-135M took 54 s on an Ada GPU against 4 s to load the saved
    engine, so this is the difference between a usable artifact and a rebuild.
    """
    torch.manual_seed(0)
    source = torch.nn.Sequential(
        torch.nn.Linear(16, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    ).eval()
    example = torch.randn(8, 16, device="cuda")
    expected = copy.deepcopy(source).cuda()(example)

    artifact = lm7.export(
        source,
        args=(example,),
        target="nvidia",
        backend="tensorrt",
        output=tmp_path / "trt.lm7",
    )

    assert artifact.manifest.compiled_file == "compiled_model.trt.pt2"
    requirements = artifact.manifest.runtime_requirements or {}
    # An engine is tuned for one GPU architecture; the manifest has to say which.
    assert requirements["device_bound"] is True
    assert requirements["compute_capability"].startswith("sm")
    assert requirements["torch-tensorrt"]
    torch.testing.assert_close(artifact(example), expected, rtol=2e-3, atol=2e-3)

    reloaded = lm7.load_artifact(artifact.path)
    torch.testing.assert_close(reloaded(example), expected, rtol=2e-3, atol=2e-3)


def test_corrupt_tensorrt_engine_fails_checksum_validation(tmp_path):
    artifact = lm7.export(
        torch.nn.Sequential(torch.nn.Linear(16, 4)).eval(),
        args=(torch.randn(8, 16, device="cuda"),),
        target="nvidia",
        backend="tensorrt",
        output=tmp_path / "trt.lm7",
    )
    engine = artifact.path / "compiled_model.trt.pt2"
    engine.write_bytes(engine.read_bytes() + b"corrupt")

    with pytest.raises(ArtifactLoadError, match="checksum does not match"):
        lm7.load_artifact(artifact.path)
