"""Real TensorRT-LLM: engine build, streaming generation, and cache reuse.

Skips itself unless the package is importable and an Ampere-or-newer GPU is
present. Both are required and neither is available in CI: GitHub's GPU runners
are gated to Team/Enterprise organizations, and TensorRT-LLM needs its own
environment anyway (`torch>=2.9.1,<=2.10.0a0`, `transformers==4.57.3`), so it
cannot share the CUDA venv the rest of the suite uses.

    uv venv --python 3.12 .venv-trtllm
    uv pip install --python .venv-trtllm/bin/python \
        --extra-index-url https://pypi.nvidia.com tensorrt-llm
    .venv-trtllm/bin/python -m pytest tests/test_tensorrt_llm_integration.py -m tensorrt_llm
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from lm7.runtimes import ServeConfig, engine_dir, engine_identity, reusable, write_manifest
from lm7.runtimes.tensorrt_llm import TensorRTLLMRuntime
from lm7.targets import TargetSpec

pytestmark = pytest.mark.tensorrt_llm

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _unavailable() -> str | None:
    if importlib.util.find_spec("tensorrt_llm") is None:
        return "TensorRT-LLM is not installed"
    if not torch.cuda.is_available():
        return "no CUDA GPU"
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return "TensorRT-LLM needs Ampere (sm80) or newer"
    return None


@pytest.fixture(scope="module")
def target() -> TargetSpec:
    reason = _unavailable()
    if reason:
        pytest.skip(reason)
    major, minor = torch.cuda.get_device_capability()
    return TargetSpec("nvidia", "gpu", architecture=f"sm{major}{minor}")


@pytest.fixture(scope="module")
def runtime() -> TensorRTLLMRuntime:
    return TensorRTLLMRuntime()


def test_probe_reports_available_and_pins_versions(runtime, target):
    info = runtime.probe()
    assert info.available is True, info.reason
    assert info.version
    # These are what an engine is keyed on, so an engine built here is not
    # reusable after an upgrade -- the point of recording them.
    assert info.pinned["tensorrt_llm"]
    assert info.pinned["torch"]


def test_supports_this_card(runtime, target):
    support = runtime.supports(target, MODEL, ServeConfig())
    assert support.supported is True, support.reason


def test_streams_generation_and_the_deltas_reassemble(runtime, target):
    """The runtime yields a cumulative string per step and LM7 converts it to a
    delta, so the test that matters is that concatenating the deltas gives back
    a sensible completion rather than a repeated prefix."""
    config = ServeConfig(max_batch_size=1, max_input_len=64, max_output_len=32)
    prepared = runtime.prepare(target, MODEL, config)

    chunks = list(runtime.generate(prepared, "The capital of France is", max_new_tokens=16))
    assert chunks, "no chunks streamed"
    assert chunks[-1].finished is True
    text = "".join(chunk.text for chunk in chunks)
    assert text.strip(), "streamed only empty deltas"
    # A cumulative-vs-delta mistake shows up as the same prefix repeated.
    assert text.count(text[:8]) == 1 if len(text) >= 8 else True


def test_engine_cache_round_trips_on_this_card(runtime, target, tmp_path):
    info = runtime.probe()
    identity = engine_identity(info, target, MODEL, ServeConfig())
    directory = engine_dir(identity, tmp_path)
    write_manifest(directory, identity)

    ok, why = reusable(directory, identity)
    assert ok is True, why

    # The guard that matters: the same engine directory against a different card.
    other = TargetSpec("nvidia", "gpu", architecture="sm80" if "90" in str(target) else "sm90")
    ok, why = reusable(directory, engine_identity(info, other, MODEL, ServeConfig()))
    assert ok is False
    assert "key" in why
