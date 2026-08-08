"""Mocked tests for the TensorRT-LLM runtime: the LM7-owned half.

Everything here runs without TensorRT-LLM installed, because everything here is
LM7's responsibility -- dependency reporting, target and configuration
validation, and engine cache identity. The runtime's own half (kernels, paged KV
cache, scheduler, decode loop) is exercised in
`tests/test_tensorrt_llm_integration.py`, which skips itself when the package is
absent.
"""

from __future__ import annotations

import json

import pytest

from lm7.runtimes import (
    ServeConfig,
    engine_dir,
    engine_identity,
    inspect_runtimes,
    read_manifest,
    registry,
    reusable,
    write_manifest,
)
from lm7.runtimes.base import RuntimeInfo
from lm7.runtimes.tensorrt_llm import TensorRTLLMRuntime
from lm7.targets import TargetSpec


def nvidia(architecture: str | None) -> TargetSpec:
    return TargetSpec("nvidia", "gpu", architecture=architecture)


def info(**overrides) -> RuntimeInfo:
    base = {
        "name": "tensorrt-llm",
        "version": "1.2.1",
        "available": True,
        "reason": "installed",
        "pinned": {"tensorrt_llm": "1.2.1", "torch": "2.9.1"},
    }
    return RuntimeInfo(**{**base, **overrides})


def test_probe_without_the_package_names_the_install_and_the_conflict():
    """A serving runtime's dependency set is large enough that "unavailable" on
    its own is unactionable, and this one cannot share an environment with the
    rest of the repo -- so the message has to say both."""
    probe = TensorRTLLMRuntime().probe()
    if probe.available:  # pragma: no cover - only on a box with it installed
        pytest.skip("TensorRT-LLM is installed here")
    assert "pypi.nvidia.com" in probe.reason
    assert "own venv" in probe.reason
    assert "torch" in probe.reason and "transformers" in probe.reason
    # Reported even when unavailable, so `doctor` on a laptop still says what an
    # engine would be keyed on.
    assert "tensorrt_llm" in probe.pinned


@pytest.mark.parametrize(
    ("target", "config", "expected"),
    [
        (TargetSpec("cpu", "cpu"), ServeConfig(), "NVIDIA only"),
        (nvidia("sm75"), ServeConfig(), "Ampere"),
        (nvidia("sm90"), ServeConfig(dtype="float32"), "dtype must be"),
        (nvidia("sm90"), ServeConfig(quantization="nvfp4"), "quantization must be"),
        (nvidia("sm80"), ServeConfig(quantization="fp8"), "Ada (sm89)"),
        (nvidia("sm90"), ServeConfig(kv_cache_free_gpu_memory_fraction=0.0), "must be in (0, 1]"),
        (nvidia("sm90"), ServeConfig(kv_cache_free_gpu_memory_fraction=1.5), "must be in (0, 1]"),
    ],
)
def test_supports_refuses_with_a_reason(target, config, expected):
    support = TensorRTLLMRuntime().supports(target, "some/model", config)
    assert support.supported is False
    assert expected in support.reason


@pytest.mark.parametrize("architecture", ["sm80", "sm89", "sm90", "sm120"])
def test_supports_accepts_ampere_and_newer(architecture):
    support = TensorRTLLMRuntime().supports(nvidia(architecture), "some/model", ServeConfig())
    assert support.supported is True


def test_fp8_is_admitted_from_ada_upward():
    runtime = TensorRTLLMRuntime()
    config = ServeConfig(quantization="fp8")
    assert runtime.supports(nvidia("sm89"), "m", config).supported is True
    assert runtime.supports(nvidia("sm90"), "m", config).supported is True


def test_an_unqualified_nvidia_target_is_not_gated_on_capability():
    """`nvidia` has no architecture until it resolves against hardware, so
    refusing it here would reject the common case."""
    support = TensorRTLLMRuntime().supports(nvidia(None), "m", ServeConfig(quantization="fp8"))
    assert support.supported is True


def test_engine_identity_changes_with_anything_an_engine_is_pinned_to():
    """A TensorRT-LLM engine is valid only for the architecture, versions and
    shape bounds it was built with, so each of those has to move the key."""
    base = engine_identity(info(), nvidia("sm90"), "some/model", ServeConfig())
    assert base.key() == engine_identity(info(), nvidia("sm90"), "some/model", ServeConfig()).key()

    different = [
        engine_identity(info(), nvidia("sm80"), "some/model", ServeConfig()),
        engine_identity(info(), nvidia("sm90"), "other/model", ServeConfig()),
        engine_identity(info(), nvidia("sm90"), "some/model", ServeConfig(dtype="float16")),
        engine_identity(info(), nvidia("sm90"), "some/model", ServeConfig(max_batch_size=16)),
        engine_identity(info(), nvidia("sm90"), "some/model", ServeConfig(quantization="fp8")),
        engine_identity(
            info(pinned={"tensorrt_llm": "1.1.0"}), nvidia("sm90"), "some/model", ServeConfig()
        ),
    ]
    assert len({item.key() for item in different}) == len(different)
    assert base.key() not in {item.key() for item in different}


def test_engine_dir_is_namespaced_by_runtime_and_key(tmp_path):
    identity = engine_identity(info(), nvidia("sm90"), "some/model", ServeConfig())
    directory = engine_dir(identity, tmp_path)
    assert directory.parent == tmp_path
    assert directory.name.startswith("tensorrt-llm-")
    assert identity.key() in directory.name


def test_reusable_refuses_an_empty_directory_with_a_reason(tmp_path):
    identity = engine_identity(info(), nvidia("sm90"), "some/model", ServeConfig())
    ok, why = reusable(tmp_path, identity)
    assert ok is False
    assert "lm7-engine.json" in why


def test_reusable_accepts_a_manifest_it_wrote(tmp_path):
    identity = engine_identity(info(), nvidia("sm90"), "some/model", ServeConfig())
    directory = engine_dir(identity, tmp_path)
    write_manifest(directory, identity)

    ok, why = reusable(directory, identity)
    assert ok is True, why
    assert read_manifest(directory)["architecture"] == "sm90"


def test_reusable_refuses_a_manifest_from_another_architecture(tmp_path):
    """The failure this exists to prevent: an engine built on one card reused on
    another, which fails deep inside the runtime if it fails at all."""
    built = engine_identity(info(), nvidia("sm90"), "some/model", ServeConfig())
    directory = engine_dir(built, tmp_path)
    write_manifest(directory, built)

    wanted = engine_identity(info(), nvidia("sm80"), "some/model", ServeConfig())
    ok, why = reusable(directory, wanted)
    assert ok is False
    assert wanted.key() in why


def test_reusable_survives_a_corrupt_manifest(tmp_path):
    identity = engine_identity(info(), nvidia("sm90"), "some/model", ServeConfig())
    directory = engine_dir(identity, tmp_path)
    directory.mkdir(parents=True)
    (directory / "lm7-engine.json").write_text("{not json", encoding="utf-8")

    ok, why = reusable(directory, identity)
    assert ok is False
    assert "lm7-engine.json" in why


def test_manifest_is_json_and_records_the_config(tmp_path):
    identity = engine_identity(
        info(), nvidia("sm90"), "some/model", ServeConfig(max_batch_size=4, quantization="fp8")
    )
    path = write_manifest(engine_dir(identity, tmp_path), identity)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["config"]["max_batch_size"] == 4
    assert written["config"]["quantization"] == "fp8"
    assert written["schema_version"] == 1


def test_registry_names_what_is_registered_when_asked_for_something_else():
    with pytest.raises(KeyError, match="tensorrt-llm"):
        registry.get("vllm")


def test_inspect_runtimes_reports_availability():
    reported = inspect_runtimes()
    assert [item["name"] for item in reported] == ["tensorrt-llm"]
    assert isinstance(reported[0]["available"], bool)
    assert reported[0]["reason"]
