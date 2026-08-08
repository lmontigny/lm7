"""The vLLM runtime's translation layer, exercised without vLLM installed.

``serve_argv`` is deliberately free of vLLM imports so that the mapping LM7
owns can be checked on any machine. The launch path is not covered here and is
not covered anywhere yet -- see docs/serving.md.
"""

from __future__ import annotations

import json

import pytest

from lm7.serving.base import ServeRequest
from lm7.serving.runtimes.vllm import VLLMServingRuntime, serve_argv
from lm7.targets import TargetSpec

CPU = TargetSpec("cpu", "cpu")
NVIDIA = TargetSpec("nvidia", "gpu", architecture="sm90")


def _request(**kwargs: object) -> ServeRequest:
    return ServeRequest(model="hf://owner/model", target=NVIDIA, **kwargs)  # type: ignore[arg-type]


def _paired(argv: list[str]) -> dict[str, str]:
    return {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}


def test_argv_carries_the_model_and_the_core_constraints() -> None:
    argv = serve_argv(_request(max_model_len=8192, max_num_seqs=64, port=9000))
    paired = _paired(argv)
    assert paired["--model"] == "owner/model"
    assert paired["--max-model-len"] == "8192"
    assert paired["--max-num-seqs"] == "64"
    assert paired["--port"] == "9000"


def test_optional_constraints_are_absent_when_not_asked_for() -> None:
    """An unset constraint must not become a flag; vLLM's own default wins."""
    argv = serve_argv(_request())
    assert "--gpu-memory-utilization" not in argv
    assert "--max-num-batched-tokens" not in argv
    assert "--enable-prefix-caching" not in argv
    assert "--enable-lora" not in argv
    assert "--speculative-config" not in argv


def test_chunked_prefill_and_memory_fraction_are_passed_through() -> None:
    argv = serve_argv(_request(max_batched_tokens=4096, kv_cache_fraction=0.85))
    paired = _paired(argv)
    assert paired["--max-num-batched-tokens"] == "4096"
    assert paired["--gpu-memory-utilization"] == "0.85"


def test_prefix_caching_is_a_bare_flag() -> None:
    argv = serve_argv(_request(prefix_caching=True))
    assert "--enable-prefix-caching" in argv
    assert argv[argv.index("--enable-prefix-caching") + 1 :][:1] != ["True"]


def test_lora_adapters_enable_lora_and_list_the_modules() -> None:
    argv = serve_argv(_request(lora_adapters=("a=/models/a", "b=/models/b")))
    assert "--enable-lora" in argv
    index = argv.index("--lora-modules")
    assert argv[index + 1 : index + 3] == ["a=/models/a", "b=/models/b"]


def test_speculative_model_becomes_a_json_config() -> None:
    argv = serve_argv(_request(speculative_model="hf://owner/draft"))
    payload = json.loads(argv[argv.index("--speculative-config") + 1])
    assert payload == {"model": "hf://owner/draft"}


def test_extra_options_become_flags_with_dashes() -> None:
    argv = serve_argv(_request(extra={"swap_space": 8, "enforce_eager": True}))
    assert _paired(argv)["--swap-space"] == "8"
    assert "--enforce-eager" in argv


def test_extra_options_that_are_false_are_dropped() -> None:
    argv = serve_argv(_request(extra={"enforce_eager": False, "seed": None}))
    assert "--enforce-eager" not in argv
    assert "--seed" not in argv


def test_a_non_huggingface_model_uri_is_rejected() -> None:
    from lm7.errors import UnsupportedModelError

    with pytest.raises(UnsupportedModelError, match="hf://"):
        serve_argv(ServeRequest(model="/local/path", target=NVIDIA))


def test_probe_reports_the_install_instruction_when_absent() -> None:
    runtime = VLLMServingRuntime()
    info = runtime.probe()
    if info.available:
        pytest.skip("vLLM is installed; this covers the absent case.")
    assert "pip install vllm" in info.reason
    assert runtime.supports(_request()).supported is False


def test_capabilities_claim_the_full_serving_feature_set() -> None:
    capabilities = VLLMServingRuntime().capabilities()
    assert capabilities.continuous_batching
    assert capabilities.paged_kv_cache
    assert capabilities.prefix_caching
    assert capabilities.chunked_prefill


def test_describe_reports_argv_as_unvalidated_without_vllm() -> None:
    runtime = VLLMServingRuntime()
    if runtime.probe().available:
        pytest.skip("vLLM is installed; this covers the absent case.")
    described = runtime.describe(_request())
    assert described["validated"] is False
    assert "--model" in described["argv"]


def test_unsupported_vendors_are_declined_by_name() -> None:
    runtime = VLLMServingRuntime()
    if not runtime.probe().available:
        pytest.skip("vLLM is not installed; vendor gating is unreachable.")
    request = ServeRequest(model="hf://owner/model", target=TargetSpec("apple", "gpu"))
    support = runtime.supports(request)
    assert support.supported is False
    assert "apple" in support.reason


@pytest.mark.vllm
def test_argv_is_accepted_by_vllms_own_parser() -> None:
    """The anti-drift check: vLLM must accept what LM7 produced.

    Modelled on Ray Serve LLM's config-congruence test. It needs vLLM importable
    but no GPU, because parsing happens before any device is touched.
    """
    pytest.importorskip("vllm")
    from lm7.serving.runtimes.vllm import build_namespace

    args = build_namespace(_request(max_model_len=4096, max_num_seqs=16, prefix_caching=True))
    assert args.max_model_len == 4096
    assert args.max_num_seqs == 16
