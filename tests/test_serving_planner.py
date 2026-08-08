from __future__ import annotations

import json

import pytest

from lm7.backends.base import Support
from lm7.cli import main
from lm7.errors import BackendUnavailableError
from lm7.serving.base import Capabilities, ServeRequest, ServerHandle, unmet_capabilities
from lm7.serving.planner import plan_serving
from lm7.serving.registry import RuntimeRegistry
from lm7.targets import TargetSpec

CPU = TargetSpec("cpu", "cpu")


class _FakeRuntime:
    def __init__(self, name: str, support: Support, capabilities: Capabilities | None = None):
        self.name = name
        self._support = support
        self._capabilities = capabilities or Capabilities()

    def probe(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def capabilities(self) -> Capabilities:
        return self._capabilities

    def supports(self, request: ServeRequest) -> Support:
        return self._support

    def describe(self, request: ServeRequest):  # type: ignore[no-untyped-def]
        return {"runtime": self.name}

    def launch(self, request: ServeRequest) -> ServerHandle:
        return ServerHandle(self.name, "http://test", request.target)


def _registry(*runtimes: _FakeRuntime) -> RuntimeRegistry:
    registry = RuntimeRegistry()
    for runtime in runtimes:
        registry.register(runtime)
    return registry


def _request(**kwargs: object) -> ServeRequest:
    return ServeRequest(model="hf://owner/model", target=CPU, **kwargs)  # type: ignore[arg-type]


def test_highest_priority_supported_runtime_wins() -> None:
    registry = _registry(
        _FakeRuntime("slow", Support(True, "ok", priority=0)),
        _FakeRuntime("fast", Support(True, "ok", priority=90)),
    )
    selected, plan = plan_serving(_request(), "auto", registry)
    assert selected.name == "fast"
    assert plan.selected == "fast"
    assert {c.runtime for c in plan.candidates} == {"fast", "slow"}


def test_explicitly_requested_runtime_beats_priority() -> None:
    registry = _registry(
        _FakeRuntime("slow", Support(True, "ok", priority=0)),
        _FakeRuntime("fast", Support(True, "ok", priority=90)),
    )
    selected, _ = plan_serving(_request(), "slow", registry)
    assert selected.name == "slow"


def test_requesting_an_unavailable_runtime_reports_its_reason() -> None:
    registry = _registry(_FakeRuntime("vllm", Support(False, "vLLM is not installed.")))
    with pytest.raises(BackendUnavailableError, match="vLLM is not installed"):
        plan_serving(_request(), "vllm", registry)


def test_requesting_an_unregistered_runtime_lists_what_exists() -> None:
    registry = _registry(_FakeRuntime("eager", Support(True, "ok")))
    with pytest.raises(BackendUnavailableError, match="Available: eager"):
        plan_serving(_request(), "sglang", registry)


def test_no_supported_runtime_reports_every_reason() -> None:
    """The reasons are the message. "Nothing works" is not an actionable error."""
    registry = _registry(
        _FakeRuntime("eager", Support(False, "does not implement continuous_batching")),
        _FakeRuntime("vllm", Support(False, "vLLM is not installed")),
    )
    with pytest.raises(BackendUnavailableError) as excinfo:
        plan_serving(_request(), "auto", registry)
    message = str(excinfo.value)
    assert "does not implement continuous_batching" in message
    assert "vLLM is not installed" in message


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, ()),
        ({"prefix_caching": True}, ("prefix_caching",)),
        ({"lora_adapters": ("a=/tmp/a",)}, ("lora",)),
        ({"speculative_model": "hf://owner/draft"}, ("speculative_decoding",)),
        ({"max_batched_tokens": 2048}, ("chunked_prefill",)),
        ({"max_num_seqs": 8}, ("continuous_batching",)),
    ],
)
def test_requested_capabilities_track_the_constraints_asked_for(
    kwargs: dict[str, object], expected: tuple[str, ...]
) -> None:
    assert _request(**kwargs).requested_capabilities() == expected


def test_a_request_that_asks_for_nothing_needs_no_capabilities() -> None:
    """Otherwise the reference runtime could never serve anything at all."""
    assert unmet_capabilities(_request(), Capabilities()) == ()


def test_unmet_capabilities_names_only_what_is_missing() -> None:
    request = _request(prefix_caching=True, max_num_seqs=4)
    capabilities = Capabilities(continuous_batching=True)
    assert unmet_capabilities(request, capabilities) == ("prefix_caching",)


def test_cli_runtimes_lists_capabilities(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["runtimes", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    names = {runtime["name"] for runtime in data["runtimes"]}
    assert {"eager", "vllm"} <= names
    eager = next(r for r in data["runtimes"] if r["name"] == "eager")
    assert eager["capabilities"]["streaming"] is True
    assert eager["capabilities"]["paged_kv_cache"] is False


def test_cli_serve_explain_reports_the_plan(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["serve", "hf://owner/model", "--target", "cpu", "--explain", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert data["selected_runtime"] == "eager"
    assert data["resolved_config"]["model"] == "owner/model"


def test_cli_serve_explain_still_prints_candidates_when_nothing_fits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The failing plan is the one whose candidate table matters most."""
    exit_code = main(
        [
            "serve",
            "hf://owner/model",
            "--target",
            "cpu",
            "--max-num-seqs",
            "8",
            "--explain",
            "--json",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert data["selected_runtime"] is None
    assert len(data["candidates"]) >= 2
    eager = next(c for c in data["candidates"] if c["runtime"] == "eager")
    assert "continuous_batching" in eager["reason"]
