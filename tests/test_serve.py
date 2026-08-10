"""Engine, sampling and vLLM-launcher tests. No HTTP, no downloads, no FastAPI.

Everything here runs on a plain ``[dev]`` install, which is what the `quality`
CI job has. The HTTP surface is exercised in ``test_serve_integration.py``,
which needs the ``serve`` extra and is marked accordingly.

The model is scripted rather than mocked in the usual sense: a fake runner that
emits a fixed sequence of token ids and a fake tokenizer whose vocabulary is a
handful of string pieces. That is enough to test everything this layer is
actually responsible for -- stop sequences, EOS, capacity refusal, the lock --
without asserting anything about PyTorch.
"""

from __future__ import annotations

import asyncio
import json
import platform
import time
from pathlib import Path

import pytest
import torch

import lm7.serve.vllm as vllm_module
from lm7.cli import _build_parser, _cors_origins
from lm7.errors import UnsupportedModelError
from lm7.serve.cli import serve_plan
from lm7.serve.engine import (
    LM7ServeEngine,
    ServeConfig,
    resolve_model_source,
    select_token,
)
from lm7.serve.validation import unsupported_fields
from lm7.serve.vllm import vllm_argv, vllm_platform
from lm7.targets import parse_target

# id 0 is EOS; the rest are the string pieces a "detokenizer" glues together.
VOCAB = ["", "Hello", " world", "!", " and", " again"]
EOS = 0


class FakeTokenizer:
    """Just enough of a Hugging Face tokenizer for the engine to drive.

    ``chat_template`` is present so the engine takes the same branch it takes for
    an instruct checkpoint; the template itself is trivial because what is being
    tested is that the engine *uses* it, not what it renders.
    """

    eos_token_id = EOS
    chat_template = "fake"

    def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        # One token per word, which makes prompt length predictable in tests.
        count = max(len(prompt.split()), 1)
        return {"input_ids": torch.arange(count, dtype=torch.long).reshape(1, count)}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(VOCAB[token_id] for token_id in token_ids)

    def apply_chat_template(
        self, conversation: list[dict[str, str]], tokenize: bool = False, **kwargs: object
    ) -> str:
        return " ".join(f"<{turn['role']}>{turn['content']}" for turn in conversation)


class FakeState:
    def __init__(self, logits: torch.Tensor, sequence_length: int) -> None:
        self.logits = logits
        self.sequence_length = sequence_length
        self.next_token = logits[:, -1].argmax(dim=-1, keepdim=True)


class ScriptedRunner:
    """A ``GenerationRunner`` stand-in that replays ``script`` one token per call.

    Also records how many generations were in flight at once, which is how the
    ``asyncio.Lock`` is checked: the static KV cache is a single set of buffers,
    so a peak above one would mean two requests writing into the same cache.
    """

    target = parse_target("cpu")
    backend = "eager"
    cache_bytes = 1024

    def __init__(self, script: list[int], delay: float = 0.0) -> None:
        self.script = script
        self.delay = delay
        self.index = 0
        self.resets = 0
        self.in_flight = 0
        self.peak_in_flight = 0

    def reset(self) -> None:
        self.resets += 1
        self.index = 0

    def _logits(self) -> torch.Tensor:
        token_id = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        logits = torch.full((1, 1, len(VOCAB)), -10.0)
        logits[0, 0, token_id] = 10.0
        return logits

    def _busy(self) -> None:
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        if self.delay:
            time.sleep(self.delay)
        self.in_flight -= 1

    def prefill(self, input_ids: torch.Tensor) -> FakeState:
        self._busy()
        return FakeState(self._logits(), int(input_ids.shape[-1]))

    def decode(self, token: torch.Tensor, state: FakeState) -> tuple[torch.Tensor, FakeState]:
        self._busy()
        next_state = FakeState(self._logits(), state.sequence_length + 1)
        return next_state.next_token, next_state


def build_engine(
    script: list[int], *, delay: float = 0.0, max_model_len: int = 64
) -> LM7ServeEngine:
    config = ServeConfig(model="hf://owner/fake", max_model_len=max_model_len)
    return LM7ServeEngine(
        ScriptedRunner(script, delay),
        FakeTokenizer(),
        config,
        model_id="owner/fake",
    )


def collect(
    engine: LM7ServeEngine, prompt: str = "a prompt", **kwargs: object
) -> tuple[str, str, int]:
    kwargs.setdefault("max_tokens", 8)
    kwargs.setdefault("temperature", 0.0)
    return asyncio.run(engine.complete(prompt, **kwargs))  # type: ignore[arg-type]


def stream(engine: LM7ServeEngine, prompt: str = "a prompt", **kwargs: object) -> list[str]:
    kwargs.setdefault("max_tokens", 8)
    kwargs.setdefault("temperature", 0.0)

    async def run() -> list[str]:
        return [
            delta.text
            async for delta in engine.generate(prompt, **kwargs)  # type: ignore[arg-type]
            if not delta.finished
        ]

    return asyncio.run(run())


# -- generation -----------------------------------------------------------


def test_generation_stops_at_eos() -> None:
    engine = build_engine([1, 2, 3, EOS, 4])
    text, reason, tokens = collect(engine)
    assert text == "Hello world!"
    assert reason == "stop"
    assert tokens == 3


def test_generation_stops_at_the_token_budget() -> None:
    engine = build_engine([1, 2, 3, 4, 5])
    text, reason, tokens = collect(engine, max_tokens=2)
    assert text == "Hello world"
    assert reason == "length"
    assert tokens == 2


def test_prefill_alone_answers_a_single_token_request() -> None:
    engine = build_engine([1, 2, 3])
    text, _, tokens = collect(engine, max_tokens=1)
    assert (text, tokens) == ("Hello", 1)


def test_streamed_deltas_reassemble_into_the_completion() -> None:
    engine = build_engine([1, 2, 3, EOS])
    assert "".join(stream(engine)) == "Hello world!"


def test_a_stop_sequence_truncates_and_is_never_streamed() -> None:
    """The stop text must not reach the client, in either response shape.

    This is what the hold-back buffer in `generate` is for: " world" is only
    recognizable once its last character has been decoded, and by then a naive
    implementation has already sent it.
    """
    engine = build_engine([1, 2, 3, 4])
    text, reason, _ = collect(engine, stop=" world")
    assert (text, reason) == ("Hello", "stop")
    assert "".join(stream(engine, stop=" world")) == "Hello"


def test_a_stop_sequence_is_found_across_a_token_boundary() -> None:
    """ "d!" spans the end of ' world' and the whole of '!' -- two tokens."""
    engine = build_engine([1, 2, 3, 4])
    text, reason, _ = collect(engine, stop="d!")
    assert (text, reason) == ("Hello worl", "stop")


def test_a_stop_list_takes_the_earliest_match() -> None:
    engine = build_engine([1, 2, 3, 4])
    text, _, _ = collect(engine, stop=["!", " world"])
    assert text == "Hello"


def test_the_reported_backend_is_the_one_that_compiled_not_the_one_requested() -> None:
    """`--backend auto` is a question; reporting it back would be reporting the question.

    Nothing has compiled until the first request, so until then the requested
    value is the only truthful answer available.
    """
    engine = build_engine([1])
    assert engine.backend == "eager"
    engine.runner.cudagraphs = {"decode": {"backend": "inductor"}}
    assert engine.backend == "inductor"


def test_the_server_reports_whether_the_compile_cost_has_been_paid() -> None:
    """`warm` is what lets a client say "compiling" instead of looking hung."""
    engine = build_engine([1, EOS])
    assert engine.warm is False
    collect(engine)
    assert engine.warm is True


def test_graph_stats_surface_a_token_that_triggered_a_compile() -> None:
    """`steady_frames > 0` is the regression the prefill/decode split prevents.

    It lives in `runner.counters`, where a server would never notice it, so the
    engine lifts it onto `/metrics`.
    """
    engine = build_engine([1, EOS])
    assert engine.graph_stats() == {"prefill_lengths": 0, "steady_frames": 0}

    engine.runner.compiled_prefill_lengths = [7, 13]
    engine.runner.counters = {"steady": {"frames": 2}}
    assert engine.graph_stats() == {"prefill_lengths": 2, "steady_frames": 2}


def test_graph_stats_tolerate_a_runner_without_counters() -> None:
    """`backend="eager"` compiles nothing, so the counters may not exist at all."""
    engine = build_engine([1, EOS])
    engine.runner.counters = None
    assert engine.graph_stats()["steady_frames"] == 0


def test_each_request_resets_the_shared_cache() -> None:
    engine = build_engine([1, 2, EOS])
    collect(engine)
    collect(engine)
    # Once per request from the engine, plus `prefill`'s own reset each time.
    assert engine.runner.resets >= 2


# -- capacity -------------------------------------------------------------


def test_a_prompt_the_static_cache_cannot_hold_is_refused() -> None:
    engine = build_engine([1, 2], max_model_len=8)
    with pytest.raises(ValueError, match="exceeds the 8-token static cache"):
        engine.resolve_budget("one two three four five", 8)


def test_capacity_counts_the_completion_too() -> None:
    engine = build_engine([1, 2], max_model_len=8)
    engine.resolve_budget("one two", 6)
    with pytest.raises(ValueError):
        engine.resolve_budget("one two", 7)


def test_an_explicit_budget_is_never_quietly_narrowed() -> None:
    # The whole reason `None` exists: a caller that asked for 7 and received 6
    # would have been misled, so the ask is refused and the refusal says 6.
    engine = build_engine([1, 2], max_model_len=8)
    with pytest.raises(ValueError, match="Ask for at most 6"):
        engine.resolve_budget("one two", 7)


def test_omitting_the_budget_takes_whatever_the_cache_has_left() -> None:
    engine = build_engine([1, 2], max_model_len=8)
    assert engine.resolve_budget("one two", None)[1] == 6
    # And it shrinks as the conversation grows, instead of becoming impossible.
    assert engine.resolve_budget("one two three four", None)[1] == 4


def test_a_prompt_that_fills_the_cache_is_a_different_failure() -> None:
    # Not "ask for fewer tokens" -- there is no number of tokens that would fit,
    # so the message has to point at the conversation instead.
    engine = build_engine([1, 2], max_model_len=4)
    with pytest.raises(ValueError, match="leaves no room"):
        engine.resolve_budget("one two three four five", None)


# -- concurrency ----------------------------------------------------------


def test_concurrent_requests_never_share_the_kv_cache() -> None:
    """Two overlapping generations must serialize, not interleave.

    The runner owns one static cache; two decode loops running against it would
    write into each other's buffers and return two wrong answers without raising.
    """
    engine = build_engine([1, 2, EOS], delay=0.01)

    async def both() -> list[tuple[str, str, int]]:
        return list(
            await asyncio.gather(
                engine.complete("a prompt", max_tokens=4, temperature=0.0),
                engine.complete("a prompt", max_tokens=4, temperature=0.0),
            )
        )

    results = asyncio.run(both())
    assert engine.runner.peak_in_flight == 1
    assert [text for text, _, _ in results] == ["Hello world", "Hello world"]


def test_the_event_loop_stays_responsive_during_generation() -> None:
    """The whole point of `asyncio.to_thread`: /health answers mid-decode.

    A blocking decode loop would let the ticker run zero times, since nothing
    would yield to the loop between the first token and the last.
    """
    engine = build_engine([1, 2, 3, EOS], delay=0.02)
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.001)
            ticks += 1

    async def run() -> None:
        ticker = asyncio.create_task(tick())
        await engine.complete("a prompt", max_tokens=4, temperature=0.0)
        ticker.cancel()

    asyncio.run(run())
    assert ticks > 0


# -- cancellation ---------------------------------------------------------


def test_a_disconnected_client_stops_the_decode_loop() -> None:
    engine = build_engine([1, 2, 3, 4, 5, 5, 5, 5])
    calls = 0

    async def gone() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    text, reason, _ = collect(engine, max_tokens=8, is_disconnected=gone)
    assert reason == "cancelled"
    assert text == "Hello world"
    # The metrics of a cancelled request are still recorded -- see the `finally`
    # in `generate`.
    assert engine.metrics.requests == 1


# -- token selection ------------------------------------------------------


def test_zero_temperature_is_greedy() -> None:
    logits = torch.tensor([[[0.1, 5.0, 0.2]]])
    assert int(select_token(logits, temperature=0.0, top_p=1.0).item()) == 1


def test_sampling_is_reproducible_for_a_seed() -> None:
    logits = torch.tensor([[[1.0, 1.1, 0.9]]])
    first = [
        int(select_token(logits, temperature=1.0, top_p=1.0, generator=g).item())
        for g in [torch.Generator().manual_seed(7)]
    ]
    second = [
        int(select_token(logits, temperature=1.0, top_p=1.0, generator=g).item())
        for g in [torch.Generator().manual_seed(7)]
    ]
    assert first == second


def test_top_p_keeps_the_token_that_crosses_the_threshold() -> None:
    """A top_p below the largest single probability must still keep that token.

    Dropping everything at or above the cumulative threshold would leave an
    all-zero distribution, and `torch.multinomial` raises on one.
    """
    logits = torch.tensor([[[10.0, 0.0, 0.0]]])
    generator = torch.Generator().manual_seed(0)
    assert int(select_token(logits, temperature=1.0, top_p=0.1, generator=generator).item()) == 0


# -- chat templating ------------------------------------------------------


def test_the_tokenizer_chat_template_is_used_when_there_is_one() -> None:
    engine = build_engine([1])
    prompt = engine.apply_chat_template([{"role": "user", "content": "hi"}])
    assert prompt == "<user>hi"


def test_a_base_model_without_a_template_gets_a_labelled_transcript() -> None:
    engine = build_engine([1])
    engine.tokenizer.chat_template = None
    prompt = engine.apply_chat_template(
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    )
    assert prompt == "system: be brief\nuser: hi\nassistant:"


# -- request validation ---------------------------------------------------


def test_fields_that_would_change_the_answer_are_named() -> None:
    assert unsupported_fields({"n": 4}) == ["n"]
    assert unsupported_fields({"logit_bias": {"5": 10}}) == ["logit_bias"]
    assert set(unsupported_fields({"n": 2, "tools": [{"type": "function"}]})) == {"n", "tools"}


def test_the_defaults_every_openai_sdk_sends_are_not_refused() -> None:
    body = {
        "n": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0.0,
        "logprobs": False,
        "tools": [],
        "tool_choice": "auto",
        "response_format": {"type": "text"},
    }
    assert unsupported_fields(body) == []


# -- vLLM handover --------------------------------------------------------


def test_the_vllm_command_line_carries_the_lm7_flags() -> None:
    config = ServeConfig(
        model="hf://owner/model", target="cpu", backend="vllm", port=9001, max_model_len=4096
    )
    argv = vllm_argv(config)
    assert argv[:3] == ["vllm", "serve", "owner/model"]
    assert argv[argv.index("--port") + 1] == "9001"
    assert argv[argv.index("--max-model-len") + 1] == "4096"
    assert "--dtype" not in argv


def test_an_explicit_dtype_reaches_vllm() -> None:
    config = ServeConfig(model="hf://owner/model", target="cpu", backend="vllm", dtype="bfloat16")
    argv = vllm_argv(config)
    assert argv[argv.index("--dtype") + 1] == "bfloat16"


def test_a_target_vllm_has_no_backend_for_is_refused_rather_than_launched() -> None:
    """vLLM falls back to whatever platform plugin loads, so an unmapped target
    would otherwise start a server on the wrong device and never say so."""
    for unsupported in ("tenstorrent", "intel:npu", "qualcomm:sm8750"):
        with pytest.raises(UnsupportedModelError, match="vLLM has no backend"):
            vllm_platform(parse_target(unsupported))


def test_a_target_vllm_supports_is_translated_to_its_platform_name() -> None:
    """Checked for every mapped vendor, since the map is the whole of the claim.

    The target is patched in rather than resolved, because resolving `nvidia`
    needs an NVIDIA GPU attached to the machine running the test.
    """
    assert vllm_platform(parse_target("nvidia")) == "cuda"
    assert vllm_platform(parse_target("amd")) == "rocm"
    assert vllm_platform(parse_target("cpu")) == "cpu"
    assert vllm_platform(parse_target("tpu")) == "tpu"
    # Apple Silicon via the vllm-metal platform plugin — see docs/serving.md.
    assert vllm_platform(parse_target("apple")) == "metal"


def test_a_missing_vllm_names_the_install_command_rather_than_failing_opaquely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vllm_module, "vllm_executable", lambda: None)
    config = ServeConfig(model="hf://owner/model", target="cpu", backend="vllm")
    with pytest.raises(UnsupportedModelError, match="pip install vllm"):
        vllm_module.serve_with_vllm(config)


def test_vllm_is_looked_for_where_it_is_actually_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importable in LM7's interpreter is not the test that matters.

    The handover is a subprocess, and on Apple Silicon vLLM normally lives in
    vllm-metal's own venv -- deliberately isolated, because vLLM pins a torch
    version. An import check alone reports "not installed" on a machine where
    `vllm serve` runs perfectly well.
    """
    monkeypatch.setattr(vllm_module.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(vllm_module.shutil, "which", lambda _: None)
    monkeypatch.setattr(vllm_module.Path, "exists", lambda _: False)
    assert vllm_module.vllm_executable() is None
    assert vllm_module.vllm_available() is False

    monkeypatch.setattr(vllm_module.shutil, "which", lambda _: "/somewhere/bin/vllm")
    assert vllm_module.vllm_executable() == "/somewhere/bin/vllm"

    monkeypatch.setattr(vllm_module.shutil, "which", lambda _: None)
    monkeypatch.setattr(vllm_module.Path, "exists", lambda _: True)
    # Compared as a path rather than a suffix string: Windows renders the same
    # location with backslashes, and this suite runs there too.
    expected = Path(vllm_module._VLLM_METAL_VENV).expanduser()
    assert Path(vllm_module.vllm_executable()) == expected


def test_a_loopback_server_pins_vllm_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """vLLM's gloo init picks the LAN address and hangs on macOS otherwise.

    No error, no timeout — startup simply stops after "PyTorch device set to:
    mps". Measured here as a >10-minute hang becoming a 130-second startup.
    """
    monkeypatch.delenv("VLLM_HOST_IP", raising=False)
    loopback = ServeConfig(model="hf://owner/model", target="cpu", backend="vllm")
    assert vllm_module.vllm_environment(loopback)["VLLM_HOST_IP"] == "127.0.0.1"


def test_a_server_bound_to_all_interfaces_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real address is required off-loopback, and LM7 cannot guess it."""
    monkeypatch.delenv("VLLM_HOST_IP", raising=False)
    exposed = ServeConfig(model="hf://owner/model", target="cpu", backend="vllm", host="0.0.0.0")
    assert "VLLM_HOST_IP" not in vllm_module.vllm_environment(exposed)


def test_an_explicit_vllm_host_ip_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A multi-node deployment has already said what the address is."""
    monkeypatch.setenv("VLLM_HOST_IP", "10.0.0.7")
    config = ServeConfig(model="hf://owner/model", target="cpu", backend="vllm")
    assert vllm_module.vllm_environment(config)["VLLM_HOST_IP"] == "10.0.0.7"


def _kernel(release: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Report ``release`` as the kernel version, whatever this machine runs."""
    uname = platform.uname()
    monkeypatch.setattr(
        vllm_module.platform, "uname", lambda: uname._replace(release=release), raising=True
    )


def test_a_modern_wsl2_kernel_gets_pinned_memory_switched_back_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without it, vLLM 0.26+ dies with `RuntimeError: UVA is not available`.

    vLLM turns pinned memory off whenever it detects WSL, and its CUDA worker
    allocates a UVA buffer that needs it, so startup fails before a model loads.
    Measured on WSL2 6.18.33.2 with an RTX 4070 SUPER.
    """
    monkeypatch.delenv("VLLM_WSL2_ENABLE_PIN_MEMORY", raising=False)
    _kernel("6.18.33.2-microsoft-standard-WSL2", monkeypatch)
    config = ServeConfig(model="hf://owner/model", target="cpu", backend="vllm")
    assert vllm_module.vllm_environment(config)["VLLM_WSL2_ENABLE_PIN_MEMORY"] == "1"


def test_an_old_wsl2_kernel_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Below 4.19.121 the default is a real limitation, not a cautious one."""
    monkeypatch.delenv("VLLM_WSL2_ENABLE_PIN_MEMORY", raising=False)
    _kernel("4.19.104-microsoft-standard", monkeypatch)
    config = ServeConfig(model="hf://owner/model", target="cpu", backend="vllm")
    assert "VLLM_WSL2_ENABLE_PIN_MEMORY" not in vllm_module.vllm_environment(config)


def test_a_kernel_that_is_not_wsl_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_WSL2_ENABLE_PIN_MEMORY", raising=False)
    _kernel("6.8.0-51-generic", monkeypatch)
    config = ServeConfig(model="hf://owner/model", target="cpu", backend="vllm")
    assert "VLLM_WSL2_ENABLE_PIN_MEMORY" not in vllm_module.vllm_environment(config)


def test_an_explicit_pin_memory_setting_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Including switching it off, which is the reason to read it at all."""
    monkeypatch.setenv("VLLM_WSL2_ENABLE_PIN_MEMORY", "0")
    _kernel("6.18.33.2-microsoft-standard-WSL2", monkeypatch)
    config = ServeConfig(model="hf://owner/model", target="cpu", backend="vllm")
    assert vllm_module.vllm_environment(config)["VLLM_WSL2_ENABLE_PIN_MEMORY"] == "0"


def test_passthrough_arguments_reach_vllm_last(monkeypatch: pytest.MonkeyPatch) -> None:
    """Last, so a caller spelling a flag out beats what LM7 translated.

    `--gpu-memory-utilization` is the case that motivated this: vLLM's default
    asks for more of a 12 GiB card than a desktop leaves free, and LM7 models no
    flag for it.
    """
    config = ServeConfig(
        model="hf://owner/model",
        target="cpu",
        backend="vllm",
        max_model_len=4096,
        vllm_args=("--gpu-memory-utilization", "0.8", "--max-model-len", "2048"),
    )
    argv = vllm_argv(config)
    assert argv[-4:] == ["--gpu-memory-utilization", "0.8", "--max-model-len", "2048"]
    # Spelled twice, and argparse takes the last -- which is the caller's.
    assert argv.index("--max-model-len") < argv.index("--gpu-memory-utilization")


def test_passthrough_arguments_are_refused_by_lm7s_own_server() -> None:
    """Ignoring them would start a server that is not the one asked for."""
    from lm7.serve.cli import serve_model

    config = ServeConfig(model="hf://owner/model", target="cpu", vllm_args=("--enforce-eager",))
    with pytest.raises(UnsupportedModelError, match="--vllm-arg"):
        serve_model(config)


def test_the_parser_collects_repeated_vllm_arguments() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "model",
            "serve",
            "hf://owner/model",
            "--backend",
            "vllm",
            "--vllm-arg=--gpu-memory-utilization",
            "--vllm-arg",
            "0.8",
        ]
    )
    assert args.vllm_args == ["--gpu-memory-utilization", "0.8"]


# -- the plan -------------------------------------------------------------


def test_the_plan_describes_the_lm7_server_without_loading_anything() -> None:
    plan = serve_plan(ServeConfig(model="hf://owner/model", target="cpu", port=1234))
    assert plan["runtime"] == "lm7"
    assert plan["model"] == "owner/model"
    assert "/v1/chat/completions" in plan["endpoints"]


def test_the_plan_shows_the_command_vllm_would_be_given() -> None:
    plan = serve_plan(ServeConfig(model="hf://owner/model", target="cpu", backend="vllm"))
    assert plan["runtime"] == "vllm"
    assert plan["argv"][:2] == ["vllm", "serve"]
    assert isinstance(plan["vllm_installed"], bool)
    # "Not installed" is usually "installed in a different environment", so the
    # plan names which vllm was found rather than only whether one was.
    assert "vllm_executable" in plan
    assert plan["environment"].get("VLLM_HOST_IP") == "127.0.0.1"


# -- where the model comes from -------------------------------------------


def _saved_model_dir(tmp_path: Path, name: str = "checkpoint") -> Path:
    """A directory shaped like one `save_pretrained` wrote, without weights.

    `resolve_model_source` decides what to hand `from_pretrained`; it never loads
    anything, so `config.json` existing is the whole fixture.
    """
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    return directory


def test_a_local_directory_is_served_as_itself(tmp_path: Path) -> None:
    directory = _saved_model_dir(tmp_path)
    assert resolve_model_source(str(directory)) == str(directory.resolve())


def test_a_relative_local_directory_resolves_to_an_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _saved_model_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    # The served id has to survive the server changing directory later, and a
    # client reading /v1/models cannot resolve "./checkpoint" against our cwd.
    assert resolve_model_source("./checkpoint") == str(directory.resolve())


def test_a_hub_uri_still_resolves_to_the_model_id() -> None:
    assert resolve_model_source("hf://owner/model") == "owner/model"


def test_a_directory_without_a_config_says_so_rather_than_failing_in_transformers(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "not-a-model"
    empty.mkdir()
    with pytest.raises(UnsupportedModelError, match="does not contain config.json"):
        resolve_model_source(str(empty))


def test_a_path_that_does_not_exist_is_not_reported_as_a_bad_hub_uri(tmp_path: Path) -> None:
    # The Hub error ("expected a Hugging Face URI") sends someone who typed a
    # path to the wrong place entirely.
    with pytest.raises(UnsupportedModelError, match="No such directory"):
        resolve_model_source(str(tmp_path / "missing"))


def test_a_file_is_refused_with_the_reason(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_text("")
    with pytest.raises(UnsupportedModelError, match="is a file, not a directory"):
        resolve_model_source(str(weights))


def test_a_bare_name_is_still_refused_because_a_hub_id_needs_its_prefix() -> None:
    with pytest.raises(UnsupportedModelError, match="hf://"):
        resolve_model_source("owner/model")


def test_the_plan_reports_a_local_directory(tmp_path: Path) -> None:
    directory = _saved_model_dir(tmp_path)
    plan = serve_plan(ServeConfig(model=str(directory), target="cpu"))
    assert plan["model"] == str(directory.resolve())


def test_vllm_is_handed_the_local_directory(tmp_path: Path) -> None:
    directory = _saved_model_dir(tmp_path)
    argv = vllm_argv(ServeConfig(model=str(directory), target="cpu", backend="vllm"))
    assert argv[:3] == ["vllm", "serve", str(directory.resolve())]


# -- deployment flags ------------------------------------------------------


def test_the_two_cache_flags_are_one_setting() -> None:
    # Two spellings sharing a dest, so a user who reaches for vLLM's name and a
    # user who reaches for compile_generation's name configure the same cache.
    parser = _build_parser()
    long_form = parser.parse_args(["model", "serve", "hf://owner/model", "--max-model-len", "77"])
    alias = parser.parse_args(["model", "serve", "hf://owner/model", "--max-sequence-length", "77"])
    assert long_form.max_model_len == alias.max_model_len == 77


def test_the_cache_default_is_reported_by_the_parser_not_only_the_dataclass() -> None:
    parser = _build_parser()
    args = parser.parse_args(["model", "serve", "hf://owner/model"])
    assert args.max_model_len == ServeConfig(model="x").max_model_len == 4096


def test_cors_origins_are_split_and_stripped() -> None:
    assert _cors_origins("*") == ("*",)
    assert _cors_origins("http://localhost:3000, http://localhost:8080") == (
        "http://localhost:3000",
        "http://localhost:8080",
    )
    # A trailing comma is a typo, not a request for an empty origin.
    assert _cors_origins("http://localhost:3000,") == ("http://localhost:3000",)


def test_disabling_cors_is_expressible() -> None:
    # "" has to mean "no origins", not "fall back to the wildcard default", or
    # there is no way to turn the default off.
    assert _cors_origins("") == ()


def test_the_plan_never_prints_the_api_key() -> None:
    config = ServeConfig(model="hf://owner/model", target="cpu", api_key="s3cret")
    plan = serve_plan(config)
    assert plan["api_key"] is True
    assert "s3cret" not in json.dumps(plan)
    assert "s3cret" not in json.dumps(config.to_dict())


def test_the_plan_reports_quantization_and_origins() -> None:
    plan = serve_plan(
        ServeConfig(
            model="hf://owner/model",
            target="cpu",
            quantize="int8",
            cors_origins=("http://localhost:3000",),
        )
    )
    assert plan["quantize"] == "int8"
    assert plan["cors_origins"] == ["http://localhost:3000"]


# -- the chat page against another server ---------------------------------


def test_the_page_defaults_to_the_server_that_sent_it() -> None:
    """LM7 serves the page and the API from one origin, so paths stay relative."""
    from lm7.serve.ui import render

    assert 'const API = "";' in render()


def test_the_page_can_be_pointed_at_another_server() -> None:
    """`--ui-port` beside `--backend vllm`: vLLM owns the API port and has no page."""
    from lm7.serve.ui import render

    page = render("http://127.0.0.1:8200/")
    assert 'const API = "http://127.0.0.1:8200";' in page
    assert "__LM7_API_BASE__" not in page


def test_a_page_pointed_elsewhere_is_still_self_contained() -> None:
    """The API base is the one outward reference, and it is a local server."""
    from lm7.serve.ui import render

    page = render("http://127.0.0.1:8200")
    assert page.count("http://") == 1
    for marker in ("https://", "//cdn", "integrity=", "@import"):
        assert marker not in page


def test_the_page_server_serves_only_the_page() -> None:
    import urllib.error
    import urllib.request

    from lm7.serve.ui import serve_page

    server = serve_page(0, "http://127.0.0.1:8200")
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            body = response.read().decode()
        assert response.status == 200
        assert "<title>lm7 serve</title>" in body
        assert 'const API = "http://127.0.0.1:8200";' in body
        # It hands out one file; the API is somewhere else entirely.
        with pytest.raises(urllib.error.HTTPError, match="404"):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
    finally:
        server.shutdown()
        server.server_close()


def test_the_ui_port_is_refused_where_the_page_is_already_served() -> None:
    """LM7's own server has the page at `/`; a second copy would be a puzzle."""
    from lm7.serve.cli import serve_model

    config = ServeConfig(model="hf://owner/model", target="cpu", ui_port=8201)
    with pytest.raises(UnsupportedModelError, match="serves the chat page itself"):
        serve_model(config)


def test_the_plan_names_the_chat_page_port() -> None:
    plan = serve_plan(
        ServeConfig(model="hf://owner/model", target="cpu", backend="vllm", ui_port=8201)
    )
    assert plan["ui_port"] == 8201


def test_quantizing_a_local_directory_is_refused_by_id_not_by_path(tmp_path: Path) -> None:
    # The gate keys on a Hugging Face id; a directory has none, and the refusal
    # has to say that rather than print a path where an id was promised.
    directory = _saved_model_dir(tmp_path)
    config = ServeConfig(model=str(directory), target="cpu", quantize="int8")
    with pytest.raises(UnsupportedModelError, match="not available for a local directory"):
        LM7ServeEngine.load(config)


def test_serving_a_local_directory_unquantized_is_not_blocked_by_that(tmp_path: Path) -> None:
    # The refusal must be about quantization only: the same directory with no
    # --quantize has to get past this check and fail later, on the real load.
    directory = _saved_model_dir(tmp_path)
    config = ServeConfig(model=str(directory), target="cpu")
    with pytest.raises(Exception) as caught:
        LM7ServeEngine.load(config)
    assert "not available for a local directory" not in str(caught.value)


def test_a_quantized_load_asks_for_the_quantized_compute_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`dtype="auto"` means BF16 once a weight-only mode is in play, not FP16.

    The two differ only on NVIDIA -- on CPU both resolve to FP32 -- and the gate
    refuses an *explicit* `--dtype float16` alongside `--quantize`, so `auto`
    resolved without the mode is the one way into a combination nothing rejects.
    Served that way on an RTX 4070 SUPER (Ada `sm89`), INT8 weights under FP16
    compute produced NaN logits: every token was an argmax over NaN, and
    SmolLM2-135M-Instruct -- whose EOS is not token 0 -- ran to its full budget
    and returned an empty string with `finish_reason: "length"` and no error.

    Checked here rather than in `test_serve_load_integration.py` because it needs
    an NVIDIA target, which CI has nowhere; the target is faked and only the
    dtype handed to `from_pretrained` is asserted.
    """
    import lm7.generation
    import lm7.huggingface
    import lm7.serve.engine as engine_module

    requested: dict[str, object] = {}

    class FakeTransformers:
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model_id: str) -> FakeTokenizer:
                return FakeTokenizer()

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(model_id: str, dtype: torch.dtype) -> object:
                requested["dtype"] = dtype
                return type("FakeModel", (), {"eval": lambda self: self})()

    monkeypatch.setattr(engine_module, "resolve_target", lambda _: parse_target("nvidia:sm89"))
    monkeypatch.setattr(engine_module, "_load_transformers", lambda: FakeTransformers)
    monkeypatch.setattr(lm7.huggingface, "_apply_quantization", lambda *_: (0.0, 0))
    monkeypatch.setattr(lm7.generation, "compile_generation", lambda *_, **__: ScriptedRunner([1]))

    LM7ServeEngine.load(
        ServeConfig(model="hf://HuggingFaceTB/SmolLM2-135M-Instruct", quantize="int8")
    )
    assert requested["dtype"] is torch.bfloat16
