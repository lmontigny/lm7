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
import time

import pytest
import torch

import lm7.serve.vllm as vllm_module
from lm7.errors import UnsupportedModelError
from lm7.serve.cli import serve_plan
from lm7.serve.engine import LM7ServeEngine, ServeConfig, select_token
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
        engine.check_capacity("one two three four five", 8)


def test_capacity_counts_the_completion_too() -> None:
    engine = build_engine([1, 2], max_model_len=8)
    engine.check_capacity("one two", 6)
    with pytest.raises(ValueError):
        engine.check_capacity("one two", 7)


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
    with pytest.raises(UnsupportedModelError, match="vLLM has no backend"):
        vllm_platform(parse_target("apple"))


def test_a_target_vllm_supports_is_translated_to_its_platform_name() -> None:
    """Checked for every mapped vendor, since the map is the whole of the claim.

    The target is patched in rather than resolved, because resolving `nvidia`
    needs an NVIDIA GPU attached to the machine running the test.
    """
    assert vllm_platform(parse_target("nvidia")) == "cuda"
    assert vllm_platform(parse_target("amd")) == "rocm"
    assert vllm_platform(parse_target("cpu")) == "cpu"
    assert vllm_platform(parse_target("tpu")) == "tpu"


def test_a_missing_vllm_names_the_install_command_rather_than_failing_opaquely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vllm_module, "vllm_available", lambda: False)
    config = ServeConfig(model="hf://owner/model", target="cpu", backend="vllm")
    with pytest.raises(UnsupportedModelError, match="pip install vllm"):
        vllm_module.serve_with_vllm(config)


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
