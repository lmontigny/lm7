"""A live round trip against LM7's reference runtime.

Marked ``serve`` because it downloads a small model and binds a socket. It runs
on an ordinary CPU runner, which is the whole reason the reference runtime
exists: the HTTP contract is checked somewhere, even though no third-party
serving engine installs in CI.
"""

from __future__ import annotations

import json
import time

import pytest

import lm7

pytestmark = pytest.mark.serve

MODEL = "hf://HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def server():  # type: ignore[no-untyped-def]
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    pytest.importorskip("transformers")
    # Port 0 lets the OS pick, so a developer already serving on 8000 does not
    # make this fail with a bind error that looks like a serving bug.
    handle = lm7.serve(MODEL, target="cpu", port=0, max_model_len=256)
    yield handle
    handle.stop()


@pytest.fixture(scope="module")
def client(server):  # type: ignore[no-untyped-def]
    httpx = pytest.importorskip("httpx")
    with httpx.Client(base_url=server.base_url, timeout=300) as client:
        yield client


def test_health_and_model_listing(client) -> None:  # type: ignore[no-untyped-def]
    assert client.get("/health").json() == {"status": "ok"}
    listed = client.get("/v1/models").json()
    assert listed["data"][0]["id"] == "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_chat_completion_returns_openai_shape(client) -> None:  # type: ignore[no-untyped-def]
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 8},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]


def test_text_completion_returns_openai_shape(client) -> None:  # type: ignore[no-untyped-def]
    response = client.post(
        "/v1/completions", json={"prompt": "The capital of France is", "max_tokens": 6}
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"]


def test_streaming_emits_deltas_then_done(client) -> None:  # type: ignore[no-untyped-def]
    events = []
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Count to three."}],
            "max_tokens": 8,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(line.removeprefix("data: "))

    assert events[-1] == "[DONE]"
    payloads = [json.loads(event) for event in events[:-1]]
    assert all(payload["object"] == "chat.completion.chunk" for payload in payloads)
    assert "".join(p["choices"][0]["delta"].get("content", "") for p in payloads)
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_a_bad_request_is_rejected_rather_than_served(client) -> None:  # type: ignore[no-untyped-def]
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400
    assert client.post("/v1/completions", json={"prompt": 7}).status_code == 400


def test_a_prompt_longer_than_the_cache_is_refused(client) -> None:  # type: ignore[no-untyped-def]
    """The static cache cannot grow, so this has to fail before it decodes."""
    response = client.post("/v1/completions", json={"prompt": "word " * 200, "max_tokens": 200})
    assert response.status_code == 400
    assert "max-model-len" in response.json()["detail"]


def test_abandoning_a_stream_stops_the_decode_loop(client) -> None:  # type: ignore[no-untyped-def]
    """Cancellation has to be measured, not asserted from the design.

    Two signals, because the token count alone cannot tell "stopped early" from
    "the model hit EOS": generation holds a lock, so a follow-up request cannot
    start until the abandoned one has actually stopped. The baseline is measured
    here rather than hard-coded so this is not a wall-clock threshold that a slow
    runner trips.
    """
    asked = 120
    started = time.perf_counter()
    client.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 10})
    ms_per_token = (time.perf_counter() - started) * 1000 / 10

    before = client.get("/metrics").json()["generated_tokens"]
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Write a long story."}],
            "max_tokens": asked,
            "stream": True,
        },
    ) as response:
        seen = 0
        for line in response.iter_lines():
            if line.startswith("data: "):
                seen += 1
                if seen >= 3:
                    break

    disconnected_at = time.perf_counter()
    client.post("/v1/completions", json={"prompt": "Hi", "max_tokens": 2})
    waited_ms = (time.perf_counter() - disconnected_at) * 1000

    # Half the projected remaining time is a wide margin; observed locally the
    # follow-up unblocks in ~40 ms against a ~26 s projection.
    assert waited_ms < asked * ms_per_token * 0.5
    # Recorded in a `finally`, so a cancelled request still counts.
    assert client.get("/metrics").json()["generated_tokens"] - before < asked


def test_metrics_report_what_was_served(client) -> None:  # type: ignore[no-untyped-def]
    metrics = client.get("/metrics").json()
    assert metrics["runtime"] == "eager"
    assert metrics["requests"] >= 1
    assert metrics["generated_tokens"] >= 1
    assert metrics["ttft_ms"] > 0
    assert metrics["memory"]["kv_bytes_per_token"] > 0
