"""The HTTP surface, over a scripted model. Needs the ``serve`` extra.

Marked rather than always-run because FastAPI, Uvicorn and Pydantic live behind
an optional extra, and the `quality` CI job installs none of them. What is
exercised here is the wire format an OpenAI client actually sees -- status
codes, SSE framing, the shape of every JSON body -- against the same scripted
runner ``test_serve.py`` uses, so nothing here waits on a download or a compile.
"""

from __future__ import annotations

import json

import pytest

# `tests/` is not a package, so pytest puts this directory on sys.path and the
# sibling module imports by bare name.
from test_serve import EOS, FakeTokenizer, ScriptedRunner

from lm7.serve.engine import LM7ServeEngine, ServeConfig

fastapi = pytest.importorskip("fastapi", reason="the serve extra is not installed")
from fastapi.testclient import TestClient

pytestmark = pytest.mark.serve


def client(
    script: list[int], *, max_model_len: int = 64, config: ServeConfig | None = None
) -> TestClient:
    from lm7.serve.server import build_app

    engine = LM7ServeEngine(
        ScriptedRunner(script),
        FakeTokenizer(),
        config or ServeConfig(model="hf://owner/fake", max_model_len=max_model_len),
        model_id="owner/fake",
    )
    # A context-managed TestClient runs every request on one event loop, which is
    # what the engine's `asyncio.Lock` is bound to. Callers must use `with`.
    return TestClient(build_app(engine))


def events(text: str) -> list[dict]:
    """Parse an SSE body into its JSON payloads, dropping the [DONE] sentinel."""
    payloads = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        body = line.removeprefix("data: ")
        if body == "[DONE]":
            continue
        payloads.append(json.loads(body))
    return payloads


def chat(messages: list[dict] | None = None, **extra: object) -> dict:
    # `max_tokens` is set because the schema default is 256, which does not fit
    # the small caches these tests allocate -- the server would refuse it, which
    # is correct but not what most of these are checking.
    body: dict = {
        "messages": messages or [{"role": "user", "content": "hi"}],
        "temperature": 0,
        "max_tokens": 8,
    }
    body.update(extra)
    return body


# -- the chat page --------------------------------------------------------


def test_the_root_serves_a_chat_page() -> None:
    with client([1, EOS]) as http:
        response = http.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>lm7 serve</title>" in response.text


def test_the_chat_page_makes_no_external_requests() -> None:
    """The page must render on an airgapped box, which is where local inference matters.

    A CDN script tag or a web font would work on the laptop it was written on
    and fail on exactly the machine `lm7 model serve` is for, so this asserts
    the absence rather than trusting review to catch a later addition.
    """
    from lm7.serve.ui import PAGE

    for marker in ("http://", "https://", "//cdn", "integrity=", "@import"):
        assert marker not in PAGE, f"the chat page reaches outside itself: {marker!r}"


def test_the_chat_page_drives_the_documented_endpoints() -> None:
    """It is a client, not a privileged path -- so it uses the public routes."""
    from lm7.serve.ui import PAGE

    for route in ("/health", "/metrics", "/v1/chat/completions"):
        assert route in PAGE


def test_the_chat_page_is_not_in_the_openapi_schema() -> None:
    with client([1]) as http:
        schema = http.get("/openapi.json").json()
    assert "/" not in schema["paths"]
    assert "/v1/chat/completions" in schema["paths"]


# -- discovery ------------------------------------------------------------


def test_health_reports_what_is_loaded() -> None:
    with client([1, 2, EOS]) as http:
        response = http.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "owner/fake",
        "target": "cpu",
        "backend": "eager",
    }


def test_models_lists_the_one_model_this_server_holds() -> None:
    with client([1]) as http:
        body = http.get("/v1/models").json()
    assert body["object"] == "list"
    assert [card["id"] for card in body["data"]] == ["owner/fake"]


# -- chat completions -----------------------------------------------------


def test_a_chat_completion_comes_back_in_openai_shape() -> None:
    with client([1, 2, 3, EOS]) as http:
        response = http.post("/v1/chat/completions", json=chat(max_tokens=8))
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello world!"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 3
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 3


def test_the_token_budget_is_reported_as_a_length_finish() -> None:
    with client([1, 2, 3, 4, 5]) as http:
        body = http.post("/v1/chat/completions", json=chat(max_tokens=2)).json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["choices"][0]["message"]["content"] == "Hello world"


def test_the_requested_model_name_is_echoed_back() -> None:
    with client([1, EOS]) as http:
        body = http.post("/v1/chat/completions", json=chat(model="gpt-4o-mini")).json()
    assert body["model"] == "gpt-4o-mini"


def test_a_stop_sequence_is_honoured_over_http() -> None:
    with client([1, 2, 3, 4]) as http:
        body = http.post("/v1/chat/completions", json=chat(max_tokens=8, stop=" world")).json()
    assert body["choices"][0]["message"]["content"] == "Hello"
    assert body["choices"][0]["finish_reason"] == "stop"


# -- streaming ------------------------------------------------------------


def test_a_streamed_chat_completion_is_well_formed_sse() -> None:
    with client([1, 2, 3, EOS]) as http:
        response = http.post("/v1/chat/completions", json=chat(max_tokens=8, stream=True))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.endswith("data: [DONE]\n\n")

    chunks = events(response.text)
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    # One id and one timestamp for the whole stream, as the format requires.
    assert len({chunk["id"] for chunk in chunks}) == 1
    # The role arrives first, alone; the finish reason arrives last, alone.
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"][0]["delta"] == {}

    content = "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks)
    assert content == "Hello world!"


def test_streamed_chunks_have_openai_key_presence_not_merely_openai_values() -> None:
    """`role` is omitted after the first chunk; `finish_reason` is present as null.

    A plain `exclude_none` would get the first right and the second wrong, and
    clients read both by key presence.
    """
    with client([1, 2, EOS]) as http:
        chunks = events(
            http.post("/v1/chat/completions", json=chat(max_tokens=8, stream=True)).text
        )
    assert all("finish_reason" in chunk["choices"][0] for chunk in chunks)
    assert all("role" not in chunk["choices"][0]["delta"] for chunk in chunks[1:])


def test_a_streamed_completion_reassembles_to_the_same_text_as_a_buffered_one() -> None:
    with client([1, 2, 3, EOS]) as http:
        buffered = http.post("/v1/chat/completions", json=chat(max_tokens=8)).json()
        streamed = http.post("/v1/chat/completions", json=chat(max_tokens=8, stream=True))
    joined = "".join(
        chunk["choices"][0]["delta"].get("content") or "" for chunk in events(streamed.text)
    )
    assert joined == buffered["choices"][0]["message"]["content"]


def test_a_streamed_stop_sequence_never_reaches_the_client() -> None:
    with client([1, 2, 3, 4]) as http:
        response = http.post(
            "/v1/chat/completions", json=chat(max_tokens=8, stream=True, stop=" world")
        )
    joined = "".join(
        chunk["choices"][0]["delta"].get("content") or "" for chunk in events(response.text)
    )
    assert joined == "Hello"


# -- legacy completions ---------------------------------------------------


def test_the_completions_endpoint_answers_in_its_own_shape() -> None:
    with client([1, 2, EOS]) as http:
        body = http.post(
            "/v1/completions", json={"prompt": "hi", "max_tokens": 8, "temperature": 0}
        ).json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "Hello world"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_the_completions_endpoint_streams() -> None:
    with client([1, 2, EOS]) as http:
        response = http.post(
            "/v1/completions",
            json={"prompt": "hi", "max_tokens": 8, "temperature": 0, "stream": True},
        )
    joined = "".join(chunk["choices"][0]["text"] for chunk in events(response.text))
    assert joined == "Hello world"
    assert response.text.endswith("data: [DONE]\n\n")


# -- refusals -------------------------------------------------------------


def test_an_oversized_request_is_a_400_and_not_a_broken_stream() -> None:
    """The refusal has to happen before the response type is chosen.

    A 200 whose SSE stream dies after one chunk is much harder to debug from a
    client than a 400 that says the cache is too small.
    """
    with client([1, 2], max_model_len=8) as http:
        response = http.post(
            "/v1/chat/completions",
            json=chat([{"role": "user", "content": "one two three four five six"}], max_tokens=8),
        )
    assert response.status_code == 400
    assert "static cache" in response.json()["detail"]


def test_an_oversized_streaming_request_is_also_a_400() -> None:
    with client([1, 2], max_model_len=8) as http:
        response = http.post(
            "/v1/chat/completions",
            json=chat(
                [{"role": "user", "content": "one two three four five six"}],
                max_tokens=8,
                stream=True,
            ),
        )
    assert response.status_code == 400


def test_a_field_that_would_change_the_answer_is_refused_not_ignored() -> None:
    with client([1, EOS]) as http:
        response = http.post("/v1/chat/completions", json=chat(n=4))
    assert response.status_code == 400
    assert "does not implement n" in response.json()["detail"]


def test_an_empty_message_list_fails_schema_validation() -> None:
    with client([1]) as http:
        response = http.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 422


# -- metrics --------------------------------------------------------------


def test_metrics_report_the_compile_state_the_page_shows() -> None:
    with client([1, 2, EOS]) as http:
        cold = http.get("/metrics").json()
        assert cold["warm"] is False
        http.post("/v1/chat/completions", json=chat())
        warm = http.get("/metrics").json()
    assert warm["warm"] is True
    # The claim the whole two-graph split exists to make checkable.
    assert warm["steady_frames"] == 0
    assert "prefill_lengths" in warm


def test_the_chat_page_reads_the_compile_state() -> None:
    from lm7.serve.ui import PAGE

    for field in ("warm", "steady_frames", "prefill_lengths"):
        assert field in PAGE


def test_metrics_count_what_actually_ran() -> None:
    with client([1, 2, 3, EOS]) as http:
        assert http.get("/metrics").json()["requests"] == 0
        http.post("/v1/chat/completions", json=chat(max_tokens=8))
        body = http.get("/metrics").json()
    assert body["requests"] == 1
    assert body["generated_tokens"] == 3
    assert body["ttft_ms"] > 0
    assert body["model"] == "owner/fake"
    assert body["max_model_len"] == 64


# -- CORS and bearer auth --------------------------------------------------


def _config(**overrides: object) -> ServeConfig:
    return ServeConfig(model="hf://owner/fake", max_model_len=64, **overrides)  # type: ignore[arg-type]


def test_a_browser_ui_on_another_port_is_allowed_by_default() -> None:
    with client([EOS]) as http:
        response = http.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_the_preflight_a_browser_sends_before_a_chat_request_succeeds() -> None:
    with client([EOS]) as http:
        response = http.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_narrowing_the_origins_excludes_everything_else() -> None:
    config = _config(cors_origins=("http://localhost:3000",))
    with client([EOS], config=config) as http:
        allowed = http.get("/health", headers={"Origin": "http://localhost:3000"})
        denied = http.get("/health", headers={"Origin": "http://evil.example"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    # The request still succeeds; it is the browser that refuses to hand the body
    # to a page whose origin is missing from the response.
    assert "access-control-allow-origin" not in denied.headers


def test_an_empty_origin_list_turns_cors_off_entirely() -> None:
    with client([EOS], config=_config(cors_origins=())) as http:
        response = http.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_without_a_key_the_server_is_open() -> None:
    with client([EOS]) as http:
        assert http.get("/v1/models").status_code == 200


def test_a_key_is_required_when_one_was_configured() -> None:
    with client([EOS], config=_config(api_key="s3cret")) as http:
        assert http.get("/v1/models").status_code == 401
        assert http.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert http.get("/v1/models", headers={"Authorization": "s3cret"}).status_code == 401
        ok = http.get("/v1/models", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_health_answers_without_a_key_so_a_probe_can_use_it() -> None:
    with client([EOS], config=_config(api_key="s3cret")) as http:
        response = http.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_a_rejected_request_still_carries_cors_headers() -> None:
    # Without this, a browser reports the 401 as a CORS failure and the person
    # debugging it goes looking in the wrong place entirely.
    with client([EOS], config=_config(api_key="s3cret")) as http:
        response = http.get("/v1/models", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == "*"


def test_a_preflight_is_never_authenticated() -> None:
    # Browsers do not send Authorization on a preflight, so requiring a key here
    # would fail every cross-origin request before the real one was sent.
    with client([EOS], config=_config(api_key="s3cret")) as http:
        response = http.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert response.status_code == 200


def test_generation_works_through_the_key() -> None:
    with client([1, 2, EOS], config=_config(api_key="s3cret")) as http:
        response = http.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer s3cret"},
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
        )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"]


def test_the_built_in_chat_page_is_unavailable_behind_a_key() -> None:
    # The page is fetched by a browser and cannot send an Authorization header,
    # so a key and the built-in UI are mutually exclusive. Asserted rather than
    # left to be discovered as a blank page.
    with client([EOS], config=_config(api_key="s3cret")) as http:
        refused = http.get("/")
        assert refused.status_code == 401
        assert "chat page" in refused.json()["detail"]
    with client([EOS]) as open_http:
        assert open_http.get("/").status_code == 200
