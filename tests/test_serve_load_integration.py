"""``LM7ServeEngine.load`` against a real model. Needs the ``serve`` and ``hf`` extras.

Everything else in the serve suite drives a scripted runner and a fake tokenizer,
which is the right trade for testing the wire format -- but it means the path
that actually loads a model is untested: resolving what the user typed, both
``from_pretrained`` calls, the quantization gate, ``compile_generation``, and the
static cache being allocated on the target.

That path is where every serve bug found by hand has been. An indexed-device
comparison made ``--target apple`` 500 on every request; the quantization gate
printed a filesystem path where its own text promised a model id. Neither was
reachable from a scripted runner.

The model is `hf-internal-testing/tiny-random-LlamaForCausalLM` -- about 15 MB,
random weights, four layers. Its *output* is gibberish and nothing here asserts
otherwise; what is being checked is that a real checkpoint travels the whole way
from a URI to a token over HTTP. Correctness of generated text belongs to
docs/kv-cache-decode.md, which compares against eager on real models.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="the serve extra is not installed")
pytest.importorskip("transformers", reason="the hf extra is not installed")

from fastapi.testclient import TestClient

from lm7.errors import UnsupportedModelError
from lm7.serve.engine import LM7ServeEngine, ServeConfig
from lm7.serve.server import build_app

pytestmark = pytest.mark.serve_load

# Small enough that downloading it costs less than the compile that follows.
TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

# `backend="eager"` on purpose: this suite is checking that a model loads and
# reaches the decode loop, and Inductor would add a compile to every test for a
# code path that tests/test_generation.py already covers.
BASE = ServeConfig(model=f"hf://{TINY_MODEL}", target="cpu", backend="eager", max_model_len=64)


@pytest.fixture(scope="module")
def engine() -> LM7ServeEngine:
    """One load for the module: the download and the cache allocation are the cost."""
    return LM7ServeEngine.load(BASE)


@pytest.fixture(scope="module")
def local_model(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The same checkpoint written to disk, as `save_pretrained` leaves it."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory = tmp_path_factory.mktemp("local-model")
    AutoTokenizer.from_pretrained(TINY_MODEL).save_pretrained(directory)
    AutoModelForCausalLM.from_pretrained(TINY_MODEL).save_pretrained(directory)
    return str(directory)


def test_a_hub_model_loads_and_allocates_its_cache(engine: LM7ServeEngine) -> None:
    assert engine.model_id == TINY_MODEL
    assert engine.target.startswith("cpu")
    # The static cache is allocated at load, not at first request -- if this is
    # zero the runner never built one and every generation would be uncached.
    assert engine.kv_cache_bytes > 0


def test_a_real_model_answers_over_http(engine: LM7ServeEngine) -> None:
    with TestClient(build_app(engine)) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["usage"]["completion_tokens"] > 0
    assert body["usage"]["prompt_tokens"] > 0
    assert body["choices"][0]["finish_reason"] in {"stop", "length"}


def test_a_real_model_streams_over_http(engine: LM7ServeEngine) -> None:
    with TestClient(build_app(engine)) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
                "stream": True,
            },
        )
    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")


def test_the_cache_limit_is_enforced_against_a_real_tokenizer(engine: LM7ServeEngine) -> None:
    # The refusal arithmetic runs on real token counts here, not on the fake
    # tokenizer's one-token-per-word.
    with TestClient(build_app(engine)) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 9999},
        )
    assert response.status_code == 400
    assert "static cache" in response.json()["detail"]


def test_a_local_directory_loads_the_same_way(local_model: str) -> None:
    # The path resolve_model_source produces has to be something
    # `from_pretrained` actually accepts, which only a real load can show.
    loaded = LM7ServeEngine.load(
        ServeConfig(model=local_model, target="cpu", backend="eager", max_model_len=64)
    )
    assert loaded.model_id == local_model
    assert loaded.kv_cache_bytes > 0


def test_an_unvalidated_model_cannot_be_quantized() -> None:
    # The tiny model is not on the validated list, so the per-model gate refuses
    # it -- and does so before the download, which is the point of the ordering.
    config = ServeConfig(model=f"hf://{TINY_MODEL}", target="cpu", quantize="int8")
    with pytest.raises(UnsupportedModelError, match="not validated"):
        LM7ServeEngine.load(config)


def test_the_runner_reports_no_decode_recompiles(engine: LM7ServeEngine) -> None:
    """The regression the two-graph split exists to prevent.

    ``steady_frames`` above zero means a *token* triggered a compile. With
    ``backend="eager"`` nothing compiles at all, so this asserts the counter is
    readable and clean rather than that Inductor behaved -- the compiled version
    of this claim is measured in docs/kv-cache-decode.md.
    """
    with TestClient(build_app(engine)) as http:
        http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 4},
        )
        stats = http.get("/metrics").json()
    assert stats["steady_frames"] == 0


def test_the_cache_is_sized_by_the_flag(engine: LM7ServeEngine) -> None:
    # The flag has to reach the allocation, not just the refusal arithmetic: a
    # server that validated against 32 while holding a 64-token cache would
    # refuse requests it could serve, and one with the error the other way would
    # accept requests that overrun.
    half = LM7ServeEngine.load(
        ServeConfig(model=f"hf://{TINY_MODEL}", target="cpu", backend="eager", max_model_len=32)
    )
    assert half.max_model_len == 32
    assert half.kv_cache_bytes * 2 == engine.kv_cache_bytes
