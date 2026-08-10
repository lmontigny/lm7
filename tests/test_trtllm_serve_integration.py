"""``--backend trtllm`` against a real TensorRT-LLM and a real GPU.

Everything LM7 owns here is a translation -- config to argv, target to a refusal
-- and ``test_serve.py`` covers all of it with no GPU. What that cannot cover is
the only claim that matters: that the argv LM7 prints is one ``trtllm-serve``
actually accepts, and that what comes up on the port answers the OpenAI API.

So this launches the real thing as a subprocess, exactly as ``lm7 model serve``
does, and talks to it over HTTP. It is slow -- TensorRT-LLM loads a model,
allocates a paged KV cache and warms up -- which is why it is marked and skipped
by default.

TensorRT-LLM must be installed in its own environment (it pins a torch, a
transformers and a tensorrt that conflict with every other environment in this
repo), so this suite runs against ``trtllm-serve`` on PATH and does *not* import
``tensorrt_llm``. Point PATH at that venv's ``bin`` and run:

    python -m pytest tests/test_trtllm_serve_integration.py -m trtllm

See docs/tensorrt-llm.md.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from lm7.serve.cli import serve_plan
from lm7.serve.engine import ServeConfig

pytestmark = pytest.mark.trtllm

# Small enough to load in well under the startup budget below, and the same
# model the rest of this repo smoke-tests causal LMs with -- see CLAUDE.md.
MODEL = "hf://HuggingFaceTB/SmolLM2-135M-Instruct"

# TensorRT-LLM's startup is a model load, a paged-cache allocation and a warmup.
# Generous because the failure this guards against is a hang, not slowness.
STARTUP_TIMEOUT = 900.0


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    """A real ``trtllm-serve``, launched from LM7's own argv. Yields its base URL.

    The argv comes from :func:`serve_plan` rather than being written out here,
    so that what is exercised is the command LM7 would actually run. A hand-
    written command line would pass while ``lm7 model serve`` was broken.
    """
    if shutil.which("trtllm-serve") is None:
        pytest.skip("trtllm-serve is not on PATH; see docs/tensorrt-llm.md")

    port = _free_port()
    config = ServeConfig(model=MODEL, target="nvidia", backend="trtllm", port=port)
    argv = serve_plan(config)["argv"]

    log = open(  # noqa: SIM115 - closed in the finally below, after the process
        os.environ.get("LM7_TRTLLM_LOG", os.devnull), "w"
    )
    # Its own process group, because TensorRT-LLM spawns MPI workers that hold
    # the GPU: terminating only the parent leaves a worker owning several GiB of
    # paged KV cache, and the next test to want a GPU finds none. Observed here,
    # not theorized -- see docs/tensorrt-llm.md.
    process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(
                    f"trtllm-serve exited with {process.returncode} before answering. "
                    "Re-run with LM7_TRTLLM_LOG=/tmp/trtllm.log to see why."
                )
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(2.0)
        else:
            pytest.fail(f"trtllm-serve did not answer within {STARTUP_TIMEOUT:.0f}s")
        yield base
    finally:
        _terminate_group(process)
        log.close()


def _terminate_group(process: subprocess.Popen[bytes]) -> None:
    """Signal the whole group, so no MPI worker outlives the run holding the GPU."""
    for signal_number in (signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal_number)
        except (ProcessLookupError, PermissionError):  # pragma: no cover - already gone
            return
        try:
            process.wait(timeout=60)
            return
        except subprocess.TimeoutExpired:  # pragma: no cover - escalates to SIGKILL
            continue


def _post(base: str, path: str, body: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return dict(json.load(response))


def test_the_argv_lm7_prints_is_one_trtllm_serve_accepts(server: str) -> None:
    """The whole of LM7's contribution, checked end to end.

    The fixture built this server from `serve_plan`'s argv. Reaching a 200 on
    /health means that command line parsed, the model loaded and the runtime
    came up -- which is the claim `--dry-run` makes when it prints it.
    """
    with urllib.request.urlopen(f"{server}/health", timeout=30) as response:
        assert response.status == 200


def test_the_served_model_is_the_one_lm7_resolved(server: str) -> None:
    """`hf://owner/model` reaches TensorRT-LLM as the id it names."""
    with urllib.request.urlopen(f"{server}/v1/models", timeout=30) as response:
        listed = json.load(response)
    ids = [entry["id"] for entry in listed["data"]]
    assert any(MODEL.removeprefix("hf://") in str(served) for served in ids), ids


def test_a_chat_completion_comes_back_from_tensorrt_llm(server: str) -> None:
    body = _post(
        server,
        "/v1/chat/completions",
        {
            "model": MODEL.removeprefix("hf://"),
            "messages": [{"role": "user", "content": "The capital of France is"}],
            "max_tokens": 16,
            "temperature": 0.0,
        },
    )
    content = body["choices"][0]["message"]["content"]  # type: ignore[index]
    assert isinstance(content, str) and content.strip()


def test_a_streamed_completion_reassembles(server: str) -> None:
    """SSE deltas, which is what a client actually consumes.

    Asserted because streaming is the mode the chat page and every OpenAI SDK
    use by default, and a server that answers non-streaming requests can still
    frame SSE wrongly.
    """
    request = urllib.request.Request(
        f"{server}/v1/chat/completions",
        data=json.dumps(
            {
                "model": MODEL.removeprefix("hf://"),
                "messages": [{"role": "user", "content": "Count: one two"}],
                "max_tokens": 16,
                "temperature": 0.0,
                "stream": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    pieces = []
    saw_done = False
    with urllib.request.urlopen(request, timeout=120) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if payload == "[DONE]":
                saw_done = True
                break
            delta = json.loads(payload)["choices"][0].get("delta", {})
            pieces.append(delta.get("content") or "")
    assert saw_done
    assert "".join(pieces).strip()
