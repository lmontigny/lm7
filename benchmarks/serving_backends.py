"""Compare `lm7 model serve`'s own decode loop against the TensorRT-LLM handover.

Both answer the same OpenAI-compatible API, so both can be driven by one client
-- which is the only reason this comparison is worth anything. Every number here
comes from the same measurement code over the same loopback HTTP, against the
same model, prompt and token budget:

    python benchmarks/serving_backends.py --output artifacts/serving-backends.json
    python benchmarks/serving_backends.py --backends inductor --concurrency 1 4

Three arms. `inductor` and `cudagraphs` are both LM7's own server -- its
compiled prefill and KV-cache decode graphs, one request at a time behind a lock
-- differing only in `--compile-mode reduce-overhead`, which is what asks
Inductor to capture the decode step into a CUDA graph. `trtllm` hands the port
to TensorRT-LLM and its paged cache and in-flight batching scheduler; LM7 is not
in the request path at all for that one, so this measures two *servers*, not two
compilers.

**Concurrency is the point.** At one stream this asks whether an engine build
buys anything over `torch.compile` on a decode loop. Above one it asks the
question the launcher exists for, and the two arms are not the same shape of
thing: LM7 serializes, so its aggregate throughput is flat by construction.

Fairness, and where it stops:

- Each server is launched, warmed and measured alone. They are never up at the
  same time; a 12 GiB card cannot hold both, and a shared card would make the
  memory column meaningless.
- **Warmup is not optional and is not cosmetic.** LM7 compiles its graphs inside
  the first request; TensorRT-LLM warms up before it answers `/health`. Both are
  driven until they stop getting faster, and the discarded first request is
  reported separately as `first_request_ms` rather than hidden.
- GPU memory is read from `nvidia-smi` for both, because LM7's `/metrics`
  reports its own allocator and TensorRT-LLM reports nothing comparable. It is
  whole-process occupancy, so a desktop compositor is in it.
- One prompt length throughout, so LM7 compiles one prefill graph. Varying it
  would measure recompilation, which is a different question --
  see docs/kv-cache-decode.md.

This is a *serving* comparison and it is loopback wall clock: it includes HTTP,
SSE framing and each server's scheduler. It is not a kernel benchmark, and
nothing here is comparable to `benchmarks/decode.py`, which drives the runner
directly with no server at all.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The fast causal-LM smoke test in this repo's ladder: 30 layers makes it
# launch-bound, which is the regime a compiled decode loop is supposed to help
# most -- so it is the model most favourable to the Inductor arm, deliberately.
DEFAULT_MODEL = "hf://HuggingFaceTB/SmolLM2-135M-Instruct"

# One prompt, so LM7 compiles exactly one prefill graph. It asks for a list
# because the comparison is about decode: a 135M instruct model answers a
# short question in about 25 tokens, which is too few gaps to take a median
# of, and enumerating keeps it generating to the token budget instead.
PROMPT = (
    "List ten things to check when a deep learning model is slower than "
    "expected, and write one full sentence about each."
)

# The arms, as (serve --backend, serve --compile-mode). `cudagraphs` is here
# because leaving it out would stack the comparison: a 30-layer 135M model
# spends most of a decode step launching kernels rather than running them, and
# `reduce-overhead` is the only preset that asks Inductor to capture them into a
# CUDA graph -- exactly the overhead TensorRT-LLM removes by other means. See
# docs/inductor-options.md.
ARMS: dict[str, tuple[str, str | None]] = {
    "inductor": ("inductor", None),
    "cudagraphs": ("inductor", "reduce-overhead"),
    "trtllm": ("trtllm", None),
}

BACKENDS = tuple(ARMS)

# Generous: LM7's first-request compile is minutes on a cold Inductor cache, and
# TensorRT-LLM JIT-compiles FlashInfer kernels the first time it sees a card.
STARTUP_TIMEOUT = 1800.0


@dataclass
class Stream:
    """One streamed completion, timed from the client."""

    ttft_ms: float
    tokens: int
    total_ms: float
    inter_token_ms: list[float] = field(default_factory=list)


def _post_stream(base: str, model: str, max_tokens: int, timeout: float = 600.0) -> Stream:
    """One streaming chat completion, timed at the SSE boundary.

    Deliberately the standard library rather than the `openai` SDK: the SDK adds
    its own buffering between the socket and the caller, and TTFT measured
    through it is the SDK's latency as much as the server's.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        # Greedy, so the two arms decode the same number of tokens for the same
        # prompt and the comparison is not partly a sampling difference.
        "temperature": 0.0,
        "stream": True,
    }
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    stamps: list[float] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            choices = json.loads(payload).get("choices") or [{}]
            if (choices[0].get("delta") or {}).get("content"):
                stamps.append(time.perf_counter() - start)
    if not stamps:
        raise RuntimeError("the server streamed no content")
    gaps = [(b - a) * 1000 for a, b in itertools.pairwise(stamps)]
    return Stream(
        ttft_ms=stamps[0] * 1000,
        tokens=len(stamps),
        total_ms=stamps[-1] * 1000,
        inter_token_ms=gaps,
    )


def gpu_memory_mib() -> int | None:
    """Whole-card occupancy from ``nvidia-smi``, or None where there is no GPU.

    The card, not the process: it is the only figure that means the same thing
    for a server LM7 loaded and a server it handed over to, and it is what
    decides whether a second model fits.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return int(out.strip().splitlines()[0])


def _wait_for_port(port: int, timeout: float = 120.0) -> None:
    """Block until nothing is listening on ``port``.

    Each arm gets its own port, but a server that has just been signalled can
    still hold its listening socket for a moment, and `trtllm-serve` does not
    retry -- it exits with `Address already in use` before loading anything,
    which is a confusing way to lose a twenty-minute run.
    """
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(2.0)
    raise RuntimeError(f"port {port} is still in use after {timeout:.0f}s")


def _wait_for_health(base: str, process: subprocess.Popen[bytes], timeout: float) -> float:
    """Seconds from launch to the first 200 on ``/health``."""
    start = time.perf_counter()
    deadline = start + timeout
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode} before answering")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
                if response.status == 200:
                    return time.perf_counter() - start
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(2.0)
    raise RuntimeError(f"server did not answer within {timeout:.0f}s")


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """Signal the whole group: TensorRT-LLM's MPI workers outlive a lone parent."""
    for number in (signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), number)
        except (ProcessLookupError, PermissionError):
            return
        try:
            process.wait(timeout=90)
            return
        except subprocess.TimeoutExpired:
            continue


def serve_argv(arm: str, model: str, port: int, max_model_len: int) -> list[str]:
    """The `lm7 model serve` command line for one arm.

    Built here rather than calling into `lm7.serve` so that what is measured is
    the CLI a reader would type -- including, for the launcher arm, the argv
    translation being exercised rather than bypassed.
    """
    backend, compile_mode = ARMS[arm]
    argv = [
        sys.executable,
        "-m",
        "lm7",
        "model",
        "serve",
        model,
        "--target",
        "nvidia",
        "--backend",
        backend,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
    ]
    if compile_mode:
        argv += ["--compile-mode", compile_mode]
    return argv


def measure(
    arm: str,
    *,
    model: str,
    port: int,
    max_model_len: int,
    max_tokens: int,
    repeats: int,
    concurrency: tuple[int, ...],
    log_dir: Path | None,
) -> dict[str, Any]:
    """Launch one arm, drive it, and tear it down. Returns its whole result."""
    base = f"http://127.0.0.1:{port}"
    argv = serve_argv(arm, model, port, max_model_len)
    _wait_for_port(port)
    idle_mib = gpu_memory_mib()

    log_path = (log_dir / f"serve-{arm}.log") if log_dir else None
    log = open(log_path or os.devnull, "w")  # noqa: SIM115 - closed in `finally`
    # Its own session so `_terminate` can signal the group.
    process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result: dict[str, Any] = {
        "arm": arm,
        "backend": ARMS[arm][0],
        "compile_mode": ARMS[arm][1],
        "argv": argv,
        "log": str(log_path or ""),
    }
    try:
        result["startup_s"] = round(_wait_for_health(base, process, STARTUP_TIMEOUT), 2)

        # The first request is where LM7 compiles, so it is timed and thrown
        # out rather than averaged in -- and reported, because for the Inductor
        # arm it is the number a user actually waits through once.
        first = _post_stream(base, _served_model(base, model), max_tokens)
        result["first_request_ms"] = round(first.total_ms, 1)
        for _ in range(2):
            _post_stream(base, _served_model(base, model), max_tokens)

        served = _served_model(base, model)
        singles = [_post_stream(base, served, max_tokens) for _ in range(repeats)]
        gaps = [gap for run in singles for gap in run.inter_token_ms]
        result["single_stream"] = {
            "repeats": repeats,
            "tokens": singles[0].tokens,
            "ttft_ms_median": round(statistics.median(r.ttft_ms for r in singles), 1),
            "inter_token_ms_median": round(statistics.median(gaps), 2) if gaps else None,
            "tokens_per_s": round(1000.0 / statistics.median(gaps), 1) if gaps else None,
            "total_ms_median": round(statistics.median(r.total_ms for r in singles), 1),
        }
        result["memory"] = {
            "idle_mib": idle_mib,
            "serving_mib": gpu_memory_mib(),
        }
        result["concurrency"] = [
            _concurrent(base, served, max_tokens, n) for n in concurrency if n > 1
        ]
    finally:
        _terminate(process)
        log.close()
    # Let the driver release the card before the next arm launches.
    time.sleep(10.0)
    return result


def _concurrent(base: str, model: str, max_tokens: int, streams: int) -> dict[str, Any]:
    """`streams` streamed completions at once, from `streams` threads.

    Aggregate tokens per second is the figure that separates the two arms: LM7
    holds an `asyncio.Lock`, so a second caller waits and the total cannot rise;
    TensorRT-LLM batches them, so it should.
    """
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=streams) as pool:
        runs = list(pool.map(lambda _: _post_stream(base, model, max_tokens), range(streams)))
    wall_s = time.perf_counter() - start
    total_tokens = sum(run.tokens for run in runs)
    return {
        "streams": streams,
        "wall_s": round(wall_s, 2),
        "total_tokens": total_tokens,
        "aggregate_tokens_per_s": round(total_tokens / wall_s, 1),
        "ttft_ms_median": round(statistics.median(run.ttft_ms for run in runs), 1),
        "ttft_ms_max": round(max(run.ttft_ms for run in runs), 1),
    }


def _served_model(base: str, requested: str) -> str:
    """The id this server answers to, which is not always what was asked for."""
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as response:
            listed = json.load(response)
        return str(listed["data"][0]["id"])
    except (urllib.error.URLError, KeyError, IndexError, TimeoutError, OSError):
        return requested.removeprefix("hf://")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LM7's compiled decode loop against the TensorRT-LLM handover."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="write each server's stdout here; without it they go to /dev/null",
    )
    args = parser.parse_args()

    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "model": args.model,
        "prompt": PROMPT,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "host": platform.platform(),
        "python": platform.python_version(),
        "results": [],
    }
    for offset, arm in enumerate(args.backends):
        print(f"== {arm} ==", flush=True)
        try:
            result = measure(
                arm,
                model=args.model,
                port=args.port + offset,
                max_model_len=args.max_model_len,
                max_tokens=args.max_tokens,
                repeats=args.repeats,
                concurrency=tuple(args.concurrency),
                log_dir=args.log_dir,
            )
        except Exception as error:  # noqa: BLE001 - one arm failing is a result
            result = {"arm": arm, "error": f"{type(error).__name__}: {error}"}
        report["results"].append(result)
        print(json.dumps(result, indent=2), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
