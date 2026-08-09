# Serving

`lm7 model serve` puts an OpenAI-compatible HTTP endpoint in front of
[`lm7.compile_generation`](kv-cache-decode.md), so a local client that already
speaks the OpenAI API can talk to a model compiled for whatever hardware is in
front of you.

```bash
uv pip install -e ".[serve,hf]"
lm7 model serve hf://HuggingFaceTB/SmolLM2-135M-Instruct --target auto --max-model-len 512
```

```
lm7: loading HuggingFaceTB/SmolLM2-135M-Instruct for cpu:arm64...
lm7: serving HuggingFaceTB/SmolLM2-135M-Instruct on http://127.0.0.1:8000 (cpu:arm64, backend=auto, max_model_len=512, kv cache 24 MB)
lm7: the first request compiles the prefill and decode graphs and will be slower.
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
print(
    client.chat.completions.create(
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        messages=[{"role": "user", "content": "The capital of France is"}],
        max_tokens=8,
    )
    .choices[0]
    .message.content
)
```

## What this is, and what it is not

This is a **single-user local server**. It holds one model, one pair of compiled
graphs and one static KV cache, and it serves **one request at a time** behind an
`asyncio.Lock`. A second caller waits.

That is not a temporary limitation to be fixed later — it follows from what LM7
is. `compile_generation` allocates exactly one static KV cache and every decode
step mutates it in place, so two concurrent generations would interleave writes
into the same buffers and return two wrong answers without raising. Making that
safe means paged attention and a scheduler, which means writing a serving
engine. LM7 does not write compilers, and it does not write serving engines
either.

| | |
| --- | --- |
| Implemented | streaming (SSE), cancellation on disconnect, stop sequences, temperature/top-p sampling, seeding, a token budget checked before the response starts |
| Not implemented | continuous batching, paged attention, prefix caching, chunked prefill, speculative decoding, LoRA adapters, tool calling, `n > 1`, logprobs, structured output |

Anything in the second row that a request *asks for* is refused with a 400 that
names the field. It is not silently ignored: a caller that asks for four
completions and receives one has been misled, and the whole point of an
OpenAI-compatible endpoint is that the client does not have to know what is
behind it.

When throughput matters, use `--backend vllm` (below) and LM7 steps out of the
request path entirely.

## Talking to it from a browser

`http://127.0.0.1:8000/` is a chat page, so a compiled model can be checked by
hand without writing a client — which is the job `lm7 model serve` exists for.

It is one file (`src/lm7/serve/ui.py`), about 9 KB, with **no CDN, no web font,
no build step and no external requests of any kind**. That is a requirement
rather than a preference: a page that fetches a stylesheet from a CDN renders
fine on the laptop it was written on and fails on exactly the airgapped machine
where running a model locally is the point. A test asserts the absence, so a
later addition cannot quietly reintroduce it.

The page is an ordinary client. It reads `/health` and `/metrics` for its header
and posts to `/v1/chat/completions` with `stream: true`, with no privileged
access to the engine, and it is excluded from the OpenAPI schema because it is a
convenience rather than part of the API contract. The conversation lives in the
page: LM7's engine is one static KV cache with no notion of a session, so each
turn resends the transcript exactly as an OpenAI client would.

### The status line

Above the input, the page says what the server is doing — because "slow" and
"hung" look identical otherwise, and on a cold server the first message really
is slow:

```
cold — the first message compiles the graphs and will be slow
compiling prefill and decode graphs…          ← first message, from /metrics warm:false
prefill…                                      ← subsequent messages, before the first token
generating · 88 char/s                        ← streaming
20 tokens · 412 ms to first token · 9.4 tok/s · 1 prefill graph(s) · 0 decode recompiles
```

The last line is the part no other local server can show you, and it is the
whole point of the two-graph split:

- **`0 decode recompiles`** is `steady_frames` from `runner.counters`. Above zero
  means a *token* triggered a compile — the regression separate prefill and
  decode graphs exist to prevent — and the page turns red and says `RECOMPILED`
  rather than burying it. See [prefill and KV-cache decode](kv-cache-decode.md).
- **`N prefill graph(s)`** is the cost that split accepts: the prompt pass is
  compiled per prompt length, so this climbs as prompt lengths vary. Watching it
  climb while recompiles stay at 0 is the design working as intended.

Both are also on `/metrics`, so a script can assert them without the browser.

> Token counts come from the server, which counts tokens; the live `char/s`
> reading is characters, because mid-stream the page has SSE text fragments and
> not tokens. Timings are wall clock from the page and include HTTP over
> loopback. **They are indicators, not benchmarks** — there is no serving
> benchmark in this repo.

> **What is and is not covered.** Tests assert that `/` serves the page, that it
> contains no external reference, and that it stays out of the schema; the SSE
> reassembly the page performs was checked against a real captured stream fed in
> at arbitrary read boundaries. **Nothing renders it in a browser** — there is no
> headless-browser dependency in this repo and adding one for a 9 KB dev page is
> not worth it. Treat rendering as manually verified, not CI-verified.

### Open WebUI, and other clients

For conversation history, multiple models, or RAG, point a real client at the
endpoint. LM7 implements the OpenAI chat API, so anything that speaks it works:

```bash
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  -v open-webui:/app/backend/data --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

Then open <http://localhost:3000>. On Linux, replace `host.docker.internal` with
`172.17.0.1` or run with `--network=host`. Serve with `--host 0.0.0.0` if the
client is not on this machine — the default binds to loopback, and there is no
authentication on this endpoint.

Also known to work against an OpenAI-compatible base URL: **Continue** and
**Cline** (VS Code), **Zed**'s assistant, **Aider**, **LibreChat**, and any
`openai` SDK. Expect one caveat everywhere: this server refuses `n > 1`,
logprobs, tool calling and structured output with a 400 (see below), so a client
that depends on tools will report an error rather than degrade.

> These clients have **not** been tested against LM7 — they are listed because
> they consume the same API, not because anyone here has run them. The `openai`
> SDK is what has actually been driven end to end.

## Endpoints

| | |
| --- | --- |
| `GET /` | the built-in chat page (not in the OpenAPI schema) |
| `GET /health` | model, target, and the backend that compiled the decode graph |
| `GET /metrics` | request/token counts, TTFT, TPOT, KV cache bytes, and the compile state: `warm`, `prefill_lengths`, `steady_frames` |
| `GET /v1/models` | the one model this server holds |
| `POST /v1/chat/completions` | `stream: true` or `false` |
| `POST /v1/completions` | the pre-chat endpoint, same engine |
| `GET /docs` | FastAPI's generated schema for all of the above |

`/health` answers **while a generation is running** — measured at 10–30 ms
against a live decode loop on an M-series CPU. Every PyTorch call in the engine
goes through `asyncio.to_thread`, so the event loop stays free to answer probes
and to notice a client that has hung up. If `/health` ever starts timing out
under load, something has gone back to blocking the loop.

### `/health` reports what compiled, not what was asked for

```
$ curl -s localhost:8000/health          # before any request
{"status":"ok","model":"...","target":"cpu:arm64","backend":"auto"}
$ curl -s localhost:8000/health          # after one request
{"status":"ok","model":"...","target":"cpu:arm64","backend":"inductor"}
```

`--backend auto` is a request, not an answer. `compile_generation` compiles each
graph on its first call, so until a request has run there is genuinely nothing
to report and the requested value is the only truthful one.

## The static cache is the hard limit

`--max-model-len` allocates the KV cache at startup, on the target device, and it
never grows. `prompt + max_tokens` must fit inside it:

```
$ curl -s localhost:8000/v1/chat/completions -d '{"messages":[...],"max_tokens":600}'
{"detail":"The prompt is 31 tokens and 600 more were requested, which exceeds the
512-token static cache this server allocated at startup. Restart it with a larger
--max-model-len, or ask for fewer tokens."}
```

Checked **before** the response type is chosen, so an oversized request is a 400
with a reason rather than a 200 whose SSE stream dies after one chunk. Note the
cost of raising it: the cache is `2 × layers × kv_heads × head_dim × dtype_bytes
× max_model_len` bytes, allocated whether or not it is used.

## First request compiles

The graphs compile lazily, on their first call. The first request therefore pays
for Inductor and the rest do not, which shows up as a large TTFT average until
enough requests have run to dilute it. `/metrics` reports `warm: false` until
that first request completes, which is how the chat page knows to say
"compiling" rather than looking hung.

That LM7 compiled at all is checkable rather than assumed. On Apple M-series with
SmolLM2-135M-Instruct, `--target auto` resolving to `apple:metal`:

```
$ curl -s localhost:8000/metrics          # before any request
"backend": "auto",  "warm": false,  "prefill_lengths": 0,  "steady_frames": 0
$ curl -s localhost:8000/metrics          # after one message
"backend": "inductor",  "warm": true,  "prefill_lengths": 1,  "steady_frames": 0
```

and directly from the runner, which is where those numbers come from:

```
selected backend : inductor
counters         : prefill {frames: 1, unique_graphs: 1, graph_breaks: 0}
                   decode  {frames: 1, unique_graphs: 1, graph_breaks: 0}
                   steady  {frames: 0}
```

One graph per phase, no graph breaks, and nothing compiled in steady state. `--no-compile-prefill` leaves the prompt
pass in eager, which is worth it when prompt lengths vary: a compiled prefill is
compiled *per prompt length*, so a varied workload recompiles it repeatedly
while the decode graph — the one that runs a thousand times — is compiled once
either way. See [prefill and KV-cache decode](kv-cache-decode.md).

## Sampling

`temperature` and `top_p` are honoured; `temperature=0` is greedy, which is what
`GenerationRunner` does on its own. Sampling reads `state.logits` rather than
`state.next_token` (the runner's own argmax) and runs on the CPU in float32
whatever the model's device — three tensor ops on one row, where the transfer
dominates anyway, and `torch.multinomial` with a seeded generator is not
supported on every accelerator LM7 targets. `seed` makes a run reproducible.

Stop sequences are matched against the decoded text rather than token ids,
because a stop string is tokenized however the model felt like it and can
straddle two tokens. The stream holds back one character less than the longest
stop sequence, so a stop string is never emitted and then retracted.

## `--backend vllm`: hand over the port

```bash
lm7 model serve hf://owner/model --target nvidia --backend vllm --port 8000
```

This does not proxy, wrap, or re-implement anything. LM7 translates its target
and flags into vLLM's own `vllm serve` argv and hands over the process; what
answers the port afterwards is vLLM, unmodified, with every vLLM feature working
and none of LM7's behaviour above applying. `--dry-run` prints the exact command:

```
$ lm7 model serve hf://Qwen/Qwen3.5-0.8B --target apple --backend vllm --port 8200 --dry-run
model           Qwen/Qwen3.5-0.8B
target          apple:metal
runtime         vllm
address         http://127.0.0.1:8200
max_model_len   2048
vllm            /Users/you/.venv-vllm-metal/bin/vllm
command         vllm serve Qwen/Qwen3.5-0.8B --host 127.0.0.1 --port 8200 --max-model-len 2048
env             VLLM_HOST_IP=127.0.0.1
```

vLLM is **not** an LM7 extra, deliberately: it pins a specific PyTorch, and
pinning one here would decide the torch version for everyone who installs LM7
(this repo already has extras that disagree about torch — `litert` pins
`<2.13`). Install it yourself. A target vLLM has no backend for —
`intel:npu`, `tenstorrent`, `qualcomm` — is refused rather than launched,
because vLLM would otherwise fall back to whatever platform plugin it could load
and serve happily from the wrong device.

### On Apple Silicon: vllm-metal

vLLM has no macOS wheel on PyPI, but [vllm-metal](https://github.com/vllm-project/vllm-metal)
is a **platform plugin** — not a fork — that adds Apple Silicon through MLX, so
`vllm serve` is the same command:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

It builds its own `~/.venv-vllm-metal` (native arm64 Python 3.12) holding vLLM
plus the plugin. Two consequences LM7 handles rather than leaves to you:

- **LM7 cannot import that vLLM**, because it is in a different environment
  on purpose. So availability is not an import check: LM7 looks for an importable
  `vllm`, then a `vllm` on `PATH`, then vllm-metal's default venv, and
  `--dry-run` prints which one it found. An import-only check reports "not
  installed" on a machine where `vllm serve` runs perfectly well.
- **LM7 sets `VLLM_HOST_IP=127.0.0.1`** when the server is bound to loopback.
  Without it, vLLM initializes its `gloo` process group against the host's LAN
  address and **hangs on macOS** — no error, no timeout, startup simply stops
  after `PyTorch device set to: mps`, with `distributed_init_method=tcp://192.168.x.x`
  the only clue. Measured here: a hang of over ten minutes became a 130-second
  startup. An explicit `VLLM_HOST_IP` is never overridden, and a server bound to
  `0.0.0.0` is left alone, since a real address is required there and LM7 cannot
  guess it.

> **Validated on Apple Silicon, and nowhere else.** `lm7 model serve --target
> apple --backend vllm` was run end to end on an M-series Mac against
> `Qwen/Qwen3.5-0.8B` with vLLM 0.26.0 + vllm-metal 0.3.0: `/v1/models`, chat
> completions, SSE streaming, and the official `openai` SDK. **The CUDA, ROCm and
> TPU paths have still never been run** — no GPU box was rented for this. Note
> also that vllm-metal supports a specific model list; `SmolLM2-135M` is not on
> it, which is why this example uses Qwen3.5-0.8B (already in LM7's ladder).

## What has actually been run

Apple M-series, `SmolLM2-135M-Instruct`, transformers 5.14.1 / torch 2.13.0,
driven with `curl` and with the official `openai` Python SDK 2.53.0. Both
`--target cpu` (`cpu:arm64`) and `--target auto` (`apple:metal`, resolving to
`backend=inductor`):

- `/health`, `/metrics`, `/v1/models`
- chat completions, buffered and streamed; `/v1/completions`, buffered and streamed
- greedy output byte-identical to `model.generate` on the same prompt
- `/health` at 10–30 ms during a live generation
- 400 on an oversized prompt, on `n=4`; 200 on the `n=1`/zero-penalty defaults
  every OpenAI SDK sends; 422 on an empty `messages` array

> Serving on MPS did not work until this change. `compile_generation` compiles
> with `transfers="explicit"`, and that check compared an unindexed
> `torch.device("mps")` against the `mps:0` a transfer actually produces — so
> every request on `--target apple` was a 500, with an `InputDeviceError` saying
> `expected mps, got mps:0`. That is a **core** bug rather than a serving one
> (any `lm7.compile(..., transfers="explicit")` on an indexed device hit it, CUDA
> included); it simply had no caller until now. Fixed in `module.py:_same_device`
> — credit to [#115](https://github.com/lmontigny/lm7/pull/115), which found it
> independently.

Not run: `nvidia`, `intel:npu`, `tpu`, any model above 135M, and the vLLM
handover. **No timing here is a measurement.** The KV cache is allocated at
startup and the graphs compile inside the first request, so `/metrics` TTFT and
TPOT are compile-polluted until several requests have run. No serving benchmark
exists in this repo, and no claim about serving performance should be sourced
from it.

## Tests

```bash
python -m pytest tests/test_serve.py               # portable; no extra needed
python -m pytest -m serve                          # HTTP surface; needs [serve]
```

`tests/test_serve.py` drives the engine against a scripted runner and a
fake tokenizer — stop sequences, EOS, capacity refusal, cancellation, sampling,
the lock — and runs on a plain `[dev]` install. `tests/test_serve_integration.py`
checks the wire format an OpenAI client actually sees, and is in CI's
`serve` job.

Two tests are worth knowing about, because they check claims this page makes
rather than code paths:
`test_concurrent_requests_never_share_the_kv_cache` runs two overlapping
generations and asserts the runner never had more than one in flight, and
`test_the_event_loop_stays_responsive_during_generation` asserts a background
task still ticks while a decode loop runs.
