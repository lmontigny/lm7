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
lm7: the first request compiles the decode graph and will be slower. The prompt pass stays eager; --compile-prefill compiles it too.
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

## Where the model comes from

Two forms, and the rule between them is positional rather than clever:

```bash
lm7 model serve hf://HuggingFaceTB/SmolLM2-135M-Instruct   # the Hub
lm7 model serve ./my-finetune                              # a local directory
```

A **directory that exists on disk wins**, and a Hub id is only ever accepted
with its `hf://` prefix — so there is no ambiguity to arbitrate. A bare
`owner/model` that is not a directory was never valid and still isn't.

The local form is whatever `save_pretrained` wrote: `config.json`, the weights,
and the tokenizer files beside them. That is what makes a fine-tune, a
pre-downloaded checkpoint, or an air-gapped box reachable without a Hub round
trip. The path is resolved to an absolute one, and **that resolved path is the
served model id** — it appears in `/health`, in `/v1/models`, and in the `model`
field of every response:

```console
$ lm7 model serve ./local-smollm2 --target cpu --max-model-len 256
lm7: loading /abs/path/to/local-smollm2 for cpu:arm64...
$ curl -s localhost:8000/v1/models | jq -r '.data[0].id'
/abs/path/to/local-smollm2
```

Resolving matters because the server may change directory later and a client
reading `/v1/models` cannot resolve `./local-smollm2` against a cwd it does not
share. `--backend vllm` is handed the same resolved path, since `vllm serve`
takes a directory in the same positional slot as a Hub id.

A directory that is not a model is refused with the reason — no `config.json`,
a file where a directory was expected, or a path that does not exist each get
their own message rather than a Hugging Face URI error that would send someone
looking in the wrong place.

> **This widens where a model comes from, not what shape it can be.**
> `compile_generation` requires the Hugging Face causal-LM contract — the model
> must accept `past_key_values` and `cache_position` — so a custom architecture
> that manages its KV cache differently still will not serve, from either form.
> To serve a model object you already hold in memory, build the runner yourself
> with [`lm7.compile_generation`](kv-cache-decode.md) and hand it to
> `LM7ServeEngine`, which takes a prebuilt runner and tokenizer.

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

The page shows the server's `max_model_len`, has a `Max response` input that
sends `max_tokens`, and has a `Stop` button that aborts the streaming request.
The server already treats a disconnected streaming client as cancellation, so
stopping from the page releases the one generation slot instead of waiting for
the response to finish.

### What it costs: dtype, weights and memory

The header line says what is loaded and what it is occupying, because the two
questions anyone asks of a local server are "will this fit" and "did my flag do
anything":

```
unsloth/Llama-3.2-1B-Instruct · nvidia:sm89 · backend inductor · float16
  · 2048 ctx · kv 64 MiB · weights 2.30 GiB · gpu 2.37 GiB of 11.99 GiB
```

- **`float16`** is what the weights *are*, read off the runner — not what
  `--dtype` asked for. Same rule as `backend`: `auto` is a question, and it
  answers FP32 on CPU, FP16 on NVIDIA, and BF16 on NVIDIA once a weight-only
  quantization is in play. A server showing `auto` would be telling you nothing.
- **`weights`** is parameters plus buffers, each counted once — tied weights are
  a single allocation, which most small causal LMs have. It is the one memory
  figure that means the same thing on every target, and the only one that shows
  what a `--quantize` bought: SmolLM2-135M reports 513 MiB unquantized and
  **210 MiB** under `--quantize int8`, matching the 220 MB
  [recorded independently](#apple-m-series) when INT8 was first served.
- **`gpu 2.37 GiB of 11.99 GiB`** is the accelerator's allocator. It counts live
  tensors *this process* allocated: not the several hundred MB of CUDA context,
  not the caching allocator's reserved-but-unused blocks, and nothing another
  process holds — so it is always **smaller than `nvidia-smi`** for the same
  PID. On the run above, `nvidia-smi` said 3630 MiB against this 2.37 GiB.
- **`rss 1.53 GiB`** replaces it on a CPU target, where there is no allocator to
  ask. That is the whole interpreter, and PyTorch is most of a gigabyte before
  any model loads, so it is an upper bound on the server rather than a measure
  of the model — which is why the page never calls both of these "memory". Read
  `weights` for the model.

`memory_kind` on `/metrics` says which of the last two you are looking at, so a
script does not have to infer it from the target.

> **On Windows there is no memory figure at all.** It has neither `/proc` nor
> the `resource` module, and a `psapi.GetProcessMemoryInfo` call written for it
> returned 0 on the Windows CI runner, so it was removed rather than left in
> looking like support. `memory_bytes` is 0 there and the page omits the field;
> dtype, weights and KV cache are unaffected. Fixing it needs someone who can
> actually run a Windows box, which this project does not have.

> Neither memory figure is a leak detector. The device number moves with
> whatever the allocator is holding at the moment of the request, and RSS on
> Linux does not shrink just because Python freed something.

### The status line

Above the input, the page says what the server is doing — because "slow" and
"hung" look identical otherwise, and on a cold server the first message really
is slow:

```
cold — the first message compiles the graphs and will be slow
compiling the decode graph…                   ← first message, from /metrics warm:false
prefill…                                      ← subsequent messages, before the first token
generating · 88 char/s                        ← streaming
20 tokens · 412 ms to first token · 9.4 tok/s · eager prefill · 0 decode recompiles
```

The last line is the part no other local server can show you, and it is the
whole point of the two-graph split:

- **`0 decode recompiles`** is `steady_frames` from `runner.counters`. Above zero
  means a *token* triggered a compile — the regression separate prefill and
  decode graphs exist to prevent — and the page turns red and says `RECOMPILED`
  rather than burying it. See [prefill and KV-cache decode](kv-cache-decode.md).
- **`eager prefill`** is the default, and it is why the first line says *the
  decode graph* rather than both. Under `--compile-prefill` this slot becomes
  **`N prefill graph(s)`** — the cost that split accepts, one compile per
  distinct prompt length, climbing as prompt lengths vary while recompiles stay
  at 0. The page reads `compile_prefill` from `/metrics` to know which it is
  looking at, because `prefill_lengths` climbs either way and only counts
  compiles when the prompt pass is compiled.

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
client is not on this machine — the default binds to loopback. Once it is
reachable off-machine, set `--api-key` and pass it as `OPENAI_API_KEY`, and
narrow `--cors-origins`; see [access control](#access-control-cors-and-api-keys).

Also known to work against an OpenAI-compatible base URL: **Continue** and
**Cline** (VS Code), **Zed**'s assistant, **Aider**, **LibreChat**, and any
`openai` SDK. Expect one caveat everywhere: this server refuses `n > 1`,
logprobs, tool calling and structured output with a 400 (see below), so a client
that depends on tools will report an error rather than degrade.

> These clients have **not** been tested against LM7 — they are listed because
> they consume the same API, not because anyone here has run them. The `openai`
> SDK is what has actually been driven end to end.

A local MacBook Pro smoke test also drove two `/v1/chat/completions` turns
against SmolLM2-135M on `cpu:arm64`, resending the first assistant message in the
second request. That is the multi-turn contract LM7 supports: the client owns
the transcript, LM7 compiles and answers one request at a time, and
`steady_frames` stayed at 0. It is not tool-calling agent support; requests with
`tools` or `tool_choice` are refused rather than silently ignored.

## Endpoints

| | |
| --- | --- |
| `GET /` | the built-in chat page (not in the OpenAPI schema) |
| `GET /health` | model, target, and the backend that compiled the decode graph |
| `GET /metrics` | request/token counts, TTFT, TPOT, [dtype and memory](#what-it-costs-dtype-weights-and-memory), and the compile state: `warm`, `prefill_lengths`, `steady_frames` |
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
never grows. `--max-sequence-length` is the same setting under
`compile_generation`'s name for it — one static cache, two spellings, so neither
can silently lose to the other. It defaults to **4096** tokens.
`prompt + max_tokens` must fit inside it:

```
$ curl -s localhost:8000/v1/chat/completions -d '{"messages":[...],"max_tokens":600}'
{"detail":"The prompt is 31 tokens and 600 more were requested, which exceeds the
512-token static cache this server allocated at startup. Ask for at most 481, omit
max_tokens to use whatever fits, send a shorter conversation, or restart the server
with a larger --max-model-len."}
```

Checked **before** the response type is chosen, so an oversized request is a 400
with a reason rather than a 200 whose SSE stream dies after one chunk. Note the
cost of the default: the cache is `2 × layers × kv_heads × head_dim ×
dtype_bytes × max_model_len` bytes, allocated whether or not it is used, so
4096 tokens costs twice what 2048 does on a machine that may never send a prompt
that long. Lower it on a small device; the startup line prints what it took.

### Omitting `max_tokens` asks for whatever fits

`max_tokens` is optional, and leaving it out is not "unlimited" — it is
`max_model_len − prompt_tokens`, computed per request.

The built-in chat page leaves that value unset when `Max response` is empty, or
sends it explicitly when you enter a number.

That matters for any client that resends a conversation, which is every chat
client, because the engine has one static cache and no notion of a session. The
prompt grows every turn, so **a constant `max_tokens` is a wall, not a limit**:
a client asking for half the cache on every turn succeeds until the transcript
crosses half the cache, and from that moment every single turn is arithmetically
impossible. Omitting it instead makes replies get shorter as the conversation
grows, which degrades rather than stops.

An *explicit* `max_tokens` that does not fit is still refused rather than
narrowed — a caller that asked for 512 and silently received 40 has been misled —
but the refusal now names the largest number that would have worked.

When the prompt alone fills the cache there is no budget to offer, so that is a
different message pointing at the conversation rather than at the token count.

## Access control: CORS and API keys

The [built-in page](#talking-to-it-from-a-browser) is served by this process, so
it has no origin to cross. Any *other* browser UI — Open WebUI, a Next.js app on
`:3000` — is a different origin from `127.0.0.1:8000`, and the browser will not
hand it the response body unless the server says the origin is allowed. That is
what `--cors-origins` sets:

```bash
lm7 model serve hf://owner/model --cors-origins "http://localhost:3000"
```

It defaults to `*`, because the server binds loopback, holds no credentials and
is single-user — the ordinary case is a UI on another local port, and a default
that broke it would just be turned off by everyone. Narrow it whenever the
server is reachable from anywhere but this machine, and `--cors-origins ""`
turns CORS off entirely rather than falling back to the wildcard.

`--api-key` adds a bearer check, for a server on a shared machine or behind a
tunnel:

```bash
lm7 model serve hf://owner/model --api-key s3cret
curl -H "Authorization: Bearer s3cret" localhost:8000/v1/models
```

Two deliberate holes in it, both so the thing works at all:

- **`/health` answers without a key**, so a container probe or a shell loop can
  wait for the model to finish loading without being trusted to generate.
- **A preflight `OPTIONS` is never authenticated**, because browsers do not send
  `Authorization` on one. Requiring a key there would fail every cross-origin
  request before the real one was sent.

A 401 still carries its CORS headers. Without that a browser reports the refusal
as a CORS failure, which sends whoever is debugging it to the wrong file.

> **`--api-key` turns the built-in chat page off.** A page fetched by a browser
> cannot attach an `Authorization` header, so `GET /` is refused like everything
> else and the 401 says so rather than rendering blank. A key is for a server
> reachable by something other than you; the page is for the case where it is
> not. Use one or the other.

This is a bearer check on a loopback server, not an authorization system: one
key, no rotation, no per-client identity, and the token is compared in constant
time but travels in plaintext unless something in front of it terminates TLS.

## Quantizing what gets served

`--quantize` quantizes the weights before the decode loop is compiled, so what
is compiled is what is served:

```bash
lm7 model serve hf://owner/model --target nvidia --quantize int8
```

`none`, `int8`, `fp8` and `nvfp4`, gated by exactly the rules in
[quantization](quantization.md) — the mode is checked against the target,
backend and dtype **before** the checkpoint is downloaded, and a filter that
matches no layer in the model is a refusal rather than a silent no-op that would
report a quantization that never happened. `--backend eager` plus `--quantize`
is refused for the same reason.

`lm7 model compatibility <model> --target <target>` reports each mode's gate
without downloading weights, which is the cheapest way to find out what a
machine can do. On Apple Silicon the answer is nothing: quantization runs
through TorchAO, `int8` is gated to NVIDIA and CPU targets, and `fp8`/`nvfp4` to
NVIDIA alone — so `--target apple --quantize int8` is refused and `--target cpu`
is the local option.

> **`--quantize` needs the `hf://` form.** The gate is per model as well as per
> target, keyed by Hugging Face id against a list of checkpoints someone has
> actually validated — and a local directory carries no id, because
> `save_pretrained` does not record where the checkpoint came from. A local
> directory plus `--quantize` is therefore refused, saying so. Serving that same
> directory unquantized is unaffected.

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

One graph per phase, no graph breaks, and nothing compiled in steady state.

Those two runs predate the default flipping and so are what `--compile-prefill`
now produces: a `prefill` counter with a frame in it, and a `prefill_lengths`
that is a count of compiles. **By default the prompt pass is left eager**, and
only the decode graph is compiled — so the `prefill` counter stays empty,
`prefill_lengths` still climbs with each distinct prompt length but costs
nothing, and `/metrics` reports `compile_prefill: false` so a reader can tell
those two situations apart. A compiled prefill is compiled *per prompt length*,
which a server sees a new one of on nearly every request, while the decode graph
— the one that runs a thousand times — is compiled once either way. See
[prefill and KV-cache decode](kv-cache-decode.md) for where compiling the prompt
pass does and does not pay, and [the defaults](#why-the-prompt-pass-is-eager-here)
for the measurement that moved it.

### Why the prompt pass is eager here

`lm7.compile_generation` compiles the prompt pass by default; this server does
not. The two defaults disagree on purpose, because they are aimed at different
workloads — and the flag is the same one either way, `compile_prefill`, spelled
`--compile-prefill` on the command line.

A compiled prefill is compiled **once per distinct prompt length**. That is a
good trade for a caller that sends one shape repeatedly, which is what a
benchmark harness does. It is close to the worst case for a chat client, which
resends its whole transcript every turn and therefore arrives at a length it has
never seen before on essentially every request.

Measured on an RTX 4070 SUPER (Ada `sm89`, 12 GiB) under WSL2, torch
2.13.0+cu130, `SmolLM2-135M-Instruct`, `--target nvidia`:

| | first request | a later request at a new prompt length | a repeat length |
| --- | --- | --- | --- |
| compiled prefill (`--compile-prefill`) | ~100 s | **~80 s** | 0.13–0.45 s |
| eager prefill (the default) | 11.4 s | **0.54 s** | 0.13–0.45 s |

So the browser page this server ships with used to pay a fresh ~80 s stall on
nearly every turn, with `/metrics` showing exactly that: `prefill_lengths`
climbing by one each time while `steady_frames` stayed 0. Nothing was broken —
the design was working as documented — but the default made the page look hung,
which is the one thing the status line exists to prevent.

The same conclusion arrives from the other direction in [prefill and KV-cache
decode](kv-cache-decode.md#compiling-prefill-stops-paying-at-about-2048-tokens):
compiling the prompt pass is 2.26x faster at 512 tokens, a tie at 2,048, and
*slower* beyond that. It pays for short prompts at a repeated shape, and a chat
server is neither.

`--compile-prefill` turns it back on for a workload whose prompt lengths repeat.
`--no-compile-prefill` still parses — it is what this behaviour used to be called
— and is now a no-op.

Confirmed on an Apple M3 Pro (macOS 26.5.2, torch 2.13.0, transformers 5.15.0,
`SmolLM2-135M-Instruct`, `--target cpu` → `cpu:arm64`, `--max-model-len 512`):
the first request took **22.9 s** and a second request at a different prompt
length took **0.41 s**, with `/metrics` reporting `warm: true`,
`compile_prefill: false`, `prefill_lengths: 2` and `steady_frames: 0`. Two
distinct lengths, no second compile — which is the whole of the change.

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
- **LM7 sets `VLLM_WSL2_ENABLE_PIN_MEMORY=1`** on WSL2 kernels at or above
  4.19.121. vLLM turns pinned memory off whenever it detects WSL, and since 0.26
  its CUDA worker allocates a UVA buffer that requires it, so `vllm serve` dies
  with `RuntimeError: UVA is not available` before any model loads. vLLM's own
  gate says pinned memory *works* on those kernels and is merely off by default,
  so LM7 turns it back on. Below that version the default is a real limitation
  rather than a cautious one, and LM7 leaves it alone — as it does an explicit
  setting, in either direction.

### Flags LM7 does not model

`vllm serve` has hundreds of engine flags. LM7 translates the handful that mean
the same thing on both sides — the model, host, port, `--max-model-len`,
`--dtype` — and mirroring the rest would make it a second, always-stale copy of
vLLM's CLI. `--vllm-arg` is the escape hatch:

```bash
lm7 model serve hf://owner/model --target nvidia --backend vllm \
  --vllm-arg=--gpu-memory-utilization --vllm-arg 0.8
```

Repeatable, one argument each, and appended **last** so a flag spelled out this
way beats whatever LM7 translated — argparse takes the final occurrence, and a
caller who names a flag means it. With LM7's own backend it is refused rather
than dropped, since a server started while ignoring engine arguments is not the
server that was asked for.

The motivating case is the one above: vLLM asks for 92% of the card by default,
which is more than a 12 GiB desktop GPU has free once a display server is
attached, and it exits rather than shrinking.

### A browser page for the vLLM path

vLLM serves an API and no page, so `http://127.0.0.1:8200/` is empty. `--ui-port`
puts the same chat page beside it:

```bash
lm7 model serve hf://Qwen/Qwen3.5-0.8B --target apple --backend vllm \
  --port 8200 --ui-port 8201        # then open http://127.0.0.1:8201
```

```
:8201   the chat page, served by LM7 from the standard library
:8200   vLLM  ← the browser talks to this directly
```

LM7 hands out one HTML file and **nothing else** — no proxy, no relay, still not
in the request path. The page is served by `http.server` rather than the `serve`
extra's FastAPI, because it returns one string and needs no routing, no
validation and no dependency; it comes up immediately, while vLLM is still
loading.

Two details that make it work against a server that is not LM7's:

- **The page is a client, not a view.** Its API base URL is substituted at render
  time — empty for LM7's own server (same origin, relative paths), a full origin
  here. It reads the model name from `/v1/models`, the one endpoint every
  OpenAI-compatible server has, and sends it as `model` because vLLM requires
  that field while LM7 treats it as optional.
- **LM7's own `/metrics` is treated as absent when it is.** vLLM answers
  `/metrics` in Prometheus text and `/health` with an empty body, so the page
  degrades to model name and timings rather than erroring. Compile state,
  `steady_frames` and KV bytes appear only against LM7.

No CORS flag is needed: vLLM answers `access-control-allow-origin: *` by default,
preflight included. If you narrow it with vLLM's own `--allowed-origins`, include
`http://127.0.0.1:8201`.

`--ui-port` is refused for LM7's own backend, which already serves the page at
`/` — a second copy on another port would be a puzzle, not a feature. It works
for every launcher backend, `--backend trtllm` included.

> **Validated on Apple Silicon, and nowhere else.** `lm7 model serve --target
> apple --backend vllm` was run end to end on an M-series Mac against
> `Qwen/Qwen3.5-0.8B` with vLLM 0.26.0 + vllm-metal 0.3.0: `/v1/models`, chat
> completions, SSE streaming, the official `openai` SDK, and `--ui-port`'s page
> driving all three cross-origin. Note also that vllm-metal supports a specific
> model list; `SmolLM2-135M` is not on it, which is why this example uses
> Qwen3.5-0.8B (already in LM7's ladder).
>
> **CUDA has since been run too**, on an RTX 4070 SUPER (Ada `sm89`) under WSL2
> with vLLM 0.26.0 against `unsloth/Llama-3.2-1B-Instruct`: the handover, the
> chat page on `--ui-port`, `/v1/models`, and a chat completion, with `n=4`
> answering 200 where LM7's own server refuses it. Two things had to be fixed
> for it to start at all — `VLLM_WSL2_ENABLE_PIN_MEMORY` and `--vllm-arg`, both
> above. One remains outside LM7: on this box FlashInfer JIT-compiles its sampler
> at startup, wants `nvcc` and `ninja`, and then fails because the CCCL headers
> it bundles reject CUDA 13.3 — so the server was started with
> `VLLM_USE_FLASHINFER_SAMPLER=0` in the environment, which LM7 passes through
> and does not set. **ROCm and TPU have still never been run.**

## `--backend trtllm`: the same handover, to TensorRT-LLM

```bash
lm7 model serve hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target nvidia --backend trtllm --port 8000
```

Same shape as `--backend vllm`, and deliberately so: LM7 translates its config
into `trtllm-serve`'s own argv and hands over the process. Both are *launcher
backends*, and everything the previous section says about not being in the
request path applies here word for word.

Sharing the shape is the point. `--dry-run` answers the same questions for
either — which executable was found, what argv it gets, what changes in the
environment — and `--ui-port` works the same way, because the plan is built once
for both rather than per backend:

```console
$ lm7 model serve hf://HuggingFaceTB/SmolLM2-135M-Instruct \
    --target nvidia --backend trtllm --max-model-len 2048 --dry-run
model           HuggingFaceTB/SmolLM2-135M-Instruct
target          nvidia:sm89
runtime         trtllm
address         http://127.0.0.1:8000
max_model_len   2048
trtllm          /home/you/.venv-trtllm/bin/trtllm-serve
command         trtllm-serve HuggingFaceTB/SmolLM2-135M-Instruct --host 127.0.0.1 --port 8000 --max_seq_len 2048
```

Three translation decisions worth stating, because each is a place LM7 could
have quietly done the wrong thing:

- **`--max-model-len` becomes `--max_seq_len`.** The same quantity under each
  side's own name. LM7 does not reinterpret it, and does not rename
  `trtllm-serve`'s flags to look like its own — the printed command is the real
  one, meant to be copied into a shell.
- **LM7 does not pass `--backend` through.** The flag exists on both sides and
  means different things: LM7's picks the launcher, `trtllm-serve`'s picks
  between its PyTorch runtime and a TensorRT engine. Passing one as the other
  would be a silent mistranslation, so LM7 passes neither and TensorRT-LLM keeps
  its own default (on 1.2.x, the PyTorch runtime with in-flight batching — *not*
  a prebuilt TensorRT engine).
- **`--quantize` is refused, not ignored.** It quantizes weights in LM7's own
  decode loop, which is not in this path at all. TensorRT-LLM quantizes at engine
  build time from a checkpoint NVIDIA ModelOpt has already produced; serve one of
  those instead.

Everything else reaches TensorRT-LLM through **`--trtllm-arg`**, the same
verbatim passthrough `--vllm-arg` provides for the other launcher — repeatable,
appended last so it beats anything LM7 translated. On a desktop card it is the
flag you will actually want, since TensorRT-LLM sizes its paged cache from free
GPU memory:

```bash
lm7 model serve hf://HuggingFaceTB/SmolLM2-135M-Instruct --target nvidia \
  --backend trtllm --trtllm-arg=--free_gpu_memory_fraction --trtllm-arg 0.25
```

Each passthrough belongs to one launcher and is **refused** by the other, and by
LM7's own server — the two CLIs share no spelling (`--max-model-len` against
`--max_seq_len`), so handing vLLM's flags to `trtllm-serve` could only produce an
argv that does not parse.

A non-NVIDIA target is refused, and so is a pre-Ampere NVIDIA card: TensorRT-LLM
has no kernels below `sm80` and fails during engine construction rather than
falling back, so the refusal names the card instead of arriving as a CUDA error
several minutes into a load.

### It needs its own environment

TensorRT-LLM pins `torch`, `transformers` and `tensorrt` to versions that
conflict with every other environment in this repo, so it cannot be an LM7 extra
for exactly the reason vLLM cannot. Install it in a venv of its own and put that
venv on `PATH`; LM7 looks for an importable `tensorrt_llm`, then `trtllm-serve`
on `PATH`, then `~/.venv-trtllm/bin/trtllm-serve`. The install, the version set
and what was measured are in [TensorRT-LLM](tensorrt-llm.md).

**Handing over the process is load-bearing here**, not just tidy. TensorRT-LLM
spawns MPI workers that re-execute the parent's command line; a launcher that
drove the Python API in-process would have its workers re-run `python -m lm7`,
hit argparse and `MPI_ABORT` the job *after* the model had loaded. `trtllm-serve`
is its own entry point, so its workers re-execute it and the re-exec is harmless.
[TensorRT-LLM](tensorrt-llm.md#why-this-is-a-launcher-and-not-an-in-process-runtime)
has the whole story, including what the in-process version cost.

## What has actually been run

### Apple M-series

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

The **token budget** was checked against the failure that motivated it, on a
1024-token cache: a 660-token transcript with an explicit `max_tokens: 512` is
refused and told to ask for at most 364, and the same transcript with
`max_tokens` omitted generates (660 prompt + 19 completion, `finish: stop`).

The **deployment flags** were exercised against that same local checkpoint on
`cpu:arm64`: `/health` answering without a key while `/v1/models` returned 401
without one, 401 with a wrong one and 200 with the right one; a 401 still
carrying `access-control-allow-origin`; a preflight `OPTIONS` succeeding with no
`Authorization` header and echoing back `authorization, content-type`; a
disallowed origin getting no CORS header at all; and a generation completing
through the key.

**`--quantize int8` has now been served**, on `cpu:arm64` with
SmolLM2-135M-Instruct: 210 modules converted, model storage 538 MB → 220 MB
(2.44x), and a served request answered correctly by the quantized model. Both
refusals were checked the same way — `--target apple --quantize int8` on the
vendor gate, `--backend eager --quantize int8` on the backend gate, each firing
before the checkpoint downloads. `fp8` and `nvfp4` are unreachable from an
M-series machine — they are gated to NVIDIA — and were served later on
[`sm89`](#on-nvidia-rtx-4070-super-ada-sm89).

The **local-directory form** was run the same way, on `cpu:arm64`: the same
checkpoint written out with `save_pretrained` and served as `./local-smollm2`,
with `--dry-run` resolving the relative path, `/health` and `/v1/models`
reporting the absolute one, a buffered completion, an SSE stream, and a round
trip through the `openai` SDK. `/health` reported `backend=auto` before the
first request and `backend=inductor` after it, as it does for a Hub model.

> Serving on MPS did not work until this change. `compile_generation` compiles
> with `transfers="explicit"`, and that check compared an unindexed
> `torch.device("mps")` against the `mps:0` a transfer actually produces — so
> every request on `--target apple` was a 500, with an `InputDeviceError` saying
> `expected mps, got mps:0`. That is a **core** bug rather than a serving one
> (any `lm7.compile(..., transfers="explicit")` on an indexed device hit it, CUDA
> included); it simply had no caller until now. Fixed in `module.py:_same_device`
> — credit to [#115](https://github.com/lmontigny/lm7/pull/115), which found it
> independently.

### On NVIDIA (RTX 4070 SUPER, Ada `sm89`)

The same surface, on the local dev GPU — an RTX 4070 SUPER (Ada `sm89`, 12 GiB)
under WSL2, driver 595.45.03 / CUDA 13.2, torch 2.13.0+cu130, transformers
5.14.1, fastapi 0.141.1 / uvicorn 0.52.1, `openai` SDK 2.53.0. `--target nvidia`
resolves to `nvidia:sm89` and `backend=auto` becomes `inductor` after the first
request, exactly as it does on Metal. Driven with `curl`, `httpx` and the
`openai` SDK against `SmolLM2-135M-Instruct` and `unsloth/Llama-3.2-1B-Instruct`:

- `/health`, `/metrics`, `/v1/models`; the chat page at `/` (14 KB, no external
  reference, absent from `/openapi.json`)
- chat completions and `/v1/completions`, buffered and streamed, through `curl`
  and through the SDK
- **greedy output byte-identical to `model.generate`** on the same GPU, for both
  models, on two prompts each
- `/health` answered in 1.9–12.1 ms (median 2.7) while a generation was running
- two overlapping generations serialized rather than interleaving — 0.15 s and
  0.32 s, wall clock 0.32 s — and both returned the same greedy answer
- a client hanging up mid-SSE stopped the generation, which was still counted in
  `/metrics`, and the next request was served normally
- 400 on `n=4`, on `logprobs`, and on an oversized `max_tokens` (naming the 993
  that would have fit); 422 on an empty `messages`; 200 on the SDK's defaults
- the deployment flags, checked the same way as on `cpu:arm64` above, including
  `GET /` returning 401 while a key is set

`steady_frames` stayed **0** across every request of every run above, which is
the claim the two-graph split exists to support. `prefill_lengths` climbed with
the number of distinct prompt lengths, as designed — and on this box that is
expensive: the first request compiled for **~100 s** and each new prompt length
cost roughly **80 s** more, against a warm request of 0.13–0.45 s.
Leaving the prompt pass eager brought the first request to 11.4 s and a second,
unseen prompt length to 0.54 s.

**This run is why eager prefill is now the default**, and why the numbers above
are what `--compile-prefill` costs rather than what the server does out of the
box — see [why the prompt pass is eager here](#why-the-prompt-pass-is-eager-here).
It was measured before that change, so the run itself is unmodified.

**Both quantized modes NVIDIA gates now serve**, from `--dtype auto`:
`--quantize int8` and `--quantize fp8` on SmolLM2-135M-Instruct, and
`--quantize nvfp4` on Llama-3.2-1B-Instruct — the per-model gate refuses NVFP4
for SmolLM2, so that pairing is the one the validated list allows. `fp8` needs
`sm89`, which this card is; weight-only `nvfp4` carries no capability floor
(only `nvfp4-dynamic` requires `sm100`, and `--quantize` does not offer it).

> **INT8 and FP8 did not work until this change**, and failed silently rather
> than loudly. `LM7ServeEngine.load` resolved its dtype without telling
> `_resolve_dtype` which quantization was coming, so `--dtype auto` on NVIDIA
> gave the FP16 an *unquantized* model gets instead of the BF16
> `_QUANTIZED_COMPUTE_DTYPE` mandates. INT8 weights under FP16 compute produce
> NaN logits, so every token was an argmax over NaN — token 0, which is
> `<|endoftext|>` and *not* SmolLM2-Instruct's EOS. The server therefore ran to
> its full budget and returned **HTTP 200 with an empty string** and
> `finish_reason: "length"`. The gate refuses an explicit `--dtype float16`
> alongside `--quantize`, so `auto` was the only way in. It never showed on CPU,
> where quantized and unquantized `auto` are both FP32 — which is why the
> `cpu:arm64` validation above missed it. Fixed in `serve/engine.py`, with a
> portable regression test in `tests/test_serve.py`.

### On x86-64 CPU (Intel Coffee Lake, AVX2)

The `cpu:arm64` runs above are Apple's vector unit; this is the other one. An
Intel i7-8086K (Coffee Lake, 12 threads, **AVX2 and no VNNI**) under WSL2, torch
2.13.0+cu130, transformers 5.14.1, SmolLM2-135M-Instruct, `--target cpu`
resolving to `cpu:x86_64` and `backend=auto` becoming `inductor`:

- `/health`, `/metrics`, `/v1/models`, and the chat page at `/`
- chat completions and `/v1/completions`, buffered and streamed
- **greedy output byte-identical to `model.generate`** on two prompts, against an
  FP32 eager reference on the same machine
- 400 on `n=4` and on an oversized `max_tokens` (naming the 481 that would have
  fit), 422 on an empty `messages`
- `steady_frames` 0, `prefill_lengths` 1
- **`--quantize int8` served correctly**, which had only been run on `cpu:arm64`

Two things this target makes visible that the others do not:

- **The KV cache is twice the size of the same setting on a GPU.** 23.6 MB at
  `--max-model-len 512` against 11.8 MB on `nvidia:sm89`, because `--dtype auto`
  is FP32 on CPU and FP16 on NVIDIA. Worth knowing before sizing a cache from a
  number read off a GPU run.
- **INT8 buys nothing here.** Warm requests were 0.53 s unquantized and
  0.44–0.76 s across three INT8 samples — noise, not a speedup. This CPU has AVX2
  and no VNNI, which is exactly the case
  [quantization](quantization.md) says INT8 latency does not transfer out of.
  The point of running it was that the quantized path *works*, not that it is
  fast.

Compiling the first request took ~2m10–2m17s, against ~100 s for the same model
on the 4070.

Not run: the vLLM CPU handover, `--api-key`/`--cors-origins` on this target
(they are transport middleware and target-independent), and anything above 135M.

### On Linux Arm (Arm Neoverse N3)

The `cpu:arm64` runs above are Apple Silicon, which is macOS. This is the same
architecture family on the other operating system: a GCP `n4a-standard-8`
(Google Axion, **Arm Neoverse N3**, 8 vCPU, Debian 12), torch 2.13.0+cpu,
transformers 5.15.0, `SmolLM2-135M-Instruct`, `--target cpu`.

```
lm7: loading HuggingFaceTB/SmolLM2-135M-Instruct for cpu:aarch64...
lm7: serving ... on http://127.0.0.1:8124 (cpu:aarch64, backend=auto, max_model_len=512, kv cache 24 MB)
lm7: the first request compiles the prefill and decode graphs and will be slower.
```

That last line is the banner as it read when this run was made, with the prompt
pass compiled by default; today it names the decode graph alone unless
`--compile-prefill` is passed. The transcript is left as it was recorded.

**The target string in the API is `cpu:aarch64`, not `cpu:arm64`.** Every other
`arm64` line on this page came from a Mac, and `platform.machine()` spells the
same family `arm64` on macOS and `aarch64` on Linux — see
[CPU inference](cpu.md#on-aarch64-the-kernel-prints-less). It is one target
family with two spellings and no separate code path, but the spelling is visible
in `/health` and `/metrics` output, so a client that string-matches `cpu:arm64`
to decide anything will not match a Linux Arm server:

```json
{"status":"ok","model":"...","target":"cpu:aarch64","backend":"auto"}
```

What ran:

- `/health`, `/metrics`, `/v1/models`
- chat completions, buffered and streamed; `/v1/completions`, buffered
- **greedy output byte-identical to `model.generate`** on the same prompt,
  against an FP32 eager reference on the same host
- 400 on `n=4`, 422 on an empty `messages` array, and 400 on an oversized
  `max_tokens` naming the 481 that would have fit

`/metrics` reports `dtype: float32`, `weights_bytes` 538,060,288 (513 MiB) and
`kv_cache_bytes` 23,592,960 at `--max-model-len 512` — the same cache size as
the x86-64 row above, and for the same reason: `--dtype auto` is FP32 on every
CPU, so the KV cache is twice what the same setting allocates on a GPU. Nothing
about the Arm target changes that.

**`--quantize int8` has been served here too.** `/metrics` reports
`weights_bytes` 220,332,928 against 538,060,288 unquantized — 513 to 210 MiB,
the same 2.44x the offline path gets — with `dtype` still `float32`, and the
completion is byte-identical to the unquantized one above. It is a footprint
change and not a speed one: neither INT8 mechanism is faster than FP32 on this
part, which is [measured in
quantization](quantization.md#int8-on-cpu-has-two-mechanisms-and-they-differ-by-4x)
rather than assumed from `i8mm` being absent from the kernel's reach.

Not run here: the deployment flags (`--api-key`, `--cors-origins`), the
local-directory form, the vLLM handover, and anything above 135M. The first two
are transport middleware and target-independent, and were checked on the two CPU
targets above.

Still unrun anywhere: `intel:npu` and `tpu`, and the ROCm and TPU vLLM handovers.
Nothing above 1B has been served on any target. **No timing on this page is a
measurement.** The KV cache is allocated at startup and the graphs compile inside
the first request, so `/metrics` TTFT and TPOT are compile-polluted until several
requests have run; the figures quoted throughout are wall clock from a client on
loopback, and they are here to show which order of magnitude a flag moves things
by. No serving benchmark exists in this repo, and no claim about serving
performance should be sourced from it.

### The TensorRT-LLM handover, on the same card

`--backend trtllm` was run end to end on that same RTX 4070 SUPER under WSL2,
against TensorRT-LLM 1.2.1 and `SmolLM2-135M-Instruct`: the server came up from
LM7's own argv, `/v1/models` listed the model as `owned_by: tensorrt_llm`, a
chat completion answered correctly, and an SSE stream reassembled. Four
integration tests pass in 120 s, startup included.

Cold start was ~125 s and the launched server held 11.9 GiB of the 12 GiB card
for a 135M model, because TensorRT-LLM sizes its paged cache from free memory —
422,048 tokens of it, against the 2049 that `--max-model-len` bounds a single
request to. The install needed six things beyond `pip install tensorrt-llm`
before it would run at all outside NVIDIA's container. Both stories, and the
timings, are in [TensorRT-LLM](tensorrt-llm.md).

**Both launchers now start on this card**, which is the useful thing to know
about the shared layer: `--dry-run`, `--ui-port` and the refusals come from one
implementation, and each backend has now put a real server on the port. vLLM
still has no throughput measurement here; TensorRT-LLM does — see below.

### When the single-stream server stops being the right answer

[`benchmarks/serving_backends.py`](../benchmarks/serving_backends.py) drives
this server and the TensorRT-LLM handover from one client, over the same HTTP,
so the two are comparable. On an RTX 4070 SUPER with SmolLM2-135M, the answer
has a clear shape:

| | this server, `reduce-overhead` | `--backend trtllm` |
| --- | --- | --- |
| time to first token | **13 ms** | 50 ms |
| one stream | 143 tok/s | 174 tok/s |
| **eight streams, aggregate** | 143 tok/s | **1,139 tok/s** |
| worst TTFT at eight streams | 6.3 s | **0.10 s** |
| GPU held, over idle | **615 MiB** | 10,268 MiB |

For one caller this server is the better answer — sooner to the first token, 17x
less memory, and within 1.21x per token. It stops being an answer somewhere
between one caller and two, and the flat line is the `asyncio.Lock` described
above doing exactly what it says. Full table, method and caveats in
[TensorRT-LLM](tensorrt-llm.md#against-the-inductor-path).

The two needed different amounts of help to get there, and the difference is
instructive. vLLM needed two changes *inside* LM7 (`VLLM_WSL2_ENABLE_PIN_MEMORY`
and `--vllm-arg`) because its failures were things a launcher could fix.
TensorRT-LLM needed none, and instead needed six things fixed in its own
environment before it would import — which is why LM7 sets nothing for it and
documents the environment as a prerequisite.

## Tests

```bash
python -m pytest tests/test_serve.py               # portable; no extra needed
python -m pytest -m serve                          # HTTP surface; needs [serve]
python -m pytest -m serve_load                     # a real model; needs [serve,hf]
python -m pytest -m trtllm                         # a real trtllm-serve and a GPU
```

`tests/test_serve.py` drives the engine against a scripted runner and a
fake tokenizer — stop sequences, EOS, capacity refusal, cancellation, sampling,
the lock — and runs on a plain `[dev]` install. It also covers both launchers'
whole contribution: the argv they build and what they refuse, with neither vLLM
nor TensorRT-LLM installed. `tests/test_serve_integration.py` checks the wire
format an OpenAI client actually sees, and is in CI's `serve` job.

`tests/test_trtllm_serve_integration.py` is the one that cannot be faked: it
launches a real `trtllm-serve` **from `serve_plan`'s own argv** and talks to it
over HTTP, so what is checked is the command LM7 would actually run rather than
a hand-written one that could pass while `lm7 model serve` was broken. It needs
an Ampere-or-newer GPU and TensorRT-LLM's own environment, so it is not in CI.

### The load path needs a real model to test at all

Both files above use fakes, which is right for testing the wire format and wrong
for testing what happens before it. Nothing in them executes
`LM7ServeEngine.load` — resolving what the user typed, both `from_pretrained`
calls, the quantization gate, `compile_generation`, the static cache landing on
the target — and **that is where every serve bug found by hand has been**: an
indexed-device comparison that made `--target apple` 500 on every request; a
quantization gate that printed a filesystem path where its own text promised a
model id.

`tests/test_serve_load_integration.py` closes that, using
`hf-internal-testing/tiny-random-LlamaForCausalLM` — about **15 MB** of random
weights. Its output is gibberish and nothing asserts otherwise; what is checked
is that a real checkpoint travels from a URI to a token over HTTP, from the Hub
and from a `save_pretrained` directory, that the cache is sized by
`--max-model-len` rather than merely validated against it, and that the
quantization gate refuses a model nobody has validated. Generated-text
correctness belongs to [prefill and KV-cache decode](kv-cache-decode.md), which
compares against eager on real models.

It runs in CI's `serve` job, in a step **after** the fakes-only run, so the
existing signal — that the HTTP surface works without Transformers installed —
is preserved rather than absorbed.

Two tests are worth knowing about, because they check claims this page makes
rather than code paths:
`test_concurrent_requests_never_share_the_kv_cache` runs two overlapping
generations and asserts the runner never had more than one in flight, and
`test_the_event_loop_stays_responsive_during_generation` asserts a background
task still ticks while a decode loop runs.
