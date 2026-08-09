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

## Endpoints

| | |
| --- | --- |
| `GET /health` | model, target, and the backend that compiled the decode graph |
| `GET /metrics` | request count, token counts, TTFT, TPOT, KV cache bytes |
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
enough requests have run to dilute it. `--no-compile-prefill` leaves the prompt
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
$ lm7 model serve hf://owner/model --target cpu --backend vllm --dry-run
model           owner/model
target          cpu:arm64
runtime         vllm
address         http://127.0.0.1:8000
max_model_len   2048
vllm            NOT INSTALLED
command         vllm serve owner/model --host 127.0.0.1 --port 8000 --max-model-len 2048
```

vLLM is **not** an LM7 extra, deliberately: it pins a specific PyTorch, and
pinning one here would decide the torch version for everyone who installs LM7
(this repo already has extras that disagree about torch — `litert` pins
`<2.13`). Install it into the environment yourself. A target vLLM has no backend
for — `apple`, `intel:npu`, `tenstorrent`, `qualcomm` — is refused rather than
launched, because vLLM would otherwise fall back to whatever platform plugin it
could load and serve happily from the wrong device.

> **Not validated.** The handover has never been run. vLLM does not install on
> Apple Silicon and no GPU box was rented for this. `vllm_argv` is unit-tested;
> `serve_with_vllm` is not. See [limitations](limitations.md#serving).

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
