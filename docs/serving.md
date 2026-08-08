# Serving

`lm7 serve` starts an OpenAI-compatible HTTP endpoint for a causal LM. LM7 does
not implement a serving engine to do it: it resolves the target, checks the
request against what each registered runtime can actually do, and hands the work
to whichever one wins.

This is the same split as `lm7.compile()`. LM7 writes no kernels, and it writes
no scheduler either.

```
model URI + target + constraints
            ↓
      LM7 serving planner     ← capability check, memory preflight, explain
            ↓
        vLLM  |  eager (reference)
```

## Quick start

```bash
pip install -e ".[serve,hf]"

lm7 runtimes                      # what is installed and what each implements
lm7 serve hf://HuggingFaceTB/SmolLM2-135M-Instruct --target cpu --explain
lm7 serve hf://HuggingFaceTB/SmolLM2-135M-Instruct --target cpu --max-model-len 512
```

```python
import lm7

with lm7.serve("hf://HuggingFaceTB/SmolLM2-135M-Instruct", target="cpu", port=0) as server:
    print(server.base_url)  # http://127.0.0.1:54321
```

Endpoints: `/health`, `/metrics`, `/v1/models`, `/v1/completions`,
`/v1/chat/completions`. The two completion endpoints accept `"stream": true` and
emit server-sent events terminated by `data: [DONE]`.

## Why serving is not a compile backend

`lm7.backends.base.Backend` takes an `nn.Module` that is already in memory and
returns a callable. A serving runtime takes a *checkpoint reference* and owns
model loading, weight quantization, KV allocation, and the socket — there is no
module to hand it and no callable to get back. So `lm7.serving` is a parallel
package with its own protocol, mirroring the `backends/` layout (protocol,
registry, planner, one file per runtime) without sharing the `Backend` protocol.

The one piece with no counterpart in `backends/` is `Capabilities`, and it is
why the layer earns its place. A compile backend either compiles a model or
declines it. A serving runtime can accept a model and silently ignore half the
request — so LM7 checks the constraints that were asked for against what the
runtime implements, and refuses rather than dropping a flag:

```
$ lm7 serve hf://owner/model --target cpu --max-num-seqs 8 --explain
Selected nothing for cpu:arm64 serving hf://owner/model

Candidates:
  eager: unavailable (priority 0) - The reference runtime does not implement
    continuous_batching. It serves one request at a time; use a real serving runtime.
  vllm: unavailable (priority 0) - vLLM is not installed; install it separately
    with 'pip install vllm'.
```

Only constraints that were actually requested are required, so a plain request
still falls back to the reference runtime.

## Runtimes

| Runtime | Targets | Status |
| --- | --- | --- |
| `vllm` | `nvidia`, `amd`, `tpu`, `cpu` | Implemented, **not validated on hardware** |
| `eager` | anything PyTorch runs | Validated on Apple M-series CPU |

### vLLM

LM7 runs vLLM **in-process through vLLM's own OpenAI server**, not as a
supervised subprocess. vLLM ships `run_server`, `build_app` and
`build_async_engine_client_from_engine_args` as importable functions, so the
engine and its FastAPI app run inside the LM7 interpreter. LM7 reimplements no
part of the OpenAI schema and polls no health endpoint. Isolation is not lost:
vLLM V1 already runs its `EngineCore` in a subprocess, so that boundary is
inherited rather than rebuilt.

The translation is still real, but vLLM validates it. `serve_argv()` maps a
`ServeRequest` onto vLLM's flags and is free of vLLM imports, so it is
unit-tested on machines where vLLM cannot install at all. `build_namespace()`
then feeds that argv through vLLM's own `make_arg_parser` and
`validate_parsed_serve_args`, which is what makes `--dry-run` worth printing:

```bash
lm7 serve hf://meta-llama/Llama-3.1-8B --target nvidia:sm90 --runtime vllm --dry-run
```

`validated: false` in that output means vLLM is not installed and nothing has
checked the argv — LM7 says so rather than implying a confirmation that did not
happen.

**There is deliberately no `vllm` extra.** vLLM pins a specific PyTorch, and
this project already has extras that disagree about torch (`litert` pins
`torch>=2.4,<2.13`), so a pin here would decide the torch version for everyone
installing LM7. Install vLLM yourself; the runtime probes for it.

**What is not validated:** the launch path has never run. No GPU was available
when this landed, and vLLM does not install on Apple Silicon. `serve_argv` is
tested; `launch()` is not. Treat the vLLM runtime as implemented, not proven.

### The reference runtime

`eager` is LM7's own single-stream server over `compile_generation`. It exists
for three reasons, and none of them is performance:

1. `--fallback` should mean the same thing for `serve` as for `compile`.
2. The HTTP contract needs to be testable on a CPU runner, and no third-party
   serving engine installs on one.
3. Its capability row is the most honest description of what LM7 implements by
   itself.

It serves **one request at a time**, behind a lock, because the runner owns
exactly one static KV cache and two concurrent generations would interleave
writes into the same buffers. It implements streaming, cancellation and metrics;
it implements no continuous batching, no paged KV cache, no prefix caching, no
chunked prefill, no speculative decoding and no LoRA. It is not a serving
system and should never be described as one.

Cancellation is measured, not inferred from the design. Each decode step runs in
a worker thread, so the event loop stays free to notice a client that
disconnected. Abandoning a 400-token stream after three chunks on an Apple
M-series CPU stopped it at 5 tokens, and a follow-up request — which has to
queue behind it for the lock — unblocked in 40 ms against a 26 s projection for
the full generation. `test_abandoning_a_stream_stops_the_decode_loop` measures
its own baseline rather than hard-coding that threshold.

Cancelled requests are counted: the metrics are recorded in a `finally`, because
an abandoned stream is closed *at* the `yield` and everything after the loop is
otherwise skipped — losing exactly the requests a server most wants counted.

## Memory preflight

`lm7.serving.budget` costs a configuration before anything is allocated:

```
kv_bytes_per_token = 2 (K and V) × layers × kv_heads × head_dim × dtype_bytes
kv_bytes           = kv_bytes_per_token × max_model_len × max_num_seqs
```

`num_key_value_heads` is the field that matters and the one often absent — a
model without grouped-query attention does not set it, and falling back to
`num_attention_heads` is what keeps the estimate right for MHA.

When the weight size is unknown, `fits` is `None` and only the KV cost is
reported. A verdict without the weights would be a guess, and this refuses to
give one.

## Not yet done

- **SGLang and TensorRT-LLM.** Planned as one file each, same shape as the vLLM
  runtime. TensorRT-LLM likely should never be selected by `--runtime auto`,
  because it needs an ahead-of-time engine build and auto-selection that
  silently triggers a multi-minute build is hostile.
- **`--isolate`.** A subprocess mode, justified by dependency conflict rather
  than architecture: when the installed runtime cannot share LM7's interpreter,
  spawning it against its own venv is the escape hatch.
- **`--detach`, `lm7 serve status`, `lm7 serve stop`.** The handle is currently
  foreground-only.
- **Any measurement.** No TTFT/TPOT/throughput numbers exist for any runtime on
  any GPU. `benchmarks/serving.py` does not exist yet.

## Testing

```bash
pip install -e ".[dev,serve,hf]"
python -m pytest tests/test_serving_planner.py tests/test_serving_budget.py \
                 tests/test_vllm_runtime.py            # no model, no network
python -m pytest -m serve                              # live CPU round trip
python -m pytest -m vllm                               # needs vLLM importable
```

The `vllm` marker covers the anti-drift check that vLLM's own parser accepts
what LM7 produced — modelled on Ray Serve LLM's `test_config_congruence.py`,
which asserts its config path and vLLM's CLI path agree. It needs vLLM
importable but no GPU, because parsing happens before any device is touched.
