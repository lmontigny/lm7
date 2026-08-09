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
        vLLM  |  builtin (reference)
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
  builtin: unavailable (priority 0) - The reference runtime does not implement
    continuous_batching. It serves one request at a time; use a real serving runtime.
  vllm: unavailable (priority 0) - vLLM is not installed; install it separately
    with 'pip install vllm'.
```

Only constraints that were actually requested are required, so a plain request
still falls back to the reference runtime.

## Runtimes

| Runtime | Targets | Status |
| --- | --- | --- |
| `vllm` | `nvidia`, `amd`, `tpu`, `cpu` | Served real requests on Apple Silicon via vllm-metal; **no GPU has run it** |
| `builtin` | anything PyTorch runs | Validated on Apple M-series CPU and MPS |

### vLLM

LM7 runs vLLM **in-process through vLLM's own OpenAI-compatible app**, not as a
supervised subprocess. It composes the functions vLLM exposes for embedding --
`build_async_engine_client_from_engine_args`, `build_app`, `init_app_state` --
and drives uvicorn itself, so the engine and its FastAPI app run inside the LM7
interpreter. LM7 reimplements no part of the OpenAI schema and polls no health
endpoint. Isolation is not lost: vLLM V1 already runs its `EngineCore` in a
subprocess, so that boundary is inherited rather than rebuilt.

It deliberately does *not* call `run_server`. That is vLLM's CLI entry point and
it installs signal handlers, which only work on the main thread -- and a library
cannot take the caller's main thread. See below.

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

#### Running it on Apple Silicon

vLLM does have an Apple Silicon story: [vllm-metal](https://github.com/vllm-project/vllm-metal),
a community-maintained platform plugin that puts MLX under vLLM. It needs a
**native arm64 Python 3.12** — a Rosetta/x86-64 interpreter is rejected, and a
Homebrew or conda Python that reports `x86_64` will not do. Its `install.sh`
fetches a prebuilt macOS arm64 vLLM wheel rather than building from source, so
the same thing can be done directly:

```bash
uv venv --python 3.12 .venv-vllm && VIRTUAL_ENV=.venv-vllm uv pip install \
  "https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl" \
  vllm-metal lm7
```

Running LM7 against it is what turned this runtime from unproven into tested,
and it found three things that no test on an unmodified Mac could have:

- **`FlexibleArgumentParser` moved** from `vllm.utils` to
  `vllm.utils.argparse_utils` by 0.26. LM7 tries both, because it pins no vLLM
  version.
- **`run_server` cannot be called off the main thread.** It installs a SIGTERM
  handler, which raises `ValueError: signal only works in main thread of the
  main interpreter`. It is vLLM's *CLI* entry point; the embedding path is the
  layer under it, so LM7 composes `build_async_engine_client_from_engine_args`,
  `build_app` and `init_app_state` and drives uvicorn itself.
- **`lm7.serve()` needs a `__main__` guard with this runtime.** vLLM V1 runs its
  `EngineCore` in a *spawned* subprocess, so the child re-imports the calling
  module; a script that calls `lm7.serve()` at module scope starts a second
  engine while importing and dies with a `freeze_support` message that mentions
  nothing relevant. LM7 detects that failure and says so. The CLI is unaffected.

```python
import lm7


def main():
    with lm7.serve("hf://…", runtime="vllm") as server:
        ...


if __name__ == "__main__":  # required: vLLM spawns its engine core
    main()
```

**Still not validated:** everything above was measured on the vllm-metal CPU
path on an 18 GiB M-series Mac. No NVIDIA GPU has run this runtime, so
tensor parallelism, paged-KV throughput, and every performance claim in the
capability table remain vLLM's rather than LM7's measurements.

**LM7's memory preflight does not cover this runtime.** It runs in the built-in
runtime only, and the first real launch here failed on exactly what it exists to
prevent: vLLM's default `gpu-memory-utilization` of 0.92 asked for 16.56 GiB of
an 18 GiB machine with 3.92 GiB free, and the engine discovered that ~40 seconds
in. Pass `--kv-cache-fraction` to lower it. Extending the preflight across
runtimes is not done.

### The built-in runtime

`builtin` is LM7's own single-stream server over `compile_generation`. It exists
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

#### It serves a compiled model

This is the one place where `lm7 serve` and `lm7.compile` meet. The built-in
runtime drives `compile_generation`, so its decode graph goes through LM7's own
planner and compiler:

```bash
lm7 serve hf://… --target apple                          # auto: the planner picks
lm7 serve hf://… --target apple --compile-backend eager  # opt out
```

**The decode graph compiles; the prompt pass does not.** `GenerationRunner`
compiles prefill *per prompt length*, so compiling it in a server means a fresh
compile the first time each new prompt length arrives — invisible to a benchmark
that sends one length, constant for a server. The decode graph compiles once, at
a fixed shape, and every token of every request reuses it.

`/metrics` reports what actually ran, not what was asked for: `auto` resolves
through the planner and appears as the backend it chose.

Measured on an Apple M3 Pro, SmolLM2-135M-Instruct, `max_model_len=512`,
steady-state after warmup (the compile cost is excluded from the per-token
figure and reported separately). Two to three runs each; a laptop under load is
noisy, hence the ranges:

| Target | `--compile-backend eager` | `auto` (→ `inductor`) | Speedup | One-time compile |
| --- | --- | --- | --- | --- |
| `cpu` | 23.6–27.4 ms/token | 22.0–23.9 ms/token | 1.0–1.2x | ~5.5–7 s |
| `apple` (MPS) | 19.1–23.7 ms/token | 11.2–13.7 ms/token | 1.4–2.1x | ~4.5–17 s |

So MPS is worth it and CPU is close to noise. `auto` is the default anyway,
because the cost is paid once per process and the CPU case never *lost* — but a
short-lived server that answers a handful of requests should pass
`--compile-backend eager` and skip the wait.

Third-party runtimes **refuse** this flag rather than ignoring it: vLLM is handed
a checkpoint and compiles internally, so there is nothing for an LM7 compile
backend to drive.

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
