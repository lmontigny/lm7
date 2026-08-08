# Plan: `lm7 serve`

Design plan for adding a serving layer to LM7. Not yet implemented; this is the
shape the work should take and the order it should land in.

## The claim this layer makes

`lm7.compile()` returns a callable. A callable is not a serving system: there is
no scheduler, no paged KV cache, no continuous batching, no admission control,
no HTTP surface. Building those is a multi-year effort that vLLM, SGLang and
TensorRT-LLM have already done.

So LM7 does not implement a serving engine. It becomes the **control plane above
one**: a target string plus constraints in, a running OpenAI-compatible endpoint
out, with the choice of engine explained rather than assumed.

```
model URI + target + constraints
            ↓
      LM7 serve planner        ← memory preflight, capability check, explain
            ↓
  ┌─────────┼──────────┬─────────────┐
vLLM     SGLang   TensorRT-LLM    eager (built-in reference)
```

This is exactly the `compile` value proposition applied to serving: LM7 writes
no kernels, and now it writes no scheduler either.

## Why serving is not a `Backend`

The temptation is to add `vllm.py` to `src/lm7/backends/`. That is wrong, and
the protocol shows why. `Backend` is:

```python
def compile(self, request, example_args, example_kwargs) -> Artifact
def load(self, artifact) -> Callable[..., Any]
```

It takes an `nn.Module` already in memory and returns a callable. vLLM takes a
*checkpoint reference* and owns model loading, weight quantization, KV
allocation, and the socket. There is no `nn.Module` to hand it and no callable
to get back — the deliverable is an endpoint with a lifetime.

So: a **parallel** protocol in a new `src/lm7/serving/` package, mirroring the
`backends/` layout (protocol, registry, planner, one file per runtime) so the
two halves read the same way, but not sharing the `Backend` protocol itself.

## Protocol

`src/lm7/serving/base.py`:

```python
@dataclass(frozen=True)
class ServeRequest:
    model: str                  # hf://owner/model, a local checkpoint dir
    target: TargetSpec
    host: str = "127.0.0.1"
    port: int = 8000
    dtype: str = "auto"
    quantization: str = "none"  # reuse huggingface.normalize_quantization
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    max_batched_tokens: int | None = None   # chunked prefill
    tensor_parallel_size: int = 1
    kv_cache_fraction: float | None = None
    prefix_caching: bool | None = None
    lora_adapters: tuple[LoRASpec, ...] = ()
    speculative: SpeculativeSpec | None = None
    fallback: str = "warn"
    extra: Mapping[str, Any] = field(default_factory=dict)   # --runtime-arg k=v

@dataclass(frozen=True)
class Capabilities:
    continuous_batching: bool = False
    paged_kv_cache: bool = False
    prefix_caching: bool = False
    chunked_prefill: bool = False
    speculative_decoding: bool = False
    lora: bool = False
    streaming: bool = False
    cancellation: bool = False
    metrics: bool = False

@dataclass(frozen=True)
class RuntimeInfo:            # the BackendInfo analogue
    name: str
    version: str | None
    available: bool
    reason: str

class ServingRuntime(Protocol):
    name: str
    def probe(self) -> RuntimeInfo: ...
    def capabilities(self) -> Capabilities: ...
    def supports(self, request: ServeRequest) -> Support: ...   # reuse backends.base.Support
    def launch(self, request: ServeRequest) -> ServerHandle: ...
```

`Capabilities` is the piece that has no counterpart in `backends/`, and it is
the reason the layer earns its place. Constraints differ per runtime: asking for
LoRA on a runtime that does not serve adapters must **refuse**, not silently
drop the flag. It also gives `lm7 runtimes` a comparison table, and it makes the
built-in fallback runtime honest — it reports nearly every field `False`, which
documents precisely what LM7 does not implement.

## Prior art: how other projects actually wrap vLLM

Checked before choosing, because the first draft of this plan guessed wrong. The
ecosystem splits on exactly one question — **does the wrapper own the HTTP
surface?**

**Camp A — supervise a process or container. Wrapper does not own HTTP.**

- *KServe* does not even set a command: `VLLMBackend` relies on the container
  image's `ENTRYPOINT` already being `vllm serve`. Its own tracker records the
  consequence — the runtime breaks on any non-stock image, whereas the SGLang
  backend sets `["python3", "-m", "sglang.launch_server"]` explicitly and so
  works on custom images.
- *SkyPilot* puts `vllm serve …` in a YAML `run:` block and detects readiness by
  grepping the log for `Uvicorn running on`.
- *production-stack* and *llm-d* run pods and put a KV-aware router in front.

Note what Camp A has in common: they are all **cluster orchestrators**, and a
container boundary already exists for reasons that have nothing to do with vLLM.
Supervision is not their design choice, it is their substrate. LM7 is a library
in a Python process — it does not have that substrate, so copying the pattern
means inventing the boundary *and* the log-grepping.

**Camp B — import the Python API. Wrapper owns or re-serves HTTP.**

- *Ray Serve LLM* constructs `AsyncLLM(vllm_config=…)` in-process and — the
  detail that matters — reuses vLLM's own `OpenAIServingChat`,
  `OpenAIServingCompletion`, `OpenAIServingModels` handlers instead of
  reimplementing the OpenAI schema. Its `vllm_engine.py` is still ~1100 lines,
  because Ray additionally owns replicas, routing, LoRA resolution, and
  prefill/decode disaggregation.
- *NVIDIA Dynamo* does `from vllm.v1.engine.async_llm import AsyncLLM` and
  `AsyncLLM.from_vllm_config(vllm_config=…, stat_loggers=factory, …)`, with the
  HTTP frontend owned by Dynamo's own Rust runtime. Dynamo is the closest
  analogue to LM7's framing — a layer above vLLM *and* SGLang *and* TRT-LLM —
  and it is squarely in Camp B.
- *BentoML / OpenLLM* build the engine in the service constructor.

## Launch strategy: embed vLLM's own server, don't shell out

There is a third option the first draft missed, and it is the right one. vLLM
ships its OpenAI server as **importable functions**, not just a CLI:

```python
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,  # (engine_args: AsyncEngineArgs) -> AsyncIterator[EngineClient]
    build_app,        # (args: Namespace) -> FastAPI
    init_app_state,   # (engine_client, state, args) -> None
    build_and_serve,  # (engine_client, listen_address, sock, args) -> asyncio.Task
    run_server,       # (args, **uvicorn_kwargs)
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
```

So LM7 runs **vLLM's own FastAPI app inside the LM7 process**. That gets the
whole OpenAI surface — streaming, cancellation, tool calls, `/metrics` — with no
argv string, no `/health` poll, no log grepping, and no reimplemented schema. It
is Camp B without Ray's 1100 lines, because LM7 does not own routing or
replicas.

Process isolation is not lost: vLLM V1 already runs `EngineCore` in its own
subprocess. LM7 inherits that boundary instead of inventing one.

**The translation does not disappear, it gets type-checked.** `build_app` /
`init_app_state` / `run_server` take an argparse `Namespace`, so LM7 still maps
`ServeRequest` onto vLLM's arguments — but through
`make_arg_parser(FlexibleArgumentParser())` plus `validate_parsed_serve_args`,
i.e. **vLLM's own parser and validator**. A bad constraint is rejected by vLLM
at plan time, in-process, before any GPU is touched. This is what makes
`--dry-run` meaningful: it prints a *validated resolved config*, not a hopeful
string. Engine-side config goes through the `AsyncEngineArgs` dataclass, which
is typed already.

### Steal Ray's anti-drift test

Ray maintains `test_config_congruence.py`, which asserts that its own config
path and vLLM's CLI path produce **identical `VllmConfig` objects** (modulo an
`EXPECTED_DIFF_FIELDS` allowlist for `instance_id` and placement groups). LM7
should carry the same test: `ServeRequest → VllmConfig` must equal
`vllm serve <equivalent argv> → VllmConfig`. That single test is what keeps the
capability/constraint surface honest across vLLM upgrades, and it needs no GPU.

### Keep subprocess as a mode, for a packaging reason not a design one

`--isolate` retains the spawn-a-child path, justified by dependency conflict
rather than architecture: vLLM pins a specific PyTorch, TensorRT-LLM more so,
and LM7 depends only on `torch>=2.0` and must coexist with `[tensorrt]`,
`[litert]`, and friends. When the installed runtime cannot share LM7's
interpreter, spawning it against its own venv is the escape hatch. This is the
*only* argument for supervision that survives, and it should be documented as an
environment workaround.

Consequences for the extras: a `vllm` extra would drag a hard torch pin into a
project whose other extras already fight over torch (`litert` pins
`torch>=2.4,<2.13`). Prefer probe-only — no `vllm` extra, `pip install vllm`
documented separately, `probe()` reporting the version it found — with a comment
in `pyproject.toml` saying why, in the style of the existing extras.

`ServerHandle` becomes: base URL, chosen runtime, resolved config, the serving
`asyncio.Task`, `stop()`, `metrics()`. In `--isolate` mode it additionally
carries pid and log path.

### Metrics get better, not just equal

Dynamo passes `stat_loggers=factory` into `AsyncLLM.from_vllm_config`. In-process,
LM7 can **inject its own stat logger** rather than scraping and re-parsing
Prometheus text. TTFT/TPOT/queue depth/KV utilization arrive as objects, in the
normalized LM7 schema, at the source. The scrape path is then only needed for
`--isolate`.

## What LM7 adds on top

Four things, all of which are LM7-shaped rather than engine-shaped:

- **Target resolution and runtime selection.** The existing `detect_targets()` /
  `resolve_target()` path already knows the card. `lm7 serve --explain` prints
  the ranked candidates and the reason each was or was not chosen, exactly like
  `lm7 explain` does for backends.
- **Memory preflight / admission.** Before allocating anything, estimate weight
  bytes (parameter count × dtype, already computable via
  `huggingface._model_storage_bytes` logic) plus KV bytes per token
  (`2 × layers × kv_heads × head_dim × dtype_bytes`, the shape
  `generation._allocate_static_cache` already builds) × `max_model_len` ×
  `max_num_seqs`, compare against `DeviceInfo.total_memory_bytes`, and either
  refuse with a specific number or derive `--gpu-memory-utilization`. This is
  pure arithmetic, unit-testable with no GPU, and it turns the single most
  common serving failure (OOM 40 seconds into startup) into an instant message.
- **Capability negotiation.** Above.
- **Normalized metrics.** Every runtime names its statistics differently. LM7
  normalizes to one schema — TTFT, TPOT, tokens/s, running/waiting requests, KV
  utilization — so `lm7 serve status --json` reads the same across engines.
  In-process this is an injected stat logger rather than a scrape. Do not
  reimplement the measurement, only the naming.

## The fallback runtime

`serving/runtimes/eager.py`: a built-in single-stream OpenAI-compatible server
over the existing `generation.GenerationRunner`. No batching, no paging, one
request at a time.

It exists for three reasons: `--fallback=warn` should mean the same thing for
`serve` as it does for `compile`; it makes the HTTP contract and the CLI
testable on an ordinary CPU CI runner where vLLM cannot install; and its
all-`False` `Capabilities` row is the clearest possible statement of what LM7
itself does and does not do. It is ~200 lines on `fastapi`/`uvicorn` behind a
`serve` extra. It must never be described as a serving system.

If this proves to be scope creep, it is the piece to cut — the cost is that CI
can then only test config translation, never a live request.

## Runtime coverage

| runtime | targets | notes |
| --- | --- | --- |
| vLLM | `nvidia`, `amd`, `tpu`, `cpu` (source build) | broadest model coverage; the default |
| SGLang | `nvidia`, `amd` | RadixAttention prefix caching |
| TensorRT-LLM | `nvidia` only | fastest on NVIDIA, needs an engine build step |
| eager | anything torch runs | fallback, not a serving system |

`apple`, `qualcomm`, `tenstorrent`, `intel:npu` have no third-party serving
runtime: they resolve to the fallback or refuse.

**Pick vLLM to prove the idea.** It has the widest model support, an importable
server (above) rather than only a CLI, standard metrics, and it is what a reader
will expect the default to be. TensorRT-LLM is the more interesting NVIDIA
number but adds an ahead-of-time engine build that muddies the first
implementation.

## Artifacts do not cross over

`lm7 export` produces `.lm7` / AOTInductor `.pt2` packages. **No third-party
serving runtime consumes those.** vLLM and SGLang take HF checkpoints;
TensorRT-LLM takes its own engine directory. `lm7 serve` therefore takes a model
URI, not an `.lm7` path, and `docs/limitations.md` must say so explicitly — this
is precisely the kind of assumed-compatibility claim this repo keeps having to
correct later.

## CLI surface

Top-level `serve`, not a `--serve` flag — it is a mode with its own lifetime,
and it matches `doctor` / `targets` / `model` / `bundle`:

```
lm7 serve hf://meta-llama/Llama-3.1-8B \
    --target nvidia:sm120 --runtime auto \
    --max-model-len 8192 --max-num-seqs 64 --port 8000
lm7 serve … --explain          # ranked runtimes + memory preflight, no launch
lm7 serve … --dry-run          # print the validated resolved runtime config and exit
lm7 serve status [--json]      # normalized metrics from a running server
lm7 serve stop
lm7 runtimes [--json]          # the `lm7 backends` analogue, + capability table
```

Python API mirrors `lm7.compile`:

```python
with lm7.serve("hf://…", target="nvidia:sm120", runtime="vllm") as server:
    server.base_url
```

`--dry-run` and `--explain` are worth building first: they make the whole layer
reviewable without a GPU.

## Landing order

One branch per step, `agent/<name>`, PR into `main`.

1. **`agent/serving-protocol`** — `serving/{base,registry,planner,budget}.py`,
   `lm7 runtimes`, `lm7 serve --explain`/`--dry-run` (no launch). Generalize
   `planner.plan` into a request-type-agnostic `select()` shared by both halves
   rather than duplicating the 25 lines. Unit tests only, all run in CI.
   `docs/serving.md` + index entry.
2. **`agent/serving-vllm`** — probe, `supports`, `capabilities`, `ServeRequest →
   AsyncEngineArgs` + validated `Namespace`, in-process `build_and_serve`, LM7
   stat logger. Target ~200 lines; if it approaches Ray's 1100, LM7 has started
   owning something it should not. The config-congruence test against
   `vllm serve` runs wherever vLLM imports, no GPU needed;
   `tests/test_vllm_integration.py` (marked `vllm`) covers a live request and
   needs CUDA.
3. **`agent/serving-eager-fallback`** — the reference runtime, the `serve`
   extra, a CPU CI job, and a live request/streaming/cancel integration test.
4. **`agent/serving-python-api`** — `lm7.serve()` handle, context manager,
   `--detach`, `serve status`/`stop`, and the `--isolate` subprocess mode with
   its Prometheus scrape fallback.
5. **`agent/serving-sglang`**, **`agent/serving-trtllm`** — one file each, same
   shape as step 2.
6. **Measurement** — `benchmarks/serving.py` (TTFT / TPOT / tokens-s / p99 under
   a fixed request trace) on the RTX PRO 6000 Blackwell, JSON into `artifacts/`
   on the rented box, prose written locally afterwards. Never mixed with numbers
   from `nvidia_matrix.py` or `moe.py`.

Per `CLAUDE.md`'s backend checklist, each runtime PR also needs: a pytest marker
in `pyproject.toml`, split mocked/integration test files, a `docs/` page linked
from `docs/README.md`, a CI matrix entry *or* a comment saying why it cannot
have one (vLLM/SGLang/TRT-LLM all need a GPU GitHub will not provide), and a
`docs/changelog.md` line.

## Open questions

- Does `--runtime auto` ever pick TensorRT-LLM? It is fastest on NVIDIA but
  needs an engine build first; auto-selection that silently triggers a
  multi-minute build is hostile. Probably: auto never selects it, matching how
  `tensorrt` sits below `inductor` in compile priority.
- Multi-node / TP>1 is out of scope for the first pass — a single `--tensor-
  parallel-size` passthrough, and no claim that it was tested (only the 96 GiB
  card is available, single GPU).
- Speculative decoding and LoRA are passthrough flags gated by `Capabilities`;
  LM7 validates nothing about them beyond that the runtime accepts them.
