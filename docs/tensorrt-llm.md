# TensorRT-LLM (experimental)

`--backend trtllm` hands `lm7 model serve`'s port to NVIDIA's
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). LM7 resolves the target,
refuses the hardware TensorRT-LLM has no kernels for, translates its config into
`trtllm-serve`'s own argv, and hands over the process. What answers the port
afterwards is TensorRT-LLM, unmodified.

It is the second *launcher backend*, after `--backend vllm`, and it shares that
one's plumbing rather than paralleling it — see
[serving](serving.md#--backend-trtllm-the-same-handover-to-tensorrt-llm) for the
user-facing half.

> [!WARNING]
> **Experimental, one card, one model.** Run on an RTX 4070 SUPER (Ada `sm89`,
> 12 GiB) under WSL2 with SmolLM2-135M. Nothing here is a benchmark: see
> [what was and was not measured](#what-was-and-was-not-measured).

**The short version:** for one caller on a small model, LM7's own server with
CUDA Graphs answers sooner and holds 17x less GPU memory. Past one caller it
stops scaling at all, and TensorRT-LLM reaches 8x its aggregate throughput by
eight streams. [The measurement](#against-the-inductor-path) is what that
sentence rests on.

## Why this is a launcher, and not an in-process runtime

The first revision of this work ([#110](https://github.com/lmontigny/lm7/pull/110))
was not a launcher. It drove `tensorrt_llm._tensorrt_engine.LLM` in-process
behind a second protocol in `src/lm7/runtimes/` — `probe`/`supports`/`prepare`/
`generate` — parallel to `src/lm7/backends/` and with its own `ServeConfig`.
The review asked for "1 layer in between to allow different serving backend",
and by then `src/lm7/serve/` had grown exactly that layer for vLLM. So the
adapter became a launcher and the parallel protocol went away.

That was the right trade for three reasons beyond having one layer instead of
two. **The first two are the first revision's findings, reported from an H100 and
not re-run here** — they are recorded because they are the reasons the design
changed, not because this branch reproduced them:

**`tensorrt_llm.LLM` is not the TensorRT engine path on 1.2.x.** The public class
became the *PyTorch* backend and rejects `build_config` outright, pointing at
`_tensorrt_engine` instead. An in-process adapter that wants engine execution has
to import a leading-underscore module and pin itself to a private API. A launcher
imports nothing: `trtllm-serve` is a supported entry point, and *which* runtime
it uses is TensorRT-LLM's decision to make, not LM7's. What this branch *did*
confirm on 1.2.1 is the consistent half — `trtllm-serve serve --backend` offers
`pytorch`, `tensorrt` and `_autodeploy`, and its default is `pytorch`.

**The MPI re-exec is fatal in-process and harmless out of it.** TensorRT-LLM
spawns MPI workers that **re-execute the parent's command line**. Under
`python -m lm7` those workers re-ran the CLI with no arguments, hit argparse, and
`MPI_ABORT`ed the job *after* the engine had finished building — a successful
30-second build followed by `error: the following arguments are required:
command`. `mpirun -n 1` did not fix it; the shipped `trtllm-llmapi-launch`
wrapper did, which meant `lm7 serve` could only be run under another launcher.
Handing over the process removes the problem rather than working around it:
`trtllm-serve` is its own entry point, so its workers re-execute *it* — and on
this branch `lm7 model serve --backend trtllm` runs under no wrapper at all.

**The in-process version had no OpenAI endpoint.** It streamed tokens to one
caller through LM7's own code. `trtllm-serve` answers the same API as everything
else in this repo's serving story, so a client does not have to know which
backend is behind the port — which is the whole point of `lm7 model serve`.

What was lost is real and worth naming: the first revision measured TTFT and
inter-token latency *itself*, around the runtime's stream, so the numbers meant
the same thing as LM7's elsewhere. A launcher cannot do that, because it is not
in the request path. Those numbers now have to come from a client, and they
measure a different thing.

## The boundary

| LM7 owns | TensorRT-LLM owns |
| --- | --- |
| target resolution and hardware gating | attention kernels |
| launcher selection (`--backend`) | paged KV-cache management |
| finding the executable | batch scheduler and in-flight batching |
| config → argv translation | the decode loop |
| refusing what does not translate | engine build and execution |
| | the HTTP surface |

Anything from the right column appearing in `src/lm7/serve/trtllm.py` is a bug in
the boundary. The module is 198 lines, most of them comments, for that reason —
the same size as the vLLM launcher beside it.

## Installing

TensorRT-LLM needs **its own environment**. It pins versions that conflict with
every other environment in this repo, which is the same reason vLLM is not an
LM7 extra: pinning a torch here would decide the torch version for everyone who
installs LM7.

```bash
python3 -m venv .venv-trtllm
.venv-trtllm/bin/python -m pip install tensorrt-llm==1.2.1
```

That lands `tensorrt_llm 1.2.1`, `torch 2.9.1+cu128`, `transformers 4.57.3`,
`tensorrt 10.14.1`, `flashinfer-python 0.6.4` — 210 packages and **19 GB**.

### On a bare WSL2 box, that install does not run

`import tensorrt_llm` fails, and so does `trtllm-serve`. The pip wheel assumes
the environment NVIDIA's NGC container provides, and outside it the missing
pieces surface one at a time, each as a different error several minutes apart.
What follows is the full set, in the order they appeared on this machine —
Ubuntu 24.04 under WSL2, driver 595.71 / CUDA 13.2, **no `sudo`**:

```bash
# libmpi.so.40: mpi4py 4.x resolves the MPI ABI at import and there is no
# system MPI. The PyPI OpenMPI wheel is the no-root answer.
.venv-trtllm/bin/python -m pip install openmpi

# libcublasLt.so.13, then nvcc, then nv/target, then curand_kernel.h. The
# bindings are built against CUDA 13 while torch brought cu12, and FlashInfer
# JIT-compiles its kernels at startup, so headers and a compiler are runtime
# dependencies rather than build-time ones.
.venv-trtllm/bin/python -m pip install nvidia-cuda-nvcc nvidia-cuda-cccl "cuda-toolkit[all]"
```

Two things pip cannot supply:

```bash
# libnuma.so.1 is a system library, and there is no sudo here. Extracting the
# .deb into the venv needs no root and is contained to it.
apt-get download libnuma1 && dpkg-deb -x libnuma1_*.deb /tmp/numa
cp -P /tmp/numa/usr/lib/x86_64-linux-gnu/libnuma.so.1* .venv-trtllm/lib/

# FlashInfer links against $CUDA_HOME/lib64 and $CUDA_HOME/lib64/stubs, but the
# wheels ship lib/ with no stubs and no unversioned .so. On WSL the real
# libcuda lives in /usr/lib/wsl/lib rather than beside the toolkit.
CU=.venv-trtllm/lib/python3.12/site-packages/nvidia/cu13
ln -sfn lib "$CU/lib64"
ln -sf libcudart.so.13 "$CU/lib/libcudart.so"
mkdir -p "$CU/lib/stubs" && ln -sf /usr/lib/wsl/lib/libcuda.so.1 "$CU/lib/stubs/libcuda.so"
```

And the environment every invocation needs, because `_find_cuda_home` reads
`CUDA_HOME` and the loader will not find any of the above without it:

```bash
CU="$PWD/.venv-trtllm/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME="$CU"
export LD_LIBRARY_PATH="$CU/lib:$PWD/.venv-trtllm/lib:$LD_LIBRARY_PATH"
export PATH="$PWD/.venv-trtllm/bin:$PATH"
```

> **LM7 does none of this.** Repairing a vendor's environment is on the other
> side of the boundary, and a launcher that silently set `CUDA_HOME` would be
> guessing. `trtllm_environment()` returns the environment unchanged, and
> `--dry-run` reports that it changes nothing — which is why the list above is a
> prerequisite and not a flag.

LM7 does not need to import it. Put that venv's `bin` on `PATH` — LM7 looks for
an importable `tensorrt_llm`, then `trtllm-serve` on `PATH`, then
`~/.venv-trtllm/bin/trtllm-serve`, and `--dry-run` prints which one it found:

```bash
PATH="$PWD/.venv-trtllm/bin:$PATH" \
  lm7 model serve hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target nvidia --backend trtllm --dry-run
```

An import check alone would report "not installed" on a machine where
`trtllm-serve` runs perfectly well, which is the normal state of a working box.

## Running it

```bash
PATH="$PWD/.venv-trtllm/bin:$PATH" \
  lm7 model serve hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target nvidia --backend trtllm --max-model-len 2048
```

```
lm7: handing HuggingFaceTB/SmolLM2-135M-Instruct to trtllm on 127.0.0.1:8000
...
[TRT-LLM] [I] Estimated max memory in KV cache : 9.06 GiB
[TensorRT-LLM][INFO] Max KV cache blocks per sequence: 65 [window size=2049],
  tokens per block=32, primary blocks=13189, max sequence length=2049
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 9.06 GiB for max tokens in paged
  KV cache (422048).
INFO:     Application startup complete.
```

Those two numbers are the reason this backend exists, and they are worth reading
carefully. `--max-model-len 2048` became `--max_seq_len`, which bounds **one
request** at 2049 tokens. The cache TensorRT-LLM then allocated holds **422,048**
tokens in 32-token pages, shared across every request in flight. LM7's own server
allocates one static cache for one sequence, because that is what a compiled
callable can own; the difference between those two lines is the whole argument
for handing the port over.

It also means the launched server takes essentially the whole card: 11.9 GiB of
this 12 GiB GPU, for a 135M model. `--free_gpu_memory_fraction` is TensorRT-LLM's
flag to turn that down, and LM7 does not model it — it passes it through:

```bash
lm7 model serve hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target nvidia --backend trtllm --max-model-len 2048 \
  --trtllm-arg=--free_gpu_memory_fraction --trtllm-arg 0.25
```

Measured on this card, that turns the 9.06 GiB / 422,048-token cache into
**2.52 GiB / 117,216 tokens**, and the server's total footprint from 11.9 GiB
into **4.2 GiB**, still answering correctly. `--trtllm-arg` is repeatable and
appended last, so it wins over anything LM7 translated — the same escape hatch,
with the same rules, as `--vllm-arg` beside it.

## What was and was not measured

**RTX 4070 SUPER (Ada `sm89`, 12 GiB) under WSL2**, driver 595.71, TensorRT-LLM
1.2.1, SmolLM2-135M-Instruct, `--max-model-len 2048`, one stream, greedy.

What ran end to end: `lm7 model serve --backend trtllm` came up, `/v1/models`
listed the model as `owned_by: tensorrt_llm`, a chat completion answered *"The
capital of France is Paris."*, and an SSE stream reassembled. The integration
suite — which builds its server from `serve_plan`'s own argv rather than a
hand-written command line — is **4 passed**, including startup: 120 s on the
first run of the day, 54 s once FlashInfer's kernel cache and the page cache are
warm. The portable suite is 688 passed, 86 skipped.

| | |
| --- | --- |
| cold start (launch → `/health` 200) | ~50 s warm, ~125 s on a fresh box |
| GPU memory held | 11.9 GiB of 12 GiB |

**Latency is not quoted here**, because quoting it twice is how a document ends
up disagreeing with itself. Every timing lives in
[against the Inductor path](#against-the-inductor-path), measured by one harness
against 128-token completions. An earlier revision of this page reported 62.5 ms
TTFT and 7.3 ms inter-token from an ad-hoc script on 7–8 token answers; those
were too few gaps to take a median of, and the harness supersedes them.

> The first revision of this PR measured 105 ms TTFT and 1.27 ms inter-token on
> an H100 *around the runtime's own stream* rather than over HTTP. That is a
> different quantity on different hardware and must not be compared with
> anything on this page.

The cold start is dominated by TensorRT-LLM's own startup, not by anything LM7
does; the first launch on a fresh box is slower still, because FlashInfer
JIT-compiles its `sm89` kernels into `~/.cache/flashinfer` before the server can
answer.

**Killing the launcher does not kill the server.** LM7 hands over with
`subprocess.call`, so a `SIGTERM` aimed at the `lm7` process alone leaves
`trtllm-serve` running and holding 11.9 GiB. In a terminal this does not arise —
Ctrl-C goes to the whole foreground process group — but it is why the integration
test starts the server in its own session and signals the group, and it is worth
knowing before scripting around it.

## Against the Inductor path

The measurement that decides whether this backend is worth reaching for.
[`benchmarks/serving_backends.py`](../benchmarks/serving_backends.py) drives
**three arms from one client**, which is the only reason they are comparable:
both servers answer the same OpenAI API, so the same code times both over the
same loopback HTTP, against the same model, prompt and token budget.

- **`inductor`** — LM7's own server, compiled prefill and KV-cache decode graphs.
- **`cudagraphs`** — the same, plus `--compile-mode reduce-overhead`, which is
  what asks Inductor to capture the decode step into a CUDA graph. **Leaving this
  arm out would have stacked the comparison**, and it changes the answer: see
  below.
- **`trtllm`** — the handover.

RTX 4070 SUPER (Ada `sm89`, 12 GiB) under WSL2, SmolLM2-135M-Instruct,
`--max-model-len 2048`, 128 tokens greedy, median of 5 after warmup:

| | `inductor` | `cudagraphs` | `trtllm` |
| --- | --- | --- | --- |
| startup to `/health` | 10.1 s | 10.0 s | 50.0 s |
| first request (compile) | 19.4 s | 75.7 s | 1.1 s |
| TTFT | 23.0 ms | **13.0 ms** | 50.4 ms |
| inter-token | 16.6 ms | 6.97 ms | **5.76 ms** |
| one stream | 60.2 tok/s | 143.5 tok/s | **173.5 tok/s** |
| GPU held, over idle | **572 MiB** | 615 MiB | 10,268 MiB |

**CUDA Graphs close most of the single-stream gap.** Against plain `inductor`
TensorRT-LLM decodes 2.9x faster per token; against `cudagraphs` it is **1.21x**,
and LM7 answers the *first* token nearly four times sooner (13.0 ms against
50.4 ms) while holding **17x less memory**. On one stream, on a small model, the
engine is not the reason to switch.

### Concurrency is the reason to switch

Aggregate tokens/s across N simultaneous streams, and the median and worst TTFT
inside them:

| N | `cudagraphs` | `trtllm` | `cudagraphs` TTFT (med / max) | `trtllm` TTFT (med / max) |
| --- | --- | --- | --- | --- |
| 1 | 143.5 | 173.5 | 13 ms | 50 ms |
| 2 | 141.6 | 223.4 | 466 / 917 ms | 187 / 190 ms |
| 4 | 138.4 | 538.9 | 1,401 / 2,787 ms | 114 / 116 ms |
| 8 | 142.6 | **1,139.4** | 3,159 / 6,292 ms | **97 / 100 ms** |

LM7's aggregate is **flat** — 143 tok/s at every N — and its worst TTFT grows
linearly to 6.3 seconds. That is not a defect to fix: `compile_generation`
allocates one static KV cache that every decode step mutates in place, so the
server holds an `asyncio.Lock` and a second caller waits. The line is flat
because the design says it will be.

TensorRT-LLM scales close to linearly to **8.0x** LM7's aggregate at N=8, and its
TTFT *falls* as concurrency rises — 50 ms alone, 97 ms across eight streams that
each started immediately — because requests join a batch instead of a queue.

**So the two are not competing at the same thing**, which is the useful result:
LM7's server is the better answer for one caller on a small model, and it stops
being an answer at all somewhere between one and two. That boundary is the whole
argument for the handover, and it is now measured rather than asserted.

> **Indicators, not a benchmark.** Wall clock from a Python client over loopback
> HTTP, so every figure includes framing, the OpenAI schema and each server's
> scheduler. One model, one card, one prompt length, one token budget. Nothing
> here is comparable to [`benchmarks/decode.py`](../benchmarks/decode.py), which
> drives the runner directly with no server in the way — see CLAUDE.md on not
> mixing harnesses. Reproduce with:
>
> ```bash
> python benchmarks/serving_backends.py --output artifacts/serving-backends.json
> ```

## What is not done

- **The comparison is one model on one card.** SmolLM2-135M is 30 layers and
  launch-bound, which is the case most favourable to a compiled decode loop; a
  larger model spends proportionally more time in GEMMs and less in launches, so
  the single-stream columns above should be expected to move. Nothing above 135M
  has been compared, and neither has a second card.
- **Continuous batching is exercised only up to 8 streams**, all of them the same
  prompt and the same token budget arriving at the same instant. That is the
  easiest possible batch: nothing here measures a mixed-length queue, arrivals
  spread over time, or the point where the scheduler starts preempting. The
  scaling above should be read as "it batches", not as a throughput ceiling.
- **No quantization.** `--quantize` is refused rather than passed through:
  TensorRT-LLM quantizes at engine build time from an NVIDIA ModelOpt checkpoint,
  which is a different mechanism from LM7's weight-only path. Serving a
  pre-quantized checkpoint is untried.
- **One GPU.** No tensor parallelism and no multi-card run of any kind.
- **Only three flags are translated**, and the rest go through verbatim.
  `--host`, `--port` and `--max-model-len` are the ones LM7 spells; everything
  else — `--max_batch_size`, `--tp_size`, `--extra_llm_api_options` — reaches
  TensorRT-LLM through `--trtllm-arg` rather than an LM7 flag, deliberately, for
  the reason `--vllm-arg` exists: mirroring a vendor CLI would make LM7 a second,
  always-stale copy of it. What is genuinely not done is *modelling* any of them,
  so LM7 makes no claim about what they do.
- **Not in CI.** GitHub's GPU runners are gated to Team/Enterprise organizations,
  and this needs both an Ampere-or-newer GPU and an environment of its own.

## Reference

- `src/lm7/serve/trtllm.py` — the launcher
- `src/lm7/serve/cli.py` — `LAUNCHER_BACKENDS`, the layer shared with vLLM
- `benchmarks/serving_backends.py` — the three-arm comparison above
- `tests/test_serve.py` — the translation and the refusals, no GPU needed
- `tests/test_trtllm_serve_integration.py` — a real server, `-m trtllm`
