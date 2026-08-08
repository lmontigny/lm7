# TensorRT-LLM runtime (experimental)

A serving runtime, not a compiler backend. `lm7 serve` hands a decoder-only model
to TensorRT-LLM and streams tokens back; TensorRT-LLM owns the engine, the paged
KV cache, the scheduler and the decode loop.

> [!WARNING]
> **Experimental, single GPU, and one model family exercised.** Measured on one
> H100 with SmolLM2-135M. The engine cache is metadata only — see
> [what LM7 does not own yet](#what-lm7-does-not-own-yet).

## Why it is not the `tensorrt` backend

`tensorrt` compiles a module through Torch-TensorRT and returns something
callable: `probe`/`supports`/`compile`/`load`, driven by the caller. That shape
cannot express continuous batching, because scheduling is a property of a server
holding many in-flight requests, not of a callable one caller invokes. So this is
a separate concept — `src/lm7/runtimes/` — with its own protocol.

| LM7 owns | TensorRT-LLM owns |
| --- | --- |
| target resolution and hardware gating | attention kernels |
| runtime selection | paged KV-cache management |
| dependency checks | batch scheduler |
| configuration (`ServeConfig`) | decode loop |
| engine cache identity and manifest | TensorRT engine build and execution |
| TTFT / inter-token measurement | |

Anything in the right column appearing in `src/lm7/runtimes/` is a bug in the
boundary. This adapter is ~200 lines for that reason.

## Installing

TensorRT-LLM needs **its own environment**. It pins `torch>=2.9.1,<=2.10.0a0`
against 2.13 in the CUDA venv and 2.12.1 in the TensorRT one, `transformers`
to exactly 4.57.3 against 5.14.1, `tensorrt~=10.14.1` against 10.16.1, plus
`triton==3.5.1` and `torchao<0.16`. None of that overlaps.

```bash
uv venv --python 3.12 .venv-trtllm
uv pip install --python .venv-trtllm/bin/python \
    --extra-index-url https://pypi.nvidia.com tensorrt-llm
uv pip install --python .venv-trtllm/bin/python --no-deps -e .
```

Measured on the H100 box: resolves and installs 118 packages, landing
`tensorrt_llm 1.2.1`, `torch 2.9.1+cu128`, `transformers 4.57.3`, `triton 3.5.1`.

```console
$ lm7 runtimes
Registered runtimes (1):
  tensorrt-llm: available, version 1.2.1
```

Without it, the same command names the conflict and the install line rather than
just reporting "unavailable", because a 118-package dependency set is not
something to leave a reader to work out.

## Running it

**`lm7 serve` must be launched through `trtllm-llmapi-launch`.**

```bash
trtllm-llmapi-launch python -m lm7 serve HuggingFaceTB/SmolLM2-135M-Instruct \
    --runtime tensorrt-llm --target nvidia --max-new-tokens 48 --json
```

This is not optional and the failure is confusing without it. TensorRT-LLM
spawns MPI workers that **re-execute the parent's command line**; under
`python -m lm7` those workers re-run the CLI with no arguments, hit argparse, and
`MPI_ABORT` the job *after* the engine has finished building. The symptom is a
successful 30-second build followed by `lm7: error: the following arguments are
required: command`. `mpirun -n 1` does not fix it; the shipped launcher does.

The Python API has no such constraint — `tests/test_tensorrt_llm_integration.py`
drives `prepare` and `generate` directly under pytest and passes.

## Measured on an H100

SmolLM2-135M, BF16, batch 1, 48 tokens, `sm90`:

| | |
| --- | --- |
| engine build (`prepare`) | 30.2 s |
| TTFT | 105.4 ms |
| inter-token latency (median) | 1.27 ms |
| steady-state rate | ~790 tokens/s |

TTFT and inter-token latency are measured by LM7 around the runtime's stream
rather than reported by the runtime, so the same numbers mean the same thing if a
second runtime is ever added.

Integration tests: 4 passed in 68 s, including a real engine build and a
streaming generation whose deltas reassemble into a completion.

## What LM7 does not own yet

- **The engine cache is metadata only.** LM7 computes an identity from the model,
  architecture, config and pinned versions, writes `lm7-engine.json`, and refuses
  a manifest built for a different card. It does **not** hand TensorRT-LLM an
  engine path, so the runtime rebuilds every time. Measured: 30.2 s on the first
  run and 30.7 s on the second, with the manifest matching. The JSON reports
  `engine_manifest_matched` and a hard-coded `engine_build_skipped: false`
  precisely so this cannot be misread as a cache hit. Wiring the directory
  through is the obvious next step and is not done.
- **No continuous batching is exercised.** The scheduler is TensorRT-LLM's and it
  is running, but `lm7 serve` submits one prompt, so nothing here measures it.
- **No OpenAI-compatible endpoint.** TensorRT-LLM ships `trtllm-serve` for that.
- **No comparison against the Inductor path.** Engine build time, TTFT,
  inter-token latency, tokens/s, peak memory and batch scaling against
  `lm7.compile` is the measurement that would justify this runtime existing, and
  it needs a harness that can drive both from one place. Not written.
- **One model, one card.** SmolLM2-135M on one H100. FP8 is accepted by the
  configuration gate on `sm89`+ and has not been run.

## Reference

- `src/lm7/runtimes/base.py` — the protocol, and why a runtime is not a `Backend`
- `src/lm7/runtimes/engines.py` — engine identity and manifest
- `src/lm7/runtimes/tensorrt_llm.py` — the adapter
- `tests/test_tensorrt_llm_runtime.py` — LM7's half, no GPU needed
- `tests/test_tensorrt_llm_integration.py` — real engine, `-m tensorrt_llm`

Not in CI: GitHub's GPU runners are gated to Team/Enterprise organizations, and
this needs both an Ampere-or-newer GPU and its own environment.
