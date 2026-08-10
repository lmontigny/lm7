# TensorRT-LLM (experimental)

`--backend trtllm` hands `lm7 model serve`'s port to NVIDIA's
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). LM7 resolves the target,
refuses the hardware TensorRT-LLM has no kernels for, translates its config into
`trtllm-serve`'s own argv, and hands over the process. What answers the port
afterwards is TensorRT-LLM, unmodified.

It is the second *launcher backend*, after `--backend vllm`, and it shares that
one's plumbing rather than paralleling it — see
[serving](serving.md#-backend-trtllm-the-same-handover-to-tensorrt-llm) for the
user-facing half.

> [!WARNING]
> **Experimental, one card, one model.** Run on an RTX 4070 SUPER (Ada `sm89`,
> 12 GiB) under WSL2 with SmolLM2-135M. Nothing here is a benchmark: see
> [what was and was not measured](#what-was-and-was-not-measured).

## Why this is a launcher, and not an in-process runtime

The first revision of this work ([#110](https://github.com/lmontigny/lm7/pull/110))
was not a launcher. It drove `tensorrt_llm._tensorrt_engine.LLM` in-process
behind a second protocol in `src/lm7/runtimes/` — `probe`/`supports`/`prepare`/
`generate` — parallel to `src/lm7/backends/` and with its own `ServeConfig`.
The review asked for "1 layer in between to allow different serving backend",
and by then `src/lm7/serve/` had grown exactly that layer for vLLM. So the
adapter became a launcher and the parallel protocol went away.

That was the right trade for three reasons beyond having one layer instead of
two, and they are worth recording because each cost real time to find:

**`tensorrt_llm.LLM` is not the TensorRT engine path on 1.2.x.** The public class
became the *PyTorch* backend and rejects `build_config` outright, pointing at
`_tensorrt_engine` instead. An in-process adapter that wants engine execution has
to import a leading-underscore module and pin itself to a private API. A launcher
imports nothing: `trtllm-serve` is a supported entry point, and *which* runtime
it uses is TensorRT-LLM's decision to make, not LM7's.

**The MPI re-exec is fatal in-process and harmless out of it.** TensorRT-LLM
spawns MPI workers that **re-execute the parent's command line**. Under
`python -m lm7` those workers re-ran the CLI with no arguments, hit argparse, and
`MPI_ABORT`ed the job *after* the engine had finished building — a successful
30-second build followed by `error: the following arguments are required:
command`. `mpirun -n 1` did not fix it; the shipped `trtllm-llmapi-launch`
wrapper did, which meant `lm7 serve` could only be run under another launcher.
Handing over the process removes the problem rather than working around it:
`trtllm-serve` is its own entry point, so its workers re-execute *it*.

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
the boundary. The module is ~180 lines, most of it comments, for that reason.

## Installing

TensorRT-LLM needs **its own environment**. It pins versions that conflict with
every other environment in this repo, which is the same reason vLLM is not an
LM7 extra: pinning a torch here would decide the torch version for everyone who
installs LM7.

```bash
python3 -m venv .venv-trtllm
.venv-trtllm/bin/python -m pip install tensorrt-llm==1.2.1
```

<!-- MEASURED-INSTALL -->

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

<!-- MEASURED-RUN -->

## What was and was not measured

<!-- MEASURED-RESULTS -->

## What is not done

- **No comparison against the Inductor path.** TTFT, inter-token latency,
  tokens/s, peak memory and batch scaling against `lm7.compile` is the
  measurement that would justify choosing one over the other, and it needs a
  harness driving both from one place. Not written. This remains the largest gap,
  and it was the largest gap in the first revision too.
- **No continuous batching exercised.** TensorRT-LLM's scheduler is running and
  it is the reason to reach for this backend, but nothing here submits concurrent
  requests, so nothing here measures it.
- **No quantization.** `--quantize` is refused rather than passed through:
  TensorRT-LLM quantizes at engine build time from an NVIDIA ModelOpt checkpoint,
  which is a different mechanism from LM7's weight-only path. Serving a
  pre-quantized checkpoint is untried.
- **One card, one model, one GPU.** No tensor parallelism, no `--tp_size`
  passthrough, and nothing above SmolLM2-135M.
- **Not in CI.** GitHub's GPU runners are gated to Team/Enterprise organizations,
  and this needs both an Ampere-or-newer GPU and an environment of its own.

## Reference

- `src/lm7/serve/trtllm.py` — the launcher
- `src/lm7/serve/cli.py` — `LAUNCHER_BACKENDS`, the layer shared with vLLM
- `tests/test_serve.py` — the translation and the refusals, no GPU needed
- `tests/test_trtllm_serve_integration.py` — a real server, `-m trtllm`
