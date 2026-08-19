# What `torch.compile` wins on sm89, and what the timing loop decides

A `torch.compile` speedup is usually quoted as one number against eager. On a
small-batch model that number is mostly not about compilation: it is about who
launches the kernels, and it moves by 2x depending on where the benchmark
synchronizes. This is the measurement behind that claim.

Everything here is one card — the local **RTX 4070 SUPER (Ada `sm89`, 12 GiB)**
under WSL2, PyTorch 2.13.0+cu130, Transformers 5.14.1, FP16 — through
[`benchmarks/compile_modes.py`](../benchmarks/compile_modes.py). For what the
four Inductor presets *are* and how LM7 forwards them, see
[TorchInductor options](inductor-options.md); this page is only about how much
they are worth here and how easily the measurement misreports it.

## Three arms

The script times three arms, none of which put LM7 in the timed call:

| arm | what it is |
| --- | --- |
| `eager` | PyTorch as written |
| `inductor` | `torch.compile(mode="default")` — fused kernels, ordinary launches |
| `reduce-overhead` | `torch.compile(mode="reduce-overhead")` — fused kernels **and** CUDA Graphs |

The middle arm is the one that makes the result readable. `eager → inductor` is
what better kernels buy; `inductor → reduce-overhead` is what removing per-launch
CPU work buys, because CUDA Graphs change nothing else.

## Results

Batched mean is one synchronization around 200 calls; per-call median is
`lm7.benchmark_callable`, which synchronizes before and after every call and is
what `benchmarks/gpu.py` reports. Both time identical work.

| model | batch | arm | batched | speedup | per-call | speedup |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| MLP | 1 | eager | 0.244 ms | 1.00x | 0.278 ms | 1.00x |
| | | `inductor` | 0.300 ms | 0.81x | 0.292 ms | 0.95x |
| | | `reduce-overhead` | 0.267 ms | **0.91x** | 0.340 ms | **0.82x** |
| ResNet-18 | 1 | eager | 7.021 ms | 1.00x | 5.836 ms | 1.00x |
| | | `inductor` | 2.518 ms | 2.79x | 2.648 ms | 2.20x |
| | | `reduce-overhead` | 0.523 ms | **13.43x** | 0.678 ms | **8.61x** |
| ResNet-18 | 64 | eager | 9.359 ms | 1.00x | 9.442 ms | 1.00x |
| | | `inductor` | 6.022 ms | 1.55x | 6.858 ms | 1.38x |
| | | `reduce-overhead` | 6.014 ms | **1.56x** | 6.860 ms | **1.38x** |
| SmolLM2-135M | 1 | eager | 57.530 ms | 1.00x | 61.178 ms | 1.00x |
| | | `inductor` | 19.665 ms | 2.93x | 17.337 ms | 3.53x |
| | | `reduce-overhead` | 1.918 ms | **30.00x** | 3.487 ms | **17.55x** |
| Llama-3.2-1B | 1 | eager | 45.666 ms | 1.00x | 47.926 ms | 1.00x |
| | | `inductor` | 17.615 ms | 2.59x | 22.112 ms | 2.17x |
| | | `reduce-overhead` | 7.051 ms | **6.48x** | 8.048 ms | **5.96x** |

## The headline number is CUDA Graphs, not codegen

On SmolLM2-135M, fusion accounts for 57.5 → 19.7 ms and CUDA Graphs for
19.7 → 1.92 ms. The launch-overhead step is four times the codegen step. Any
description of that 30x as "the compiler optimized the model" is wrong about
where it came from.

The control is ResNet-18 at batch 64. Same model, same kernels, 64x the
arithmetic per launch — and `reduce-overhead` becomes indistinguishable from
`inductor` (6.014 against 6.022 ms). Once the GPU is saturated there is no
launch gap left to hide, and the total speedup collapses from 13.43x to 1.56x.
That is the whole mechanism in one row: these small-batch multiples are a
statement about CPU dispatch, not about the arithmetic.

The model ladder says the same thing in a different way. SmolLM2-135M has 30
small decoder layers and gains 30.00x; Llama-3.2-1B has 16 larger ones, spends
more of each call inside kernels, and gains 6.48x.

And the MLP loses. Two linears and a GELU give fusion nothing to work with, and
CUDA Graph replay does not pay for its own overhead at this size — 0.91x
batched, 0.82x per-call. It is in the ladder precisely for this. Read the exact
ratio loosely: every MLP arm is under half a millisecond, and shortening the
warmup moves it as far as 0.30x. That compiling loses here is reproducible; the
size of the loss is not.

## The timing loop moves the answer by 2x

The two policies disagree, and they disagree most exactly where the win is
largest — SmolLM2 reports 30.00x batched against 17.55x per-call.

A per-call barrier is the honest measure of one isolated call's latency, and it
is also the pattern that denies CUDA Graphs their advantage: what they remove is
CPU launch overhead, and a loop that blocks on the GPU after every iteration
never lets that overhead overlap with anything.
[TorchInductor options](inductor-options.md#measured-behaviour) already flagged
this as a caveat on the `sm120` MLP result. It is not a caveat here, it is the
dominant term: on a launch-bound model the choice of policy is worth more than
the choice of preset.

Neither number is wrong. They answer different questions — "what does one
request cost when nothing else is queued" against "what does this cost inside a
loop that keeps the GPU fed" — and a serving loop that queues several steps
before synchronizing is closer to the second. Quote which one you measured.

## Warmup has to outlast the clock ramp

An idle RTX 4070 SUPER sits at **300 MHz** against a 3105 MHz boost clock. A
fixed five-iteration warmup does not leave that state, so an eager arm measured
first is charged for the ramp while a compiled arm measured after a 30-second
compile never pays it.

Measured on SmolLM2 with the naive ordering — five warmup iterations, eager
first — the reported gain was **60.24x**. The same box with a time-based warmup
on both arms reports 30.00x. Half the headline was the eager baseline being
measured on a cold card.

The script therefore warms for `--warm-seconds` (default 10) per arm rather than
a fixed count. During eager measurement the card still drops to P5 and ~35%
utilization, because a launch-bound model genuinely cannot keep it busy — that
is a real property of the workload, not an artifact, and it is why the eager
figures carry the clock they do.

## Reconciling with the existing sm89 table

[TorchInductor options](inductor-options.md#rtx-4070-super-eager-versus-reduce-overhead)
records an earlier eager-versus-`reduce-overhead` comparison on this same card.
Re-running `benchmarks/gpu.py` unchanged on this box today, alongside the new
script's per-call arm:

| model | source | eager | compiled | speedup |
| --- | --- | ---: | ---: | ---: |
| SmolLM2-135M | recorded in `inductor-options.md` | 70.867 ms | 4.455 ms | 15.91x |
| | `benchmarks/gpu.py` today | 51.981 ms | 3.453 ms | 15.06x |
| | `compile_modes.py`, per-call | 61.178 ms | 3.487 ms | 17.55x |
| Llama-3.2-1B | recorded in `inductor-options.md` | 38.721 ms | **17.982 ms** | 2.15x |
| | `benchmarks/gpu.py` today | 30.848 ms | **7.907 ms** | 3.90x |
| | `compile_modes.py`, per-call | 47.926 ms | **8.048 ms** | 5.96x |

The SmolLM2 row reconciles: three harnesses land between 15.06x and 17.55x, and
the compiled latency agrees to within 0.04 ms across the two harnesses run today.

The Llama row does not. The recorded 17.982 ms does not reproduce — `gpu.py`,
its own harness, now returns 7.907 ms on the same card and model. Note where
17.982 ms does land: on top of this page's `inductor` arm for Llama, 17.615 ms
batched, which is the no-CUDA-Graph number. The most likely reading is that
CUDA Graph capture was not active in the run that produced the recorded row,
which is exactly the confusion `cudagraphs_active` was added to settle. That is
a hypothesis from the number's position, not a diagnosis of a session that
cannot be re-run, and the recorded row is left in place rather than edited.

The eager column is a separate, smaller disagreement: `compile_modes.py` builds
its input as `input_ids` alone, while `gpu.py` passes the tokenizer's full
output including `attention_mask`, which can select a different attention path.
The compiled arms agree across both; the eager arms differ by up to 1.55x. This is the
reason this repo refuses to mix harnesses in one comparison: the two scripts
are each comparable within themselves and not across.

## What this does not show

- **WSL2 inflates it.** Kernel-launch overhead is higher under WSL2 than native
  Linux, which raises the eager baseline and therefore every multiple on this
  page. Nothing here was run on bare-metal Linux, so treat the small-batch
  numbers as an sm89-under-WSL2 result rather than an RTX 4070 result.
- **Batch 1 at five tokens is not a serving shape.** With `use_cache=False`
  these are full forward passes over `The capital of France is` — neither
  autoregressive decode against a KV cache (see
  [prefill and KV-cache decode](kv-cache-decode.md)) nor a realistic prefill
  length. It is close to a lower bound on per-call work, which is why the
  multiples are large.
- **Only ResNet-18 was swept over batch size.** Both causal LMs were measured at
  batch 1 only, so "the LMs are launch-bound" rests on the
  `inductor → reduce-overhead` step, not on a measured batch curve for them.
- **One card, one process each.** No `sm90`/`sm120` comparison, and no claim
  that these ratios transfer to another architecture.

## Reproducing

```bash
python benchmarks/compile_modes.py --model smollm2 \
  --output artifacts/compile-modes-rtx4070-smollm2-b1.json
```

`--model` takes `mlp`, `resnet18`, or any key from the shared `HF_MODELS` set;
`--batch-size` is what produces the saturation control above. Run each
configuration in a fresh process — Inductor caches compiled code, and a warm
cache makes a later arm look cheaper than it is.

`--output` writes the full result, including the host and PyTorch build, as
JSON. `artifacts/` is gitignored, so those files stay on the machine that
produced them; every table above was authored from one.
