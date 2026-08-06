# NVIDIA H100 (`sm90`, Hopper)

What LM7 does on the GPU most production inference actually runs on, measured on
a single **NVIDIA H100 80GB HBM3** (`sm90`, driver 580.173.02, Intel Xeon
Platinum 8470 host) with `torch 2.13.0+cu130`, `torchao 0.17.0` and
`transformers 5.14.1` — the same software the Blackwell page used, so the two
cards compare directly.

Every other NVIDIA number in this repo comes from a gaming card (RTX 4070 SUPER,
Ada `sm89`) or a workstation card (RTX PRO 6000, Blackwell `sm120`). This page
exists because neither is what anyone deploys on.

## Nothing had to be special-cased

`nvidia:sm90` resolves, selects `inductor`, and runs:

```console
$ lm7 targets
Detected targets (2):
  nvidia:sm90: NVIDIA H100 80GB HBM3 (Hopper), 79.2 GiB
    precision: native fp32, fp16, bf16, int8, fp8
  cpu:x86_64: Intel(R) Xeon(R) Platinum 8470, 108.1 GiB
```

The generation is named, `fp8` is reported native, and `fp4` is correctly absent
— all of it from the plain-integer capability gates described on the
[Blackwell page](nvidia-blackwell.md#nothing-had-to-be-special-cased), with no
Hopper-specific code anywhere. That row was previously a prediction from the
gates; it is now an observation.

Concretely, `90` is one row of a descending threshold table in
`src/lm7/detection.py`, and the precision gates are integer comparisons:
`fp8` at `>= 89`, `fp4` at `>= 100`. Hopper clears the first and misses the
second, which is the whole of the sm90 support story. `tests/test_detection.py`
already pinned both — `("sm90", "Hopper")` in the generation table and an
explicit `fp4 == "absent"` assertion — before any H100 was available to run on,
so this hardware confirmed the tests rather than the other way around.

`lm7 doctor` on the same box, trimmed to the backends that resolve:

```console
$ lm7 doctor
LM7 0.1.0 diagnostics
Python: 3.12.3
PyTorch: 2.13.0+cu130
Platform: Linux-6.8.0-110-generic-x86_64-with-glibc2.39

Detected targets (2):
  nvidia:sm90: NVIDIA H100 80GB HBM3 (Hopper), 79.2 GiB
    precision: native fp32, fp16, bf16, int8, fp8
  cpu:x86_64: Intel(R) Xeon(R) Platinum 8470, 108.1 GiB

Registered backends (16):
  aot_inductor: available, version 2.13.0+cu130
  eager: available, version 2.13.0+cu130
  inductor: available, version 2.13.0+cu130
```

The remaining thirteen report `unavailable` with an install hint, which on this
box is accurate rather than a fault: it is a CUDA venv, and `coreml`,
`executorch`, `openxla`, `qnn`, `tvm`, `zentorch` and the rest were never
installed into it. `tensorrt` is among them — Torch-TensorRT sits in a separate
`.venv-trt` on this machine that was never exercised, so **no TensorRT result on
`sm90` exists**. The repo's TensorRT numbers come from other cards entirely:
Ada `sm89` in [the TensorRT evaluation](nvidia-tensorrt-evaluation.md) and
Blackwell `sm120` in [the wider sweep](tensorrt-validation.md).

## What compiling buys

`benchmarks/gpu.py`, BF16, median of 30 after 5 warmups.

**SmolLM2-135M:**

| batch | `eager` | `inductor` | speedup | throughput |
| --- | --- | --- | --- | --- |
| 1 | 14.940 ms | 7.320 ms | 2.04x | 137 samples/s |
| 8 | 14.875 ms | 7.252 ms | 2.05x | 1,103 samples/s |
| 32 | 15.207 ms | 7.200 ms | 2.11x | 4,445 samples/s |
| 64 | 14.869 ms | 7.013 ms | 2.12x | **9,126 samples/s** |

**Llama-3.2-1B:**

| batch | `eager` | `inductor` | speedup | throughput |
| --- | --- | --- | --- | --- |
| 1 | 8.548 ms | 3.329 ms | 2.57x | 300 samples/s |
| 32 | 8.766 ms | 3.452 ms | 2.54x | **9,271 samples/s** |

## Latency is flat from batch 1 to batch 64

This is the number to read first, and it is not about LM7 at all.

Median latency moves from 14.94 ms to 14.87 ms on `eager`, and 7.32 ms to
7.01 ms on `inductor`, while throughput scales close to linearly — 137 to 9,126
samples per second. Peak device memory moves from 289 MiB to 319 MiB on `eager`
across the same range: 64x the work for 10% more memory. See
[what the card actually holds](#what-the-card-actually-holds) for the full
memory picture.

**At batch 1 an H100 running a small model is measuring launch overhead, not the
card.** Sixty-four times the work costs nothing because there was nothing to
contend with in the first place. Any batch-1 latency figure quoted for this class
of model on this class of hardware — including the ones above — is a statement
about the CPU-side dispatch path.

The practical consequence for a serving deployment: batch aggressively before
reaching for a faster GPU. That advice has a ceiling, measured in
[where the flat-latency regime ends](#where-the-flat-latency-regime-ends) —
and the compile speedup does not survive to it.

## What the card actually holds

Peak device memory, same runs as the tables above, from
`torch.cuda.max_memory_allocated` via `benchmarks/gpu.py`. The card reports
79.2 GiB usable.

| model | batch | `eager` | `inductor` | compile cost | % of card |
| --- | --- | --- | --- | --- | --- |
| SmolLM2-135M | 1 | 289.1 MiB | 393.4 MiB | +104.3 MiB | 0.49% |
| | 8 | 292.4 MiB | 339.1 MiB | +46.7 MiB | 0.42% |
| | 32 | 303.8 MiB | 340.8 MiB | +37.0 MiB | 0.42% |
| | 64 | 318.9 MiB | 343.1 MiB | +24.2 MiB | 0.42% |
| | 256 | 410.0 MiB | 410.0 MiB | +0.0 MiB | 0.51% |
| | 1024 | 774.3 MiB | 774.3 MiB | +0.0 MiB | 0.95% |
| | 4096 | 2231.4 MiB | 2231.9 MiB | +0.5 MiB | 2.75% |
| Llama-3.2-1B | 1 | 2391.6 MiB | 2944.1 MiB | +552.5 MiB | 3.63% |
| | 32 | 2437.9 MiB | 2452.5 MiB | +14.6 MiB | 3.02% |
| | 256 | 2772.2 MiB | 2772.2 MiB | +0.0 MiB | 3.42% |
| | 1024 | 3918.2 MiB | 3918.2 MiB | +0.0 MiB | 4.83% |

**Nothing here fills the card.** The largest number in the table is 3.9 GiB
against 79.2 GiB available — 4.8%, at batch 1024. An 80 GB H100 running a 1B
model is 95% idle memory even when pushed well past any batch size these
latencies justify, and that is the honest context for every figure on this page.

**Batch size barely moves memory until it suddenly does.** SmolLM2-135M goes
289 → 319 MiB from batch 1 to batch 64: 64x the work for 10% more memory,
because weights dominate and activations are rounding error at a six-token
prompt. Past batch 64 activations start to matter and the curve turns roughly
linear — 410 MiB at 256, 774 MiB at 1024, 2231 MiB at 4096.

**The memory cost of compiling is real at batch 1 and exactly zero past batch
256.** +552 MiB on Llama-3.2-1B at batch 1 falls to +15 MiB at batch 32, then to
byte-identical peaks from 256 upward — `eager` and `inductor` allocate the same
2772.2 MiB and 3918.2 MiB. Whether the batch-1 excess is compile-time autotuning
workspace or allocator fragmentation has **not** been isolated; both would
produce this shape, and telling them apart needs a memory-snapshot run that was
not done.

That agrees with the one large-model datapoint this repo has: on `sm120`,
[Mixtral-8x7B occupying 92% of a 95 GiB card](limitations.md#what-torchcompile-actually-does-to-a-sparse-moe)
cost 0.4 GB to compile, or +0.4%. A near-full card is not an argument against
compiling — but a peak-memory figure measured at batch 1 overstates what
compilation costs anywhere near production batch sizes.

## Where the flat-latency regime ends

The batch 1–64 range above is flat because nothing is being asked of the card.
Pushing to batch 4096 finds the other end, and it is compute that runs out, not
memory — the card is still 97% empty when latency starts climbing.

| model | batch | `eager` | `inductor` | speedup | best throughput |
| --- | --- | --- | --- | --- | --- |
| SmolLM2-135M | 256 | 15.231 ms | 7.666 ms | 1.99x | 33,393 samples/s |
| | 1024 | 16.327 ms | 20.370 ms | **0.80x** | 62,718 samples/s (`eager`) |
| | 4096 | 55.687 ms | 74.851 ms | **0.74x** | **73,554 samples/s** (`eager`) |
| Llama-3.2-1B | 256 | 12.742 ms | 10.648 ms | 1.20x | 24,043 samples/s |
| | 1024 | 44.819 ms | 38.462 ms | 1.16x | **26,624 samples/s** |

**The Inductor speedup does not survive to large batch, and on SmolLM2-135M it
inverts.** 1.99x at batch 256 becomes 0.80x at 1024 and 0.74x at 4096 — compiled
is meaningfully *slower* than eager. This is the same lesson as the MLP result
below, arrived at from the other direction: Inductor's win here was removing
per-launch overhead, and once each kernel has enough work to hide that overhead
there is nothing left to remove. On Llama-3.2-1B the speedup decays more gently,
1.20x to 1.16x, without crossing over in the range measured.

**The 2.0x–2.6x headline speedups on this page are therefore a small-batch
result.** They are real, and they are what a batch-1 or batch-32 deployment
sees. They are not a property of the model, the card, or LM7.

**Peak throughput and peak speedup do not occur at the same batch size.**
SmolLM2-135M's best throughput, 73,554 samples/s, is *eager* at batch 4096 — a
configuration where compiling costs 34% more latency. Optimizing for the
compile speedup would have picked batch 256 and left 2.2x of throughput behind.

The practical consequence, and it points the opposite way from the batch-1
advice earlier on this page: past some batch size, measure whether compiling is
still buying anything. It stops, and then it reverses.

## The speedup is stable across generations; the hardware is what moves

The same workloads on the Ada `sm89` dev card, through the same harness:

| model | | Ada `sm89` | Hopper `sm90` | H100 advantage |
| --- | --- | --- | --- | --- |
| SmolLM2-135M | `eager` | 36.981 ms | 14.940 ms | 2.48x |
| | `inductor` | 18.298 ms | 7.320 ms | 2.50x |
| | **speedup** | **2.02x** | **2.04x** | — |
| Llama-3.2-1B | `eager` | 22.249 ms | 8.548 ms | 2.60x |
| | `inductor` | 8.622 ms | 3.329 ms | 2.59x |
| | **speedup** | **2.58x** | **2.57x** | — |

Two independent models, two generations, and the compile speedup lands within
0.02x each time while the absolute numbers improve ~2.5x. Both columns are
batch 1, so this is a statement about holding batch fixed and changing the card
— not about the speedup surviving a change of batch size, which
[it does not](#where-the-flat-latency-regime-ends).

What Inductor removes here is Python and kernel-launch overhead, and a faster
card shortens the kernels without changing how many there are. So the ratio
survives the hardware change — which is the useful property, because it means a
speedup measured on a cheap card is a reasonable predictor for an expensive one.

Note this is the opposite of what the [MoE measurements](limitations.md#what-torchcompile-actually-does-to-a-sparse-moe)
show across *model sizes*, where the speedup shrinks monotonically from 3.12x to
1.09x as the model grows. Scaling the hardware preserves the ratio; scaling the
model does not.

## These workloads are launch-bound, not FLOP-bound

Llama-3.2-1B is **faster than SmolLM2-135M** on this card — 8.548 ms against
14.940 ms eager — with roughly nine times the parameters. SmolLM2-135M is 30
layers; Llama-3.2-1B is 16. Fewer, fatter layers means fewer kernel launches, and
at this scale the launch count is what the clock is measuring.

Parameter count does not order latency here. Layer count comes closer.

This ordering is itself a small-batch artifact: by batch 1024 it has reversed,
with Llama-3.2-1B at 44.819 ms against SmolLM2-135M's 16.327 ms. Once there is
enough work per launch to be FLOP-bound, the nine-times-larger model is slower
again, which is the expected ordering and the reason "launch-bound" is a claim
about a regime rather than about these models.

## The workloads a serving engine will not take

Everything above is a causal LM, which is the one workload class where H100 users
already have better options. This section is the other half: models an LLM
serving engine does not accept at all, run through the same `lm7.compile` call.

`benchmarks/tensorrt_matrix.py`, FP16, `eager` against `inductor`:

| model | batch | `eager` | `inductor` | speedup |
| --- | --- | --- | --- | --- |
| MLP (plain `nn.Module`) | 1 | 0.068 ms | 0.121 ms | **0.56x** |
| | 32 | 0.065 ms | 0.127 ms | **0.51x** |
| ResNet-18 (vision) | 1 | 1.507 ms | 0.753 ms | 2.00x |
| | 32 | 1.635 ms | 0.930 ms | 1.76x |
| BERT (encoder) | 1 | 3.711 ms | 2.226 ms | 1.67x |
| | 32 | 3.630 ms | 2.169 ms | 1.67x |

All twelve cells ran unmodified and agreed with eager to FP16 precision (max
elementwise difference 9.77e-03). Nothing needed a Hopper-specific path.

Only the MLP is *arbitrary* in the strict sense — a hand-written module with no
model library behind it, which is the case
[Optimum](../notes/competition.md) does not cover and LM7's design bet is about.
ResNet-18 and BERT are library models; what makes them relevant here is that
they are not causal LMs, so vLLM and TensorRT-LLM have nothing to offer them.

**Compiling loses on the small MLP, at both batch sizes.** 0.121 ms against
0.068 ms is not measurement noise, and it reproduces the Ada result (0.322 ms
compiled against 0.235 ms eager). Below some amount of work per call there is no
Python or launch overhead left to remove, and Inductor's own dispatch is the
larger cost. `fallback` will not save a user from this, because nothing failed —
it compiled fine and came out slower. Measure before assuming compilation is
free.

Note that this is one of *two* ways to land on the losing side. The MLP loses
because there is too little work per call; SmolLM2-135M at
[batch 1024 and above](#where-the-flat-latency-regime-ends) loses because there
is too much. Compilation pays in the middle.

**Vision and encoder workloads get real speedups**, 1.67x–2.00x, on a card where
the alternative tooling declines to run them.

**The flat-latency result holds off the causal-LM path too.** ResNet-18 goes
1.507 → 1.635 ms from batch 1 to 32, BERT 3.711 → 3.630 ms. That confirms the
finding above is a property of the hardware at this scale rather than anything
about transformers or about generation.

## Compile cost

| model | `eager` first call | `inductor` first call | Ada `inductor` |
| --- | --- | --- | --- |
| SmolLM2-135M | 0.79 s | 21.4–23.6 s | 33.8 s |
| Llama-3.2-1B | 1.10 s | 12.7–13.3 s | 20.5 s |

Compilation is CPU work, so the 24-vCPU Xeon host is what makes it ~1.5x cheaper
than the Ada box, not the GPU. It is still paid on every process start — see
[JIT vs. AOT](jit-vs-aot.md) for moving it out of the serving path.

## What this page does not say

- **One GPU.** This is a single H100. LM7 has no tensor or data parallelism; a
  model that does not fit on one card is out of scope. See
  [limitations](limitations.md#scope-of-the-project).
- **Prefill only.** These are forward-pass latencies. LM7's exported causal-LM
  artifacts do not capture a KV-cache decode loop, so none of this is a
  tokens-per-second figure for generation.
- **No FP8 result yet.** `sm90` reports `fp8: native` and LM7's FP8 path has not
  been exercised on this card. Native is not the same as used — the
  [Blackwell page](nvidia-blackwell.md#native-is-not-the-same-as-used) explains
  why that distinction matters.
- **`inductor` p95 is noisy at low batch** — 17.58 ms against a 7.32 ms median at
  batch 1, and 15.76 ms at batch 8, resolving to 7.32 ms and 7.10 ms at batch 32
  and 64. Consistent with warmup bleed into the sample window, but that is a
  hypothesis and has not been isolated.
- **No serving stack was involved.** If the workload is LLM serving specifically,
  vLLM and TensorRT-LLM exist and are better at it. LM7's case is the model that
  is not an HF causal LM.
- **The batch sweep used a six-token prompt.** Batch is the only thing scaled;
  sequence length is not. A long-context workload would reach the memory and
  compute ceilings at far smaller batch, and none of the crossover points here
  transfer to it.
- **`fp8` and quantization were not part of any of this.** Every number on this
  page is BF16 or FP16. The `artifacts/h100-quant/` directory on the test box is
  empty — an INT8/FP8 run on `sm90` was set up and never executed.
