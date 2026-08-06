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
samples per second. Peak device memory goes from 303 MB to 343 MB across the
same range.

**At batch 1 an H100 running a small model is measuring launch overhead, not the
card.** Sixty-four times the work costs nothing because there was nothing to
contend with in the first place. Any batch-1 latency figure quoted for this class
of model on this class of hardware — including the ones above — is a statement
about the CPU-side dispatch path.

The practical consequence for a serving deployment: batch aggressively before
reaching for a faster GPU.

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
0.02x each time while the absolute numbers improve ~2.5x.

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
