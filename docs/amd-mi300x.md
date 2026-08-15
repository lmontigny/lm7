# AMD MI300X (`gfx942`, CDNA 3)

What LM7 does on the first AMD GPU it has ever run on, measured on a single
**AMD Instinct MI300X VF** (`gfx942`, CDNA 3, 191.7 GiB, SPX compute partition,
amdgpu 6.16.13) rented from the AMD Developer Cloud, with ROCm 7.2.4,
`torch 2.10.0+rocm7.2.4`, `triton 3.6.0+rocm7.2.4`, `torchao 0.17.0` and
`transformers 5.15.0`, on an Intel Xeon Platinum 8568Y+ host with 235.9 GiB.

Every AMD claim in this repo said "implemented", never "validated" — four files
said it in nearly the same words. This page is what replaced that.

The ROCm PyTorch on this host is a **pre-pulled Docker image**, not a system
install: the host's `python3` has no torch at all. Everything below ran inside
`rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`.

## Nothing had to be special-cased, and the predictions held

`amd:gfx942` resolves, names its silicon, and selects `inductor`:

```console
$ lm7 targets
Detected targets (2):
  amd:gfx942: AMD Instinct MI300X VF (CDNA 3), 191.7 GiB
    precision: native fp32, fp16, bf16, int8, fp8
  cpu:x86_64: INTEL(R) XEON(R) PLATINUM 8568Y+, 235.9 GiB

$ lm7 explain --target amd
Selected inductor for amd:gfx942
```

`examples/rocm_mlp.py` — the first thing [the AMD guide](amd-rocm.md) tells a
reader to run, and until now the first thing nobody had — compiles and agrees
with eager:

```console
$ python examples/rocm_mlp.py
GPU: AMD Instinct MI300X VF
ROCm: 7.2.53211
Target: amd:gfx942
Backend: inductor
Output: shape=(8, 4), device=cuda:0
TorchInductor output matches eager ROCm.
```

The `gfx` tables in `src/lm7/detection.py` were written from AMD's ISA
documentation and pinned by unit tests before any AMD GPU was available, exactly
as the `sm90` table was before an H100 was. Every value they predicted is what
the hardware reports: generation `CDNA 3`, `fp8` native, `fp4` absent, and
`fp8_format: fnuz` — the CDNA 3 encoding, which is *not* the OCP `e4m3` that
`sm89`+ implements, so an FP8 number from this card and one from an H100 were
not produced in the same format.

`cuda_build` answers the question that matters more on ROCm than on CUDA, since
a missing `gfx` target fails hard at load rather than JIT-ing from PTX:

```json
{"arch_list": ["gfx908","gfx90a","gfx942","gfx1030","gfx1100","gfx1101",
               "gfx1200","gfx1201","gfx950","gfx1151","gfx1150"],
 "native_kernels": true, "architecture_specific": null}
```

`architecture_specific` is `null` rather than `false` on purpose: ROCm expresses
the `sm_90a` equivalent as feature suffixes on the target string
(`gfx942:sramecc+:xnack-`), so that is a question this vendor does not answer.

## What compiling buys

`benchmarks/nvidia_matrix.py --plan core`, BF16, median of 20 after 5 warmups,
one cell per process. **20 of 20 cells `ok`**, no failures.

| model | eager | inductor | inductor-cudagraphs | inductor-max-autotune |
| --- | --- | --- | --- | --- |
| mlp (8.4M, hand-built) | **0.094** | 0.133 | 0.112 | 0.100 |
| resnet18 | 1.771 | 1.518 | **1.067** | 1.157 |
| bert-base-uncased | 3.330 | 2.253 | **0.845** | 1.361 |
| SmolLM2-135M | 14.657 | 7.396 | **2.016** | 7.302 |
| Llama-3.2-1B | 7.197 | 3.575 | **1.824** | 2.575 |

Median ms. Bold is the fastest path for that model.

### HIP graph capture behaves like CUDA's

This is the finding. `backends/inductor.py` has emitted
`cudagraphs_requested` / `cudagraph_skips` / `cudagraphs_active` on AMD since it
was written, reading the same Dynamo counter, and nothing had ever checked
whether the numbers meant anything. They do: **every `inductor-cudagraphs` cell
captured, with zero skips**, on all five models.

It is also the largest win available on this card — 7.3x over eager on
SmolLM2-135M and 3.9x on Llama-3.2-1B, against 2.0x for plain Inductor on both.
The gap is widest exactly where the repo's model notes predict it should be: a
30-layer 135M model is launch-bound, and removing launch overhead is what a graph
does. SmolLM2-135M is *slower in eager* than the 9x larger Llama-3.2-1B (14.657
against 7.197) for the same reason.

### `max-autotune` is not worth its compile time here

It beats plain Inductor on Llama-3.2-1B (2.575 against 3.575) and on bert, ties
on SmolLM2 (7.302 against 7.396), and never beats CUDA Graphs. It costs 29–34 s
of first-call compile against 8–15 s. Triton autotunes with AMD-specific
parameters — `matrix_instr_nonkdim`, `waves_per_eu`, `kpack` — so the path is
real, not a silent fallback.

### The MLP is where compiling loses

0.094 ms eager against 0.133 ms compiled. The hand-built MLP is in the ladder
precisely to catch this case, and it reproduces on a third vendor.

## Quantization, and why none of it changes the gate

`benchmarks/quantization.py`, Llama-3.2-1B, BF16 baseline, four prompts.
`_QUANTIZATION_VENDORS` refuses every mode on `amd`; this harness calls
`_apply_quantization` directly and never reaches that gate, so these numbers
exist without the gate having moved.

| mode | median | vs BF16 | storage | top-1 | max logit diff |
| --- | --- | --- | --- | --- | --- |
| none (BF16) | 3.910 ms | — | 2.472 GB | — | — |
| `int8` | 38.027 ms | **9.72x slower** | 1.500 GB (1.65x) | 4/4 | 0.73 |
| `fp8` | 4.897 ms | 1.25x slower | 1.668 GB (1.48x) | 4/4 | 1.02 |
| `nvfp4` | 10.548 ms | 2.70x slower | 1.073 GB (2.30x) | **3/4** | 4.62 |
| `fp8-dynamic` | 5.311 ms | 1.36x slower | 1.666 GB (1.48x) | 4/4 | 1.09 |
| `fp8-dynamic-rowwise` | 4.743 ms | 1.21x slower | 1.668 GB (1.48x) | 4/4 | 1.12 |
| `nvfp4-dynamic` | refused | — | — | — | — |

Four things to read out of that, and only the last is a recommendation.

**`nvfp4-dynamic` is refused by torchao's own gate**, with *"NVFP4 DYNAMIC mode
is only supported on sm100+ machines"*. That is `precision_support`'s
`fp4: absent` confirmed a second time by an independent mechanism.

**FP8 is the mode this silicon is built for, and it is still a latency loss.**
Per-row dynamic FP8 is the best of the six at 1.21x slower than BF16 while
holding 4/4 top-1 — the same shape as every weight-only result in this repo,
where footprint is the reliable benefit and speed is not.

**`nvfp4` weight-only fails the accuracy bar at 3/4**, which is how the NVIDIA
table treats a rejected (model, mode) pair, and its 4.62 max logit difference is
the largest here by a factor of four.

**The INT8 number was blamed on the toolchain, and the toolchain was innocent.**
torchao 0.17.0 prints `Skipping import of cpp extensions due to incompatible
torch version` against torch 2.10 (it wants ≥ 2.11), so the first reading of
this table was that the dequantization ran unfused and the 9.72x was an
artifact. That hypothesis was stated here and it is wrong — see the re-run
below.

### Which modes this admitted, and which it did not

The latency table alone reads like a fallback. It is not: `benchmarks/fp8_kernel_check.py`
on this card shows both dynamic modes emitting `_scaled_mm` and no plain `mm`,
the same generated code the `sm90` and `sm120` rows show — see
[quantization](quantization.md#the-same-check-on-cdna-3). **CDNA 3 computes in
FP8**, and it is still slower than BF16 on this model and shape.

That is enough to admit them, because it is the bar the NVIDIA table already
uses: three of the four admitted NVIDIA activation pairs also cost latency and
are admitted on accuracy. So `_QUANTIZATION_VENDORS` now reaches AMD for `fp8`,
`fp8-dynamic` and `fp8-dynamic-rowwise`, and for nothing else.

| mode | on AMD | why |
| --- | --- | --- |
| `fp8`, `fp8-dynamic`, `fp8-dynamic-rowwise` | **admitted** | real FP8 GEMM, 4/4 top-1, logit differences 1.11–1.40 against the H100's 1.09–1.33 |
| `int8` | refused | 4/4, and ~10x slower on both PyTorch versions — a mode whose best case is a regression |
| `nvfp4` | refused | 3/4 top-1 |
| `nvfp4-dynamic` | refused | no FP4 silicon; torchao refuses it first |

Two gates had to be built rather than reused. `compute_capability` is `None` for
every `gfx` and every capability check reads `None` as "do not gate" — correct
for an unresolved NVIDIA target, and enough to hand FP8 to a `gfx90a` that has
none. Both `_supports_fp8` and `supports_native_bf16` now read the same
`precision_support` table `lm7 targets` prints.

### The same sweep on torch 2.13, which settles it

Rebuilt in a separate venv from `download.pytorch.org/whl/rocm7.2` —
`torch 2.13.0+rocm7.2`, same ROCm 7.2.4, same `torchao 0.17.0`, which now loads
its extensions instead of skipping them wholesale:

| mode | 2.10 median | 2.13 median | 2.10 ratio | 2.13 ratio |
| --- | --- | --- | --- | --- |
| none (BF16) | 3.910 ms | **3.410 ms** | — | — |
| `int8` | 38.027 ms | **37.245 ms** | 9.72x | **10.92x** |
| `fp8` | 4.897 ms | 4.343 ms | 1.25x | 1.27x |
| `nvfp4` | 10.548 ms | **6.690 ms** | 2.70x | 1.96x |
| `fp8-dynamic` | 5.311 ms | 4.290 ms | 1.36x | 1.26x |
| `fp8-dynamic-rowwise` | 4.743 ms | **4.184 ms** | 1.21x | 1.23x |

Read the `int8` row twice. **It barely moved in absolute terms** — 38.027 ms to
37.245 ms, under 3% — while the BF16 baseline got 13% faster, so the *ratio got
worse*. The compiled extensions were never the explanation. INT8 weight-only is
roughly 10x slower than BF16 on this silicon, and the earlier caveat on this page
overstated the confound.

What torch 2.13 did change is `nvfp4`, from 2.70x to 1.96x — a 37% improvement
that never reached the accuracy problem, which is unchanged at 3/4 and a 4.62
logit difference. And every ratio here is against a faster baseline, so the two
runs compare by ratio and not row-for-row.

The honest summary is the one the NVIDIA table already reaches: **footprint is
the reliable benefit of weight-only quantization and speed is not**, on a third
vendor now. `fp8-dynamic-rowwise` remains the least-bad at 1.23x.

## An AOTInductor artifact, and the bug it found

`aot_inductor` accepts `amd` as of [#173](https://github.com/lmontigny/lm7/pull/173).
The open question was whether the C++ wrapper links against ROCm. It does:

```console
$ lm7 explain --target amd --backend aot_inductor
Selected aot_inductor for amd:gfx942

$ python -m pytest tests/test_amd_aot_integration.py -q -m rocm
6 passed
```

The manifest records the ROCm pair, which is the whole reason that PR did not
simply reuse the CUDA fields:

```json
{"torch": "2.10.0+rocm7.2.4.git3d3aa833", "device": "amd", "device_bound": true,
 "hip": "7.2.53211", "gcn_architecture": "gfx942",
 "device_name": "AMD Instinct MI300X VF"}
```

No `cuda`, no `compute_capability` — `torch.cuda.get_device_capability()` answers
`(9, 4)` on this card, and recording it would have written `sm94`, an NVIDIA
architecture that has never existed.

### `aoti_load_package` does not import what it uses

The first run was 5 of 6. `test_amd_aot_artifact_reloads_without_compiling`
failed with `module 'torch._inductor' has no attribute 'codecache'`.

`aoti_load_package` reaches for `torch._inductor.codecache` without importing
it, and that submodule is lazy. So the failure appears **only** in a process that
did nothing but `import torch` — which is exactly the fresh-process reload the
AOT path exists for. Compiling in the same process hides it, because Inductor
pulls codecache in on the way through. Nothing about this is AMD-specific; a
CUDA box on a torch that imports codecache eagerly would never see it, which is
why no NVIDIA lifecycle run in this repo ever did.

The mismatch hint added in the same PR is what made it diagnosable rather than
mysterious — it correctly took its no-differences branch:

> The artifact was built with PyTorch 2.10.0+rocm7.2.4, ROCm runtime 7.2.53211,
> GPU architecture gfx942, which is what this process has, so the package or its
> dependencies are at fault rather than the environment.

Fixed by importing the submodule before the load.

### The lifecycle across a real process boundary

`benchmarks/aot_artifact_lifecycle.py run --model smollm2 --dtype bfloat16`,
each stage its own process:

| stage | wall | load | to first inference | steady |
| --- | --- | --- | --- | --- |
| export | 39.29 s | — | — | — |
| `lm7.load_artifact`, cold | 15.94 s | 9.62 s | 13.06 s | **4.045 ms** |
| `lm7.load_artifact`, warm | 15.73 s | 9.64 s | 12.87 s | 3.996 ms |
| `aoti_load_package`, cold | — | — | — | **failed** |
| `inductor` JIT, cold cache | 23.53 s | — | 22.01 s | 7.969 ms |
| `inductor` JIT, warm cache | 12.33 s | — | 11.17 s | 8.565 ms |

The artifact is 0.54 GB and 29.60 s of build. Its steady-state call is **1.97x
faster than the JIT it came from** (4.045 ms against 7.969 ms), which is the same
per-call framework-overhead effect the NVIDIA page records.

**The two `aoti_load_package` rows are the codecache bug, reproduced.** That arm
calls PyTorch's loader directly and bypassed LM7's, so it failed where
`lm7.load_artifact` succeeded — the clearest possible demonstration that the bug
is in the raw call and not in the artifact. The benchmark has since been given
the same import, so a re-run would fill those rows in; the table above is the
run that found it.

All six mismatch cases were **rejected with a clear message**, which is the bar
[artifact compatibility](aot-artifact-compatibility.md) sets — including the
architecture guard, firing for the first time on a `gfx` pair:

> its aot_inductor payload was built for `amd:sm89`, but this machine is
> `amd:gfx942`. Kernels are compiled per architecture, so it cannot run here.

## What 191.7 GiB holds

Two models this project could not previously measure, both at their natural
size on one card.

**OLMoE-1B-7B** (6.92B total, 1B active), `benchmarks/moe.py`, BF16 — **not** the
matrix harness, and the two disagree by 2.3x, so do not read this beside the
table above:

| | graphs | breaks | ops | eager | inductor | speedup | peak VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OLMoE-1B-7B | 1 | **0** | 1055 | 30.093 ms | 19.277 ms | 1.56x | 13.9 / 14.1 GB |

Zero graph breaks, and both backends agree on the next token. That is the
transformers 5.x behaviour [limitations](limitations.md) records — *"on
transformers 5.x there are no breaks at all"* — now confirmed on a third vendor
under `transformers 5.15.0`. It also contradicts nothing about the older claim,
because the claim was always about the (model, transformers version) pair.

**Mistral-7B-Instruct-v0.3** (7.25B), `lm7 model run`, FP16 — the first *dense*
7B measured anywhere in this repo. [Model
coverage](limitations.md#model-coverage) listed it as named-but-unmeasured with
the note that *"measuring it needs a rented card"*:

| | params | latency | storage | peak VRAM | next token |
| --- | --- | --- | --- | --- | --- |
| Mistral-7B-v0.3 | 7.25B | 7.035 ms | 14.50 GB | 14.85 GB | `Paris` |

First call 21.9 s. Peak VRAM is **7.7% of this card**, which is the sense in
which 191.7 GiB is not the constraint for anything on the current ladder.

**The two entries that needed this card**, same harness, same BF16:

| | total / active | graphs | breaks | ops | eager | inductor | speedup | peak VRAM | % of card |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLMoE-1B-7B | 6.9B / 1B | 1 | 0 | 1055 | 30.093 ms | 19.277 ms | 1.56x | 14.1 GB | 7.4% |
| Qwen3-30B-A3B | 30.5B / 3.3B | 1 | 0 | 3247 | 99.235 ms | 65.950 ms | 1.50x | 61.8 GB | 32.2% |
| Mixtral-8x7B | 46.7B / 12.9B | 1 | 0 | 1919 | 44.187 ms | 29.066 ms | 1.52x | 93.8 GB | 48.9% |

**Qwen3-30B-A3B had never run anywhere in this repo.** It was reachable by name
in `benchmarks/moe.py` and nothing had pointed a harness at it.

**Mixtral-8x7B lands on 93.8 GB, which is the number the Blackwell run
reported to the decimal** — and there it was 98.7% of a 96 GiB card, measured
with 1.2 GB of headroom. Here it is 48.9% of the card. That is the sense in which
this hardware was the point.

It also **contradicts the shrinking-speedup claim** that
[limitations](limitations.md) records from Blackwell, where compiling Mixtral
bought 1.09x and the sequence read 3.12x → 1.60x → 1.09x as models grew. On this
card all three MoE models sit between 1.50x and 1.56x with no size trend at all,
including the one that reproduces the Blackwell footprint exactly. Same harness,
same dtype, different card and different PyTorch — so the honest reading is that
the trend was a property of that measurement rather than of MoE size, and neither
run alone can say which.

Zero graph breaks throughout, and every backend pair agrees on the next token.

## Decode, the memory-bound half

Everything above is a forward pass. Generation is two problems, and the second
one — a long run of one-token passes bounded by how fast weights come out of
memory — is the half [prefill and KV-cache decode](kv-cache-decode.md) measures
on an H100 and this page previously left blank.

`benchmarks/decode.py` on `unsloth/Llama-3.2-1B-Instruct`, BF16, 100 decode steps
per cell, prompts of tiled English, in the `torch 2.13.0+rocm7.2` venv with
`transformers 5.15.0` — the same torch minor the H100 page used
(`2.13.0+cu130`), chosen so the comparison is not also a version comparison.

**This is 16 cells of the H100 page's 60**, and they were not produced the way
that page's were. Read [the two caveats below](#what-the-decode-numbers-are-not)
before comparing anything.

At 512 tokens, batch 1 — the shape the H100 page leads with:

| Arm | Prefill | Decode | Throughput | vs eager | MI300X ÷ H100 |
| --- | --- | --- | --- | --- | --- |
| `eager` | 9.3 ms | 9.318 ms/token | 107 tok/s | 1.00x | **1.47x faster** |
| `inductor` | 6.9 ms | 3.539 ms/token | 283 tok/s | 2.63x | 1.26x faster |
| `cudagraphs` | 14.0 ms | **2.288 ms/token** | **437 tok/s** | **4.07x** | 1.29x *slower* |
| `decode-only` | 10.2 ms | 2.290 ms/token | 437 tok/s | 4.07x | 1.28x *slower* |

**The card is faster than an H100 at the arm that does not measure the card, and
slower at the arm that does.** Eager decode here is 9.32 ms/token against the
H100's 13.72 — but the H100 page's own reading of that column is that eager
decode is "Python and kernel launches almost end to end", so the MI300X winning
it says the host loop is cheaper, not that the GPU is faster. Once graphs remove
the launches, the ordering inverts: 2.288 ms/token against 1.77.

That is also why compiling buys less here — **4.07x against the H100's 7.75x**.
Not because HIP graph capture underperforms, but because there was less overhead
to delete. Both machines land in the same place by a different route:
`cudagraphs` and `decode-only` are within 0.1% of each other on every cell, so
compiling the prompt pass buys nothing at these lengths on either vendor.

Per-token decode, `eager`, across the shapes that ran:

| Prompt | batch 1 | batch 4 | batch 8 |
| --- | --- | --- | --- |
| 512 | 9.318 | 8.928 | 9.117 |
| 1024 | 9.203 | 9.228 | — |

Flat, exactly as on the H100 — 8.93 to 9.32 ms/token across eight times the
batch. The same grid with HIP graphs, where the work becomes visible:

| Prompt | batch 1 | batch 4 | batch 8 |
| --- | --- | --- | --- |
| 512 | 2.288 (4.07x) | 2.420 (3.69x) | 2.649 (3.44x) |
| 1024 | 2.681 (3.43x) | — | — |

Decode throughput under `cudagraphs`, tokens/second across the batch: 437 at
batch 1, 1653 at batch 4, 3021 at batch 8 (512-token prompt). Peak memory never
exceeded **2.79 GiB** — 1.5% of the card.

Three counters held on every one of the 16 cells: `cudagraphs_active` true on
both graphs for the two capture arms, `recompiled_during_decode` false, and
`steady` frames zero. **HIP graph capture drives a KV-cache decode loop for 100
consecutive tokens without recapturing** — the claim [HIP graph
capture](#hip-graph-capture-behaves-like-cudas) makes for a forward pass, now
also for the stateful path.

### The harness cannot complete a sweep on this machine

The 60-cell sweep the H100 page runs **does not finish here.** It died after 13
cells with `segfault at a9 ... in python3.12`, and re-running it reproduced a
crash every time, with three different signatures — `segfault at a9`,
`free(): invalid pointer`, `corrupted double-linked list`. All three are glibc
heap-corruption symptoms, the fault address is inside the CPython binary rather
than `libamdhip64` or `libtorch`, VRAM unwinds to zero, and the host had 227 GiB
free. It is not an OOM and not a GPU fault.

It is also **not deterministic and not shape-dependent**, which took four
narrowing runs to establish — the first two readings, that it was a 1024-token
problem and then that it was HIP capture at 1024, were both wrong:

| Arms, one process | Result |
| --- | --- |
| `cudagraphs` alone, s1024 and s2048 | clean |
| `inductor cudagraphs`, s1024 | clean |
| `eager cudagraphs`, s1024 | **crash** |
| all four, s512 — a cell that had just passed | **crash** |

The common factor is the `eager` arm running *before* another arm in the same
process; the crash then lands on the later arm. Isolating one process per
(arm, shape) sidesteps it completely — **every cell launched that way exited 0,
18 of them before the run was stopped by hand** — and that is how the numbers
above were produced.

What this does *not* establish is where the bug lives. The same harness completes
all 60 cells on an H100, so it is not universal, but one machine and one ROCm
build cannot separate LM7, `torch 2.13.0+rocm7.2`, and this container's glibc. No
minimal reproducer outside `decode.py` was built, so nothing has been filed
upstream. See [limitations](limitations.md#compilation-and-artifacts).

### What the decode numbers are not

- **16 cells, not 60.** Prompt lengths 2048, 4096 and 8192 never ran, and batch 8
  ran only at 512. The H100's most interesting decode result is that the speedup
  *collapses* as context grows — 7.75x at 512/batch 1 down to 1.86x at
  8192/batch 8. **Nothing here reaches the lengths where that happens**, so this
  page says how compiling pays at short context and is silent about the slope.
  The 3.44x at 512/batch 8 is the only hint that the same decline starts.
- **A different execution mode from the H100 rows.** Those 60 cells ran four arms
  in one process; these 16 ran one arm per process out of necessity. A fresh
  allocator and a cold cache per cell is not obviously worth nothing, so treat
  cross-vendor ratios as indicative rather than exact.
- **Token agreement was recovered offline, not asserted by the harness.** With
  one arm per process each cell is its own reference, so its `same_tokens` field
  is self-comparing and vacuous. Comparing the recorded token ids across cells
  afterwards, every arm matches `eager` exactly at all four complete shapes — the
  check holds, but it was reconstructed rather than enforced during the run.

## Serving

`lm7 model serve` had been [validated on five
targets](serving.md) and AMD was not one of them. It is now, on
SmolLM2-135M-Instruct at FP16, backend resolved to `inductor`:

```console
$ curl -s localhost:8123/health
{"status":"ok","model":"HuggingFaceTB/SmolLM2-135M-Instruct",
 "target":"amd:gfx942","backend":"auto"}
```

`/v1/chat/completions` returns real text — which is the check that matters,
because [the NVIDIA failure](limitations.md) this repo records was an endpoint
answering HTTP 200 with an empty string and NaN logits. Here the model answers
in words. `pytest -m serve` (40) and `-m serve_load` (8) both pass.

Warm requests, 32 tokens each, measured end to end through HTTP:

| request | wall | per token |
| --- | --- | --- |
| 1 | 589 ms | 18.4 ms |
| 2 | 587 ms | 18.3 ms |
| 3 | 715 ms | 22.3 ms |

**Read `/metrics` cumulatively or not at all.** Its `ttft_ms` and `tpot_ms` are
running averages over every request including the first, which pays for
compiling the decode loop — after three requests they read 14606 ms and 234 ms,
against the ~18 ms/token the warm requests actually take. The first request cost
19.5 s to first token.

Two fields are worth reading directly:

- **`steady_frames: 0`** — no token triggered a compile, which is the regression
  the separate prefill and decode graphs exist to prevent and the reason that
  counter is exposed over HTTP at all. `prefill_lengths: 2` is the expected
  cost of two distinct prompt lengths.
- **`memory_kind: device`, `memory_total_bytes: 205822885888`** — the
  `torch.cuda.mem_get_info` path reports ROCm memory correctly, so the
  whole-card figure in `/metrics` is real on AMD and not a CUDA-only field
  returning nothing.

### The vLLM ROCm handover, which had never been run

`docs/limitations.md` said the ROCm and TPU handovers had never been run. The
ROCm one has now.

Not from PyPI: `pip install vllm` fetches the CUDA build and would displace the
ROCm torch. The route is AMD's own image, `rocm/vllm:latest` — vLLM
`0.11.2.dev673` on `torch 2.9.0a0+rocm`, hip 7.0 — with LM7 installed beside it,
which leaves torch alone. `runtime_installed` then flips from `false` to `true`
and resolves an executable.

```console
$ lm7 model serve hf://HuggingFaceTB/SmolLM2-135M-Instruct \
    --target amd --backend vllm --port 8200
(APIServer) INFO: Application startup complete.
```

`/v1/models` answers with `"owned_by": "vllm"`, chat completions return text, and
`stream: true` delivers chunked deltas. **It needed no LM7 fix to start** — worth
recording, because CUDA needed two (`VLLM_WSL2_ENABLE_PIN_MEMORY` and
`--vllm-arg`) plus a FlashInfer workaround before it would come up at all.

What that does *not* establish: nothing about vLLM's throughput was measured,
here or on any other platform, and the model is a 135M. LM7's involvement ends
when the process starts.

## What this page does not say

- **One card, one session, one partition.** SPX, so a single logical device;
  nothing here says anything about CPX or about multi-GPU.
- **This is an MI300X `VF`** — an SR-IOV virtual function. It reports the full
  191.7 GiB and a single SPX partition, so it behaves as the whole card, but it
  is not a bare-metal MI300X and no comparison against one was possible.
- **Two PyTorch versions.** Everything except the quantization re-run ran on
  `torch 2.10.0+rocm7.2`; the re-run used `2.13.0+rocm7.2` in a separate venv, and
  the two are compared by ratio rather than row-for-row because the BF16 baseline
  moved 13%. The matrix and MoE numbers are 2.10 and were not repeated.
- **No MIGraphX.** The native bindings import on the host and not inside the
  ROCm PyTorch container, and chasing that was not worth the metered time. The
  [evaluation plan](amd-migraphx.md) is unchanged.
- **No CI, and there will not be.** GitHub's hosted runners provide no AMD GPU,
  and the GPU-hosted runners that exist are gated to Organizations on
  Team/Enterprise Cloud.
- **Serving is one model at 135M, single stream.** No concurrency, no larger
  model, no quantized serve — which is the configuration that was silently wrong
  on NVIDIA. The vLLM handover starts and answers; **its throughput is
  unmeasured**, as it is on every platform in this repo.
- **Decode is measured at short context only.** [16 of the H100 page's 60
  cells](#what-the-decode-numbers-are-not) — nothing past a 1024-token prompt,
  because the harness [cannot complete a sweep on this
  machine](#the-harness-cannot-complete-a-sweep-on-this-machine). The ~18
  ms/token in the serving section above is an end-to-end HTTP figure and is not
  comparable to either those numbers or the H100's 1.77 ms/token.
