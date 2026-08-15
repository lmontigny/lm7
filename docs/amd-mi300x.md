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

## Quantization, and why none of it should change the gate yet

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

**The INT8 number is not safe to attribute to CDNA 3.** torchao 0.17.0 prints
`Skipping import of cpp extensions due to incompatible torch version` against
this container's torch 2.10 (it wants ≥ 2.11), so the dequantization runs
unfused in pure PyTorch. That confound is more than large enough to explain a
9.72x gap on its own, and it does not touch the FP8 modes the same way.
**No `_QUANTIZATION_VENDORS` entry should change on this evidence.** The run to
do first is the same sweep against torch ≥ 2.11 with the compiled path
available.

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

Qwen3-30B-A3B and Mixtral-8x7B were *not* run — they were queued behind the
tiers that answer questions, and the session ended first. They remain the two
entries this card could uniquely have measured.

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

**The vLLM ROCm handover still has not been run.** `--backend vllm --dry-run`
produces correct argv and reports `runtime_installed: false`; vLLM is not in this
image, and installing it was not worth the metered time. The argv translation is
therefore exercised and the handover itself is not — say "implemented", not
"validated", exactly as before.

## What this page does not say

- **One card, one session, one partition.** SPX, so a single logical device;
  nothing here says anything about CPX or about multi-GPU.
- **This is an MI300X `VF`** — an SR-IOV virtual function. It reports the full
  191.7 GiB and a single SPX partition, so it behaves as the whole card, but it
  is not a bare-metal MI300X and no comparison against one was possible.
- **torch 2.10, not the 2.13 the NVIDIA pages use.** Rows compare against the
  H100 and Blackwell pages for "does it work" and only roughly for speed. The
  quantization confound above is a direct consequence.
- **No MIGraphX.** The native bindings import on the host and not inside the
  ROCm PyTorch container, and chasing that was not worth the metered time. The
  [evaluation plan](amd-migraphx.md) is unchanged.
- **No CI, and there will not be.** GitHub's hosted runners provide no AMD GPU,
  and the GPU-hosted runners that exist are gated to Organizations on
  Team/Enterprise Cloud.
- **Serving is one model at 135M, single stream.** No concurrency, no larger
  model, no quantized serve — which is the configuration that was silently wrong
  on NVIDIA. The vLLM ROCm handover is still unrun.
- **No decode benchmark.** `benchmarks/decode.py` was not run, so the
  memory-bound half of generation is unmeasured here; the ~18 ms/token above is
  an end-to-end HTTP figure and not comparable to the 1.77 ms/token
  `compile_generation` reaches on an H100.
