# NVIDIA validation suite

A reusable way to ask "what does LM7 actually do on this NVIDIA GPU?" and get an
answer another machine can check, rather than a page of numbers from one card.

`benchmarks/nvidia_matrix.py` already ran one cell per process. This documents it
as a suite: an environment block every result set carries, named cell plans, and
a summary. The H100 numbers below are the first instantiation, not the point —
the point is that the same six commands produce the equivalent page for an A100,
an L40S, or a Blackwell part.

## Why the environment block exists

Every latency figure in this repo is only comparable against another taken on the
same silicon **and** the same wheel, and this project has already been caught by
the second half: `tensorrt` pins PyTorch 2.12 while everything else runs 2.13, so
rows from the two venvs compare for "does it work" and only roughly for speed.

```bash
python benchmarks/nvidia_matrix.py --environment
```

```json
{
  "gpu": "NVIDIA H100 80GB HBM3",
  "compute_capability": "sm90",
  "driver": "13.0",
  "cuda": "13.0",
  "pytorch": "2.13.0+cu130",
  "triton": "3.7.1",
  "torchao": "0.17.0",
  "supported_precisions": {
    "bf16": true, "fp16": true, "fp32": true,
    "fp4": false, "fp8": true, "int8": true
  },
  "cuda_build": {
    "arch_list": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
    "native_kernels": true,
    "architecture_specific": false
  }
}
```

The same block is written to `environment.json` in any results directory, so a
set of JSON files is still self-describing months later.

### It describes an AMD GPU too, which is the point of having it

ROCm reaches the GPU through the same `torch.cuda` API, so `--target amd` runs
this suite unchanged and its numbers are comparable to the NVIDIA ones above —
which matters, because [`benchmarks/moe.py` and this harness disagree by
2.3x](#do-not-compare-these-numbers-to-benchmarksmoepy) and picking the same one
is the whole reason the comparison holds.

What it used to produce on AMD was a block with `gpu` filled in and
`compute_capability`, `driver`, `cuda`, `supported_precisions` and `cuda_build`
all `null` — results with no record of what produced them, which is the one thing
this block exists to prevent. Three keys fix that:

- **`architecture`** is the vendor-neutral answer: `sm90` or `gfx942`.
  `compute_capability` stays NVIDIA-only rather than being generalized, because
  the `environment.json` files already sitting beside the H100 and Blackwell
  results use it with that meaning.
- **`hip`** is the runtime version. `torch.version.cuda` is `None` on ROCm, so
  recording only that left an AMD report claiming no runtime at all; `driver`
  now falls back to it.
- **`fp8_format`** is `fnuz` or `ocp`. CDNA 3 implements the `fnuz` FP8
  encoding and every NVIDIA card from `sm89` uses the OCP one, so
  `"fp8": true` on both sides of a comparison does not mean the two numbers were
  produced in the same format. See [AMD GPU
  support](amd-rocm.md#what-lm7-targets-says-about-the-card-and-how-much-to-trust-it).

Three paths do not exist off NVIDIA — `tensorrt`, `tensorrt-export` and
`onnxruntime` — and those cells now record `skipped` with a reason instead of
running and failing. The distinction is not cosmetic: a `works: false` cell with
a traceback reads as "this broke here", when the truth is the path was never
available on this vendor. The skip also returns before the model is built, so
learning it does not cost an 8B checkpoint download.

The same matrix has now run on one AMD GPU, a rented MI300X (`gfx942`). All 20
core cells passed, HIP graph capture worked, and the AMD row is recorded in
[AMD MI300X](amd-mi300x.md). Other AMD architectures remain unexercised; see
[limitations](limitations.md#hardware-validation).

### `supported_precisions` is the silicon; `cuda_build` is the install

These answer different questions and they disagree usefully.

**From compute capability 9.0 NVIDIA splits the compilation target in two.**
`sm_90` is portable and forward compatible; `sm_90a` carries the
architecture-specific instructions — Hopper's `wgmma` and TMA among them — and is
deliberately *not* forward compatible, so a kernel that needs them must be built
for the `a` variant explicitly.
([NVIDIA CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html))

The measured consequence: **`torch 2.13.0+cu130` ships no `a` variant for any
architecture.** An H100 on this wheel runs correctly, reports `fp8: native`, and
cannot reach Hopper's architecture-specific instructions at all, because nothing
in the build was compiled to use them. That is invisible from `lm7 targets`
alone, which is why `cuda_build` now sits beside the precision report.

The second finding is sharper, and it is about the card this project developed
on. **`sm_89` is not in that list.** The RTX 4070 SUPER — the primary dev GPU
behind most NVIDIA numbers in this repo — has no natively compiled kernels in the
stock wheel and runs by JIT-ing PTX from an older target:

```json
{"arch_list": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
 "native_kernels": false, "architecture_specific": false}
```

That is legal and correct — CUDA's forward compatibility is doing exactly its job
— but it is a real difference between the two machines, and nothing surfaced it
before. It is now in `lm7 doctor --json` and `lm7 targets --json` under
`capabilities` → `cuda_build`, for any NVIDIA target.

## Running it

Cells are one per process, deliberately: TensorRT and Inductor can abort the
interpreter and a large model can poison the CUDA context, so a crashed cell
should cost that cell rather than the sweep.

```bash
python benchmarks/nvidia_matrix.py --plan core            # cell list, one per line
python benchmarks/nvidia_matrix.py --model smollm2 --path inductor-cudagraphs \
    --results-dir artifacts/h100-suite
python benchmarks/nvidia_matrix.py --summarize artifacts/h100-suite
```

| plan | cells | environment |
| --- | --- | --- |
| `core` | 5 models × eager, inductor, cudagraphs, max-autotune | default CUDA venv |
| `artifacts` | AOTInductor export + reload per model | default CUDA venv |
| `quant` | 3 causal LMs × unquantized, `fp8`, `fp8-dynamic` | default CUDA venv |
| `large` | Llama-3.1-8B across the portable paths | default CUDA venv |
| `tensorrt` | 4 models × JIT and export | `.venv-trt` (torch 2.12) |
| `onnxruntime` | 4 models | `onnxruntime-gpu` venv |

Each cell records compile time, warm latency, throughput, peak VRAM, output
parity against eager on the same device, artifact reload where the path exports,
CUDA Graph status, and the converted-module count for the FP8 paths.

> [!NOTE]
> The FP8 paths are causal-LM only. Their layer selector matches on `.mlp.`
> module paths, so `mlp`, `resnet18` and `bert` have nothing to convert; the
> suite refuses those cells rather than silently measuring an unquantized model
> and labelling it FP8.

## First results, on an H100

Eight cells, BF16, on the environment block above. Seven ran; the eighth is the
BERT+FP8 refusal described in the note.

| model | path | median | peak VRAM | parity vs eager | CUDA Graphs |
| --- | --- | --- | --- | --- | --- |
| MLP | `inductor` | 0.142 ms | 0.10 GB | 0.00e+00 | not requested |
| MLP | `inductor-cudagraphs` | 0.258 ms | 0.05 GB | 0.00e+00 | **captured** |
| ResNet-18 | `inductor-cudagraphs` | 0.794 ms | 0.10 GB | 4.69e-02 | **captured** |
| SmolLM2-135M | `inductor` | 7.901 ms | 0.41 GB | 1.50e+00 | not requested |
| SmolLM2-135M | `inductor-cudagraphs` | **1.999 ms** | 0.30 GB | 1.50e+00 | **captured** |
| SmolLM2-135M | `inductor-fp8` | 7.580 ms | 0.28 GB | 5.94e-01 | not requested |
| SmolLM2-135M | `inductor-fp8-dynamic` | 6.780 ms | 0.28 GB | 2.25e+00 | not requested |

**CUDA Graphs are worth 3.95x on SmolLM2-135M** — 7.901 ms to 1.999 ms — which is
larger than any other single change measured on this card, and roughly double
what compiling itself buys. That is the launch-bound result from
[the H100 page](nvidia-h100.md#these-workloads-are-launch-bound-not-flop-bound)
arriving from the other direction: if the clock is measuring kernel launches,
the thing that removes kernel launches wins.

It also means **the headline figures on that page understate the card**, because
they use the default Inductor preset, which does not request CUDA Graphs.

**And it does not generalize downward.** The same preset makes the small MLP
*slower*, 0.142 → 0.258 ms, capture succeeding all the while. Below some amount
of work per call, graph replay overhead exceeds the launch overhead it removes —
the same shape as the MLP compile result already on the H100 page.

`cudagraphs_active` in each record is the honest field: `reduce-overhead`
*requests* capture and Inductor declines it for mutated inputs, dynamic shapes
and CPU scalars, bumping a skip counter rather than raising. All three cells here
captured with zero skips, so the 3.95x is graphs and not something else.

## Sparse MoE

The `moe` plan covers two hand-built two-layer configs, which cost no download,
and `allenai/OLMoE-1B-7B-0924-Instruct` (6.92B total, 1B active), the largest
sparse MoE that fits an 80 GB card. `qwen3-30b-a3b` and `mixtral-8x7b` are
reachable by name for a bigger one — Mixtral-8x7B peaks at 93.4 GB in BF16, so
it is deliberately outside the plan.

| model | `eager` | `inductor` | `+cudagraphs` | peak VRAM |
| --- | --- | --- | --- | --- |
| mixtral-tiny | 2.748 ms | 0.957 ms (2.87x) | **0.438 ms** (6.27x) | 0.03 GB |
| olmoe-tiny | 3.048 ms | 0.993 ms (3.07x) | **0.362 ms** (8.42x) | 0.03 GB |
| OLMoE-1B-7B | 18.847 ms | 4.861 ms (3.88x) | 4.290 ms (4.39x) | 13.87 GB |

**Sparse routing does not prevent CUDA Graph capture.** All three captured with
zero skips, including the 6.92B model. That is worth pinning rather than
assuming: the router is exactly the sort of data-dependent control flow capture
declines, and on transformers 4.57.3 `aten.nonzero` in that router broke Dynamo
[eight or nine times per model](limitations.md#what-torchcompile-actually-does-to-a-sparse-moe).
The `grouped_mm` rewrite in 5.x removed both problems at once.

**The CUDA Graph gain shrinks as work per launch grows**, and it does not track
parameter count. The tiny configs gain 6–8x, SmolLM2-135M gains 3.95x at 30
layers, and OLMoE-1B-7B gains 1.13x over plain `inductor` at 16 layers and 50x
the parameters. Graph replay removes launch overhead, so what predicts the win is
how many launches there are and how little each one does.

### FP8 on a sparse MoE is refused, correctly

Every FP8 cell fails, on both architectures and at both scales, with:

```text
fp8 matched no quantizable layers in this model, so quantization would silently
do nothing. It selects linears whose module path contains '.mlp.', and this
model has none. Use quantization='none'. Try int8, which selects every linear
except lm_head.
```

The cause is the same `grouped_mm` rewrite. On transformers 5.x the per-expert
`nn.Linear` modules are gone, replaced by the parameter tensors `grouped_mm`
consumes, so a two-layer MoE has **nine** linears — attention plus `lm_head` —
and none of them under `.mlp.`. There is nothing for the FP8 selector to match.

The plan keeps one such cell per tiny architecture, because a refusal that stays
a refusal is a regression test: silently converting nothing and reporting an FP8
latency would be far worse than failing.

> [!NOTE]
> The suggested `int8` fallback selects every linear except `lm_head`, which on
> an MoE is attention only — the experts hold most of the parameters and are not
> linears. Expect a much smaller footprint saving than on a dense model. That
> follows from the module structure above and has **not** been measured.

### Do not compare these numbers to `benchmarks/moe.py`

`moe.py` and this suite build their inputs differently, and the gap is large
enough to invert a conclusion. On the same H100, same model, same session:

| harness | eager | inductor | speedup | peak VRAM |
| --- | --- | --- | --- | --- |
| `nvidia_matrix.py` | 18.847 ms | 4.861 ms | **3.88x** | 13.9 GB |
| `moe.py` | 26.520 ms | 15.379 ms | **1.72x** | 27.7 GB |

Taking the suite's 3.88x against the published Blackwell 1.60x would have read as
Hopper crushing Blackwell on MoE. Measured through the *same* harness, Blackwell
is ahead — 17.88 ms eager and 11.20 ms compiled against Hopper's 26.520 and
15.379 — and the speedup ratios are close, 1.60x against 1.72x.

`moe.py` also reports `graphs=1, breaks=0` for the 6.92B model here, reproducing
on `sm90` the zero-graph-break result previously measured only on `sm120`.

## What this suite does not do

- **No multi-GPU.** One device, matching LM7's scope.
- **The `tensorrt` and `onnxruntime` plans need their own environments**, and
  nothing checks that you are in the right one — a wrong venv shows up as a cell
  that fails to import, which is a legible failure but not a helpful one.
- **Parity is against eager on the same device**, so it measures the backend and
  not the transfer or the dtype. A large `max_abs_diff` on a causal LM is
  expected at BF16 and is not by itself a defect; the number is there to be
  compared across cards, not against zero.
- **No decode loop.** Every cell is a forward pass.
