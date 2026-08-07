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
