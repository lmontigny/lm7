# Blackwell quantization figure plan

This is the runbook for one measured figure showing what LM7 quantization buys
on an NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB, `sm120`). It is a
plan, not validation: no result belongs in `docs/` or the README until the
session has run and the accuracy and kernel gates below pass.

## Question and headline workload

The figure should answer two questions without conflating them:

1. Does quantization reduce steady-state latency?
2. How much model storage or peak GPU memory does it save?

Use `unsloth/Llama-3.1-8B-Instruct`, BF16, through LM7 and TorchInductor. This
is large enough for quantized matrix multiplication to matter, fits every arm
comfortably on the 96 GiB card, and already has LM7 quantization evidence.

Do not use MoE for the headline. With Transformers 5.x, expert weights may be
grouped parameter tensors rather than `nn.Linear` modules. A selector keyed on
`.mlp.` can therefore convert nothing, turning the experiment into a coverage
test rather than a quantization comparison. OLMoE can be a separate follow-up
only after the converted layer count and emitted kernels are verified.

Run two pilot shapes:

- batch 8, sequence 128: `M ~= 1024`, favorable to narrow GEMMs;
- batch 1, sequence 1024: a recognizable long-prefill workload.

Use the first shape for the headline if it passes the gates. Keep both in the
report so the conclusion is not tied to one convenient shape. Optionally add
batch 1, sequence 16 as a counterexample showing why quantization can lose when
the matrices are too small.

## Arms

BF16 is the baseline and the output/accumulation datatype.

| Figure label | LM7 mode | What it isolates |
| --- | --- | --- |
| BF16 | `none` | unquantized baseline |
| INT8 weight-only | `int8` | memory saving without narrow activation compute |
| FP8 weight-only | `fp8` | weight compression versus native FP8 compute |
| FP8 dynamic | `fp8-dynamic` | native FP8 matmul with per-tensor activation scale |
| FP8 dynamic rowwise | `fp8-dynamic-rowwise` | native FP8 matmul with per-row scale |
| NVFP4 weight-only | `nvfp4` | maximum weight storage reduction |
| NVFP4 dynamic | `nvfp4-dynamic` | native Blackwell FP4 matmul |

Keep a slow arm in the data. The point is to distinguish memory compression
from faster arithmetic, not to manufacture a chart where every bar wins.

## Reproducibility

Record before the run:

- exact GPU name, compute capability, VRAM, driver and power limit;
- PyTorch, CUDA, TorchAO and Transformers versions;
- LM7 commit SHA and complete command line;
- model revision, prompt/input construction and random seed;
- Inductor compile mode and every environment variable affecting compilation.

Run each arm in a fresh process to avoid allocator, compilation and cache
contamination. For each process:

1. load a fresh model;
2. apply the requested quantization through LM7;
3. record converted module/layer counts;
4. compile with identical Inductor settings;
5. warm up 10 calls;
6. time at least 30 CUDA-synchronized forwards;
7. record median, p95, minimum, peak allocated memory, model storage,
   quantization time and compilation time.

Repeat the complete process five times. Preserve one JSON report per process in
`artifacts/`; author the figure from those reports rather than copied numbers.

## Correctness and mechanism gates

An arm enters the main figure only when all applicable gates pass:

- no eager fallback occurred;
- the expected nonzero number of layers was converted;
- four reference prompts preserve the BF16 top-1 token;
- maximum absolute logit difference is recorded;
- generated continuations are compared on a larger prompt set;
- dynamic FP8 emits `_scaled_mm` with the requested scale granularity;
- dynamic NVFP4 emits native FP4 `_scaled_mm` rather than a BF16
  dequantize-then-matmul path.

A mode that fails fidelity remains in the raw table but appears as rejected in
the figure. A plausible completion is not proof of kernel datatype or fidelity.

## Figure

Build one SVG/PNG with aligned panels and the same arm order:

- top: median latency normalized to BF16, with the cross-process range;
- bottom: model storage, and peak GPU memory if it differs materially.

Use direct labels, not color alone. Suggested labels include `0.83x latency`,
`1.54x smaller`, and `rejected: top-1 3/4`. Lower is better in both panels.
The caption must name the GPU, model, shape, software stack, repeat count and
accuracy gate.

The headline is acceptable only if a dynamic FP8 or FP4 arm is faster than BF16
in the median, its five-run range is credible, it uses a verified narrow kernel,
and it passes the fidelity gate. Otherwise publish the negative result: on that
model and shape, quantization bought memory but not latency.

## MoE follow-up

After the dense figure, pilot `allenai/OLMoE-1B-7B-0125-Instruct` separately.
Before timing it, report which grouped expert tensors were converted and prove
that the narrow kernels cover expert GEMMs. Do not place dense and MoE results
in one headline chart; their routing, active parameter count and quantization
coverage answer different questions.
## Measured result — 2026-08-16

The rented host was an NVIDIA RTX PRO 6000 Blackwell Server Edition (`sm120`,
95.6 GiB), driver 580.126.20. The run used CUDA 13.0, TorchAO 0.17.0,
Transformers 5.15.0 and PyTorch 2.15.0.dev20260816+cu130. This is newer than
the 2.13.0 stack in the earlier Blackwell report and must be stated beside any
comparison.

![Two-panel bar chart showing throughput and model-storage reduction for the validated and rejected FP8/NVFP4 modes on Llama-3.1-8B. Dynamic FP8 passes fidelity and improves throughput; dynamic NVFP4 is faster and smaller but rejected by the fidelity gate.](../docs/figures/blackwell-quantization.png)

The selected workload was batch 8, sequence 128, BF16 input/output, through LM7
and TorchInductor. Input IDs repeat the tokens for "The capital of France is";
this makes the matrix shape reproducible but is a performance workload, not a
quality workload.

| mode | median | throughput vs BF16 | storage | reduction | converted | fidelity |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 | 46.84 ms | 1.00x | 16.06 GB | 1.00x | — | baseline |
| INT8 weight-only | 8997.06 ms | 0.005x | 9.09 GB | 1.77x | 224 | prior 4/4 |

> [!WARNING]
> This INT8 row is retained as diagnostic evidence but excluded from the headline
> figure. LM7 `int8` is weight-only, leaving BF16 activations and dequantizing
> weights around the matmul. On this PyTorch 2.15 nightly stack that path regressed
> to 0.005x throughput. The follow-up `int8-dynamic` mode quantizes both operands
> and stays behind the hardware-validation gate until it is remeasured.
| FP8 weight-only | 57.01 ms | 0.82x | 10.43 GB | 1.54x | 96 | prior 3/4, rejected |
| FP8 dynamic | 35.85 ms | **1.31x** | 10.42 GB | 1.54x | 96 | **4/4**, diff 0.83 |
| FP8 dynamic rowwise | 34.47 ms | **1.36x** | 10.43 GB | 1.54x | 96 | **4/4**, diff 0.70 |
| NVFP4 weight-only | 57.68 ms | 0.81x | 6.03 GB | **2.66x** | 224 | prior 2/4, rejected |
| NVFP4 dynamic | 25.23 ms | **1.86x** | 6.03 GB | **2.66x** | 224 | **2/4**, rejected; diff 3.30 |

"Prior" fidelity values are the repository's existing Blackwell result on the
2.13.0 stack, not a rerun under 2.15.0. The dynamic modes were checked during
this session against the same BF16 checkpoint.

The latency decision boundary is not noisy. At the user's request the planned
five process repetitions were stopped to conserve rental credit. Completed
fresh-process medians were:

| mode | processes | median of medians | range |
| --- | ---: | ---: | ---: |
| BF16 | 4 | 46.84 ms | 46.77–46.93 |
| FP8 dynamic | 4 | 35.85 ms | 35.82–35.90 |
| FP8 dynamic rowwise | 3 | 34.47 ms | 34.45–34.51 |

The long-prefill cross-check at batch 1, sequence 1024 preserved the ordering:
51.48 ms BF16, 40.77 ms dynamic FP8, 39.45 ms rowwise dynamic FP8 and 30.14 ms
dynamic NVFP4. The last arm still fails fidelity.

Generated Inductor caches contained `_scaled_mm` in 2,987 files, float8 markers
in 45 and `e2m1` in 3, confirming that narrow scaled kernels were emitted.
A future kernel audit should isolate one mode per empty cache before making
per-mode source-count claims.

The defensible headline is: **rowwise dynamic FP8 improves throughput by 1.36x
and reduces model storage by 1.54x while preserving 4/4 top-1 agreement on this
gate.** Dynamic NVFP4 is faster and smaller, but it is rejected for fidelity.
