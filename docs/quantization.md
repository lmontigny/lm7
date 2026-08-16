# Quantization

Quantization stores or computes a tensor in fewer bits than the model was trained
in. There are two halves to it, and **LM7 implements both** — the second only
recently, and only on NVIDIA.

## Weight versus activation quantization

**Weight quantization** shrinks the stored parameters. Weights are converted
once, up front, and dequantized on the fly inside the matmul, so the arithmetic
still happens in a higher precision. What you gain is memory footprint and memory
bandwidth; what you avoid is calibration, because a weight tensor's range is
known statically.

**Activation quantization** also narrows the tensors flowing between layers, so
the matmul itself executes in low precision. That is where the larger speedups
come from, because it cuts arithmetic work rather than just bytes moved. The cost
is that activation ranges depend on the input, so the scales have to be chosen at
runtime or from calibration data, and accuracy degrades more readily.

The distinction is not cosmetic, and it is the reason every weight-only mode in
this document is *slower* than its BF16 baseline while the activation modes are
the only ones that have ever been faster. A weight-only kernel still issues a
BF16 matmul; it just reads narrower bytes on the way in. Only an activation mode
asks the tensor cores to multiply in the narrow format. This was
[verified at kernel level](#verifying-that-the-narrow-kernel-actually-ran) rather
than assumed.

## Supported data types

Bare names are weight-only. The `-dynamic` names quantize activations too.

| `--quantize` | Weight storage | Activations | Targets | Compute dtype |
| --- | --- | --- | --- | --- |
| `none` (default) | as loaded | — | all | FP32 / FP16 / BF16 |
| `int8` | INT8 | BF16 | NVIDIA Ampere (`sm80`) or newer, **CPU** | BF16 on NVIDIA, FP32 on CPU |
| `int8-dynamic` | INT8 | **INT8, quantized per call** | NVIDIA Ampere (`sm80`) or newer | BF16 accumulate |
| `fp8` | FP8 | BF16 | NVIDIA Ada (`sm89`), Hopper (`sm90`), or newer | BF16 |
| `nvfp4` | NVFP4 — 4-bit, one FP8 scale per 16 values | BF16 | NVIDIA Ampere (`sm80`) or newer | BF16 |
| `fp8-dynamic` | FP8 | **FP8, quantized per call**, one scale per tensor | NVIDIA Ada (`sm89`) or newer | BF16 accumulate |
| `fp8-dynamic-rowwise` | FP8 | **FP8, quantized per call**, one scale per row | NVIDIA Ada (`sm89`) or newer | BF16 accumulate |
| `nvfp4-dynamic` | NVFP4 | **NVFP4, quantized per call** | NVIDIA **Blackwell** (`sm100`, `sm120`) | BF16 accumulate |

`int8-dynamic` fixes the operand mismatch in the weight-only path: it uses
TorchAO's `Int8DynamicActivationInt8WeightConfig(version=2)` so the compiled
matmul receives INT8 activations and INT8 weights. It remains outside the
validated-model allowlist until its fidelity and generated kernels are measured
on real hardware.

The two FP8 dynamic rows differ only in scale granularity, and the difference is
worth a mode of its own because it is not visible from the call site: TorchAO's
`Float8DynamicActivationFloat8WeightConfig` resolves an omitted `granularity` to
per-tensor rather than raising, so `fp8-dynamic` has always been the per-tensor
one. `fp8-dynamic-rowwise` asks for `PerRow` — a scale per weight output row and
per activation token. On `sm90` it is faster than per-tensor at every shape
measured; see [the H100 numbers](#fp8-granularity-on-h100).

Note that weight-only `fp8` already stores a per-row *weight* scale. What the
rowwise mode changes is the **activation** scale, which is where the per-tensor
default costs range.

The two NVFP4 rows have different hardware floors, which looks inconsistent and
is not. Weight-only NVFP4 never issues an FP4 matmul, so it runs anywhere BF16 is
native; `nvfp4-dynamic` asks the tensor cores to multiply in FP4, which exists
only on Blackwell.

> [!NOTE]
> `nvfp4` was **not** redefined to mean the activation mode. An existing
> `--quantize nvfp4` command does exactly what it did before. `nvfp4-weight-only`
> is accepted as the explicit spelling, matching `int8-weight-only` and
> `fp8-weight-only`.

Quantization pins the compute dtype so measurements stay comparable, and the
dtype is target-specific: BF16 on NVIDIA, FP32 on CPU. `--dtype` must be `auto`
or that target's dtype, and `auto` resolves to it. For the TorchAO path
`--backend` must be `auto` or `inductor`. Anything else raises
`UnsupportedModelError` rather than silently degrading — including a mode on a
target it was not measured on, an FP8 request on pre-Ada hardware, and any
(model, mode) pair outside the validated list.

**`--backend openvino --quantize int8` is a second, faster route on Intel CPU**,
and it is not the TorchAO path — NNCF compresses the OpenVINO IR instead of
converting torch modules. It accepts `int8` only, gates on
`VALIDATED_OPENVINO_INT8`, and reports its saving as *compiled weights* because
the torch module is left untouched. See
[the comparison below](#int8-on-cpu-has-two-mechanisms-and-they-differ-by-4x).

CPU uses FP32 rather than BF16 because x86-64 without AVX-512 has no native BF16
path, so forcing BF16 there would measure emulation rather than the format.

## Why weight-only quantization needs Ampere or newer

The same emulation argument rules out pre-Ampere NVIDIA, which is why every mode
above requires at least `sm80`. Measured on a Tesla T4 (Turing, `sm75`) with
SmolLM2-135M at sequence length 16, against an unquantized baseline in the same
compute dtype:

| configuration | prefill | top-1 | note |
| --- | --- | --- | --- |
| unquantized FP16 | **17.5 ms** | 4/4 | the baseline to beat |
| unquantized BF16 | 60.3 ms | 4/4 | 3.4x slower — BF16 is emulated on `sm75` |
| INT8 + BF16 | 58.0 ms | **3/4** | 3.3x slower than not quantizing |
| INT8 + FP16 | 13.5 ms | **0/4** | **NaN logits** |

Neither compute dtype works. BF16 is the numerically sound choice but Turing fakes
it, so INT8 ends up 3.3x *slower* than simply running FP16 — and it drops a top-1
token on a model that scores 4/4 on `sm89`. FP16 is genuinely fast but produces NaN:
dequantized INT8 products leave FP16's 5-bit exponent range, and every prompt then
returns `<|endoftext|>`. The 13.5 ms is the cost of propagating NaN.

So LM7 raises `UnsupportedModelError` on `sm75` and below rather than offering a
mode whose best case is a regression. `torch.cuda.is_bf16_supported()` cannot be
used for this check — it returns True on a T4 — so the gate reads the capability
number, as the FP8 gate does.

Also verified on that hardware: FP8 is correctly refused on `sm75`, which until now
had only ever been exercised against a synthetic `TargetSpec`.

> [!NOTE]
> `lm7 model export --quantize int8` is a **different mechanism**: calibrated
> XNNPACK post-training quantization inside the ExecuTorch backend, which
> quantizes activations too. Everything on this page is the TorchAO weight-only
> path used by `lm7 model run`. See the [ExecuTorch guide](executorch.md).

`--quantize` replaced `--quantization`, and the long-form values it took
(`int8-weight-only`, `fp8-weight-only`) are still accepted and normalized onto
the short names.

## Which layers are converted

This differs between modes, and it changes the memory saving you should expect:

- `int8` converts **every `nn.Linear` except `lm_head`** — attention projections
  included.
- `fp8` converts **only the MLP linears** (`.mlp.` in the module path), leaving
  attention and `lm_head` in BF16.
- `nvfp4` converts **every `nn.Linear` except `lm_head` whose last two
  dimensions are both multiples of 16**. NVFP4 stores one scale per 16-element
  block, so TorchAO raises on any other shape; LM7 filters those layers out
  rather than failing the whole model. The run reports how many layers were
  actually converted.

`lm_head` is left alone in every mode because it is both large and
accuracy-sensitive: it maps hidden states to the full vocabulary, so error there
lands directly on the token distribution.

## TorchAO

The conversion itself is [TorchAO](https://github.com/pytorch/ao)'s, not LM7's.
LM7 pins `torchao==0.17.0` and calls `torchao.quantization.quantize_()` with
`Int8WeightOnlyConfig`, `Float8WeightOnlyConfig`, or `NVFP4WeightOnlyConfig`,
passing a module filter that selects the layers above.

> [!NOTE]
> `NVFP4WeightOnlyConfig` lives in `torchao.prototype.mx_formats`. Prototype
> carries no API stability promise, and the symbol has moved between releases,
> which is why LM7 pins TorchAO exactly and reports a pin-specific error rather
> than letting an `ImportError` escape.

The quantized model is then compiled and executed through LM7's normal
`inductor` path, so quantization composes with the rest of the stack instead of
being a separate execution route. That is also why `--backend` is restricted:
TensorRT and OpenXLA have their own quantization stories that LM7 has not
integrated.

```bash
uv pip install -e ".[hf,torchao]"
lm7 model run hf://unsloth/Llama-3.2-1B-Instruct \
  --target nvidia --backend inductor --dtype bfloat16 \
  --quantize nvfp4
```

The run reports model storage bytes, converted layer count, quantization time,
first-call and steady-state latency, and peak GPU memory, so the
footprint-versus-latency trade is measured rather than assumed. Add `--json` for
structured output.

## NVFP4 across two GPU generations

NVFP4 is NVIDIA's 4-bit block-scaled float format. Hardware FP4 matmul exists
only on Blackwell (`sm100`, `sm120`), so the question that matters is whether
that hardware changes the trade. It has now been measured on both sides:

| mode | Ada `sm89` eager | Ada `sm89` compiled | Blackwell `sm120` compiled |
| --- | --- | --- | --- |
| `bf16` baseline | 27.66 ms | **8.92 ms** | **3.11 ms** |
| `int8` | 40.62 ms | 65.13 ms (7.30x) | 18.77 ms (6.04x) |
| `fp8` | 47.21 ms | 14.61 ms (1.64x) | 3.64 ms (1.17x) |
| `nvfp4` | 260.98 ms | **22.27 ms** (2.50x) | **3.84 ms** (1.24x) |

Llama-3.2-1B, single prompt, BF16, `inductor`. Ada is an RTX 4070 SUPER, median
of 10 after warmup. Blackwell is an RTX PRO 6000 Blackwell Server Edition,
median of 30 after 5 warmup calls, through
[`benchmarks/quantization.py`](../benchmarks/quantization.py). Parenthesised
ratios are against each machine's own BF16 baseline.

The Blackwell column reproduces with:

```bash
python benchmarks/quantization.py --model llama32-1b --dtype bfloat16 \
  --output artifacts/quantization.json
```

That harness sweeps every mode against one baseline in a single process and
reports footprint, accuracy, and latency together. It measures accuracy eagerly
and latency compiled, on the reasoning that weight-only error belongs to the
weights while the latency anyone cares about belongs to the compiled path.
Before it existed these numbers were collected one `lm7 model run` at a time,
which is why the Ada column cannot be reproduced prompt-for-prompt.

The Blackwell stack was `torch 2.13.0+cu130`, `torchao 0.17.0+cu130`,
`transformers 5.14.1`, driver 580.126.20. PyTorch ships `sm_120` kernels, so no
source build was needed; LM7 resolved `nvidia:sm120` and selected `inductor`
without any change to the capability handling in `src/lm7/huggingface.py`, which
compares `smXX` numerically and so already ordered Blackwell above Ada. The
transformers version differs from whatever the Ada run used, which is one more
reason to treat cross-column comparisons as indicative rather than controlled.

**FP4 tensor cores do not rescue weight-only NVFP4, and cannot.** Weight-only
quantization never issues an FP4 matmul — the weight is unpacked to BF16 inside
the kernel, so the FP4 units are never asked to multiply anything. The
measurement agrees with the mechanism: were they engaged, `nvfp4` would beat the
BF16 baseline instead of trailing it by 1.24x. Reaching those units requires FP4
*activations* as well.

That is now implemented, as `nvfp4-dynamic`, and inspecting the emitted kernels
confirms both halves of this paragraph: weight-only NVFP4 emits a BF16
`extern_kernels.mm` with dequantization around it, and the dynamic mode emits
`_scaled_mm` with `e2m1` operands and no plain `mm` at all. See
[activation quantization](#activation-quantization).

**What Blackwell changes is how much the mode costs.** NVFP4 went from 2.50x
slower than its own BF16 baseline to 1.24x, and gained 5.79x in absolute terms
where the baseline itself gained only 2.87x — the largest generational
improvement of any mode measured. The plausible cause is memory bandwidth and
better fused-unpack codegen rather than arithmetic, since the unpack is the part
of the kernel that 4-bit storage actually touches. That attribution is a
hypothesis, not a measurement.

The practical consequence is that the same footprint saving costs far less. On
Ada, 2.30x smaller cost 150% more latency. On Blackwell, 2.30x smaller costs
24%. Blackwell is the first configuration where weight-only NVFP4 is a
reasonable default rather than a footprint-only last resort — accuracy
permitting, which is a separate question answered below.

Two things did not improve. Eager execution stays pathological, because the
unpack is not fused and every matmul pays it in full, which is why `--backend`
is restricted to `inductor` — NVFP4 without it is unusable. And INT8 remains the
worst latency of any mode on both generations by a wide margin.

Compilation is not free either. On Ada the first `nvfp4` call took **72 seconds**
against a 20 ms steady-state call. On Blackwell it ranged from **31 to 60
seconds** across runs against a 3.8 ms steady-state call, so the ratio got worse
even as both numbers improved. The mode still only makes sense where the process
is long-lived or the artifact is reused.

## Activation quantization

The dynamic modes quantize activations at runtime alongside the weights, so the
matmul executes in the narrow format. They need `sm89` (FP8) or `sm100` (NVFP4),
and they are the only modes here that have ever beaten their BF16 baseline.

```bash
lm7 model run hf://unsloth/Llama-3.2-1B-Instruct \
  --target nvidia --backend inductor --dtype bfloat16 \
  --quantize fp8-dynamic
```

### On plain linears, where the shape is visible

One `nn.Linear` reached through a transformer-shaped module path, BF16 in and
out, `torch.compile(fullgraph=True)`, median of 30 after 5 warmup calls on an
RTX PRO 6000 Blackwell (`sm120`), through
[`benchmarks/activation_quant.py`](../benchmarks/activation_quant.py). Every cell
is a ratio against that shape's BF16 baseline:

| M × K × N | `int8` | `fp8` | `nvfp4` | `fp8-dynamic` | `nvfp4-dynamic` |
| --- | --- | --- | --- | --- | --- |
| 128 × 4096 × 4096 | 53.3x | 1.18x | 1.37x | 1.72x | 2.55x |
| 256 × 4096 × 4096 | 81.3x | 1.20x | 1.33x | 1.03x | 1.50x |
| 1024 × 4096 × 4096 | 159.2x | 1.10x | 1.16x | **0.83x** | **0.85x** |
| 128 × 8192 × 8192 | 109.3x | 2.24x | 2.08x | **0.86x** | 1.08x |

Relative error against the BF16 output: `int8` 0.5–0.7%, `fp8` 2.4–2.9%,
`fp8-dynamic` 3.4–4.0%, `nvfp4` 9.7–10.3%, `nvfp4-dynamic` 12.4–14.8%.

**Every sub-1.00x cell in that table is an activation mode.** No weight-only mode
beats the baseline at any shape measured, which is what the mechanism predicts:
dequantizing narrow weights into a BF16 matmul cannot reduce arithmetic, only
bytes moved.

**The win is shape-dependent and arrives late.** Both dynamic modes lose at
M=128 and only cross over around M=1024. Below that there is not enough
arithmetic to pay for quantizing the activations on every call.

**INT8 weight-only is pathological here** — 53x to 159x slower than BF16, far
worse than the 6.2x the same mode costs on a whole model. Something about the
Blackwell INT8 dequantization path is badly served at these shapes. Recorded
rather than explained; it has not been investigated.

### On a real model, where the shape is small

Llama-3.2-1B, BF16, single 5-token prompt, `sm120`:

| mode | steady | vs baseline | storage | top-1 | max logit diff |
| --- | --- | --- | --- | --- | --- |
| `none` | 3.068 ms | — | 2.472 GB | — | — |
| `int8` | 18.964 ms | 6.18x | 1.65x smaller | 4/4 | 0.65 |
| `fp8` | 3.568 ms | 1.16x | 1.48x smaller | 4/4 | 0.89 |
| `nvfp4` | 3.885 ms | 1.27x | 2.30x smaller | 3/4 | 4.62 |
| **`fp8-dynamic`** | **2.978 ms** | **0.97x** | 1.48x smaller | **4/4** | 1.59 |
| `nvfp4-dynamic` | 4.528 ms | 1.48x | 2.30x smaller | 3/4 | 5.03 |

`fp8-dynamic` is the first mode in this document to come out faster than its BF16
baseline on a real model, and it keeps its top-1 token on all four prompts. It is
admitted to the gate for this model.

`nvfp4-dynamic` is *slower* here than weight-only NVFP4, and the linear table
above says why: a 5-token prompt at batch 1 is M=5, an order of magnitude below
where the FP4 path starts winning. It also drops a top-1 token, so it is **not**
admitted for this model. Both facts are about this shape and this model, not
about the mode.

### Verifying that the narrow kernel actually ran

A plausible output does not prove the tensor cores multiplied in FP8 or FP4.
Inspecting the code Inductor emits for one 1024 × 4096 × 4096 linear:

| mode | weight type | packed dtype | emitted |
| --- | --- | --- | --- |
| `none` | `Parameter` | bfloat16 | `extern_kernels.mm` |
| `fp8` | `Float8Tensor` | `float8_e4m3fn` | `extern_kernels.mm` + dequant |
| `nvfp4` | `NVFP4Tensor` | `uint8`, 4096 × 2048 | `extern_kernels.mm` + dequant |
| **`fp8-dynamic`** | `Float8Tensor` | `float8_e4m3fn` | **`_scaled_mm`, no `mm`** |
| **`nvfp4-dynamic`** | `NVFP4Tensor` | `uint8`, 4096 × 2048 | **`_scaled_mm` + `e2m1`, no `mm`** |

The weight-only rows keep a BF16 `extern_kernels.mm` and add dequantization
around it. The dynamic rows emit `_scaled_mm` — the scaled narrow-format GEMM —
and no plain `mm` at all, with `e2m1` (the FP4 element format) appearing in the
NVFP4 case. That is the difference between reading narrow bytes and computing in
narrow arithmetic, and it is visible in generated code rather than inferred from
a timing.

Both NVFP4 rows show the weight packed as `uint8` at half the column count, which
is the two-values-per-byte packing.

#### The same check on CDNA 3

`benchmarks/fp8_kernel_check.py` on an MI300X (`gfx942`), same 1024 × 4096 × 4096
linear, identical under `torch 2.10.0+rocm7.2` and `2.13.0+rocm7.2`:

| mode | weight scale | activation | emitted |
| --- | --- | --- | --- |
| `none` | — | none | `mm` → BF16 GEMM |
| `fp8` | `[4096, 1]` per-row | none | `mm` + dequant → BF16 GEMM |
| **`fp8-dynamic`** | `[1, 1]` per-tensor | per-tensor | **`_scaled_mm`, no `mm`** |
| **`fp8-dynamic-rowwise`** | `[4096, 1]` per-row | per-row | **`_scaled_mm`, no `mm`** |

Structurally identical to the NVIDIA table above: weight-only keeps a BF16 GEMM
and adds dequantization, and both dynamic modes emit the scaled narrow GEMM with
no plain `mm`. **CDNA 3 computes in FP8 rather than reading FP8 bytes into a BF16
multiply**, and that is what admitted the three FP8 modes on AMD rather than the
API returning without raising.

It is worth being precise about what this does *not* say. The FP8 there is the
`fnuz` encoding rather than the OCP `e4m3` `sm89`+ uses — see [AMD
MI300X](amd-mi300x.md) — so the two `_scaled_mm` rows are the same *shape* of
result and not the same arithmetic.

> [!NOTE]
> **The fused Triton activation-scaling kernel is not available here.** TorchAO's
> `use_triton_kernel=True` requires MSLK, and `pip install mslk` resolves to an
> empty 0.0.0 placeholder on PyPI that installs nothing importable — the real
> project is a source build from github.com/pytorch/MSLK. Asking for the Triton
> kernel without it raises `mslk is required for NVFP4 triton quantization` on
> the first call, so LM7 checks and requests the path that runs.
> `lm7.huggingface.nvfp4_dynamic_kernel()` reports `triton-mslk` or
> `torch-fallback`. Every measurement above is `torch-fallback`, and as the kernel
> table shows, that fallback still issues a native FP4 GEMM — what is missing is
> the *fused* activation scaling, not FP4 arithmetic.

## FP8 granularity on H100

Everything above is `sm120`. This section is `sm90` — a single **NVIDIA H100 80GB
HBM3**, driver 580.173.02, `torch 2.13.0+cu130`, `torchao 0.17.0` — and it exists
because H100 is the card FP8 is usually sold on, and because the per-tensor and
per-row modes had never been compared anywhere.

### Per-row wins at every shape, and neither beats BF16

[`benchmarks/activation_quant.py`](../benchmarks/activation_quant.py), same four
shapes and method as the `sm120` table above. Ratios are against that shape's BF16 baseline, so **lower is
better and below 1.00x means faster than not quantizing**:

| M × K × N | `fp8` | `fp8-dynamic` | `fp8-dynamic-rowwise` |
| --- | --- | --- | --- |
| 128 × 4096 × 4096 | 1.26x | 1.51x | **1.27x** |
| 256 × 4096 × 4096 | 1.19x | 1.52x | **1.32x** |
| 1024 × 4096 × 4096 | 1.15x | 1.41x | **1.04x** |
| 128 × 8192 × 8192 | 1.78x | 1.15x | **1.14x** |

**Per-row is faster than per-tensor at all four shapes**, and the gap is largest
where the sm120 table said the dynamic path should be winning: at M=1024,
per-tensor costs 1.41x and per-row 1.04x. Passing `granularity=PerRow()` is the
entire difference between those two columns.

**But no mode beats the BF16 baseline at these shapes.** The best cell in the
table is 1.04x — a 4% loss. On `sm120` the same benchmark had `fp8-dynamic` at
0.83x and `nvfp4-dynamic` at 0.85x at this shape, both genuine wins. H100's BF16
tensor cores are fast enough at these sizes that the cost of quantizing
activations on every call is not repaid.

That is a statement about these four shapes, and it does **not** survive contact
with a real model: on Llama-3.2-1B below, `fp8-dynamic-rowwise` comes out at
0.94x, faster than not quantizing. An isolated linear is not a transformer, and
this benchmark disagreeing with the model-level one is a reason to trust the
model-level one.

Relative error against the BF16 output: `fp8` 2.30–2.79%, `fp8-dynamic`
3.16–4.12%, `fp8-dynamic-rowwise` 3.30–3.82%. Per-row is slightly *more* accurate
than per-tensor at three of four shapes, which is the expected direction — a
scale per row fits the data better than one scale for the whole tensor.

### The scale shape is what proves the granularity

[`benchmarks/fp8_kernel_check.py`](../benchmarks/fp8_kernel_check.py),
1024 × 4096 × 4096, reading the quantized weight and Inductor's generated code:

| mode | weight scale | activation scale | emitted |
| --- | --- | --- | --- |
| `none` | — | — | `mm` |
| `fp8` | `(4096, 1)` | — | `mm` + dequant |
| **`fp8-dynamic`** | **`(1, 1)`** | per-tensor | **`_scaled_mm`, no `mm`** |
| **`fp8-dynamic-rowwise`** | **`(4096, 1)`** | per-row | **`_scaled_mm`, no `mm`** |

Both dynamic modes compute in FP8 on `sm90` — a scaled GEMM and no plain `mm`.
What separates them is the scale tensor: `(1, 1)` against `(4096, 1)`.

This check is the reason the mode exists as a mode. TorchAO does not raise when
`granularity` is omitted, so "I called the FP8 dynamic config" and "it scaled per
row" are independent claims, and only the scale shape settles the second.

Note the `fp8` row: weight-only FP8 already carries a per-row **weight** scale.
The rowwise mode's contribution is the per-row **activation** scale, which is the
one the per-tensor default was flattening.

### On real models, including the first 8B activation-mode measurement

[`benchmarks/quantization.py`](../benchmarks/quantization.py), BF16, single
5-token prompt, `sm90`. Accuracy is eager over four prompts; latency is compiled.

**Llama-3.2-1B:**

| mode | steady | vs baseline | storage | top-1 | max logit diff |
| --- | --- | --- | --- | --- | --- |
| `none` | 4.321 ms | — | 2.472 GB | — | — |
| `int8` | 21.350 ms | 4.94x | 1.65x smaller | 4/4 | 0.72 |
| `fp8` | 4.206 ms | 0.97x | 1.48x smaller | 4/4 | 0.92 |
| `nvfp4` | 7.091 ms | 1.64x | 2.30x smaller | 3/4 | 4.59 |
| `fp8-dynamic` | 4.417 ms | 1.02x | 1.48x smaller | 4/4 | 1.33 |
| **`fp8-dynamic-rowwise`** | **4.076 ms** | **0.94x** | 1.48x smaller | **4/4** | 1.09 |
| `nvfp4-dynamic` | — | — | — | — | rejected: needs `sm100` |

**Llama-3.1-8B** — no activation mode had ever been measured on this model:

| mode | steady | vs baseline | storage | top-1 | max logit diff |
| --- | --- | --- | --- | --- | --- |
| `none` | 7.677 ms | — | 16.061 GB | — | — |
| `fp8` | 14.524 ms | 1.89x | 1.54x smaller | **3/4** | 0.56 |
| `fp8-dynamic` | 8.686 ms | 1.13x | 1.54x smaller | 4/4 | 0.81 |
| **`fp8-dynamic-rowwise`** | **8.293 ms** | **1.08x** | 1.54x smaller | **4/4** | 0.78 |

**`fp8-dynamic-rowwise` is the fastest mode on the 1B, and the only one that
beats not quantizing** — 0.94x against `fp8-dynamic`'s 1.02x. Per-row is faster
*and* closer to the baseline than per-tensor on both models, which is the
direction the mechanism predicts.

**On the 8B, the dynamic modes are more accurate than the weight-only one.**
Weight-only `fp8` drops a top-1 token at 3/4 while both dynamic modes hold 4/4.
That reproduces on `sm90` the rejection [recorded on `sm120`](#llama-31-8b-finally-on-a-gpu),
and it is the opposite of the intuition that quantizing *more* things costs more
accuracy: a per-call activation scale adapts to the data, where a weight-only
mode has to survive whatever activations arrive.

**Neither dynamic mode beats BF16 on the 8B**, at 1.08x and 1.13x. So the
win is not simply "bigger model, better FP8" — the 1B wins and the 8B does not,
at this sequence length.

All four (model, mode) pairs clear the 4/4 bar and are admitted to
`VALIDATED_ACTIVATION`, which is what makes `--quantize fp8-dynamic-rowwise`
usable on them. Three of the four are admitted on **accuracy while costing
latency**; only rowwise-on-1B is faster than not quantizing at all.

> [!NOTE]
> TorchAO quotes roughly 1.46x prefill and 1.21x decode throughput for
> Llama-3.1-8B on H100. Nothing here reproduces that, and nothing here contradicts
> it: these are 5-token prefills at batch 1, which is the shape regime where
> [the H100 measures launch overhead rather than the card](nvidia-h100.md#these-workloads-are-launch-bound-not-flop-bound).
> A throughput claim at serving batch and sequence length is a different
> measurement that this repo has not made.

## INT8 on CPU

INT8 is the only mode measured off NVIDIA. It converts the same layers, keeps
the same 4/4 top-1 agreement, and gives a slightly *better* footprint ratio than
on GPU, because the CPU baseline is FP32 rather than BF16:

| model | top-1 | max logit diff | storage | FP32 | INT8 |
| --- | --- | --- | --- | --- | --- |
| SmolLM2-135M | 4/4 | 1.36 | 513 → 210 MiB (2.44x) | 49.9 ms | 75.8 ms (**1.52x**) |
| Llama-3.2-1B | 4/4 | 0.76 | 4943 → 2026 MB (2.44x) | 411.2 ms | 928.5 ms (**2.26x**) |
| DeepSeek-Coder-1.3B | 4/4 | 0.67 | 5386 → 1746 MB (3.09x) | — | — |
| Llama-3.1-8B | 4/4 | **0.50** | 32121 → 11190 MB (2.87x) | 2237.1 ms | 5957.5 ms (**2.66x**) |

Intel i7-8086K, 6 threads, sequence length 16, compiled through `inductor`,
median of 20 after warmup, one configuration per process. The DeepSeek row was
measured on a different host for accuracy and footprint only; its latency is left
blank rather than filled with a number from another machine. Its
better-than-2.44x ratio comes from a wider model with proportionally less weight
sitting in the untouched `lm_head`.

**The Llama-3.1-8B row is from a different host** — an AMD EPYC 7B13 (Zen 3, 8
cores, AVX2), where a 30 GiB FP32 baseline fits in RAM. Its FP32 and INT8 legs
were measured in the same process on that machine, so the 2.66x ratio is
internally valid; the absolute milliseconds are not comparable with the rows
above.

On the shared `"The capital of France is"` prompt this model's greedy next token
is `" a"`, not the `" Paris"` the smaller models produce. That is the instruct
checkpoint continuing raw text without its chat template, and it is not a
quantization artifact: FP32 and INT8 agree on it, which is all this check asks.

### Size makes INT8 more accurate and no faster

The 8B row was run to test an obvious hypothesis: that weight-only INT8 should
pay off at scale, where a decode step is more bandwidth-bound and the
dequantization overhead is amortized over more arithmetic. **It does not.**

| model | parameters | max logit diff | INT8 latency |
| --- | --- | --- | --- |
| SmolLM2-135M | 0.135B | 1.36 | 1.52x slower |
| Llama-3.2-1B | 1.24B | 0.76 | 2.26x slower |
| Llama-3.1-8B | 8.03B | 0.50 | 2.66x slower |

Across a 60x range of model size the regression gets *worse*, not better, though
the growth is flattening. It never reverses. The explanation in the VNNI section
below is size-independent — an FP32 GEMM against a dequantize-then-FP32-GEMM is
strictly more work at any scale — so there is no model size at which this path
becomes a speedup on CPU.

Accuracy moves the other way, and cleanly: the maximum logit difference falls
monotonically as the model grows, and 8B is the tightest INT8 result measured
anywhere in this document, on either target. That matches the usual account of
weight redundancy rising with scale, and it is the part of the "quantization
works better on big models" intuition that survives contact with these
measurements.

The practical read: on CPU, INT8 weight-only buys footprint — 32.1 GB down to
11.2 GB, which is the difference between a model fitting in RAM and not — at a
cost of roughly 2.7x latency. That is a good trade when the alternative is not
running the model at all, and a bad one otherwise.

> [!NOTE]
> An earlier revision of this table reported SmolLM2-135M INT8 at `~50 ms`, i.e.
> parity. Re-measured with the numbers above it is **1.52x slower**. The FP32
> column reproduced exactly (49.9 ms against the previous ~49 ms), so the
> methodology held and the INT8 figure was the stale one — most likely a
> different torchao version or sequence length. INT8 is not at parity at 135M
> either; it is cheaper there, not free.

### VNNI does not rescue it, which rules out the obvious explanation

The previous revision of this page attributed the 1B regression to missing
hardware: this CPU is AVX2-only, so "INT8 GEMM needs VNNI's `vpdpbusd` to beat an
AVX2 FP32 path", and a Cascade Lake or Sapphire Rapids part "would likely land
differently". That was a guess, and it was wrong. Re-running the same script on a
Cascade Lake Xeon with `avx512f` **and** `avx512_vnni`:

| model | ISA | FP32 | INT8 | ratio (median) | ratio (min) |
| --- | --- | --- | --- | --- | --- |
| SmolLM2-135M | AVX2 only | 49.9 ms | 75.8 ms | 1.52x slower | 1.38x slower |
| SmolLM2-135M | **AVX-512 + VNNI** | 133.8 ms | 76.3 ms | 0.57x *faster* | 1.11x slower |
| Llama-3.2-1B | AVX2 only | 411.2 ms | 928.5 ms | 2.26x slower | 2.69x slower |
| Llama-3.2-1B | **AVX-512 + VNNI** | 623.7 ms | 1212.1 ms | 1.94x slower | 3.44x slower |

The 1B regression survives VNNI intact — 2.3–2.7x on AVX2 against 1.9–3.4x on
VNNI, ranges that overlap completely.

**Why the hardware cannot help.** TorchAO's `Int8WeightOnlyConfig` is
*weight-only*: activations stay FP32, so the kernel dequantizes the weights and
runs an **FP32** GEMM. `vpdpbusd` computes INT8×INT8→INT32 and therefore needs
quantized *activations* to be reachable at all. There is no INT8 GEMM in this path
for VNNI to accelerate, and the extra time is the dequantization bolted onto a
matmul that was going to run in FP32 regardless. That is consistent with the
OpenVINO section below, which correctly notes that it is *activation*
quantization that wants VNNI.

Read the footprint as the reliable benefit and treat latency as something to
measure per model. Two caveats on the VNNI column: that host is a 4-vCPU
virtualised instance and it is noisy — the identical FP32 configuration produced
a 134 ms median in one run and 275 ms in another, with within-run spreads up to
8x — and it runs at 2.6 GHz against the i7's 4.0 GHz, so only the ratios compare.
The SmolLM2 VNNI row is genuinely ambiguous between the two estimators and no
"faster" claim is made from it. What survives the noise is the direction: on both
ISAs, by every estimator, weight-only INT8 is slower for the 1B model.

### i8mm does not rescue it either, and here is the kernel proving why

The two rows above are two x86 ISAs, which left open whether the explanation was
about weight-only quantization or about Intel. An Arm Neoverse N3 (GCP
`n4a-standard-8`, 8 vCPU, `torch 2.13.0+cpu`) reports `i8mm` — the Arm INT8
matrix-multiply instructions, the analogue of `amx_int8` — and answers it.
Llama-3.2-1B, FP32 against INT8, eager, through
[`benchmarks/quantization.py`](../benchmarks/quantization.py):

| model | ISA | FP32 | INT8 | ratio | footprint |
| --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B | **Arm Neoverse N3, `i8mm`** | 179.1 ms | 242.1 ms | **1.35x slower** | 4.943 GB → 2.026 GB (2.44x) |

A third ISA, the same direction. Top-1 agreement is 4/4 with a maximum
last-token logit difference of 0.51, so the weights are fine; only the latency
disappoints. The footprint ratio is 2.44x, identical to every other row on this
page, because it is a property of the weights and not of the host.

Read only the ratio. This leg ran **eager**, with 3 warmups and 10 repeats,
where the table at the top of this section ran `inductor` at sequence length 16
with a median of 20 — so its 179.1 ms and the 411.2 ms above are not two
measurements of the same thing. Eager was chosen deliberately: the question is
which kernel oneDNN picks, and eager is the shortest path to it. On this part
that costs little, since compiling a GEMM-bound CPU workload
[buys nothing measurable](cpu.md#latency-on-a-neoverse-n3).

**This time the kernel says so directly rather than by inference.** Running one
forward pass under `ONEDNN_VERBOSE=all`, with and without `--quantize int8`:

```
quantize=None    113x  matmul  gemm:acl  f32/f32/f32
quantize='int8'  113x  matmul  gemm:acl  f32/f32/f32
                       declined: lowp_gemm:acl -> unsupported datatype combination
```

The two are **identical**. Quantizing the weights changes neither the number of
matmuls, nor the kernel, nor the datatypes it runs at: 113 FP32 GEMMs either
way. oneDNN's INT8 matmul kernels, `lowp_gemm:acl` and `lowp_gemm_sq:acl`, are
declined in *both* runs for `unsupported datatype combination` — they are never
offered INT8 operands to work on, because weight-only quantization dequantizes
before the GEMM.

So the VNNI explanation above was right, and it was never about VNNI. Weight-only
INT8 issues no INT8 GEMM on any of the three ISAs measured, which is why no
INT8 hardware — `vpdpbusd`, `i8mm`, or presumably `amx_int8` — can reach it. The
extra 35% is dequantization bolted onto a matmul that was going to run in FP32
regardless.

Untested, and therefore still unclaimed: AMX (`amx_int8`), which neither that
Xeon nor this Arm part has. It is now the only INT8 hardware in this table
without a measurement, and the mechanism above predicts it will not help either
— but that is a prediction, not a result.

What is reliable on CPU is the footprint: **2.44x smaller, at no measured
accuracy cost**, which is the difference between a 513 MiB and a 210 MiB
SmolLM2.

> [!WARNING]
> An earlier draft of this page reported INT8 on CPU as 4.5x slower. That number
> came from a harness that built six quantization configs in one process, and it
> did not reproduce: measured one configuration per process, INT8 is at parity
> for SmolLM2. The table above is the isolated measurement.

## INT8 through OpenVINO, for an artifact rather than a process

Everything above quantizes a model inside a running Python process. To ship
something instead, `lm7 model export --backend openvino --quantize int8`
compresses the OpenVINO IR's weights with
[NNCF](https://github.com/openvinotoolkit/nncf) between `convert_model` and
`save_model`. That is a **different mechanism** from the TorchAO path: it needs
no calibration data, it runs on the IR rather than on `nn.Linear` modules, and
the result is a `.xml`/`.bin` pair that loads without PyTorch.

It also compresses more. TorchAO leaves `lm_head` in full precision; NNCF
compresses every eligible layer including the embedding and the vocabulary
projection, which is where the extra saving comes from:

| model | IR weights | top-1 | max logit diff | FP32 | INT8 |
| --- | --- | --- | --- | --- | --- |
| SmolLM2-135M | 538.1 → 135.2 MB (**3.98x**) | 4/4 | 1.20 | ~36 ms | ~31 ms (**1.16x faster**) |
| Llama-3.2-1B | 4943.3 → 1237.5 MB (3.99x) | **3/4** | 1.79 | ~292 ms | ~284 ms (parity) |
| DeepSeek-Coder-1.3B | 5385.9 → 1348.4 MB (3.99x) | 4/4 | 0.79 | — | — |

Intel i7-8086K, sequence length 16, median of 30 after warmup, one IR per
process. As in the CPU table above, the DeepSeek row comes from a different host
and covers accuracy and footprint only; its FP32 figure is the model's own weight
bytes rather than a separately measured FP32 IR. Because the OpenVINO export is
static-shape, its four prompts are four *five-token* prompts rather than the
mixed-length set used elsewhere on this page. This is the only quantization path measured here that is *faster* than
its baseline rather than slower — but only on the smaller model, and only
modestly.

DeepSeek-Coder-1.3B passes this path with the smallest logit movement of the
three, so it is admitted alongside SmolLM2.

### INT8 on CPU has two mechanisms, and they differ by 4x

Both are reachable from `lm7 model run` on an Intel CPU, and they are not
interchangeable:

```bash
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct --target cpu --quantize int8
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct --target cpu \
  --backend openvino --quantize int8
```

SmolLM2-135M, 5-token prompt, each mechanism against its own FP32 baseline on the
same host:

| mechanism | i7-8086K (AVX2) | Cascade Lake Xeon (VNNI) | Arm Neoverse N3 (`i8mm`) | Xeon 8581C (AMX-INT8) |
| --- | --- | --- | --- | --- |
| TorchAO weight-only (`inductor`) | 1.5x **slower** | 1.4x **slower** | 1.35x **slower** | **1.12x faster** |
| NNCF (`openvino`) | 1.83x faster | **2.53x faster** | 1.08x **slower** | 1.86x faster |

So on Intel CPU the OpenVINO route is the one to reach for, and the gap widens with
VNNI. Absolute INT8 times were 16.0 ms on the i7 and 16.2 ms on the Xeon — the
2.6 GHz part matches the 4.0 GHz one, which is VNNI closing a 1.5x clock deficit.

### The TorchAO row inverts on Emerald Rapids, and the reason is not established

The last column is a GCP `c4-standard-8`, Intel Xeon Platinum 8581C (Emerald
Rapids), 8 vCPU / 4 physical cores, `torch 2.13.0+cpu`, `torchao 0.18.0`,
`openvino 2026.3.0` — the first part measured here that advertises `amx_int8`.
Medians over repeated `lm7 model run` invocations on the 5-token prompt, each
mechanism against the FP32 baseline of its own backend:

| path | FP32 | INT8 | |
| --- | ---: | ---: | --- |
| TorchAO weight-only (`inductor`) | 17.4 ms | 15.5 ms | 1.12x faster |
| NNCF (`openvino`) | 15.0 ms | 8.08 ms | **1.86x faster** |

The NNCF row is unremarkable — it lands between the AVX2 and Cascade Lake
results, and the story that VNNI-class hardware makes this path pay holds. **The
TorchAO row is the finding: it is the first host in this repo where weight-only
INT8 is not a regression.** Three hosts on three ISAs had it 1.35–1.5x slower.

What that is *not* is a demonstration that AMX-INT8 rescued it. The explanation
this page gives for the three losing rows — weight-only dequantizes to FP32 and
issues an FP32 GEMM, so no INT8 dot-product instruction is reachable — predicts
that `amx_int8` should make no difference at all, and a 1.12x win is not what
"the tile units are now doing the matmul" would look like either. Two candidates
are unseparated:

- **The library versions are not held fixed.** This ran `torchao 0.18.0`; the
  versions behind the other three columns were not recorded. This repo's standing
  rule is that behaviour is a property of (model, library version), and a
  weight-only path that stopped being slower is exactly the kind of thing a
  TorchAO release changes.
- **The kernel check that settled the `i8mm` question cannot be run here.**
  [That result](#i8mm-does-not-rescue-it-either-and-here-is-the-kernel-proving-why)
  rests on reading oneDNN's executed primitives. On this host oneDNN emits
  *nothing* for SmolLM2 at FP32 or INT8 — the model's linears are served by
  ATen/MKL and never enter oneDNN, as
  [the CPU page records](cpu.md#the-bf16-crossover-is-a-function-of-thread-count-not-just-row-count)
  — so there is no log in which an INT8 GEMM could be found or ruled out.

So the table's fourth column is a latency measurement and the row above it stays
the recommendation: on this part NNCF is still 1.9x and TorchAO still 1.12x, so
OpenVINO is still the route to reach for on Intel CPU. What should not be carried
away is "AMX-INT8 fixes weight-only quantization".

The footprints behave as they do everywhere else: TorchAO reports model storage
538.1 → 220.3 MB, NNCF reports compiled weights 538.1 → 135.2 MB, and all four
configurations returned the same greedy next token (`' Paris'`).

**The OpenVINO advantage is Intel's, not INT8's.** On an Arm Neoverse N3 the
NNCF route is 13.06 ms against a 12.13 ms FP32 baseline — a small regression
rather than the 1.83–2.53x win it is on Intel, measured through `lm7 model run`
on the same host. So **neither** mechanism is faster than FP32 on Arm, and the
sentence above ("the OpenVINO route is the one to reach for") is a statement
about Intel CPUs that should not be carried onto a Graviton or Axion.

What does transfer is the footprint. NNCF still reports compiled weights at
513.1 → 129.0 MiB, a 74.9% reduction, and the greedy next token is unchanged
(`' Paris'`). Both mechanisms shrink the model on Arm; neither speeds it up.

That is consistent with [the kernel evidence
above](#i8mm-does-not-rescue-it-either-and-here-is-the-kernel-proving-why) for
the TorchAO path, but it is *not* the same explanation and is not claimed to be:
NNCF compresses the OpenVINO IR rather than converting torch modules, and no
equivalent kernel-level check was run on the OpenVINO plugin. What is measured
is the latency, on one model at one prompt length.

The saving is reported differently for the two, because the OpenVINO path never
modifies the torch module: it prints `Compiled weights: 513.1 -> 129.0 MiB (74.9%
reduction)` measured off the IR that actually executes, while TorchAO prints
`Model storage`.

### VNNI *does* help this path, unlike the TorchAO one

Re-measuring SmolLM2-135M on the Cascade Lake Xeon, rebuilding all three IRs from
one exported graph so they differ only in quantization:

| IR | weights | top-1 | max logit diff | median |
| --- | --- | --- | --- | --- |
| FP32 | 538.1 MB | — | — | 45.3 ms |
| weight-only INT8 | 135.2 MB | 4/4 | 1.11 | **22.0 ms** |

That is **2.1x faster than FP32**, against the 1.16x measured on the AVX2 i7 —
the opposite of the TorchAO weight-only path, which VNNI did nothing for. The
accuracy reproduces the row above (1.11 here against 1.20 there), so this is the
same operation, not a different one.

The difference is what the runtime does at execution time. OpenVINO's CPU plugin
dynamically quantizes activations for an INT8-weight-compressed model, so it
really does issue INT8 GEMMs and `vpdpbusd` applies. "Weight-only" describes how
the IR is *stored*, not how the plugin executes it. TorchAO's weight-only path
dequantizes to FP32 and issues an FP32 GEMM, which is why the same nominal
technique gains 2.1x here and loses 1.5-2.3x there.

**Llama-3.2-1B is rejected.** It loses a top-1 token on one prompt in four, and
excluding `lm_head` from compression did not recover it, because that model ties
its output projection to its input embedding — so the shared weight is still
compressed through the embedding. The gate is therefore per model, as it is for
the runtime path.

Full post-training quantization (`nncf.quantize`, which also quantizes
activations, and does need calibration data) was measured and **not adopted**:
on SmolLM2-135M it produced a max logit difference of 11.9 against FP32 — ten
times the weight-only path — and was *slower* than weight-only at 33.0 ms,
because INT8 activations want VNNI this CPU does not have.

That decision was re-examined on the VNNI Xeon, with a 32-sample calibration set
of varied prose rather than repeats of the eval prompt, and it **stands**:

| IR | weights | top-1 | max logit diff | median |
| --- | --- | --- | --- | --- |
| weight-only INT8 | 135.2 MB | **4/4** | **1.11** | 22.0 ms |
| full INT8 PTQ | 135.8 MB | **2/4** | **11.55** | 19.1 ms |

Two things changed and one did not. The speed half of the original objection was
wrong for this hardware — on VNNI full PTQ is 13% *faster* than weight-only, not
slower — and the footprint is slightly *worse*, because the extra activation
scales outweigh nothing. What did not change is accuracy: 11.55 here against the
11.9 recorded before, so a calibration set built specifically to be representative
did not move it. Ten times the logit error of weight-only, and 2/4 on a gate that
requires 4/4, for 13% latency.

Anyone re-running this should know that `model_type=nncf.ModelType.TRANSFORMER` is
not optional: without it the same call produced 0/4 top-1 and a max logit
difference of **81.5** on this model.

One caveat about artifact size: LM7 writes the source `exported_program.pt2`
alongside the compiled IR, and that stays FP32. The deployable IR is 3.98x
smaller, but the `.lm7` directory as a whole is not.

```bash
uv pip install -e ".[hf,openvino]"
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct out.lm7 \
  --backend openvino --target cpu --quantize int8
```

The same flags work with `--target intel:npu`, which is where compressed weights
matter most — an NPU's advantage is bandwidth and power, not FP32 throughput.
That combination has not been measured: no NPU was available. See
[intel-npu.md](intel-npu.md).

## Scope and caveats

> [!NOTE]
> This path is validated per **(model, mode)** pair and rejects everything else.
> Currently validated: `HuggingFaceTB/SmolLM2-135M-Instruct` (`int8`, `fp8`),
> `unsloth/Llama-3.2-1B-Instruct` (`int8`, `fp8`, `nvfp4`),
> `deepseek-ai/deepseek-coder-1.3b-instruct` (`int8`, `fp8`), and
> `unsloth/Llama-3.1-8B-Instruct` (`int8`). Every `int8` entry below 8B was
> measured on NVIDIA sm89 *and* on x86-64 CPU.
>
> **Llama-3.1-8B is now measured on both targets.** It used to be CPU-only,
> because its 30 GiB FP32 baseline fit on no GPU here. A Blackwell `sm120` with
> 96 GB holds it — at BF16 the GPU baseline is 16.1 GB, not 30 GiB — and INT8
> passes there at 4/4. FP8 and NVFP4 were run against it for the first time in
> the same sweep and **both fail**, so the entry stays `int8`-only.

### What "validated" means here

Each pair was run on an NVIDIA RTX 4070 SUPER (Ada, sm89) against an
unquantized BF16 baseline across four prompts, comparing the top-1 next token
and the maximum last-token logit difference:

| model | mode | top-1 agreement | max logit diff | storage |
| --- | --- | --- | --- | --- |
| SmolLM2-135M | `int8` | 4/4 | 1.14 | 1.65x smaller |
| SmolLM2-135M | `fp8` | 4/4 | 1.47 | 1.42x smaller |
| SmolLM2-135M | `nvfp4` | **2/4** | **7.41** | 2.30x smaller |
| Llama-3.2-1B | `int8` | 4/4 | 0.56 | 1.65x smaller |
| Llama-3.2-1B | `fp8` | 4/4 | 0.72 | 1.48x smaller |
| Llama-3.2-1B | `nvfp4` | 4/4 | **4.34** | 2.30x smaller |
| **LFM2.5-230M** | `int8` | **0/4** | **22.41** | 1.55x smaller |
| **LFM2.5-230M** | `fp8` | 4/4 | 0.00 | **1.00x — no-op** |
| **LFM2.5-230M** | `nvfp4` | **3/4** | **6.31** | 2.04x smaller |
| DeepSeek-Coder-1.3B | `int8` | 4/4 | 1.66 | 1.82x smaller |
| DeepSeek-Coder-1.3B | `fp8` | 4/4 | 2.12 | 1.43x smaller |
| **DeepSeek-Coder-1.3B** | `nvfp4` | 4/4 | **9.25** | 2.84x smaller |

The LFM2.5 rows are why the gate is keyed on the pair rather than the model:

- **INT8 destroys it.** Not one prompt kept its top-1 token, and the logits moved
  by 22.4. A blanket "NVIDIA is supported" claim would have shipped that.
- **FP8 does nothing to it.** LFM2.5 has no `.mlp.` module paths, so the FP8
  filter matched zero layers. TorchAO quantizes nothing and raises nothing, so
  the run *looked* successful while returning byte-identical logits at 1.00x
  storage. LM7 now raises `UnsupportedModelError` when a filter matches no
  layer, instead of reporting a no-op as a success.

### NVFP4 costs much more accuracy than 8-bit

Four bits per weight is a real quality loss at these model sizes, and the table
above shows it plainly: NVFP4's logit differences are 4.3–9.3 where INT8 and FP8
sit at 0.6–2.2. SmolLM2-135M (2/4) and LFM2.5-230M (3/4) fail the top-1 check
outright, leaving Llama-3.2-1B as the only admitted pair.

DeepSeek-Coder-1.3B is the awkward case: it does keep its top-1 token on all four
prompts, yet its logits move by 9.25 — more than twice Llama's 4.34, and wider
than the differences that accompanied both rejections. It is therefore **kept
out** of the NVFP4 gate. Four greedy prompts are a coarse instrument, and a model
whose logits move that far has almost certainly changed its distribution below
the argmax; admitting it on 4/4 alone would be reading more into the signal than
it carries. Reopening this needs a stronger accuracy check than next-token
agreement, not another prompt.

### The four prompts were never recorded, and they matter

Re-measuring accuracy on Blackwell surfaced a weakness in the bar itself. The
`sm120` sweep reproduced the 8-bit results closely — Llama-3.2-1B kept 4/4 for
both `int8` (max logit difference 0.65 against Ada's 0.56) and `fp8` (0.89
against 0.72), confirming that weight-only accuracy is a property of the weights
and not of the GPU.

`nvfp4` on the same model scored **3/4, with a logit difference of 4.62** against
Ada's 4/4 and 4.34. The logit difference agrees to within 7%, so this is almost
certainly not a hardware difference: the sm89 run's four prompts were never
written down, and the four in `benchmarks/quantization.py` are a stated set
rather than a reconstruction of them. Changing the prompts moved the pair across
the 4/4 line.

That is worth stating plainly, because Llama-3.2-1B is the *only* model admitted
to the NVFP4 gate, and it was admitted on a 4/4 result that a different four
prompts do not reproduce. It reinforces the conclusion of the section above:
four greedy prompts are a coarse instrument. **The gate is left unchanged here**
— narrowing it on a prompt set the original was not measured against would be
trading one arbitrary bar for another — but it should be read as resting on
weaker evidence than a 4/4 suggests. `benchmarks/quantization.py` now records
its prompts in the JSON report so the next comparison is prompt-for-prompt.

### Llama-3.1-8B, finally on a GPU

The 8B pair was admitted on CPU evidence alone because the model fit on no GPU
here. On a 96 GB Blackwell it does — and at BF16 the GPU baseline is 16.1 GB, so
the 30 GiB figure that blocked it was always the CPU FP32 path rather than the
one a GPU would take. All four modes, `sm120`, BF16, `inductor`:

| mode | median | vs baseline | storage | top-1 | max logit diff |
| --- | --- | --- | --- | --- | --- |
| `bf16` baseline | 12.47 ms | — | 16.06 GB | — | — |
| `int8` | 88.91 ms | 7.13x | 9.09 GB (1.77x) | **4/4** | 0.39 |
| `fp8` | 19.25 ms | 1.54x | 10.43 GB (1.54x) | 3/4 | 0.58 |
| `nvfp4` | 18.47 ms | 1.48x | 6.03 GB (**2.66x**) | 2/4 | 2.78 |

**INT8 passes, so the entry that was assumed is now measured.** FP8 and NVFP4
had never been run against this model at all; both fail, so the gate keeps the
same value it had for a completely different reason.

Three things do not carry over from the 1B results:

- **The latency penalty grows with model size.** On the same GPU, `int8` went
  from 6.04x to 7.13x, `fp8` from 1.17x to 1.54x, and `nvfp4` from 1.24x to
  1.48x. The comparatively cheap NVFP4 of the section above is a 1B result and
  does not extrapolate.
- **The footprint saving grows too**, for the opposite reason: embeddings and
  norms are a smaller fraction of a larger model, so more of it is convertible.
  `nvfp4` reaches 2.66x here against 2.30x at 1B.
- **NVFP4 beats FP8 on both axes at 8B** — faster *and* 1.7x smaller — and is
  still the mode to avoid, because it is the one that loses half its top-1
  tokens.

The FP8 result is worth reading carefully rather than as a simple failure. Its
maximum logit difference is 0.58, smaller than several modes that scored 4/4
elsewhere, and the single prompt it flipped went from the baseline's `" a"` to
`" Paris"` on `"The capital of France is"`. A 0.58 logit movement flipped a
near-tie between two plausible continuations, and it flipped it toward the
answer a reader would call correct.

That is a limit of the bar, not a defence of FP8: **top-1 agreement measures
fidelity to the unquantized baseline, not quality.** A quantized model that
disagrees may be better or worse, and next-token agreement cannot tell which.
Combined with the prompt-set sensitivity above, it is the second concrete reason
in this document to replace the four-prompt check with a real accuracy metric.

This is not an artifact of which layers are selected. Narrowing the filter
recovers a little accuracy and gives back most of the footprint:

| model | every eligible linear | MLP linears only | skipping small linears |
| --- | --- | --- | --- |
| SmolLM2-135M | 2/4, diff 7.41, 2.30x | 3/4, diff 6.44, 1.74x | 3/4, diff 6.44, 1.74x |
| Llama-3.2-1B | 4/4, diff 4.34, 2.30x | 4/4, diff 3.53, 1.88x | 4/4, diff 4.34, 2.30x |
| LFM2.5-230M | 3/4, diff 6.31, 2.04x | *matches no layer* | 3/4, diff 6.16, 1.73x |

No variant turns a failing model into a passing one, so LM7 keeps the widest
filter and takes the footprint. The likelier explanation is size: block-scaled
4-bit formats are designed for models where weight redundancy is far higher than
in a 135M–1B network. Whether NVFP4 holds up better at 7B and above is
unmeasured here.

- **NVIDIA, AMD and CPU, with a different subset on each.** CPU has INT8 only.
  AMD has the three FP8 modes and *not* INT8 — measured on an MI300X, where INT8
  holds 4/4 top-1 and runs ~10x slower than BF16 on two PyTorch versions, which
  is the regression shape LM7 already refuses on Turing. NVFP4 does run on CPU
  through the same dequantization path, but it kept only 2 of 4 top-1 tokens
  there and ran 8.5x slower than compiled FP32, so it stays NVIDIA-only. No
  Apple, Intel XPU, or TPU path exists.
- **It can be slower.** Weight-only quantization trades arithmetic for
  bandwidth. At small batch sizes, where a decode step is already
  memory-bound per token but the dequantization overhead is paid on every
  matmul, the net can go the wrong way. This is why it is opt-in rather than a
  default, and why the run reports latency alongside footprint. Compiled,
  **every mode was slower than the BF16 baseline on both GPUs measured** — by
  1.6x (`fp8`), 2.5x (`nvfp4`) and 7.3x (`int8`) on sm89, and by 1.17x, 1.24x
  and 6.0x on sm120. Newer silicon shrinks the penalty substantially without
  removing it. CPU is the one place where a mode came out even: INT8 is at
  parity for SmolLM2-135M and 2.6x slower for Llama-3.2-1B. Treat the footprint
  saving as the reliable benefit and latency as something to measure per model
  *and* per target.
- **A validated pair is not a general guarantee.** It means those prompts
  agreed on that GPU or CPU. It does not establish behaviour across long
  contexts, batch sizes, or downstream task accuracy, none of which are
  measured.
- **Storage saving is not proportional to bit width.** Only the selected linears
  are converted, so a 4x narrower weight dtype does not mean a 4x smaller model —
  embeddings, norms, `lm_head`, and (for FP8) attention all stay in BF16. A
  single NVFP4 linear is 3.56x smaller than its BF16 original, but Llama-3.2-1B
  as a whole is only 2.30x smaller.
- **Activation quantization is NVIDIA-only and dynamic-only.** `fp8-dynamic`,
  `fp8-dynamic-rowwise` and `nvfp4-dynamic` quantize activations at runtime;
  static calibrated scaling, INT8 dynamic activations, and mixed pairings such as
  FP8 activations with INT4 weights are not wired up. Four pairs are admitted:
  Llama-3.2-1B and Llama-3.1-8B, each with `fp8-dynamic` and
  `fp8-dynamic-rowwise`. `nvfp4-dynamic` is admitted for nothing.
- **Per-row scaling is measured on one card.** The `fp8-dynamic-rowwise` numbers
  are all `sm90`. The mode's `sm89` floor is a capability gate, not a
  measurement — it has never been run on Ada, and the `sm120` tables above
  predate it entirely.
- **No INT4 or lower, and no quantization-aware training.** Those are unexplored
  rather than rejected.

## Related

- [Integrated targets](../README.md#integrated-targets) — where quantization can run
- [`src/lm7/huggingface.py`](../src/lm7/huggingface.py) — validation gates and
  module filters
