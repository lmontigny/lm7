# Quantization

Quantization stores or computes a tensor in fewer bits than the model was trained
in. There are two halves to it, and **LM7 currently implements only the first**.

## Weight versus activation quantization

**Weight quantization** shrinks the stored parameters. Weights are converted
once, up front, and dequantized on the fly inside the matmul, so the arithmetic
still happens in a higher precision. What you gain is memory footprint and memory
bandwidth; what you avoid is calibration, because a weight tensor's range is
known statically.

**Activation quantization** also narrows the tensors flowing between layers, so
the matmul itself executes in low precision. That is where the larger speedups
come from, because it cuts arithmetic work rather than just bytes moved. The cost
is that activation ranges depend on the input, so you need calibration data or a
quantization-aware recipe to choose per-tensor scales, and accuracy degrades more
readily. **LM7 does not do this today.**

Everything below is therefore *weight-only*: activations, and the accumulation
inside each matmul, stay in BF16.

## Supported data types

| `--quantize` | Weight storage | Targets | Compute dtype |
| --- | --- | --- | --- |
| `none` (default) | as loaded | all | FP32 / FP16 / BF16 |
| `int8` | INT8 | NVIDIA GPU, **CPU** | BF16 on NVIDIA, FP32 on CPU |
| `fp8` | FP8 | NVIDIA Ada (`sm89`), Hopper (`sm90`), or newer | BF16 |
| `nvfp4` | NVFP4 — 4-bit, one FP8 scale per 16 values | NVIDIA GPU | BF16 |

Quantization pins the compute dtype so measurements stay comparable, and the
dtype is target-specific: BF16 on NVIDIA, FP32 on CPU. `--dtype` must be `auto`
or that target's dtype, and `auto` resolves to it. `--backend` must be `auto` or
`inductor`. Anything else raises `UnsupportedModelError` rather than silently
degrading — including a mode on a target it was not measured on, an FP8 request
on pre-Ada hardware, and any (model, mode) pair outside the validated list.

CPU uses FP32 rather than BF16 because x86-64 without AVX-512 has no native BF16
path, so forcing BF16 there would measure emulation rather than the format.

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

## NVFP4 on hardware without FP4 tensor cores

NVFP4 is NVIDIA's 4-bit block-scaled float format. Hardware FP4 matmul exists
only on Blackwell (`sm100`, `sm120`). **LM7's measurements were taken on Ada
(`sm89`), which has no FP4 arithmetic at all.**

That is not a blocker, because weight-only quantization never asks the hardware
to multiply in FP4 — the weight is unpacked to BF16 inside the matmul. So on
`sm89` the mode runs correctly and buys footprint and bandwidth, and **cannot**
buy arithmetic throughput. Whether Blackwell's FP4 tensor cores change the
latency picture is unmeasured; LM7 has no such GPU.

One consequence is that eager execution is pathological. The unpack is not
fused, so every matmul pays it in full:

| mode | eager prefill | compiled prefill |
| --- | --- | --- |
| `bf16` baseline | 27.66 ms | **8.92 ms** |
| `int8` | 40.62 ms | 65.13 ms |
| `fp8` | 47.21 ms | 14.61 ms |
| `nvfp4` | 260.98 ms | **22.27 ms** |

Llama-3.2-1B, single prompt, RTX 4070 SUPER (Ada, `sm89`), median of 10 after
warmup. Inductor fuses the unpack into the matmul, which makes NVFP4 11.7x
faster than its own eager path: from 9.4x slower than eager BF16 down to 2.5x
slower than compiled BF16, and 2.9x faster than compiled INT8. This is why
`--backend` is restricted to `inductor` — NVFP4 without it is unusable.

Compilation is not free either. On that machine the first call for
`nvfp4` took **72 seconds** against a 20 ms steady-state call, so the mode only
makes sense where the process is long-lived or the artifact is reused.

## INT8 on CPU

INT8 is the only mode measured off NVIDIA. It converts the same layers, keeps
the same 4/4 top-1 agreement, and gives a slightly *better* footprint ratio than
on GPU, because the CPU baseline is FP32 rather than BF16:

| model | top-1 | max logit diff | storage | FP32 | INT8 |
| --- | --- | --- | --- | --- | --- |
| SmolLM2-135M | 4/4 | 1.36 | 513 → 210 MiB (2.44x) | 49.9 ms | 75.8 ms (**1.52x**) |
| Llama-3.2-1B | 4/4 | 0.76 | 4943 → 2026 MB (2.44x) | 411.2 ms | 928.5 ms (**2.26x**) |
| DeepSeek-Coder-1.3B | 4/4 | 0.67 | 5386 → 1746 MB (3.09x) | — | — |

Intel i7-8086K, 6 threads, sequence length 16, compiled through `inductor`,
median of 20 after warmup, one configuration per process. The DeepSeek row was
measured on a different host for accuracy and footprint only; its latency is left
blank rather than filled with a number from another machine. Its
better-than-2.44x ratio comes from a wider model with proportionally less weight
sitting in the untouched `lm_head`.

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

Untested, and therefore still unclaimed: AMX (`amx_int8`), which that Xeon does
not have, and ARM cores with dotprod/i8mm.

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

**Llama-3.2-1B is rejected.** It loses a top-1 token on one prompt in four, and
excluding `lm_head` from compression did not recover it, because that model ties
its output projection to its input embedding — so the shared weight is still
compressed through the embedding. The gate is therefore per model, as it is for
the runtime path.

Full post-training quantization (`nncf.quantize`, which also quantizes
activations, and does need calibration data) was measured and **not adopted**:
on SmolLM2-135M it produced a max logit difference of 11.9 against FP32 — ten
times the weight-only path — and was *slower* than weight-only at 33.0 ms,
because INT8 activations want VNNI this CPU does not have. Unlike the weight-only
path above, that hardware argument does apply here: quantized activations are
what make an INT8 GEMM, and an INT8 GEMM is what `vpdpbusd` accelerates. The
accuracy half of the objection is hardware-independent, so a VNNI part would have
to fix the 11.9 logit difference — probably through better calibration — before
the speed question is worth re-asking.

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
> `unsloth/Llama-3.2-1B-Instruct` (`int8`, `fp8`, `nvfp4`), and
> `deepseek-ai/deepseek-coder-1.3b-instruct` (`int8`, `fp8`). Every `int8` entry
> was measured on NVIDIA sm89 *and* on x86-64 CPU.

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

- **NVIDIA and CPU only, and only INT8 on CPU.** No AMD, Apple, Intel XPU, or
  TPU path exists. FP8 needs Ada-class tensor cores. NVFP4 does run on CPU
  through the same dequantization path, but it kept only 2 of 4 top-1 tokens
  there and ran 8.5x slower than compiled FP32, so it stays NVIDIA-only.
- **It can be slower.** Weight-only quantization trades arithmetic for
  bandwidth. At small batch sizes, where a decode step is already
  memory-bound per token but the dequantization overhead is paid on every
  matmul, the net can go the wrong way. This is why it is opt-in rather than a
  default, and why the run reports latency alongside footprint. Measured on
  sm89 and compiled, **every mode was slower than the BF16 baseline** —
  `fp8` by 1.6x, `nvfp4` by 2.5x, `int8` by 7.3x. CPU is the one place where a
  mode came out even: INT8 is at parity for SmolLM2-135M and 2.6x slower for
  Llama-3.2-1B. Treat the footprint saving as the reliable benefit and latency
  as something to measure per model *and* per target.
- **A validated pair is not a general guarantee.** It means those prompts
  agreed on that GPU or CPU. It does not establish behaviour across long
  contexts, batch sizes, or downstream task accuracy, none of which are
  measured.
- **Storage saving is not proportional to bit width.** Only the selected linears
  are converted, so a 4x narrower weight dtype does not mean a 4x smaller model —
  embeddings, norms, `lm_head`, and (for FP8) attention all stay in BF16. A
  single NVFP4 linear is 3.56x smaller than its BF16 original, but Llama-3.2-1B
  as a whole is only 2.30x smaller.
- **No activation quantization, no INT4 or lower, no quantization-aware
  training.** Those are unexplored rather than rejected.

## Related

- [Supported hardware](../README.md#supported-hardware) — where quantization can run
- [`src/lm7/huggingface.py`](../src/lm7/huggingface.py) — validation gates and
  module filters
