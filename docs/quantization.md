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

| `--quantize` | Weight storage | Compute dtype | Requires |
| --- | --- | --- | --- |
| `none` (default) | as loaded | FP32 / FP16 / BF16 | nothing |
| `int8` | INT8 | BF16 | NVIDIA GPU |
| `fp8` | FP8 | BF16 | NVIDIA Ada (`sm89`), Hopper (`sm90`), or newer |
| `nvfp4` | NVFP4 — 4-bit, one FP8 scale per 16 values | BF16 | NVIDIA GPU |

All modes force BF16 compute. `--dtype` must be `auto` or `bfloat16`, and `auto`
resolves to BF16 whenever quantization is on. `--backend` must be `auto` or
`inductor`. Anything else raises `UnsupportedModelError` rather than silently
degrading — including a non-NVIDIA target, an FP8 request on pre-Ada hardware,
and any (model, mode) pair outside the validated list.

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

## Scope and caveats

> [!NOTE]
> This path is validated per **(model, mode)** pair and rejects everything else.
> Currently validated: `HuggingFaceTB/SmolLM2-135M-Instruct` (`int8`, `fp8`) and
> `unsloth/Llama-3.2-1B-Instruct` (`int8`, `fp8`, `nvfp4`).

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
above shows it plainly: NVFP4's logit differences are 4.3–7.4 where INT8 and FP8
sit at 0.6–1.5. Only Llama-3.2-1B kept its top-1 token on all four prompts, so it
is the only pair admitted. SmolLM2-135M (2/4) and LFM2.5-230M (3/4) are rejected.

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

- **NVIDIA only.** No CPU, AMD, Apple, or TPU quantization path exists. NVFP4
  happens to run on CPU through the same dequantization path, but that is
  untested and ungated, so it stays rejected.
- **It can be slower.** Weight-only quantization trades arithmetic for
  bandwidth. At small batch sizes, where a decode step is already
  memory-bound per token but the dequantization overhead is paid on every
  matmul, the net can go the wrong way. This is why it is opt-in rather than a
  default, and why the run reports latency alongside footprint. Measured on
  sm89 and compiled, **every mode was slower than the BF16 baseline** —
  `fp8` by 1.6x, `nvfp4` by 2.5x, `int8` by 7.3x. Treat the footprint saving as
  the reliable benefit and latency as something to measure per model.
- **A validated pair is not a general guarantee.** It means those prompts
  agreed on that GPU. It does not establish behaviour across long contexts,
  batch sizes, or downstream task accuracy, none of which are measured.
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
