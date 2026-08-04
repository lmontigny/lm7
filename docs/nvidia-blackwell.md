# NVIDIA Blackwell (`sm100`, `sm120`)

What LM7 does on Blackwell, measured on an RTX PRO 6000 Blackwell Server Edition
(`sm120`, 96 GB, driver 580.126.20) with `torch 2.13.0+cu130`, `torchao
0.17.0+cu130` and `transformers 5.14.1`.

The short version: **Blackwell needed no code changes to run.** It needed code
changes to be *reported* honestly, which is what most of this page is about.

## Nothing had to be special-cased

`nvidia:sm120` resolves, selects `inductor`, and runs. That is not luck, but it
is close to it: every architecture gate in LM7 compares the `smXX` number as a
plain integer, and CUDA capabilities happen to sort correctly that way —
75 < 80 < 86 < 89 < 90 < 100 < 120. Blackwell lands above Ada by the same
comparison that admits Ada above Ampere, so the FP8 gate (`>= 89`) and the
native-BF16 gate (`>= 80`) both admitted it without being told about it.

`test_compute_capability_orders_blackwell_above_ada` pins that property, because
it is the thing that would silently break if some future gate started matching
architectures by name.

## Reporting: generation and precision

Two things were missing, and both were about a reader being able to tell what
they have.

**The generation now has a name.** `torch` reports only a capability number, and
`sm120` means nothing to anyone who has not memorized the table. The TPU path
already names its generation, and NVIDIA now does the same:

```console
$ lm7 targets
Detected targets (2):
  nvidia:sm120: NVIDIA RTX PRO 6000 Blackwell Server Edition (Blackwell), 95.6 GiB
    precision: native fp32, fp16, bf16, int8, fp8, fp4
  cpu:x86_64: ..., 176.0 GiB
```

**Precision is now reported as native, emulated, or absent.** This is the more
useful half. The distinction exists because torch will happily run an emulated
format and report success — a Tesla T4 answers `True` to
`torch.cuda.is_bf16_supported()` and then emulates BF16, which measured 3.4x
slower than FP16 on the same model. Without the label, that is an unexplained
slowdown; with it, it is an expected one.

| capability | fp16 | bf16 | int8 | fp8 | fp4 |
| --- | --- | --- | --- | --- | --- |
| `sm75` Turing | native | **emulated** | native | absent | absent |
| `sm80`–`sm87` Ampere | native | native | native | absent | absent |
| `sm89` Ada | native | native | native | native | absent |
| `sm90` Hopper | native | native | native | native | absent |
| `sm100`/`sm120` Blackwell | native | native | native | native | **native** |

Available as `lm7 targets`, in `lm7 doctor`, and under `capabilities.precision`
in the `--json` form of both.

Only NVIDIA is characterized. Every other vendor returns an empty mapping rather
than a guess — claiming "native bf16" for a CPU whose AVX-512 BF16 support was
never probed would be exactly the unmeasured assertion this report exists to
prevent.

## Native is not the same as used

This is the part worth internalizing, and the reason the table above is a
hardware fact rather than a performance promise.

**Blackwell reports `fp4: native`, and LM7's NVFP4 path issues no FP4 matmul at
all.** Weight-only quantization stores the weight in 4 bits and unpacks it to
BF16 inside the kernel, so the FP4 tensor cores are never asked to multiply
anything. Reaching them needs FP4 *activations* too, which is activation
quantization and is not implemented here.

The measurement agrees with the mechanism. On Llama-3.2-1B, BF16 baseline,
`inductor`:

| mode | Ada `sm89` | Blackwell `sm120` |
| --- | --- | --- |
| `bf16` baseline | 8.92 ms | **3.11 ms** |
| `fp8` | 14.61 ms (1.64x) | 3.64 ms (1.17x) |
| `nvfp4` | 22.27 ms (2.50x) | 3.84 ms (1.24x) |

If the FP4 units were engaged, `nvfp4` would beat the BF16 baseline instead of
trailing it. What Blackwell changes is the *cost* of the mode: the same 2.30x
footprint saving costs 150% more latency on Ada and 24% here. See
[quantization](quantization.md) for the full sweep, the 8B results, and the
accuracy figures.

So read `fp4: native` as "this silicon could, if something asked it to", and not
as "your quantized model is using it".

## Backend status

| backend | `sm120` status |
| --- | --- |
| `inductor` | **Verified.** Resolves, selects, and runs; PyTorch's `cu130` wheel ships `sm_120` kernels, so no source build is needed. |
| `aot_inductor` | **Not yet verified.** Needs the `.[cuda-aot]` extra, which packages against a CUDA toolkit the PyTorch wheel does not ship. |
| `tensorrt` | **Not yet verified.** LM7 pins `torch-tensorrt==2.12.1`, built against PyTorch 2.12, while a `sm_120` stack wants 2.13 — expect a version conflict rather than a clean run. |

The two unverified rows are unverified, not broken: nobody has run them on this
silicon. Do not read the `inductor` row as covering them.

## Scope

None of this is covered by CI, which remains CPU-only, and it is one card. A
single GPU says nothing about multi-GPU behaviour, and `sm100` (datacenter
Blackwell) is reported by the same code path but has never been executed — it
shares `sm120`'s precision row by capability number, not by measurement.
