# AMD CPU inference

LM7 needs nothing special to run on an AMD CPU: `target="cpu"` works, and the
default `inductor` backend compiles for it like any other x86-64 part. This page
covers what is *different* about AMD, measured rather than assumed.

Everything below was measured on an **AMD EPYC 7B13** (Zen 3, 8 physical cores,
16 logical, AVX2 — no AVX-512, no VNNI, no AMX), `torch 2.13.0+cpu`, Ubuntu
24.04.

## What PyTorch already uses

The stock PyTorch CPU wheel reports:

```
Intel(R) MKL-DNN v3.12.0                     # oneDNN
Intel(R) oneAPI Math Kernel Library 2024.2   # MKL
Build settings: BLAS_INFO=mkl
CPU capability usage: AVX2
```

So a PyTorch CPU install on AMD hardware runs on **Intel's MKL for BLAS and
Intel's oneDNN for primitives**. That combination invites an obvious worry, and
the folklore around it is mostly obsolete.

### MKL is not crippled on AMD

The concern is that MKL dispatches a deliberately slow generic path when
`CPUID` reports `AuthenticAMD`. Measured against NumPy's OpenBLAS on the same
machine, same FP32 GEMM, same 8 threads:

| matrix | MKL (torch) | OpenBLAS (numpy) | MKL GFLOP/s | OpenBLAS GFLOP/s | ratio |
| --- | --- | --- | --- | --- | --- |
| 1024³ | 6.14 ms | 5.98 ms | 349.6 | 359.2 | 1.03x |
| 2048³ | 24.32 ms | 31.08 ms | 706.5 | 552.7 | 0.78x |
| 4096³ | 224.07 ms | 252.38 ms | 613.4 | 544.6 | 0.89x |
| 8192³ | 1667.95 ms | 1603.64 ms | 659.2 | 685.6 | 1.04x |

MKL wins two sizes, loses two, and never by much. At ~600–700 GFLOP/s it is
close to this part's AVX2 ceiling (8 cores × 32 FLOP/cycle × ~2.4–2.7 GHz), so
there is no hidden headroom to reclaim. **Do not go looking for a replacement
BLAS on this basis.**

`MKL_DEBUG_CPU_TYPE=5`, the workaround every old forum thread recommends, does
nothing: 608.2 GFLOP/s with it against 585.4 without, which is run-to-run noise.
Intel removed that variable after MKL 2020 and this build is 2024.2.

## Thread affinity: the one real trap

`OMP_PROC_BIND=close` sounds like the right setting and is the worst one you are
likely to try. 4096³ FP32 GEMM, 8 threads, median of 9, reproduced across four
independent runs:

| `OMP_PROC_BIND` | GFLOP/s | vs unset |
| --- | --- | --- |
| unset | 585.4 | 1.00x |
| `close` | **358.0** | **0.61x** |
| `spread` | 600.8 | 1.03x |
| `master` | 86.7 | 0.15x |

`close` packs threads onto *adjacent logical CPUs*, which on an SMT part are the
two hyperthreads of one core. Eight threads then land on four physical cores,
sharing four vector units instead of using eight. The `close` figure was stable
to within 1% across runs (358.0, 363.0, 378.5, 378.6) while the unpinned figure
moved with machine load, so the gap is real and not sampling luck.

`spread` is the safe explicit choice, and leaving it unset is nearly as good —
the OpenMP runtime already does something sensible.

This is the same effect as the thread-count result in [cpu.md](cpu.md): eight
threads beat sixteen on this 8-core/16-thread part, because SMT siblings
contend for one vector unit. Both say the same thing — **count physical cores,
not logical ones**.

## ZenDNN, through zentorch

AMD ships its own PyTorch extension, and LM7 exposes it as an opt-in backend:

```bash
uv pip install -e ".[zentorch]"
```

```python
model = lm7.compile(model, target="cpu", backend="zentorch")
```

On this part it won one workload, tied another, and lost a third, so it is never
selected by `backend="auto"`. The measurements and the reasoning are in
[zentorch.md](zentorch.md).

## What is not measured here

- **Newer EPYC generations.** Genoa (Zen 4) and Turin (Zen 5) add AVX-512, BF16,
  and VNNI. Every number on this page is from a part with none of those, and
  those instructions are exactly where both oneDNN and ZenDNN claim their larger
  wins. Nothing here transfers.
- **BF16 and INT8 throughput.** The CPU compute dtype is FP32 (see
  [quantization](quantization.md)), and weight-only INT8 is a latency regression
  on CPU at every model size measured.
- **Multi-socket and NUMA.** This is a single-socket host, so no cross-socket
  memory effects appear at all.
- **AOCL.** AMD's own BLAS/BLIS was not tried, because the MKL result above
  leaves no gap for it to close on this part.

## Summary

| Question | Answer on Zen 3 |
| --- | --- |
| Does AMD need a special install? | No — `target="cpu"` is enough |
| Is MKL sabotaging AMD? | No, it is at OpenBLAS parity and near peak |
| Is `MKL_DEBUG_CPU_TYPE=5` worth setting? | No, it is a no-op since MKL 2020 |
| Should I pin threads? | Only `spread`; `close` costs 39% |
| How many threads? | Physical cores, not logical — see [cpu.md](cpu.md) |
| Is zentorch worth it? | Sometimes; measure it — see [zentorch.md](zentorch.md) |
