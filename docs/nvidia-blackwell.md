# NVIDIA Blackwell (`sm100`, `sm120`)

What LM7 does on Blackwell, measured on an RTX PRO 6000 Blackwell Server Edition
(`sm120`, 96 GB, driver 580.126.20) with `torch 2.13.0+cu130`, `torchao
0.17.0+cu130` and `transformers 5.14.1`.

The short version: **Blackwell needed no code changes to run.** All three NVIDIA
compile backends — `inductor`, `aot_inductor` and `tensorrt` — work on it
unmodified. What it needed was code to be *reported* honestly, which is what most
of this page is about.

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
  nvidia:sm120: NVIDIA RTX PRO 6000 Blackwell Server Edition (Blackwell), 95.0 GiB
    precision: native fp32, fp16, bf16, int8, fp8, fp4
  cpu:x86_64: AMD EPYC 9B45, 176.9 GiB
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

All three NVIDIA compile backends run on `sm120`. None needed a code change.

| backend | `sm120` | evidence |
| --- | --- | --- |
| `inductor` | **Verified** | `tests/test_detection.py` + `tests/test_nvidia_integration.py`, 47 passed |
| `aot_inductor` | **Verified** | `tests/test_nvidia_aot_integration.py` + `tests/test_aot_inductor.py`, 22 passed, after `uv pip install -e ".[cuda-aot]"` |
| `tensorrt` | **Verified** | `tests/test_tensorrt_integration.py` + `tests/test_tensorrt_backend.py`, 14 passed 1 skipped |

`tensorrt` needs its own environment, because `torch-tensorrt==2.12.1` pins
PyTorch 2.12 while the rest of this page runs on 2.13. That was expected to be a
blocker and is not: `torch 2.12.1+cu130` ships `sm_120` kernels too, so the pinned
pair installs and runs on Blackwell without a source build.

### TensorRT is still worth the engine build, and more so here

SmolLM2-135M, FP16, fixed shape, median of 20 after warmup:

| backend | first call | steady | vs `inductor` |
| --- | --- | --- | --- |
| `eager` | 0.31 s | 7.378 ms | — |
| `inductor` | 13.4 s | 3.849 ms | baseline |
| `tensorrt` | 20.8 s | **2.100 ms** | **1.83x faster** |

The Ada measurement was 1.76x on the same model, so the advantage holds across
generations and widens slightly. The engine build got substantially cheaper:
20.8 s here against 56 s on Ada.

The small-MLP result also reproduces — `tensorrt` at 0.114 ms against `eager` at
0.047 ms and `inductor` at 0.072 ms. TensorRT continues to lose on workloads too
small to amortize anything, which is why it stays opt-in and lower priority than
`inductor`. See the [evaluation](nvidia-tensorrt-evaluation.md).

## The backend compatibility matrix

Every NVIDIA path LM7 implements, run against one model on `sm120`.
Llama-3.2-1B-Instruct, FP16, batch 1, a 5-token prompt, median of 20 after 5
warmup calls, through
[`benchmarks/nvidia_matrix.py`](../benchmarks/nvidia_matrix.py).

| path | works | steady | build | peak VRAM | max abs diff | argmax | reload |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `eager` | ✅ | 4.626 ms | — | 2.51 GB | reference | — | — |
| `inductor` | ✅ | 2.868 ms | 7.6 s | 3.17 GB | 2.44e-2 | ✅ | — |
| `inductor` + max-autotune | ✅ | 2.645 ms | 25.5 s | 3.20 GB | **2.02e-2** | ✅ | — |
| `inductor` + CUDA Graphs | ✅ | 2.572 ms | 7.7 s | 3.17 GB | 2.44e-2 | ✅ | — |
| `aot_inductor` export/reload | ✅ | 2.704 ms | 26.9 s | 4.98 GB | 2.44e-2 | ✅ | ✅ |
| Torch-TensorRT compile | ✅ | 3.389 ms | 15.5 s | 2.51 GB | **1.26e-1** | ✅ | — |
| TensorRT export/reload | ✅ | **2.286 ms** | 78.4 s | 4.98 GB | 8.59e-2 | ✅ | ✅ |
| ONNX Runtime CUDA | ❌ | — | — | — | — | — | — |

`build` is the first-call compile for JIT paths and the export call for the two
artifact paths. Every working path agrees with eager on the greedy next token.

**Seven of eight work, and the failure is not about Blackwell.** ONNX Runtime
raises `CompilationError: Failed to serialize proto ... Models larger than 2 GiB
are outside the initial embedded-weight scope` — the 2 GiB protobuf ceiling that
[limitations](limitations.md) already records as future work. Llama-3.2-1B is
~2.5 GB at FP16, so this fails identically on Ada. It is a model-size limit
wearing a backend's name.

### Four things the matrix says that a single number would not

**TensorRT's own artifact is 1.48x faster than TensorRT JIT.** 2.286 ms reloaded
against 3.389 ms compiled in-process — the same engine reached two ways. The JIT
path carries per-call framework overhead that the serialized artifact does not,
which makes `lm7.export(backend="tensorrt")` the interesting one and
`lm7.compile(backend="tensorrt")` mostly a way to find out whether the model
converts at all.

**CUDA Graphs is the best value on this model.** 2.572 ms for a 7.7 s build:
within 13% of the fastest path overall, at the cheapest build of anything
compiled. It is the default worth reaching for.

**max-autotune is not worth it here.** 25.5 s of build — 3.3x plain Inductor —
to land at 2.645 ms, which is *slower* than CUDA Graphs at 7.7 s. It does buy
the tightest numerics of any path (2.02e-2), so it earns its place when accuracy
matters more than either latency or compile time, but not as a speed setting.

**Numerical looseness varies 6x across backends that all agree on the token.**
From 2.02e-2 (max-autotune) to 1.26e-1 (TensorRT JIT). TensorRT rewrites FP16
arithmetic far more aggressively than Inductor, and the argmax check does not see
it. A model whose top-2 logits are close would be at materially more risk under
TensorRT than under Inductor, and no amount of next-token agreement would reveal
that — the same limit of greedy checking that
[quantization](quantization.md#the-four-prompts-were-never-recorded-and-they-matter)
runs into.

The `aot_inductor` row exports and reloads inside one process, which is a weaker
claim than it reads as. For the same artifact reloaded in an interpreter that
never compiled it — with cold and warm page caches, a rejection matrix, and what
happens under a different PyTorch — see
[AOTInductor artifact compatibility](aot-artifact-compatibility.md).

### Reading these numbers

- **Three environments.** `tensorrt` pins PyTorch 2.12.1; everything else ran on
  2.13.0. Both ship `sm_120` kernels. Rows are directly comparable for
  works/fails and only roughly for latency.
- **Peak VRAM is torch's allocator only.** TensorRT and ONNX Runtime allocate
  outside it, so their figures understate real device usage. The two export paths
  peak at 4.98 GB because the artifact holds the source program alongside the
  compiled payload.
- **Build times are cold.** Inductor's on-disk FX cache survives between
  processes and `cache=False` does not clear it — that argument controls LM7's
  artifact cache. Warm, the same Inductor compile is 2.3 s rather than 7.6 s, and
  peak VRAM drops to 2.51 GB because the compile workspace is never allocated.
  The table clears `/tmp/torchinductor_*` before each Inductor row.
- **One model, one shape, one dtype.** A 5-token prefill at batch 1 is close to
  the most launch-overhead-dominated point on the curve, which flatters CUDA
  Graphs and penalizes engine builders. Larger batches or sequences would
  reorder these rows.

> [!NOTE]
> The CUDA Graphs row failed on the first attempt with `accessing tensor output
> of CUDAGraphs that has been overwritten by a subsequent run`, and the backend
> was fine. CUDA Graphs replay into one output buffer, so a harness that keeps
> the first call's tensor and reads it after the timing loop reads whatever the
> last call wrote. `benchmarks/nvidia_matrix.py` now copies off-device
> immediately. Worth knowing before recording `reduce-overhead` as broken.

## Scope

None of this is covered by CI, which remains CPU-only, and it is one card. A
single GPU says nothing about multi-GPU behaviour, and `sm100` (datacenter
Blackwell) is reported by the same code path but has never been executed — it
shares `sm120`'s precision row by capability number, not by measurement.

The matrix is one model. `benchmarks/nvidia_matrix.py` also builds `mlp`,
`resnet18`, `bert`, `smollm2` and `llama31-8b`, and running the same eight paths
across them is the obvious extension — nothing about the harness is specific to
the row above.
