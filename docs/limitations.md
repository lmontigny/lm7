# Limitations and current scope

What LM7 does not do, stated plainly. Read this before depending on it for
anything that matters.

LM7 is an early, inference-only prototype. Model coverage and compiled-artifact
compatibility are not stable, and the only path with continuous-integration
coverage is CPU.

## Scope of the project

- **Inference only.** Training and backward compilation are unsupported.
- **Local devices only.** LM7 detects PyTorch-visible devices on the machine it
  runs on. It installs no drivers, CUDA/ROCm toolchains, Xcode, PyTorch/XLA, or
  vendor SDKs, and it does not reach remote or networked accelerators.
- **Bounded by the vendor toolchains.** LM7 writes no kernels and no compiler of
  its own, so its reach is exactly what the underlying compiler already
  supports. Adding hardware means wiring up its compiler, not writing one.
- **Not a stable ABI.** Compiled artifacts are specific to compatible compiler,
  runtime, and hardware versions. Distributed inference, remote hardware, and a
  stable artifact ABI are future work.

## Compilation and artifacts

- **JIT results are process-local.** A JIT-compiled callable does not outlive
  the process. Only `aot_inductor`, `openvino`, `onnxruntime`, `tensorrt`,
  `iree_vulkan`, `litert`, `executorch`, `stablehlo`, and `lm7.export` produce
  something another process can load — and `tensorrt` only through
  `lm7.export`; its `lm7.compile` engines still die with the process.
- **Exported causal-LM artifacts are prefill-only.** `--dynamic-seq` makes the
  sequence length variable within recorded bounds; the batch dimension stays
  fixed, and a KV-cache decode loop is not captured. LM7 captures a logits-only
  graph, because `CausalLMOutputWithPast` cannot be deserialized by
  `torch.export.load`.
- See [JIT vs. AOT](jit-vs-aot.md) for the export levels, bundles, and the
  signature rules an artifact is pinned to.
- **Sparse Mixture-of-Experts models always compile; whether they *export*
  depends on the transformers version and the architecture.** Measured with
  `torch 2.13` on two-layer models through `lm7.export`:

  | transformers | Mixtral | OLMoE |
  | --- | --- | --- |
  | 4.57.3 (the CI pin) | JIT only — export fails | JIT **and** export |
  | 5.14.1 | JIT **and** export | JIT **and** export |

  The failing cell is the one this project originally generalized from. Mixtral's
  pre-5.x implementation routes tokens with a data-dependent Python loop (`for
  expert_idx in expert_hit: ...`) whose iteration count is only known at runtime.
  `torch.compile` tolerates it and `torch.export` does not: export must capture
  one static graph and hard-fails on it, which takes down every export-based
  backend (`aot_inductor`, `openvino`, `onnxruntime`, `executorch`,
  `iree_vulkan`, `litert`, `stablehlo`), while `inductor` and `tensorrt` were
  always fine.

  This loop is an **export** problem specifically. It is not what makes Dynamo
  break up a sparse MoE — that is a separate mechanism, measured
  [below](#what-torchcompile-actually-does-to-a-sparse-moe).

  Two things make the old blanket claim wrong. **OLMoE never had the problem**,
  even on the pinned transformers — its routing does not use that loop, so
  "sparse MoE cannot export" was an over-generalization from a single reference
  implementation. And **transformers 5.x removed the loop from Mixtral too**,
  replacing it with a `grouped_mm` formulation that exports cleanly. On 5.x both
  architectures export through `aot_inductor` and reload with outputs matching
  eager.

  So treat exportability as a property of the specific model and transformers
  version, not of MoE as a class, and check it rather than assuming either way.
  See [`examples/sparse_moe.py`](../examples/sparse_moe.py), which covers both
  architectures on CPU and NVIDIA.

  At real scale, `allenai/OLMoE-1B-7B-0924-Instruct` (6.92B total, 64 experts, 8
  active) runs on an AMD EPYC CPU through `eager` (525.1 ms), `inductor`
  (506.8 ms) and `zentorch` (522.1 ms), all agreeing on the greedy next token, so
  `zentorch`'s usual small-model advantage disappears. Exporting that model is
  **not** verified: the attempt destabilized a 62 GB host twice during weight
  loading and was abandoned rather than retried.

  That 1.04x used to be explained here as "Dynamo graph-breaks around the routing
  regardless of backend, so most of the model runs eagerly either way". The
  measurement holds and **the explanation was wrong** — see below.

  > [!NOTE]
  > `grouped_mm` on transformers 5.x requires tensor strides that are multiples
  > of 16 bytes, so very small hand-built configs (an `intermediate_size` of 37,
  > say) now fail in *eager* with `strides should be multiple of 16 bytes`. That
  > is a config constraint, not an LM7 or export limitation.

### What `torch.compile` actually does to a sparse MoE

The graph-break claim above was inferred from a latency result rather than from
Dynamo. Asking Dynamo directly, with `torch._dynamo.explain` through
[`benchmarks/moe.py`](../benchmarks/moe.py), gives a different answer on every
point:

| model | transformers | target | graphs | breaks | eager | inductor | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mixtral, tiny | 4.57.3 | cpu | 10 | 9 | 6.87 ms | 1.62 ms | **4.24x** |
| OLMoE, tiny | 4.57.3 | cpu | 9 | 8 | 2.17 ms | 2.21 ms | 0.98x |
| Mixtral, tiny | 5.14.1 | nvidia `sm120` | **1** | **0** | 1.84 ms | 0.59 ms | **3.12x** |
| OLMoE, tiny | 5.14.1 | nvidia `sm120` | **1** | **0** | 2.03 ms | 0.71 ms | **2.85x** |
| OLMoE-1B-7B (6.92B) | 5.14.1 | nvidia `sm120` | **1** | **0** | 17.88 ms | 11.20 ms | **1.60x** |
| OLMoE-1B-7B (6.92B) | 5.14.1 | cpu | **1** | **0** | 345.0 ms | 335.0 ms | 1.03x |

Four corrections come out of that.

**The break is `aten.nonzero`, not the Python loop.** On transformers 4.57.3 the
reported reason is *"Operator `aten.nonzero.default`'s output shape depends on
input Tensor data"* — the router's top-k scatter, which both architectures use.
The `for expert_idx in expert_hit` loop is what breaks `torch.export`. Two
different mechanisms with two different blast radii, previously described as one.

**OLMoE breaks just as much as Mixtral.** This document says OLMoE "never had the
problem", which is true of export and false of compile: on 4.57.3 it breaks eight
times, the same reason and roughly the same count as Mixtral. Exact counts drift
by ±1 between runs, so read them as "eight or nine", not as exact.

**On transformers 5.x there are no breaks at all** — one graph, zero breaks, for
both tiny architectures and for the real 6.92B model. The `grouped_mm` rewrite
that fixed export removed the `nonzero` capture problem as well.

**Graph breaks do not predict the speedup anyway.** Tiny Mixtral compiles 4.24x
faster *with* nine breaks, while tiny OLMoE gets 0.98x *with* eight. The causal
story — breaks, therefore most of the model runs eagerly, therefore no speedup —
does not survive being measured.

So the honest version of the original claim is that **it was about the target,
not about MoE.** With the same transformers and zero graph breaks, the same 6.92B
model gets 1.03x on CPU and 1.60x on an `sm120` GPU. The 1.03x reproduces the
1.04x measured on a different EPYC host, so the original number was right; what
did not generalize was reading a CPU result as a property of sparse MoE.

Every row agrees with `eager` on the greedy next token.

> [!NOTE]
> Measuring this needs care. Dynamo traces whatever it is handed, so a model left
> on the host is measured in a configuration that never runs, and
> `logger.warning_once` calls inside `forward` break the *first* trace only. An
> early revision of `benchmarks/moe.py` reported 14 breaks on a model that
> captures as one graph. The script now moves the model to the target device and
> makes one eager call before tracing.

## Per-backend scope

| Backend | Scope and caveats |
| --- | --- |
| `inductor` | The default and the best-covered path. CPU is the only target with CI. |
| `aot_inductor` | Validated for CPU, Apple Silicon (MPS), and NVIDIA GPU; uses Beta PyTorch APIs. On NVIDIA it packages against a CUDA toolkit the PyTorch wheel does not ship — install `".[cuda-aot]"`. See the [WSL linker caveat](development.md#nvidia-aot-inductor). |
| `tensorrt` | NVIDIA only. Slower engine builds and narrower model coverage than Inductor — see the [evaluation](nvidia-tensorrt-evaluation.md). `lm7.export` serializes the engine so a second process need not rebuild it; the artifact is static-shape and bound to the GPU architecture, TensorRT version, and Torch-TensorRT version that built it. |
| `openvino` | Intel CPU, plus `intel:npu` — **implemented but never run on an NPU**. Rejects bfloat16, because its runtime exchanges tensors through NumPy. Returns tensors or tuples, so a model whose `forward` returns a dataclass needs a wrapper. Optional NNCF INT8 weight compression on both `model run` and `model export`, validated per model. On the NPU: static shapes only, and FP16 compute, so expect FP16-level error. See the [guide](intel-npu.md). |
| `onnxruntime` | CPU and NVIDIA CUDA. Returns CPU tensors even after CUDA execution, because the initial adapter uses NumPy rather than I/O binding. Tensor-only inputs and flat outputs; external-data packaging above the 2 GiB protobuf limit is future work. See the [guide](onnxruntime.md). |
| `iree_vulkan` | Export-only and experimental: fixed shapes, tensor-only I/O, FP32 MLP execution is the validated scope. Causal LMs, dynamic sequences, KV caches, and WebGPU are future work. See the [guide](iree-vulkan.md). |
| `litert` | Export-only, CPU/XNNPACK only. Static tensor-only inputs, returns CPU tensors. LiteRT Torch caps PyTorch below 2.13, so conversion belongs in a separate environment. Packages generic `.tflite` graphs, not LiteRT-LM conversations. See the [guide](litert.md). |
| `executorch` | Export-only and XNNPACK-only, so the edge story is CPU. Phone NPUs (Core ML, Qualcomm QNN, MediaTek, Exynos) are not wired up; artifacts are static-shape; optional calibrated XNNPACK INT8 PTQ is available; validation is host x86-64, not a physical phone. LM7 writes the `.pte` — deploying it into an app is ExecuTorch's tooling. See the [guide](executorch.md). |
| `stablehlo` | Export-only. Needs PyTorch/XLA to lower, which pins PyTorch to a matching pair. |
| `openxla` | TPU only, single process. SPMD sharding, multi-host execution, and persistent XLA executables are not implemented, and the validated host has one chip, so the sharding paths are untested rather than merely absent. fp32 matmuls run at bf16 precision unless `options={"mat_mul_precision": ...}` says otherwise. It beats eager XLA by 27x on SmolLM2-135M and loses to it by 2.3x on a 3-layer MLP. Generation decodes eagerly and XLA recompiles per decode step, so a first `lm7 model generate` of 20 tokens costs 20 minutes and the second 2.5 s -- the documented per-shape compile cost, paid per decode graph. Warm the process before timing it. See the [guide](google-tpu.md). |
| `tenstorrent` | JIT-only and single-card: the compiled flatbuffer does not outlive the process, multi-card sharding is not exposed, and coverage is bounded by what tt-mlir lowers. See the [guide](tenstorrent.md). |
| `tvm` | CPU-only, JIT-only, positional-inputs-only, and **far slower than Inductor** — registered for reachability, not speed. Autotuning, CUDA, and artifacts are not wired up. See the [guide](tvm.md). |
| `zentorch` | AMD's ZenDNN extension: CPU-only, JIT-only, explicit-only, x86-64 Linux wheels only, and ABI-tied to a matching PyTorch. Measured on one Zen 3 EPYC at FP32 it beat Inductor on one workload, tied on another, and lost on a third; BF16, INT8, and newer EPYC generations are unmeasured. No quantization path and no artifact. See the [guide](zentorch.md). |

## Hardware validation

- **CPU is the only target with CI.** Everything else has been exercised
  manually, or not at all.
- AMD ROCm, Apple Silicon (MPS), Intel XPU, OpenXLA TPU, and Tenstorrent are
  initial single-process integrations **without physical-hardware CI**.
- OpenXLA TPU has now been **exercised on real hardware** — a single-chip TPU
  v6e — which the rest of that list has not. It still has no CI, and one chip
  cannot say anything about multi-chip behaviour. See
  [Google TPU](google-tpu.md).
- NVIDIA Inductor, quantization, and TensorRT have been exercised on two GPU
  generations: a local Ada (`sm89`) and a Blackwell (`sm120`) RTX PRO 6000.
  Detection, backend selection, and every weight-only mode worked on Blackwell
  with no code changes, and all three NVIDIA compile backends run there
  unmodified — `inductor`, `aot_inductor`, and `tensorrt`, the last in its own
  environment because it pins PyTorch 2.12. None of it has CI. See
  [NVIDIA Blackwell](nvidia-blackwell.md).
- `intel:npu` resolves, plans, and compiles through the OpenVINO NPU plugin, but
  **no Intel NPU has ever executed it**. Its integration tests skip unless
  OpenVINO reports an NPU; everything else about it is unit-tested against a
  fake runtime.
- `aws:trainium` parses as a target and is never executed.

## Evaluated, not adopted

These have measurement harnesses or written plans, and no registered backend:

- [AMD MIGraphX](amd-migraphx.md) — benchmark harness only.
- [Qualcomm Hexagon](qualcomm-hexagon.md) — lower-level Hexagon-MLIR evaluation
  plan. The initial [ExecuTorch QNN backend](qnn.md) supports static FP16 SM8750
  export but remains SDK-gated and has no automated hardware validation.
- [torch-mlir lowering](torch-mlir-lowering-evaluation.md) — would unpin
  `stablehlo` from a matching PyTorch; evaluated and **not adopted**.

## Quantization

The `lm7 model run` path is weight-only and validated per (model, mode) pair. It
reaches NVIDIA GPUs and CPU; `int8` is the only mode measured off NVIDIA, and
AMD, Apple, Intel XPU, and TPU have no path at all. Activation quantization is
not implemented here. Two export backends quantize the artifact through their
own unrelated mechanisms — ExecuTorch's calibrated XNNPACK PTQ, and OpenVINO's
NNCF weight compression, the latter validated for two models out of three tried.

Footprint is the reliable benefit, not speed. On both GPUs measured — Ada
(`sm89`) and Blackwell (`sm120`) — every mode came out *slower* than the BF16
baseline once compiled. Blackwell shrinks the penalty a long way without
removing it: `nvfp4` costs 1.24x there against 2.50x on Ada, for the same 2.30x
footprint saving. Its FP4 tensor cores are not what does that, and cannot be —
weight-only quantization unpacks to BF16 inside the kernel and never issues an
FP4 matmul. On CPU, INT8 was at parity for SmolLM2-135M and 2.6x slower for
Llama-3.2-1B, on an AVX2-only part with no VNNI — so the latency result does not
generalize to server CPUs or ARM. `nvfp4` gives the smallest footprint and the
largest accuracy loss, clearing the validation bar for one model out of four
tried — and on a second, equally arbitrary set of four prompts that one model
scores 3/4 rather than 4/4. See [quantization](quantization.md) for which layers
each mode converts and the measurements behind it.
