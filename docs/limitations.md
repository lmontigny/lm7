# Limitations and current scope

What LM7 does not do, stated plainly. Read this before depending on it for
anything that matters.

LM7 is an early, inference-only prototype. Model coverage and compiled-artifact
compatibility are not stable, and CPU and Apple Silicon (MPS) are the only
targets with continuous-integration coverage.

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
- **Arbitrary `nn.Module`, not a curated model zoo.** LM7 does not special-case
  model architectures the way Hugging Face's `optimum` family
  (`optimum-intel`, `optimum-nvidia`, `optimum-neuron`) does for its own
  `transformers`/`diffusers`/`timm`/`sentence-transformers` classes. That
  breadth comes without optimum's per-model-family tuning or production
  track record — see
  [notes/competition.md](../notes/competition.md#hugging-face-optimum) for
  the fuller comparison.

## Compilation and artifacts

- **JIT results are process-local.** A JIT-compiled callable does not outlive
  the process. The backends `lm7.export` accepts are what produce something
  another process can load: `aot_inductor`, `coreml`, `executorch`,
  `iree_vulkan`, `litert`, `onnxruntime`, `openvino`, `qnn`, `stablehlo`,
  `tensorrt`, `tvm`, and the source-only `export` level. Two of those are
  export-path-only — `tensorrt` and `tvm` both have `lm7.compile` backends whose
  compiled result still dies with the process.
- **An artifact is bound to the architecture that built it, on CPU as much as on
  GPU.** `aot_inductor`, `tensorrt` and `tvm` carry a payload compiled for one
  chip, and LM7 refuses to load one built elsewhere rather than letting it fail
  at the driver. That gate covers `cpu` and not only `nvidia`/`amd`: an
  AOTInductor CPU package holds a natively compiled `wrapper.so` — "ELF 64-bit
  LSB shared object, ARM aarch64" when built on an Axion — that no x86-64 host
  can `dlopen`, and TVM's LLVM codegen bakes in the exporting host's target
  triple the same way. CI proves it across two machines: `cross-arch-build`
  exports on `ubuntu-24.04-arm` and `cross-arch-load` gets it refused on x86-64.
  See [artifact
  compatibility](aot-artifact-compatibility.md#cpu-packages-are-architecture-bound-too).
- **Exported causal-LM artifacts are prefill-only unless `--decode` says
  otherwise.** The default capture is a logits-only prefill graph:
  `--dynamic-seq` makes the sequence length variable within recorded bounds, the
  batch dimension stays fixed, and no KV cache is in it. LM7 captures logits
  rather than the model's own output object because `CausalLMOutputWithPast`
  cannot be deserialized by `torch.export.load` — a constraint on the *output*,
  which was for a long time recorded here as the reason a decode loop could not
  be exported. It was not. See [exported decode](exported-decode.md).
- **An exported decode step runs on two backends, and prefills in one call.**
  `lm7 model export --decode` captures a KV-cache decode step that survives the
  process, holding its cache as buffers inside the artifact. The default
  `--decode-shape dynamic` binds the sequence length as a bounded dimension, so
  one graph takes a whole prompt at once and then one token at a time against the
  same cache; `--decode-shape single-token` fixes it at one token, decoding 1.29x
  faster and paying a forward pass per prompt token. There is no separate prefill
  *artifact* and cannot be: each exported program carries its own cache buffers,
  so a second artifact would fill a cache the first never sees. Validated on
  `export` and `aot_inductor` only, on CPU and float32 only, at batch 1. Every
  other export backend is refused rather than guessed at: a backend that drops the
  cache writes during lowering returns a correct first token and then diverges
  without raising.
- **`lm7 artifact generate` is greedy and single-sequence.** It drives a decode
  artifact from the manifest — tokenizer, cache length, tokens per call — and
  offers no sampling, no streaming and no server. Anything else is a caller's
  loop over the logits. An artifact exported before manifests recorded a
  `source` needs `--tokenizer` passed by hand.
- **A decode artifact is stateful, and nothing else LM7 writes is.** Two
  concurrent callers share one cache. `cache_position=0` re-anchors the write
  pointer but does not zero the slots behind it, and whether that leaks has not
  been tested.
- **The JIT decode loop is still the faster and more capable one.**
  [`lm7.compile_generation`](kv-cache-decode.md) takes a whole prompt in one
  call and reaches 1.77 ms/token on an H100; it dies with the process. Neither
  path is the other's superset.
- See [JIT vs. AOT](jit-vs-aot.md) for the export levels, bundles, and the
  signature rules an artifact is pinned to.
- **Saved compiler output is deep for AOTInductor and shallow for every other
  backend.** `lm7.export(..., debug=True)` indexes whatever the selected
  toolchain hands back, and only Inductor hands back more than the exported
  graph. Measured on an RTX 4070 SUPER (Ada `sm89`, WSL2) under `torch
  2.13.0+cu130`, on a 2-layer MLP:

  | backend | files | what they are |
  | --- | --- | --- |
  | `export` (source-only) | 3 | exported program, graph, signature |
  | `openvino` | 3 | those same three — nothing from OpenVINO's own compiler beyond the IR that is already the artifact payload |
  | `aot_inductor`, `cpu` | 11 | those three, plus FX graphs, pre- and post-fusion Inductor IR, `output_code.cpp`, and the `kernel.cpp`/`wrapper.cpp` lifted back out of the `.pt2` package |
  | `aot_inductor`, `nvidia` | 13 | the CPU set plus two `.cubin` device binaries |

  Four things follow from that.

  **PTX and assembly are classified, not produced.** LM7 labels `.ptx`, `.s`,
  `.asm`, `.cubin` and `.hsaco` when the package contains them, so the levels
  are named in the manifest — but the NVIDIA run above emitted `.cubin` only,
  Triton keeping its PTX in its own cache rather than in the package. Read the
  lower levels as best-effort, and check what a given toolchain actually wrote
  rather than assuming the list.

  **Only the export level is proved by a real compile.** The multi-level
  assertions in `tests/test_aot_inductor.py` monkeypatch
  `aoti_compile_and_package` and write the trace files themselves, so what runs
  unmarked on every commit is LM7's indexing, not Inductor's emission. The rows
  above are a hand run, not CI.

  **It is a Python-API option only.** `lm7 model export` has no `--debug`, and
  the JIT path has no LM7 API at all: the `--debug-dir` in
  [`examples/cuda_mlp.py`](../examples/cuda_mlp.py) sets `torch._inductor` trace
  config directly, and needs `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`, because a
  cache hit skips codegen and writes no trace.

  **A debug artifact carries the model's structure.** The files live inside the
  `.lm7` directory under `debug/` and are hashed into the manifest, so an
  artifact built this way ships its graph and generated source to whoever
  receives it. A failed compile discards them unless `LM7_DEBUG_FAILURE_DIR`
  names somewhere to copy them to first.

  See [compiler IR and generated
  code](development.md#compiler-ir-and-generated-code).
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
| Mixtral-8x7B (46.7B) | 5.14.1 | nvidia `sm120` | **1** | **0** | 60.43 ms | 55.62 ms | 1.09x |
| Mixtral, tiny | 5.15.0 | cpu `aarch64` | **1** | **0** | 3.431 ms | 2.291 ms | **1.50x** |
| OLMoE, tiny | 5.15.0 | cpu `aarch64` | **1** | **0** | 4.472 ms | 3.221 ms | **1.39x** |
| OLMoE-1B-7B (6.92B) | 5.15.0 | cpu `aarch64`, BF16 | **1** | **0** | 449.6 ms | 420.7 ms | 1.07x |

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

The last three rows are an Arm Neoverse N3 on transformers 5.15.0, and they say
the same things from a second CPU ISA:

- **Zero graph breaks is not an x86 property.** One graph, no breaks, on both
  tiny architectures and on the 6.92B model — a third ISA and a minor version
  past the 5.14.1 the rows above used.
- **1.07x on the 6.92B model reproduces the 1.03x on x86 CPU.** Two unrelated
  CPUs, two ISAs, the same "compiling a large MoE on a CPU is worth nothing".
  That is the claim that most deserved a second host, because it was the one
  originally over-generalized into a statement about MoE.
- **The speedup shrinks with size on CPU too**, 1.50x and 1.39x on the tiny
  configs against 1.07x at 6.92B, matching the monotonic decline documented on
  `sm120` below.
- **It also shows what the FP32 MLP could not.** [CPU
  inference](cpu.md#latency-on-a-neoverse-n3) finds Inductor worth nothing on
  this part, because that workload is 97–99% GEMM and leaves fusion under 3% to
  win. A tiny MoE is the opposite shape — many small ops — and compiles 1.4–1.5x
  faster on the same machine. Both results are about the workload, not the host.

Two caveats on those rows. The 6.92B one ran at **BF16**, not the FP32 the CPU
row above it used: 6.92B parameters at FP32 is ~28 GB against this host's 31 GB,
which is too close to risk. So its 449.6 ms is not comparable with the 345.0 ms
above it, and BF16 is not a free win on this part anyway — see [the Arm dtype
section](cpu.md#the-same-question-on-arm-where-the-logs-cannot-answer-it). The
ratio is the comparable part. And **Mixtral-8x7B was not run on Arm**: at ~93 GB
BF16 it does not fit in 31 GB, so the largest real MoE this host can hold is
OLMoE-1B-7B.

### The real Mixtral, and what compiling is worth at 46.7B

Every Mixtral claim above and below came from a hand-built two-layer config with
four experts. `mistralai/Mixtral-8x7B-Instruct-v0.1` — 46.7B total, 8 experts,
12.9B active per token — is the real one, and it fits on a 96 GB card at BF16
with room to compile:

| | eager | inductor |
| --- | --- | --- |
| latency | 60.43 ms | 55.62 ms |
| peak VRAM | **93.4 GB** | **93.8 GB** |
| graphs / breaks | 1 / 0 | 1 / 0 |

**Zero graph breaks holds at scale.** The tiny-config result was not an artifact
of the toy: 32 layers and 8 experts capture as one graph on transformers 5.14.1,
1919 ops, same as the 2-layer version.

**Compiling a model that fills the card costs 0.4 GB.** Peak device memory goes
from 93.4 GB eager to 93.8 GB compiled, against 95.0 GiB available — so Inductor
needed under half a gigabyte of headroom on a model occupying 92% of the GPU.
That is worth knowing before assuming a near-full card rules compilation out.

**The speedup shrinks as the model grows**, monotonically across everything
measured on `sm120`:

| model | parameters | inductor speedup |
| --- | --- | --- |
| Mixtral, tiny | ~2M | 3.12x |
| OLMoE-1B-7B | 6.92B | 1.60x |
| Mixtral-8x7B | 46.7B | 1.09x |

Which makes sense, and sharpens the original claim rather than reversing it
again. What Inductor removes is mostly Python and kernel-launch overhead; a
bigger model spends proportionally more of its time inside large GEMMs that cuBLAS
already handles well, so there is less left to remove. "Compiling buys almost
nothing on an MoE" was wrong about the mechanism and wrong to key on MoE, but at
46.7B the *number* it predicted is close to right — 1.09x — for a reason nobody
had stated.

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
| `inductor` | The default and the best-covered path. CPU and Apple Silicon (MPS) are the only targets with CI. |
| `aot_inductor` | Validated for CPU (x86-64 and aarch64), Apple Silicon (MPS), and NVIDIA GPU; uses Beta PyTorch APIs. On NVIDIA it packages against a CUDA toolkit the PyTorch wheel does not ship — install `".[cuda-aot]"`. See the [WSL linker caveat](development.md#nvidia-aot-inductor). Packages hold code compiled for one architecture — kernels for one GPU compute capability, or a native `.so` for one CPU ISA — and refuse to load on another (LM7 raises before loading; PyTorch's own check only warns, then hits a driver error) — see [artifact compatibility](aot-artifact-compatibility.md). |
| `tensorrt` | NVIDIA only. Slower engine builds and narrower model coverage than Inductor — see the [evaluation](nvidia-tensorrt-evaluation.md). `lm7.export` serializes the engine so a second process need not rebuild it; the artifact is static-shape and bound to the GPU architecture, TensorRT version, and Torch-TensorRT version that built it. **Four failure modes do not raise**: an export whose graph falls below the partitioner's `min_block_size` writes a TensorRT-labelled artifact containing no engine; the JIT path returns wrong numbers on BERT; `options={"dynamic": True}` is accepted and ignored by the export path, while the JIT path silently rebuilds an engine per unseen shape; and FP8 arithmetic is unreachable. See [tensorrt-validation.md](tensorrt-validation.md). |
| `openvino` | Any `cpu` target plus `intel:npu`, not Intel silicon only: the aarch64 wheel's CPU plugin loads on an Arm Neoverse N3 and the IR path is *faster* there than on either other host (4.16x over eager on SmolLM2-135M), while its INT8 advantage does not transfer at all — 1.83-2.53x faster than FP32 on Intel, 1.08x slower on Arm. `intel:npu` is **implemented but never run on an NPU**. Rejects bfloat16, because its runtime exchanges tensors through NumPy. Returns tensors or tuples, so a model whose `forward` returns a dataclass needs a wrapper. Optional NNCF INT8 weight compression on both `model run` and `model export`, validated per model. On the NPU: static shapes only, and FP16 compute, so expect FP16-level error. See the [guide](intel-npu.md). |
| `onnxruntime` | CPU and NVIDIA CUDA. Returns CPU tensors even after CUDA execution, because the initial adapter uses NumPy rather than I/O binding. Tensor-only inputs and flat outputs; external-data packaging above the 2 GiB protobuf limit is future work. See the [guide](onnxruntime.md). |
| `iree_vulkan` | Export-only and experimental: fixed shapes, tensor-only I/O, FP32 MLP execution on an RTX 4070 SUPER is the validated scope. `arm` (Mali) targets parse and compile, but **nothing has ever executed on an Arm GPU** — that needs an NDK cross-compile of the IREE runtime, which has no prebuilt Android binary, and this project owns no Mali hardware. Causal LMs, dynamic sequences, KV caches, and WebGPU are future work. See the [guide](iree-vulkan.md). |
| `litert` | Export-only, CPU/XNNPACK only from LM7's side, though the packaged `.tflite` also ran correctly on a real Snapdragon 8 Elite's GPU delegate (Adreno, OpenCL) by hand — **~660x slower per inference than its own CPU delegate** on a 3-layer MLP too small to amortise dispatch and shader compilation; unmeasured on a real model, since dynamic shapes are rejected. See [Android device testing](android-device-testing.md). Static tensor-only inputs, returns CPU tensors. LiteRT Torch caps PyTorch below 2.13, so conversion belongs in a separate environment; on Linux aarch64 even that environment cannot resolve today because `litert-converter` has no matching wheel. Packages generic `.tflite` graphs, not LiteRT-LM conversations. See the [guide](litert.md). |
| `executorch` | Export-only, XNNPACK delegate, so the edge story is CPU. MediaTek and Exynos are not wired up; artifacts are static-shape. Real ARM64 hardware runs in CI (`ubuntu-24.04-arm`), and export has also been checked against a real Snapdragon 8 Elite phone by hand — see [Android device testing](android-device-testing.md). **Calibrated XNNPACK INT8 PTQ completes but is not usable on a causal LM**: SmolLM2-135M quantizes and loads, but its logits move by 39.5 absolute and it shares 0/5 top tokens with eager — see [the measurement](executorch.md#int8-on-a-language-model). LM7 writes the `.pte` — deploying it into an app is ExecuTorch's tooling. See the [guide](executorch.md). |
| `qnn` | Export-only ExecuTorch delegate for Qualcomm Hexagon HTP, `qualcomm:sm8750` only, FP16 only, static-shape, positional-inputs-only. Deployment-only: a `.pte` refuses to run through LM7's host process and needs an Android ExecuTorch runtime built with the QNN backend. Validated on a real Snapdragon 8 Elite device — see [Android device testing](android-device-testing.md). See the [guide](qnn.md). |
| `coreml` | Export-only ExecuTorch delegate for Apple's Core ML, `target="apple"` only, static-shape, positional-inputs-only, macOS-only. Unlike `qnn`, not deployment-only: the `.pte` executes on the Mac that built it through Core ML's ANE/GPU/CPU compute units. No quantization path, no `minimum_deployment_target` control yet, and only an MLP and an embedding+linear model have been tried — not a real causal LM. See the [guide](coreml.md). |
| `stablehlo` | Export-only. Needs PyTorch/XLA to lower, which pins PyTorch to a matching pair. |
| `openxla` | TPU only, single process. SPMD sharding, multi-host execution, and persistent XLA executables are not implemented, and the validated host has one chip, so the sharding paths are untested rather than merely absent. fp32 matmuls run at bf16 precision unless `options={"mat_mul_precision": ...}` says otherwise. It beats eager XLA by 27x on SmolLM2-135M and loses to it by 2.3x on a 3-layer MLP. Generation decodes eagerly and XLA recompiles per decode step, so a first `lm7 model generate` of 20 tokens costs 20 minutes and the second 2.5 s -- the documented per-shape compile cost, paid per decode graph. Warm the process before timing it. See the [guide](google-tpu.md). |
| `tenstorrent` | JIT-only and single-card: the compiled flatbuffer does not outlive the process, multi-card sharding is not exposed, and coverage is bounded by what tt-mlir lowers. See the [guide](tenstorrent.md). |
| `tvm` | CPU-only, positional-inputs-only, and **far slower than Inductor** — registered for reachability, not speed. It does export now: `lm7.export` writes `compiled_model.tvm.so`, which reloads without `torch.export` or the Relax frontend but is bound to the exporting host's CPU architecture and gated on it. Validated on x86-64 and on an Arm Neoverse N3. Autotuning, quantization, dynamic shapes, and CUDA are still not wired up. See the [guide](tvm.md#aot-export). |
| `zentorch` | AMD's ZenDNN extension: CPU-only, JIT-only, explicit-only, x86-64 Linux wheels only, and ABI-tied to a matching PyTorch. Measured on one Zen 3 EPYC at FP32 it beat Inductor on one workload, tied on another, and lost on a third; BF16, INT8, and newer EPYC generations are unmeasured. No quantization path and no artifact. See the [guide](zentorch.md). |

## Hardware validation

- **CPU and Apple Silicon (MPS) are the only targets with CI.** Everything
  else has been exercised manually, or not at all.
- **Apple Silicon runs on real hardware in CI, for two backends.** GitHub's
  `macos-26` runner is arm64, so `tests/test_mac_integration.py` (Inductor and
  AOTInductor through MPS) and the `coreml` export/execute suite both run on an
  actual Apple GPU/ANE on every commit, not a mock. Apple is the only
  accelerator target with that property; everything below is exercised by
  hand at best.
- **CPU CI now spans three OS/architecture combinations**, not just one: Linux
  x86-64 (the original `quality` job), Linux ARM64 (`quality-arm64` on
  `ubuntu-24.04-arm`, which is also where the `executorch` export suite runs on
  real ARM64 hardware), and native Windows (`windows-2025`, with `cl.exe` on
  `PATH` for TorchInductor's C++ codegen — passing, as of the PR that added it).
  The ARM64 leg ran only the ExecuTorch export file until `quality-arm64` was
  added; the portable suite had never run on Linux Arm, which is the one
  combination a Graviton or Axion deployment actually is.
- **A real Android phone (Snapdragon 8 Elite, via a cloud rental) has checked
  three export paths by hand**: `executorch`, `litert` (both its CPU and GPU
  delegates), and `qnn`. None of this is CI — it is a one-time device check,
  not exercised on every commit. See
  [Android device testing](android-device-testing.md).
- AMD ROCm, Intel GPU/XPU, OpenXLA TPU, and Tenstorrent are initial
  single-process integrations **without physical-hardware CI**.
- OpenXLA TPU has now been **exercised on real hardware** — a single-chip TPU
  v6e — which the rest of that list has not. It still has no CI, and one chip
  cannot say anything about multi-chip behaviour. See
  [Google TPU](google-tpu.md).
- NVIDIA Inductor, quantization, and TensorRT have been exercised on three GPU
  generations: a local Ada (`sm89`), a rented Hopper H100 80GB (`sm90`), and a
  Blackwell (`sm120`) RTX PRO 6000. Detection, backend selection, and every
  weight-only mode worked on Blackwell with no code changes, and all three
  NVIDIA compile backends run there unmodified — `inductor`, `aot_inductor`, and
  `tensorrt`, the last in its own environment because it pins PyTorch 2.12. The
  H100 is where the per-row FP8 activation numbers and the batch-1–4096 Inductor
  sweep come from. None of it has CI. See [NVIDIA
  Blackwell](nvidia-blackwell.md) and [NVIDIA H100](nvidia-h100.md).
- `intel:npu` resolves, plans, and compiles through the OpenVINO NPU plugin, but
  **no Intel NPU has ever executed it**. Its integration tests skip unless
  OpenVINO reports an NPU; everything else about it is unit-tested against a
  fake runtime. This is separate from Intel GPU/XPU support: `target="intel"`
  means a PyTorch `torch.xpu` GPU path, while `target="intel:npu"` means the
  dedicated Intel AI Boost NPU.
- `aws:trainium` parses as a target and is never executed.

## Serving

`lm7 model serve` is a **single-user local server**, and the things that make a
serving engine fast are absent from it on purpose — see [serving](serving.md).

- **One request at a time.** `compile_generation` owns one static KV cache that
  every decode step mutates in place, so concurrent generations would corrupt
  each other silently. An `asyncio.Lock` serializes them; a second caller waits.
  No continuous batching, no paged attention, no prefix caching, no chunked
  prefill, no speculative decoding, no LoRA. A request that asks for one of these
  — or for `n > 1`, logprobs, tools, or structured output — is refused with a
  400 naming the field rather than served as something narrower.
- **The KV cache is allocated at startup and never grows.** `prompt + max_tokens`
  above `--max-model-len` is a 400, not a longer wait.
- **`--backend` compiles the decode loop with Inductor or not at all.**
  `auto`, `eager` and `inductor` are LM7's own server; `vllm` and `trtllm` hand
  the port to someone else's. There is no third compiler here, and `auto` is
  checked against what it *selects* rather than against the string, so a target
  whose highest-priority backend is something else — `openxla` on `tpu`, the
  Tenstorrent backend, `openvino` on `intel:npu` — is refused before the
  checkpoint downloads. Only the Inductor backend implements the `warmup: False`
  option that keeps a graph writing into a KV cache from being compiled by
  execution; see [prefill and KV-cache decode](kv-cache-decode.md#limits).
- **CI now loads a real model, but a 15 MB random-weight one.**
  `tests/test_serve_load_integration.py` runs `LM7ServeEngine.load` end to end on
  every commit, which nothing did before — the rest of the serve suite uses a
  scripted runner and a fake tokenizer. It proves the path works, not that
  output is right: the model has random weights, so no test in CI checks that a
  served answer is correct.
- **Validated on five targets and two models.** Apple M-series `cpu:arm64` and
  `apple:metal` with SmolLM2-135M-Instruct; `nvidia:sm89` (RTX 4070 SUPER, WSL2)
  with SmolLM2-135M-Instruct and Llama-3.2-1B-Instruct; `cpu:x86_64` (Intel
  Coffee Lake, AVX2) with SmolLM2-135M-Instruct; and `cpu:aarch64` (Arm Neoverse
  N3, GCP Axion) with SmolLM2-135M-Instruct — driven by `curl` and by the
  official `openai` Python SDK: both endpoints, both buffered and streamed,
  greedy output byte-identical to `model.generate`, `int8` served on all three
  CPU targets, and `fp8` and `nvfp4` on the GPU. **`intel:npu` and `tpu` have served
  nothing**, nothing above 1B has, and there is no serving benchmark in this repo
  — so no claim about serving latency or throughput can be sourced from it.
  `/metrics` TTFT and TPOT are compile-polluted until several requests have run,
  because the graphs compile inside the first one.
- **A CPU target is three different machines, and two spellings.** `cpu:arm64`
  (Apple, macOS), `cpu:x86_64` and `cpu:aarch64` (Linux Arm) are one LM7 target
  family and unrelated vector units, and a serving number does not cross between
  them: INT8 was 2.44x smaller and useful on Apple, the same 2.44x on the
  Neoverse N3 with no speed benefit, and on an AVX2-without-VNNI Intel part it
  also served correctly at no speed benefit at all. The spelling is the trap for
  clients: `platform.machine()` says `arm64` on macOS and `aarch64` on Linux, so
  the same Arm family reaches `/health` and `/metrics` under two different target
  strings and a client that string-matches `cpu:arm64` will not match a Linux Arm
  server. `--dtype auto` also means FP32 on all three, so the same
  `--max-model-len` buys a KV cache twice the size of the FP16 one a GPU
  allocates.
- **A quantized serve was silently wrong on NVIDIA until it was run there.**
  `LM7ServeEngine.load` resolved `--dtype auto` without passing the quantization
  mode, so INT8 and FP8 weights were served under FP16 compute instead of BF16;
  the logits were NaN and the endpoint returned HTTP 200 with an empty string.
  The failure is invisible on CPU, where both dtypes resolve to FP32 — which is
  the general shape of this file: a path validated on one target is not
  validated. See [serving](serving.md#on-nvidia-rtx-4070-super-ada-sm89).
- **`--backend vllm` is validated on Apple Silicon and CUDA.** LM7 translates its
  flags into vLLM's own `vllm serve` argv and hands over the process. That
  handover was run end to end on an M-series Mac against `Qwen/Qwen3.5-0.8B`
  through the [vllm-metal](https://github.com/vllm-project/vllm-metal) platform
  plugin — vLLM 0.26.0 + vllm-metal 0.3.0, chat, streaming, and the `openai` SDK
  — and on an RTX 4070 SUPER (`sm89`, WSL2) against Llama-3.2-1B-Instruct with
  vLLM 0.26.0. **The ROCm and TPU handovers have still never been run**; for
  those, say "implemented", not "validated". vLLM's own supported-model list
  applies once LM7 has handed over, and it is narrower than LM7's —
  `SmolLM2-135M` is not on vllm-metal's, for instance.
- **A handover that starts is not a handover that is tuned.** CUDA needed two
  LM7 fixes before it would start at all (`VLLM_WSL2_ENABLE_PIN_MEMORY` and
  `--vllm-arg`, both in [serving](serving.md#--backend-vllm-hand-over-the-port)),
  plus one workaround LM7 does not own: FlashInfer's startup JIT wants `nvcc` and
  `ninja` and then rejects CUDA 13.3, so that box sets
  `VLLM_USE_FLASHINFER_SAMPLER=0` in the environment. Nothing about vLLM's
  throughput has been measured here, on any platform. The same is true of
  TensorRT-LLM below: both launchers now start on this card, and neither has had
  its throughput measured.
- **`--backend trtllm` is one model on one card.** The handover was run end to
  end on an RTX 4070 SUPER (Ada `sm89`, 12 GiB) under WSL2 against TensorRT-LLM
  1.2.1 and `SmolLM2-135M-Instruct` — server up, chat, streaming, four
  integration tests, and `--trtllm-arg=--free_gpu_memory_fraction` cutting the
  paged cache from 9.06 GiB to 2.52 GiB. Only `--host`, `--port` and
  `--max-model-len` are *modelled*; everything else passes through verbatim, so
  LM7 makes no claim about what it does. Multi-GPU, quantized checkpoints, and
  anything above 135M are unrun. See [TensorRT-LLM](tensorrt-llm.md).
- **The Inductor comparison exists now, and it is one model on one card.**
  [`benchmarks/serving_backends.py`](../benchmarks/serving_backends.py) drives
  LM7's own server, the same server with CUDA Graphs, and the TensorRT-LLM
  handover from one client over the same HTTP. On SmolLM2-135M at `sm89`,
  TensorRT-LLM decodes 1.21x faster per token than `--compile-mode
  reduce-overhead` — **not** the 2.9x it beats plain Inductor by — while LM7
  answers the first token 3.9x sooner and holds 17x less GPU memory; past one
  stream LM7 is flat by design and TensorRT-LLM reaches 8x its aggregate at
  eight. **What that does not cover**: a launch-bound 135M model is the case most
  favourable to a compiled decode loop, and all eight streams share one prompt,
  one length and one arrival instant, which is the easiest batch a scheduler can
  be given. No mixed-length queue, no arrival spread, nothing larger, no second
  card. These are loopback wall-clock indicators; there is still no serving
  benchmark in this repo.

## Evaluated, not adopted

These have measurement harnesses or written plans, and no registered backend:

- [AMD MIGraphX](amd-migraphx.md) — benchmark harness only.
- [Qualcomm Hexagon](qualcomm-hexagon.md) — lower-level Hexagon-MLIR evaluation
  plan. The initial [ExecuTorch QNN backend](qnn.md) supports static FP16 SM8750
  export but remains SDK-gated and has no automated hardware validation.
- [torch-mlir lowering](torch-mlir-lowering-evaluation.md) — would unpin
  `stablehlo` from a matching PyTorch; evaluated and **not adopted**.
- [RISC-V](riscv.md) — `cpu:riscv64` parses and round-trips, and that is the
  whole of it: no PyTorch installs on a RISC-V host today, so there is nothing
  for a backend to wrap. Nothing has run on RISC-V hardware.

## Quantization

The `lm7 model run` path is validated per (model, mode) pair. It reaches NVIDIA
GPUs and CPU; `int8` is the only mode measured off NVIDIA, and AMD, Apple, Intel
XPU, and TPU have no path at all. Two export backends quantize the artifact
through their own unrelated mechanisms — ExecuTorch's calibrated XNNPACK PTQ, and
OpenVINO's NNCF weight compression, the latter validated for two models out of
three tried.

**Activation quantization exists but is narrow.** `fp8-dynamic` and
`fp8-dynamic-rowwise` (`sm89`+) and `nvfp4-dynamic` (`sm100`+) quantize
activations at runtime so the matmul executes in the narrow format, confirmed by
the emitted kernels rather than inferred. Scaling is dynamic only — there is no
static calibration path — and the admitted set is four (model, mode) pairs, all
FP8: `Llama-3.2-1B` and `Llama-3.1-8B`, each with both per-tensor and per-row
scaling, the last three added by an H100 (`sm90`) run. **`nvfp4-dynamic` is
implemented and admitted for nothing** — it scored 3/4 top-1 on the one model it
was measured against, which is where 4-bit activations on top of 4-bit weights
stop holding the token. Only per-row FP8 on the 1B is actually faster than not
quantizing (0.94x); the other three are admitted on accuracy while costing
latency. TorchAO's fused Triton scaling kernel for NVFP4 needs MSLK, which is not
installable from PyPI, so LM7 runs the torch fallback and reports which one it
used.

Footprint is the reliable benefit for the weight-only modes, not speed. On Ada
(`sm89`) and Blackwell (`sm120`) every weight-only mode came out *slower* than
the BF16 baseline once compiled. Blackwell shrinks the penalty a long way without
removing it: `nvfp4` costs 1.24x there against 2.50x on Ada, for the same 2.30x
footprint saving. **A third card broke the "always slower" half of that**:
on an H100 (`sm90`), weight-only `fp8` on Llama-3.2-1B runs at 0.97x, the first
weight-only mode measured here to beat BF16 at all. It does not carry: on the
same card `int8` is 4.94x slower, `nvfp4` 1.64x, and `fp8` on Llama-3.1-8B is
1.89x slower *and* rejected at 3/4 top-1. So the honest form of the claim is that
weight-only latency is a property of (card, model, mode) and has been a loss in
every combination but one. Its FP4 tensor cores are not what does that, and cannot be —
weight-only quantization unpacks to BF16 inside the kernel and never issues an
FP4 matmul. On CPU, INT8 was at parity for SmolLM2-135M and 2.6x slower for
Llama-3.2-1B, on an AVX2-only part with no VNNI — so the latency result does not
generalize to server CPUs or ARM. `nvfp4` gives the smallest footprint and the
largest accuracy loss, clearing the validation bar for one model out of five
tried — and on a second, equally arbitrary set of four prompts that one model
scores 3/4 rather than 4/4. The one model large enough to matter,
Llama-3.1-8B, is measured on CPU and Blackwell GPU for INT8 and rejects both
narrower modes. See [quantization](quantization.md) for which layers each mode
converts and the measurements behind it.
