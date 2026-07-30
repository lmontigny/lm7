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
- **Sparse Mixture-of-Experts models compile, but do not export.** The
  reference transformers implementation (e.g. Mixtral) routes tokens to
  experts with a data-dependent Python loop (`for expert_idx in expert_hit:
  ...`) — the number of iterations is only known at runtime. `torch.compile`
  tolerates that: Dynamo graph-breaks around the loop and falls back to eager
  for it, so `inductor` and `tensorrt` (both JIT, both Dynamo-based) compile
  and run these models correctly. `torch.export` cannot: it must capture one
  static graph and hard-fails with `GuardOnDataDependentSymNode` on the same
  loop. Every export-based backend — `aot_inductor`, `openvino`,
  `onnxruntime`, `executorch`, `iree_vulkan`, `litert`, `stablehlo` — calls
  `torch.export` internally, so none of them support this architecture today.
  See [`examples/sparse_moe.py`](../examples/sparse_moe.py) for a compiling
  example on CPU and NVIDIA.

## Per-backend scope

| Backend | Scope and caveats |
| --- | --- |
| `inductor` | The default and the best-covered path. CPU is the only target with CI. |
| `aot_inductor` | Validated for CPU, Apple Silicon (MPS), and NVIDIA GPU; uses Beta PyTorch APIs. On NVIDIA it packages against a CUDA toolkit the PyTorch wheel does not ship — install `".[cuda-aot]"`. See the [WSL linker caveat](development.md#nvidia-aot-inductor). |
| `tensorrt` | NVIDIA only. Slower engine builds and narrower model coverage than Inductor — see the [evaluation](nvidia-tensorrt-evaluation.md). `lm7.export` serializes the engine so a second process need not rebuild it; the artifact is static-shape and bound to the GPU architecture, TensorRT version, and Torch-TensorRT version that built it. |
| `openvino` | Intel CPU, plus `intel:npu` — **implemented but never run on an NPU**. Rejects bfloat16, because its runtime exchanges tensors through NumPy. Returns tensors or tuples, so a model whose `forward` returns a dataclass needs a wrapper. Optional NNCF INT8 weight compression on export, validated per model. On the NPU: static shapes only, and FP16 compute, so expect FP16-level error. See the [guide](intel-npu.md). |
| `onnxruntime` | CPU and NVIDIA CUDA. Returns CPU tensors even after CUDA execution, because the initial adapter uses NumPy rather than I/O binding. Tensor-only inputs and flat outputs; external-data packaging above the 2 GiB protobuf limit is future work. See the [guide](onnxruntime.md). |
| `iree_vulkan` | Export-only and experimental: fixed shapes, tensor-only I/O, FP32 MLP execution is the validated scope. Causal LMs, dynamic sequences, KV caches, and WebGPU are future work. See the [guide](iree-vulkan.md). |
| `litert` | Export-only, CPU/XNNPACK only. Static tensor-only inputs, returns CPU tensors. LiteRT Torch caps PyTorch below 2.13, so conversion belongs in a separate environment. Packages generic `.tflite` graphs, not LiteRT-LM conversations. See the [guide](litert.md). |
| `executorch` | Export-only and XNNPACK-only, so the edge story is CPU. Phone NPUs (Core ML, Qualcomm QNN, MediaTek, Exynos) are not wired up; artifacts are static-shape; optional calibrated XNNPACK INT8 PTQ is available; validation is host x86-64, not a physical phone. LM7 writes the `.pte` — deploying it into an app is ExecuTorch's tooling. See the [guide](executorch.md). |
| `stablehlo` | Export-only. Needs PyTorch/XLA to lower, which pins PyTorch to a matching pair. |
| `openxla` | TPU only, single process. SPMD sharding, multi-host execution, and persistent XLA executables are not implemented. |
| `tenstorrent` | JIT-only and single-card: the compiled flatbuffer does not outlive the process, multi-card sharding is not exposed, and coverage is bounded by what tt-mlir lowers. See the [guide](tenstorrent.md). |
| `tvm` | CPU-only, JIT-only, positional-inputs-only, and **far slower than Inductor** — registered for reachability, not speed. Autotuning, CUDA, and artifacts are not wired up. See the [guide](tvm.md). |

## Hardware validation

- **CPU is the only target with CI.** Everything else has been exercised
  manually, or not at all.
- AMD ROCm, Apple Silicon (MPS), Intel XPU, OpenXLA TPU, and Tenstorrent are
  initial single-process integrations **without physical-hardware CI**.
- NVIDIA Inductor, quantization, and TensorRT have been exercised on a local Ada
  (`sm89`) GPU only.
- `intel:npu` resolves, plans, and compiles through the OpenVINO NPU plugin, but
  **no Intel NPU has ever executed it**. Its integration tests skip unless
  OpenVINO reports an NPU; everything else about it is unit-tested against a
  fake runtime.
- `aws:trainium` parses as a target and is never executed.

## Evaluated, not adopted

These have measurement harnesses or written plans, and no registered backend:

- [AMD MIGraphX](amd-migraphx.md) — benchmark harness only.
- [Qualcomm Hexagon](qualcomm-hexagon.md) — evaluation plan; blocked on SDK and
  device access. ExecuTorch's QNN delegate is likely the cheaper route.
- [torch-mlir lowering](torch-mlir-lowering-evaluation.md) — would unpin
  `stablehlo` from a matching PyTorch; evaluated and **not adopted**.

## Quantization

The `lm7 model run` path is weight-only and validated per (model, mode) pair. It
reaches NVIDIA GPUs and CPU; `int8` is the only mode measured off NVIDIA, and
AMD, Apple, Intel XPU, and TPU have no path at all. Activation quantization is
not implemented here. Two export backends quantize the artifact through their
own unrelated mechanisms — ExecuTorch's calibrated XNNPACK PTQ, and OpenVINO's
NNCF weight compression, the latter validated for two models out of three tried.

Footprint is the reliable benefit, not speed. On sm89 every mode measured
*slower* than the BF16 baseline once compiled. On CPU, INT8 was at parity for
SmolLM2-135M and 2.6x slower for Llama-3.2-1B, on an AVX2-only part with no
VNNI — so the latency result does not generalize to server CPUs or ARM.
`nvfp4` gives the smallest footprint and the largest accuracy loss, clearing the
validation bar for one model out of four tried. See
[quantization](quantization.md) for which layers each mode converts and the
measurements behind it.
