# LM7 documentation

Start with the [README](../README.md) for install and the six common tasks.
This index covers everything else.

## Start here

| Document | What it covers |
| --- | --- |
| [Limitations](limitations.md) | What LM7 does not do, per backend and overall. Read before depending on it. |
| [Personal AI hardware](personal-ai-hardware.md) | How LM7 fits local assistants, edge boxes, phones, AI PCs, and new personal-device accelerators. |
| [Tested hardware](tested-hardware.md) | The physical machines LM7 has actually run on, versus the supported-hardware matrix. |
| [Architecture](architecture.md) | Targets, backends, the planner, and artifact design. |
| [JIT vs. AOT](jit-vs-aot.md) | When compilation happens, the two export levels, bundles, and signature rules. |
| [What LM7 replaces](what-this-replaces.md) | The per-vendor code you would otherwise write yourself. |
| [Development and testing](development.md) | Environment checks, GPU integration tests, compiler IR output. |
| [Hardware validation procedure](hardware-validation.md) | Reproducible checklist for adding a physical machine to tested hardware, CPU or GPU. |
| [Merged work](changelog.md) | One line per merged pull request, grouped by area. |
| [Android device testing](android-device-testing.md) | Running an exported artifact on a real phone and checking it against the host. |

## Hardware setup

| Target | Document |
| --- | --- |
| CPU | [cpu.md](cpu.md) |
| AMD CPU (EPYC, Ryzen) | [amd-cpu.md](amd-cpu.md) |
| Arm CPU (Neoverse Linux servers) | [arm-cpu.md](arm-cpu.md) |
| NVIDIA GPU | [development.md#nvidia-cuda](development.md#nvidia-cuda) |
| NVIDIA Hopper (`sm90`, H100) | [nvidia-h100.md](nvidia-h100.md) |
| NVIDIA Blackwell (`sm100`, `sm120`) | [nvidia-blackwell.md](nvidia-blackwell.md) |
| NVIDIA validation suite (any card) | [nvidia-validation.md](nvidia-validation.md) |
| AMD GPU (ROCm) | [amd-rocm.md](amd-rocm.md) |
| AMD MI300X (`gfx942`, CDNA 3) | [amd-mi300x.md](amd-mi300x.md) |
| Apple Silicon (MPS) | [apple-mps.md](apple-mps.md) |
| Intel GPU/XPU vs. Intel NPU (Core Ultra) | [intel-npu.md](intel-npu.md) |
| Google TPU | [google-tpu.md](google-tpu.md) |
| Tenstorrent | [tenstorrent.md](tenstorrent.md) |
| Qualcomm Snapdragon 8 Elite HTP | [qnn.md](qnn.md) |
| Personal AI and edge devices | [personal-ai-hardware.md](personal-ai-hardware.md) |

## Backends

| Backend | Document |
| --- | --- |
| `onnxruntime` | [onnxruntime.md](onnxruntime.md) |
| `openvino` | [openvino-evaluation.md](openvino-evaluation.md), [intel-npu.md](intel-npu.md) |
| `iree_vulkan` | [iree-vulkan.md](iree-vulkan.md) |
| `litert` | [litert.md](litert.md) |
| `executorch` (Android, iOS, embedded) | [executorch.md](executorch.md) |
| `coreml` (Apple ANE/GPU/CPU) | [coreml.md](coreml.md) |
| `qnn` (Snapdragon HTP) | [qnn.md](qnn.md) |
| `tenstorrent` | [tenstorrent.md](tenstorrent.md) |
| `tvm` | [tvm.md](tvm.md) |
| `zentorch` | [zentorch.md](zentorch.md) |
| `stablehlo` | [stablehlo-pjrt-evaluation.md](stablehlo-pjrt-evaluation.md) |

## Features

- [Hexagon-MLIR toolchain diagnostics](hexagon-toolchain-diagnostics.md) --
  actionable compile, simulator, and device preflight without opening a connection.
- [Artifact inspection](artifact-inspection.md) -- verify payload integrity,
  portability, delegate coverage, and deployment requirements without loading
  an optional runtime.
- [IR inspection](ir-inspection.md) -- retain and read exported graphs,
  Inductor IR, generated host code, and target-specific device code, with a
  measured CPU/NVIDIA `sm89` comparison of where the two stacks diverge.
- [AOTInductor artifact compatibility](aot-artifact-compatibility.md) -- what a
  packaged artifact costs to reload in a process that never compiled it, what it
  refuses to load on, and which PyTorch differences it survives.
- [Hugging Face model compatibility](model-compatibility.md) -- config-only
  preflight for model type, workflows, target/backend selection, and quantization.
- [TorchInductor options](inductor-options.md) — compile modes, CUDA Graphs,
  individual backend controls, and benchmarking guidance.
- [Quantization](quantization.md) — weight-only modes, validation gates, caveats.
- [DeepSeek coverage](deepseek.md) — one model measured across every locally
  installable backend.
- [Compiled Hugging Face generation](huggingface-generation.md) — the static
  KV-cache decode path, as `lm7 model generate` drives it through Transformers.
- [Serving](serving.md) — `lm7 model serve`: an OpenAI-compatible HTTP endpoint and a built-in chat page
  over the compiled decode loop. Single-stream by design, with `--backend vllm`
  and `--backend trtllm` as the handovers when throughput matters.
- [TensorRT-LLM](tensorrt-llm.md) — `--backend trtllm`, why it is a launcher
  rather than an in-process runtime, and what that cost to find out. Measured
  against the Inductor path on an RTX 4070 SUPER: better alone, 8x by eight
  streams.
- [Prefill and KV-cache decode](kv-cache-decode.md) — `lm7.compile_generation`:
  two separately compiled graphs, one device-resident cache, and per-phase
  compile counters. Measured on an H100.
- [Exported KV-cache decode](exported-decode.md) — `lm7 model export --decode`:
  the AOT counterpart, a decode step that outlives the process by carrying its
  cache as buffers. Why the blocker was never the one recorded, why it is
  captured strictly, and why only two backends are allowed to try.

## Evaluations

Measured investigations, including the ones that did not become backends.

- [NVIDIA TensorRT](nvidia-tensorrt-evaluation.md) — measured against Inductor.
- [TensorRT validation on Blackwell](tensorrt-validation.md) — four model
  families, four precisions, batch and sequence sweeps, and four ways the backend
  fails without raising.
- [AMD MIGraphX](amd-migraphx.md) — measured on an MI300X, not adopted.
- [Qualcomm Hexagon](qualcomm-hexagon.md) — lower-level Hexagon-MLIR plan; the QNN export path is documented in [qnn.md](qnn.md).
- [StableHLO and PJRT](stablehlo-pjrt-evaluation.md) — became the `stablehlo` backend.
- [torch-mlir lowering](torch-mlir-lowering-evaluation.md) — evaluated, not adopted.
- [RISC-V](riscv.md) — nothing to build yet; blocked on PyTorch, not on LM7.

## Reference

- [Device list](device_list.md) — the wider AI hardware and compiler landscape.
- [Architecture details](architecture_details.md) — long-form design notes.
