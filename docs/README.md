# LM7 documentation

Start with the [README](../README.md) for install and the five common tasks.
This index covers everything else.

## Start here

| Document | What it covers |
| --- | --- |
| [Limitations](limitations.md) | What LM7 does not do, per backend and overall. Read before depending on it. |
| [Architecture](architecture.md) | Targets, backends, the planner, and artifact design. |
| [JIT vs. AOT](jit-vs-aot.md) | When compilation happens, the two export levels, bundles, and signature rules. |
| [What LM7 replaces](what-this-replaces.md) | The per-vendor code you would otherwise write yourself. |
| [Development and testing](development.md) | Environment checks, GPU integration tests, compiler IR output. |
| [Merged work](changelog.md) | One line per merged pull request, grouped by area. |
| [Android device testing](android-device-testing.md) | Running an exported artifact on a real phone and checking it against the host. |

## Hardware setup

| Target | Document |
| --- | --- |
| CPU | [cpu.md](cpu.md) |
| AMD CPU (EPYC, Ryzen) | [amd-cpu.md](amd-cpu.md) |
| NVIDIA GPU | [development.md#nvidia-cuda](development.md#nvidia-cuda) |
| AMD GPU (ROCm) | [amd-rocm.md](amd-rocm.md) |
| Apple Silicon (MPS) | [apple-mps.md](apple-mps.md) |
| Intel NPU (Core Ultra) | [intel-npu.md](intel-npu.md) |
| Google TPU | [google-tpu.md](google-tpu.md) |
| Tenstorrent | [tenstorrent.md](tenstorrent.md) |
| Qualcomm Snapdragon 8 Elite HTP | [qnn.md](qnn.md) |

## Backends

| Backend | Document |
| --- | --- |
| `onnxruntime` | [onnxruntime.md](onnxruntime.md) |
| `openvino` | [openvino-evaluation.md](openvino-evaluation.md), [intel-npu.md](intel-npu.md) |
| `iree_vulkan` | [iree-vulkan.md](iree-vulkan.md) |
| `litert` | [litert.md](litert.md) |
| `executorch` (Android, iOS, embedded) | [executorch.md](executorch.md) |
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
- [Hugging Face model compatibility](model-compatibility.md) -- config-only
  preflight for model type, workflows, target/backend selection, and quantization.
- [TorchInductor options](inductor-options.md) — compile modes, CUDA Graphs,
  individual backend controls, and benchmarking guidance.
- [Quantization](quantization.md) — weight-only modes, validation gates, caveats.
- [DeepSeek coverage](deepseek.md) — one model measured across every locally
  installable backend.
- [Compiled Hugging Face generation](huggingface-generation.md) — the static
  KV-cache decode path.

## Evaluations

Measured investigations, including the ones that did not become backends.

- [NVIDIA TensorRT](nvidia-tensorrt-evaluation.md) — measured against Inductor.
- [AMD MIGraphX](amd-migraphx.md) — harness, no backend.
- [Qualcomm Hexagon](qualcomm-hexagon.md) — lower-level Hexagon-MLIR plan; the QNN export path is documented in [qnn.md](qnn.md).
- [StableHLO and PJRT](stablehlo-pjrt-evaluation.md) — became the `stablehlo` backend.
- [torch-mlir lowering](torch-mlir-lowering-evaluation.md) — evaluated, not adopted.
- [RISC-V](riscv.md) — nothing to build yet; blocked on PyTorch, not on LM7.

## Reference

- [Device list](device_list.md) — the wider AI hardware and compiler landscape.
- [Architecture details](architecture_details.md) — long-form design notes.
