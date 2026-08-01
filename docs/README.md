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

## Hardware setup

| Target | Document |
| --- | --- |
| CPU | [cpu.md](cpu.md) |
| NVIDIA GPU | [development.md#nvidia-cuda](development.md#nvidia-cuda) |
| AMD GPU (ROCm) | [amd-rocm.md](amd-rocm.md) |
| Apple Silicon (MPS) | [apple-mps.md](apple-mps.md) |
| Intel NPU (Core Ultra) | [intel-npu.md](intel-npu.md) |
| Google TPU | [google-tpu.md](google-tpu.md) |
| Tenstorrent | [tenstorrent.md](tenstorrent.md) |

## Backends

| Backend | Document |
| --- | --- |
| `onnxruntime` | [onnxruntime.md](onnxruntime.md) |
| `openvino` | [openvino-evaluation.md](openvino-evaluation.md), [intel-npu.md](intel-npu.md) |
| `iree_vulkan` | [iree-vulkan.md](iree-vulkan.md) |
| `litert` | [litert.md](litert.md) |
| `executorch` (Android, iOS, embedded) | [executorch.md](executorch.md) |
| `tenstorrent` | [tenstorrent.md](tenstorrent.md) |
| `tvm` | [tvm.md](tvm.md) |
| `zentorch` | [zentorch.md](zentorch.md) |
| `stablehlo` | [stablehlo-pjrt-evaluation.md](stablehlo-pjrt-evaluation.md) |

## Features

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
- [Qualcomm Hexagon](qualcomm-hexagon.md) — plan, blocked on SDK and device access.
- [StableHLO and PJRT](stablehlo-pjrt-evaluation.md) — became the `stablehlo` backend.
- [torch-mlir lowering](torch-mlir-lowering-evaluation.md) — evaluated, not adopted.

## Reference

- [Device list](device_list.md) — the wider AI hardware and compiler landscape.
- [Architecture details](architecture_details.md) — long-form design notes.
