# Tested hardware

[Supported hardware](../README.md#supported-hardware) is what the vendor
toolchains allow; this is what has actually run on physical hardware. CPU and
Apple Silicon (MPS) are the only targets with CI — MPS on GitHub's `macos-26`
arm64 runner; everything else below was exercised by hand, once, on one part
of its kind.

| Target | Hardware | Exercised |
| --- | --- | --- |
| `nvidia:sm89` | RTX 4070 SUPER (Ada, 12 GiB) | Primary dev GPU — Inductor, TensorRT, ONNX Runtime, IREE Vulkan, StableHLO, quantization. See [TensorRT](nvidia-tensorrt-evaluation.md). |
| `nvidia:sm120` | RTX PRO 6000 Blackwell Server Edition (96 GiB) | All three NVIDIA compile backends, unmodified, plus a 106-cell TensorRT sweep across four model families, precisions, batch sizes and dynamic shapes. See [NVIDIA Blackwell](nvidia-blackwell.md) and [TensorRT validation](tensorrt-validation.md). |
| `cpu` (AMD) | AMD EPYC 7B13 (Zen 3) | `zentorch` and CPU baselines. See [AMD CPU](amd-cpu.md). |
| `cpu`, `apple` (Apple Silicon) | M3 Pro, M4, M4 Pro | TVM, MPS compile (also in CI), an OpenVINO cross-check, ExecuTorch Core ML export and execution. See [Apple Silicon](apple-mps.md), [Core ML](coreml.md). |
| `tpu` | TPU v6e (Trillium), single chip | See [Google TPU](google-tpu.md). |
| `qualcomm:sm8750` | Snapdragon 8 Elite, physical device (cloud-rented) | ExecuTorch export and QNN. See [Android device testing](android-device-testing.md). |

AMD ROCm GPU, Intel XPU, Tenstorrent, the Intel NPU, and AWS Trainium have not
run on real hardware yet — those adapters are unit-tested against mocks. See
[hardware validation](limitations.md#hardware-validation) for the exact gaps.
