# Tested hardware

[Supported hardware](../README.md#supported-hardware) is what the vendor
toolchains allow; this is what has actually run on physical hardware. CPU and
Apple Silicon (MPS) are the only targets with CI — MPS on GitHub's `macos-26`
arm64 runner; everything else below was exercised by hand, once, on one part
of its kind, unless its row says otherwise.

| Target | Hardware | Exercised |
| --- | --- | --- |
| `nvidia:sm89` | RTX 4070 SUPER (Ada, 12 GiB) | Primary dev GPU — Inductor, TensorRT, ONNX Runtime, IREE Vulkan, StableHLO, quantization. See [TensorRT](nvidia-tensorrt-evaluation.md). |
| `nvidia:sm90` | H100 80GB HBM3 (Hopper), single card, cloud-rented | The first datacenter part, and the first NVIDIA GPU here that is neither a gaming nor a workstation card. Detection and backend selection unmodified; Inductor measured against eager on two causal LMs across batch 1–4096, plus vision/encoder/hand-written workloads. Peak VRAM never exceeds 4.8% of the card, and the Inductor speedup inverts past batch 1024. Also carries the FP8 per-row activation quantization measurements. See [NVIDIA H100](nvidia-h100.md) and [quantization](quantization.md#fp8-granularity-on-h100). |
| `nvidia:sm120` | RTX PRO 6000 Blackwell Server Edition (96 GiB) | All three NVIDIA compile backends, unmodified, plus a 106-cell TensorRT sweep across four model families, precisions, batch sizes and dynamic shapes. See [NVIDIA Blackwell](nvidia-blackwell.md) and [TensorRT validation](tensorrt-validation.md). |
| `cpu` (AMD) | AMD EPYC 7B13 (Zen 3) | `zentorch` and CPU baselines. See [AMD CPU](amd-cpu.md). |
| `cpu:aarch64` (Arm) | Arm Neoverse N2 (Azure Cobalt 100, 4 vCPU) | GitHub's `ubuntu-24.04-arm` runner, so this one is CI rather than a hand run: the ExecuTorch export suite on every commit, plus detection — `bf16`, `i8mm`, `sve2` reported, and the core named from its part number. A shared 4-vCPU runner is not a machine to time anything on; the N3 row below is. See [CPU inference](cpu.md#on-aarch64-the-kernel-prints-less). |
| `cpu:aarch64` (Arm) | Arm Neoverse N3 (GCP `n4a-standard-8`, Google Axion, 8 vCPU, 31 GiB) | Cloud-rented, and the first Linux Arm host here with measured latency: eager against Inductor on the FP32 MLP across batch 1–512, where **compiling wins nothing at any batch size** — the workload is 97–99% GEMM, so fusion has under 3% to play with. Detection names the part unmodified. Compute dtype is still FP32, so the `bf16`/`i8mm` question stays open. See [CPU inference](cpu.md#latency-on-a-neoverse-n3). |
| `cpu`, `apple` (Apple Silicon) | M3 Pro, M4, M4 Pro | TVM, MPS compile (also in CI), an OpenVINO cross-check, ExecuTorch Core ML export and execution, `lm7 model serve` on both `cpu` and `apple`, and the `--backend vllm` handover through the vllm-metal plugin. See [Apple Silicon](apple-mps.md), [Core ML](coreml.md), [serving](serving.md). |
| `tpu` | TPU v6e (Trillium), single chip | See [Google TPU](google-tpu.md). |
| `qualcomm:sm8750` | Snapdragon 8 Elite, physical device (cloud-rented) | ExecuTorch export and QNN. See [Android device testing](android-device-testing.md). |

AMD ROCm GPU, Intel XPU, Tenstorrent, the Intel NPU, and AWS Trainium have not
run on real hardware yet — those adapters are unit-tested against mocks. See
[hardware validation](limitations.md#hardware-validation) for the exact gaps.
