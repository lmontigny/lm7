# AI Inference Hardware & Compiler Ecosystem

This document summarizes the major AI inference hardware platforms and their compiler/runtime stacks from the perspective of supporting a PyTorch-based compiler.

## Datacenter Accelerators

| Vendor | Hardware | Software Stack | Open Source | Compiler / Lowering | PyTorch Support |
|---------|----------|----------------|-------------|---------------------|-----------------|
| NVIDIA | H100, H200, B200, RTX | CUDA, cuDNN, TensorRT | ❌ | NVCC, Triton, TensorRT | Native |
| AMD | Instinct MI300X, MI350 | ROCm | Mostly ✅ | HIP-Clang (LLVM), MIGraphX | Native |
| Google | TPU v4, v5e, v6 | TPU Runtime | Mostly ❌ | XLA | torch_xla |
| Intel | Gaudi 2, Gaudi 3 | SynapseAI, oneAPI, OpenVINO | Mostly ✅ | SynapseAI Compiler | Native |
| AWS | Inferentia, Inferentia2 | Neuron SDK | ❌ | Neuron Compiler (XLA-based) | torch-neuronx |
| Groq | LPU | GroqWare | ❌ | Groq Compiler | Limited |
| Cerebras | WSE-3 | Cerebras Software Platform | Partial | Cerebras Graph Compiler | Native SDK |
| Graphcore | Bow IPU | Poplar SDK | ❌ | Poplar Compiler | PopTorch |
| SambaNova | RDU | SambaFlow | ❌ | SambaFlow Compiler | Native SDK |
| Tenstorrent | Wormhole, Blackhole | TT-Metal, TT-NN | Mostly ✅ | TT-Metal Compiler (MLIR) | Native SDK |
| Huawei | Ascend 310, 910 | CANN | Partial | GE + TBE Compiler | torch_npu |
| Microsoft | Maia 100 | Azure AI Stack | ❌ | Internal MLIR/XLA Compiler | Internal |
| Meta | MTIA | Internal | ❌ | Internal Compiler | Internal |
| Tesla | FSD AI Chip | Internal | ❌ | Internal Compiler | Internal |
| Broadcom | Custom AI ASIC | Customer-specific | Varies | Customer Compiler | Customer |
| Marvell | Custom AI ASIC | Customer-specific | Varies | Customer Compiler | Customer |

---

## Edge / Embedded / Robotics

| Vendor | Hardware | Typical Devices | Software Stack | Open Source | Compiler | PyTorch Support |
|---------|----------|-----------------|----------------|-------------|----------|-----------------|
| NVIDIA | Jetson Orin, Thor | Robotics, Edge AI | CUDA, TensorRT | ❌ | TensorRT | Native |
| Qualcomm | Cloud AI 100, Robotics RB5 | Edge servers, Robotics | Qualcomm AI Stack | Partial | AI Compiler | ONNX / PyTorch |
| Intel | Core Ultra NPU, Movidius VPU | PC, Edge | OpenVINO | Mostly ✅ | OpenVINO Compiler | Native |
| Hailo | Hailo-8, Hailo-10 | Edge AI | Hailo SDK | ❌ | Dataflow Compiler | ONNX |
| Axelera AI | Metis AIPU | Vision Edge | Voyager SDK | ❌ | Voyager Compiler | ONNX |
| Kinara | Ara-2 | Industrial Edge | Ara SDK | ❌ | Ara Compiler | ONNX |
| MemryX | MX3 | Vision AI | MemryX SDK | ❌ | MemryX Compiler | ONNX |
| Kneron | KL730, KL830 | Embedded AI | Kneron PLUS | ❌ | Kneron Compiler | ONNX |
| Rockchip | RK3588 NPU | SBC, Embedded | RKNN Toolkit | Partial | RKNN Compiler | ONNX |
| Sophgo | BM1684X | Edge AI | Sophon SDK | Partial | TPU-MLIR | PyTorch via ONNX |
| Horizon Robotics | Journey Series | Automotive | Journey SDK | ❌ | Journey Compiler | ONNX |

---

## Mobile / Phone NPUs

| Vendor | Hardware | Devices | Software Stack | Open Source | Compiler | PyTorch Support |
|---------|----------|---------|----------------|-------------|----------|-----------------|
| Apple | Apple Neural Engine (ANE) | iPhone, iPad, Mac | Core ML | ❌ | Core ML Compiler | coremltools |
| Qualcomm | Hexagon NPU | Snapdragon | AI Engine Direct | Partial | AI Compiler | ExecuTorch |
| Google | Tensor G4/G5 TPU | Pixel | LiteRT (TensorFlow Lite) | Partial | XLA / LiteRT | Limited |
| Samsung | Exynos NPU | Galaxy | ENN SDK | ❌ | ENN Compiler | Limited |
| MediaTek | APU | Dimensity | NeuroPilot | Partial | NeuroPilot Compiler | ONNX |
| Huawei | Ascend Lite / Kirin NPU | Huawei Phones | CANN Lite | Partial | GE/TBE | Limited |
| UNISOC | NPU | Budget Android | UNISOC AI SDK | ❌ | Vendor Compiler | ONNX |

---

## Automotive

| Vendor | Hardware | Software | Compiler |
|---------|----------|----------|----------|
| NVIDIA | DRIVE Orin, Thor | DRIVE OS | TensorRT |
| Qualcomm | Snapdragon Ride | Ride SDK | AI Compiler |
| Tesla | FSD Chip | Internal | Internal |
| Mobileye | EyeQ | Mobileye SDK | Internal |
| Horizon Robotics | Journey | Journey SDK | Journey Compiler |
| Huawei | MDC | CANN | GE/TBE |

---

# Compiler Technologies

| Compiler | Backend |
|----------|---------|
| LLVM | Intel, AMD, Tenstorrent |
| MLIR | Tenstorrent, TensorFlow, many custom accelerators |
| XLA | TPU, AWS Neuron |
| Triton | NVIDIA, AMD (experimental) |
| TVM | Many vendors |
| IREE | Vulkan, Metal, CUDA, ROCm, CPU |
| OpenXLA | TPU, CUDA, ROCm, CPU |
| TensorRT | NVIDIA |
| MIGraphX | AMD |
| OpenVINO | Intel |
| Poplar | Graphcore |
| TT-Metal | Tenstorrent |
| SambaFlow | SambaNova |
| Groq Compiler | Groq |
| Neuron Compiler | AWS |
| CANN Compiler | Huawei |

---

# Best Targets for a New PyTorch Compiler

| Priority | Platform | Why |
|----------|----------|-----|
| ⭐⭐⭐⭐⭐ | NVIDIA CUDA | Largest deployment |
| ⭐⭐⭐⭐⭐ | AMD ROCm | Open ecosystem |
| ⭐⭐⭐⭐☆ | Intel OpenVINO / Gaudi | Open tooling |
| ⭐⭐⭐⭐☆ | Tenstorrent | Open MLIR stack |
| ⭐⭐⭐⭐☆ | IREE | Single compiler targeting many devices |
| ⭐⭐⭐⭐☆ | TVM | Widely adopted backend |
| ⭐⭐⭐☆☆ | Qualcomm | Large mobile market |
| ⭐⭐⭐☆☆ | Apple ANE | Huge install base but closed |
| ⭐⭐☆☆☆ | AWS Inferentia | Cloud only |
| ⭐⭐☆☆☆ | Groq | Specialized deployments |
| ⭐⭐☆☆☆ | Cerebras | Specialized systems |
| ⭐☆☆☆☆ | Meta MTIA | Internal only |
| ⭐☆☆☆☆ | Microsoft Maia | Internal only |
| ⭐☆☆☆☆ | Tesla FSD | Internal only |

---
