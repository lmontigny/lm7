# Personal AI hardware

LM7 is not a hardware product and does not install drivers or vendor SDKs. Its
role in a personal AI hardware stack is narrower: keep the application boundary
at a PyTorch `nn.Module`, then route compile, export, fallback, and inspection
through the compiler/runtime that matches the local device.

That is useful for local assistants, private RAG boxes, robotics controllers,
AI mini PCs, phones, and edge appliances where the same model code may need to
run on a laptop during development, a small box on a desk, a phone, or a
vendor-specific accelerator.

## What LM7 provides

- One PyTorch-facing compile/export interface for CPU, GPU, NPU, phone, and
  embedded targets.
- Runtime target detection for local devices that PyTorch or the vendor runtime
  can see.
- Export paths for devices that cannot be called from the host process, such as
  Android, iOS, and deployment-only NPUs.
- Inspectable backend selection and artifact requirements through `lm7 doctor`,
  `lm7 explain`, and `lm7 artifact inspect`.
- Controlled fallback when a preferred local backend is missing or fails during
  first-call compilation.

LM7 does not hide the vendor boundary. A QNN artifact still needs a compatible
Qualcomm runtime. A Core ML artifact still needs Apple's runtime. A TensorRT
artifact is still tied to NVIDIA's stack. LM7 keeps those requirements explicit
instead of spreading the branching logic through the application.

## Current device classes

| Device class | Preferred stack | LM7 status |
| --- | --- | --- |
| Mac mini, MacBook, iPad, iPhone | MPS for local JIT; Core ML or ExecuTorch for export | Apple Silicon and Core ML export have run on real hardware. |
| Android phones with Snapdragon | ExecuTorch for CPU; QNN for HTP/NPU | Snapdragon 8 Elite QNN export and device execution are validated. |
| Intel AI PCs and mini PCs | OpenVINO for CPU/GPU/NPU | CPU paths are validated; `intel:npu` is integrated but still mock-tested. |
| Arm Linux edge boxes | CPU Inductor, ONNX Runtime, OpenVINO, ExecuTorch | Arm Neoverse CPU and ExecuTorch paths are validated; small-board accelerators need device work. |
| NVIDIA edge GPUs and Jetson | CUDA, TensorRT, ONNX Runtime, IREE Vulkan | NVIDIA GPUs are validated, but Jetson-specific validation and setup docs are still missing. |
| AMD Ryzen/Radeon edge systems | ROCm, Vulkan/IREE, ONNX Runtime | AMD CPU and one ROCm GPU path on MI300X are validated; Radeon and AI-PC NPU coverage still need hardware runs. |
| Custom or emerging accelerators | StableHLO, IREE, TVM, MLIR/PJRT, vendor SDK | Useful integration direction; support depends on the vendor compiler exposing a usable bridge. |

See [tested hardware](tested-hardware.md) for the exact physical machines that
have run LM7 paths.

## Stacks to add next

1. **NVIDIA Jetson Orin and Thor**

   This is the clearest near-term edge target. LM7 already has most of the
   pieces through NVIDIA, TensorRT, ONNX Runtime, and IREE Vulkan integrations,
   but it needs a Jetson guide and physical validation on JetPack.

2. **Intel Core Ultra NPU**

   AI PCs and quiet mini PCs are natural personal AI hosts. LM7 already exposes
   `target="intel:npu"` through OpenVINO, but the NPU path needs real Core Ultra
   validation before the README should treat it as tested hardware.

3. **Snapdragon X Elite and X2 Elite**

   This extends the current Snapdragon phone work to Windows-on-Arm and
   developer reference devices. The likely route is QNN/QAIRT plus ONNX Runtime
   or ExecuTorch where available.

4. **Raspberry Pi 5 with Hailo-8/8L**

   This should be treated as export-first support. The likely flow is PyTorch to
   ONNX, then Hailo SDK compilation and device validation. It is most relevant
   for vision and small models, not general LLM serving.

5. **Rockchip RK3588 and RKNN**

   RK3588 boards are common low-cost edge devices. The practical first step is
   an ONNX export and diagnostics path into RKNN Toolkit, with explicit operator
   and quantization limitations.

6. **AMD Ryzen AI**

   Consumer AMD AI PCs matter for personal hardware, but the first LM7 support
   should be conservative: validate CPU/GPU paths, then evaluate the NPU stack
   once the vendor runtime is stable enough to automate.

7. **IREE, StableHLO, and TVM bridges for new silicon**

   Personal AI hardware vendors should not need application-specific glue for
   every model. If their compiler accepts StableHLO, MLIR, TVM, ONNX, or a PJRT
   bridge, LM7 can make that stack reachable from the same PyTorch-facing API.

## Validation rule

Do not promote a personal-device row from "integrated" to "validated" until it
has run on physical hardware and the result is recorded in
[tested hardware](tested-hardware.md). For deployment-only devices, validation
means both export on the host and execution on the target runtime.
