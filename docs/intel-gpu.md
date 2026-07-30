# Intel GPU (XPU)

LM7 reaches Intel Arc and Data Center GPU Max through PyTorch's `xpu` device
and TorchInductor:

```python
import lm7

model = lm7.compile(model, target="intel")
```

`intel` means the GPU. The NPU in the same Core Ultra package is a separate
target with a separate toolchain — see [intel-npu.md](intel-npu.md).

> [!WARNING]
> **No CI, no local hardware.** Like every LM7 target except CPU, this one has
> no physical-hardware coverage. See [limitations](limitations.md).

## What you need

Two pieces, and the second is the one people miss:

1. **A PyTorch build with XPU support.** `torch.xpu.is_available()` must be
   true, which needs the Intel GPU driver and a PyTorch built with XPU. Recent
   upstream PyTorch ships this; Intel Extension for PyTorch is no longer
   required for the core `torch.compile` path.
2. **`pytorch-triton-xpu`.** TorchInductor generates GPU kernels with Triton,
   and Triton's Intel code generator is **out-of-tree** — it is not in the
   stock `triton` wheel. Without it, `torch.compile` raises inside the first
   call.

```bash
python -c "import torch; print(torch.xpu.is_available())"
python -c "import triton.backends as b; print(sorted(b.backends))"
```

The second command must list `intel`. If it prints only `nvidia` or `amd`, you
have a Triton without the Intel backend.

LM7 checks this for you rather than letting it fail at the first call:

```
$ lm7 explain --target intel
  inductor: unavailable (priority 0) - The installed Triton has no Intel GPU
  code generator (it registers: amd, nvidia). That backend is out-of-tree and
  ships as pytorch-triton-xpu; install it with "pip install
  pytorch-triton-xpu", or use backend="eager". See docs/intel-gpu.md.
```

`lm7 doctor` reports the same thing under the `inductor` backend, including
which vendors the installed Triton *does* generate for.

## How the compiler reaches the hardware

```
FX graph → TorchInductor → Triton IR → Triton Intel backend → LLVM IR
        → SPIR-V → Intel Graphics Compiler (IGC) → Xe ISA → Level Zero
```

Two things follow from this that are worth knowing.

**SPIR-V here is not the SPIR-V in `iree_vulkan`.** LM7 has a second backend
that also emits SPIR-V for Intel GPUs ([iree-vulkan.md](iree-vulkan.md)), and
the two are not interchangeable. Triton emits the **compute** flavour, in the
OpenCL/Level Zero execution environment, consumed by IGC. IREE emits the
**Vulkan** flavour, a shader consumed by the Vulkan driver. Same container
format, different capabilities and memory model. A module built for one will
not load in the other.

**OpenVINO does not serve this target.** OpenVINO has a GPU plugin, but LM7 has
not evaluated it, and every non-NPU OpenVINO target in LM7 maps to the CPU
plugin. Rather than run an Intel GPU target on the CPU under a GPU label, the
`openvino` backend declines it — so `backend="openvino"` with `target="intel"`
raises instead of quietly moving your model to the CPU. Use `target="cpu"` for
the CPU plugin, or `target="intel:npu"` for the NPU.

## Backends on this target

| Backend | Status |
| --- | --- |
| `inductor` | The path. Needs `pytorch-triton-xpu`; declines with an actionable reason without it. |
| `eager` | Always available. Runs on the XPU device uncompiled — this is what a failed Inductor compile falls back to. |
| `iree_vulkan` | Export-only, Vulkan SPIR-V, unrelated to the Inductor path. |
| `openvino` | Declines; see above. |
| `aot_inductor` | Not validated for XPU — CPU, Apple, and NVIDIA only. |

## Fallback behaviour

`fallback="warn"` (the default) turns a failed Inductor compile into an eager
XPU run plus a `RuntimeWarning`. That is correct hardware at uncompiled speed,
which is easy to miss in a benchmark. Use `fallback="error"` when you are
measuring, so a missing kernel generator stops the run instead of quietly
changing what you measured.

## Not implemented

- **AOT for XPU.** `aot_inductor` is validated for CPU, Apple, and NVIDIA only,
  so there is no persistent compiled artifact for an Intel GPU. The portable
  options are `iree_vulkan` (export-only, Vulkan) and `stablehlo`.
- **SYCL-TLA.** Intel is adding a second Inductor backend for discrete GPUs
  alongside Triton. LM7 does not select or expose it.
- **The OpenVINO GPU plugin**, as above.
- **Multi-GPU.** Detection reports one entry per XPU device with its ordinal,
  but the target grammar has no way to pin one: `intel` and `intel:gpu` are the
  only spellings, and `intel:0` is rejected. Compilation always uses device 0,
  and there is no sharding.
