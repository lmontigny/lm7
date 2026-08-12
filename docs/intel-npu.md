# Intel NPU

LM7 reaches the Intel NPU — the "AI Boost" block in Core Ultra parts — through
the OpenVINO NPU plugin:

```python
import lm7

model = lm7.compile(model, target="intel:npu")
```

There is no second backend to choose. `intel:npu` is the first LM7 target with
no PyTorch device behind it, so `inductor` and `eager` both decline it and
`openvino` is the only backend that plans.

## XPU is not the NPU

Intel's names overlap across hardware and software layers:

- **XPU** is the PyTorch device family exposed as `torch.xpu`. In LM7,
  `target="intel"` means an Intel GPU reached through that path: an integrated
  Arc GPU in a Core Ultra laptop, a discrete Arc GPU, or a supported Intel
  datacenter GPU.
- **NPU** is a physical deep-learning accelerator block. In LM7,
  `target="intel:npu"` means the Intel AI Boost NPU reached through OpenVINO's
  `NPU` device.

They can sit in the same processor package, but they are not the same execution
device:

```text
Intel Core Ultra 7 258V
|
+-- CPU
|
+-- Intel Arc 140V GPU   -> PyTorch torch.xpu
|                           LM7 target="intel"
|
+-- Intel AI Boost NPU   -> OpenVINO NPU
                            LM7 target="intel:npu"
```

That makes a modern Core Ultra laptop unusually useful for LM7 validation. One
machine can exercise `target="cpu"`, `target="intel"`, and
`target="intel:npu"` against genuinely different silicon.

For new Intel test hardware, prefer a Lunar Lake/Core Ultra 200V laptop such as
a Core Ultra 7 258V or 268V with 32 GB RAM. Those parts pair an Arc 140V
integrated Xe2 GPU with Intel's NPU 4.0. A Meteor Lake/Core Ultra 100H laptop,
for example a Core Ultra 7 155H or 165H, is a cheaper fallback and still has
both an integrated Arc GPU and an Intel AI Boost NPU, but its NPU is much
smaller.

> [!WARNING]
> **Untested on hardware.** The target, its detection, and the plugin wiring
> are implemented and unit-tested, but no Intel NPU has run this code. The
> integration tests in `tests/test_openvino_integration.py` skip unless
> OpenVINO reports an NPU. Treat the numbers you get as the first data points,
> not as a validated path — see [limitations](limitations.md).

## What makes this target different

Every other LM7 target is a device `torch` can see: `torch.cuda`, `torch.xpu`,
`torch.backends.mps`, or a PJRT accelerator. The NPU is none of those.

- **No torch device.** `torch_device(intel:npu)` is `cpu`. Inputs stay host
  tensors and the plugin moves them, exactly as the OpenVINO CPU path already
  does. `transfers="explicit"` therefore checks inputs against `cpu`.
- **`kind="npu"`, not `"gpu"`.** `intel` on its own still means the Intel GPU
  (XPU), so the NPU needed its own kind rather than a second qualifier. That
  kind is what `inductor` and `eager` gate on when they decline.
- **`target="auto"` never picks it.** Automatic detection prefers a `gpu` or
  `accelerator` and otherwise takes the CPU. An NPU is a low-power part that
  wants INT8 weights and static shapes, so LM7 does not substitute it for the
  CPU silently. Name it.
- **Fallback lands on the host.** If the plugin cannot compile the model and
  `fallback="warn"` is in effect, LM7 runs PyTorch eager on the CPU and says so
  in the warning. `fallback="error"` stops instead.

## Requirements

- OpenVINO with its NPU plugin: `uv pip install -e ".[openvino]"`.
- The NPU driver. On Linux that is the `intel_vpu` kernel module (mainline since
  6.10), which publishes `/dev/accel/accel0`, plus Intel's user-mode driver. On
  Windows the NPU driver from Intel's site covers both.
- A Core Ultra (Meteor Lake or newer) part. `DEVICE_ARCHITECTURE` reports `3720`
  for Meteor Lake and `4000` for Lunar Lake.

Check what the runtime sees:

```bash
lm7 targets
```

An NPU shows up as `intel:npu` with the driver nodes recorded in its
capabilities:

```bash
lm7 doctor --json
```

If `/dev/accel/accel0` exists but no `intel:npu` appears, the driver is present
and the OpenVINO NPU plugin is not. If neither appears, the machine has no NPU
LM7 can reach.

## Constraints the plugin imposes

These are properties of the NPU plugin, not choices LM7 made. Each one is
enforced with an actionable error rather than left to surface as a plugin
exception:

- **Static shapes only.** The NPU compiler does not accept dynamic dimensions.
  LM7 already reshapes exported IR to the example inputs, so the default works;
  `options={"static_shapes": False}` is rejected, as are `dynamic_shapes=` and
  `shape_profile=` on an `intel:npu` export. One artifact per shape.
- **FP16 compute.** The plugin's supported inference precision is FP16, so LM7
  does not set the `INFERENCE_PRECISION_HINT="f32"` it pins on the CPU plugin.
  An FP32 model therefore executes in FP16 and carries FP16-level error against
  eager — the NPU integration test uses a `2e-2` tolerance where the CPU one
  uses `1e-4`. Pass `options={"inference_precision": ...}` to override.
- **No bfloat16 models.** Inherited from the OpenVINO backend: its runtime
  exchanges tensors through NumPy, which has no bfloat16 dtype.
- **No silent device fallback.** OpenVINO compiles an unavailable device onto
  the CPU plugin without complaint, which would report a CPU run as an NPU
  result. LM7 checks `Core().available_devices` first and raises instead.

## Quantization

The NPU is INT8-first: weights that stay FP32 spend the memory bandwidth the
part is trying to save. The [NNCF weight compression](quantization.md) already
wired into the OpenVINO export path applies unchanged:

```bash
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct npu.lm7 \
  --target intel:npu --backend openvino --quantize int8
```

The IR is device-neutral; the device it was exported for travels in the
manifest's `runtime_requirements.openvino_device`, so `lm7.load_artifact()`
compiles it back onto the NPU rather than dropping to the CPU.

## Measuring it

`benchmarks/openvino_eval.py` takes `--device NPU` and already controls for the
warmup, compile-accounting, and silent-fallback traps documented in the
[OpenVINO evaluation](openvino-evaluation.md):

```bash
python benchmarks/openvino_eval.py --path eager openvino_ir --device NPU \
  --output artifacts/benchmarks/npu-mlp-fp32-b8.json
```

The CPU result from the same run is the baseline that matters. An NPU is not
chosen for peak latency against a desktop CPU — it is chosen for latency per
watt while the CPU stays free — and LM7 does not measure power, so a latency
table alone will understate it.

## What is not implemented

- **Intel GPU through OpenVINO.** `intel:gpu` still compiles on the CPU plugin
  and reaches the GPU through TorchInductor's XPU path instead. Mapping it to
  the OpenVINO `GPU` device is a separate change with its own evaluation.
- **NPU-specific plugin tuning.** `NPU_COMPILATION_MODE_PARAMS`, tiling hints,
  and the compiled-blob cache are reachable through
  `options={"config": {...}}` but have no LM7 defaults.
- **Multiple NPUs.** Ordinals are parsed from OpenVINO's `NPU.<n>` naming and
  recorded, but no machine ships more than one today.
