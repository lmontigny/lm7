# AMD GPU support

LM7 has initial support for local AMD GPUs through a ROCm-enabled PyTorch build
and TorchInductor. It does not install ROCm, GPU drivers, or PyTorch itself.

Follow the
[official AMD installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
for a PyTorch build compatible with the host ROCm release and GPU architecture.
Then install LM7 without replacing that build:

```bash
uv pip install -e ".[dev]"
```

PyTorch intentionally exposes ROCm devices through the `torch.cuda` API. LM7
uses `torch.version.hip` to distinguish AMD from NVIDIA and records the
normalized `gfx` architecture reported by the runtime.

## Verify the runtime

```bash
rocminfo | grep gfx
python - <<'PY'
import torch

print("available:", torch.cuda.is_available())
print("ROCm:", torch.version.hip)
print("GPU:", torch.cuda.get_device_name(0))
print("architecture:", torch.cuda.get_device_properties(0).gcnArchName)
PY
```

## What `lm7 targets` says about the card, and how much to trust it

`lm7 targets` names the ISA family and the formats the silicon computes:

```console
$ lm7 targets
Detected targets (2):
  amd:gfx942: AMD Instinct MI300X (CDNA 3), 192.0 GiB
    precision: native fp32, fp16, bf16, int8, fp8
```

**Every AMD value in that report is read from AMD's ISA documentation and none
of it has been confirmed on hardware** — no AMD GPU has run LM7. The NVIDIA
equivalent was measured on three generations; this is a prediction that a real
`gfx942` will either confirm or correct. See
[limitations](limitations.md#hardware-validation).

Two AMD-specific things the report gets right that a `gfx`-to-`sm` analogy would
get wrong:

- **`gfx` numbers do not order by capability.** `gfx1100` is larger than
  `gfx942` and is a consumer RDNA 3 part with no FP8 at all, while `gfx942` is a
  datacenter CDNA 3 part that has it. The two are separate product lines sharing
  a numbering space, so LM7 matches the string exactly rather than comparing it
  the way it compares `smXX`. An unrecognized `gfx` costs the label and nothing
  else.
- **`fp8` names two incompatible encodings.** CDNA 3 implements the `fnuz`
  variants — no infinities, one NaN, an exponent bias one greater than OCP's —
  so `torch.float8_e4m3fnuz` is the dtype that exists on `gfx942`, not the
  `torch.float8_e4m3fn` that `sm89`+ implements. CDNA 4 and RDNA 4 moved to the
  OCP encoding. `lm7 targets --json` reports which, under `fp8_format`:

  ```console
  $ lm7 targets --json | jq '.[0].capabilities.fp8_format'
  "fnuz"
  ```

  An FP8 number from a MI300X and one from an H100 were therefore not produced
  in the same format, and are not directly comparable.

`--json` also carries `cuda_build`, which answers whether the installed wheel
ships kernels for this chip or will fall back. The key keeps its CUDA-era name
because ROCm reports through the same `torch.cuda` API, but the answer matters
more here: a missing `sm_` target still runs by JIT-ing PTX, while a missing
`gfx` target fails at load with "no kernel image is available".

## Compile and test

```bash
python examples/rocm_mlp.py
python -m pytest tests/test_amd_integration.py -q
```

The equivalent API is:

```python
compiled = lm7.compile(
    model.eval(),
    target="amd",
    backend="inductor",
    transfers="automatic",
    fallback="error",
)
output = compiled(cpu_input)
```

Use an explicit architecture when required:

```python
compiled = lm7.compile(model, target="amd:gfx942")
```

## Benchmark

```bash
python benchmarks/gpu.py \
  --target amd \
  --model mlp \
  --backend eager inductor \
  --dtype float16
```

The initial integration covers local single-GPU inference. It does not yet
provide AMD-specific AOT packages, multi-GPU execution, quantization, or CI on
physical AMD hardware.

For a possible AMD-specific compiler path beyond TorchInductor, see the
[MIGraphX evaluation plan](amd-migraphx.md).
