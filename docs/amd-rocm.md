# AMD GPU support

LM7 has initial support for local AMD GPUs through a ROCm-enabled PyTorch build
and TorchInductor. It does not install ROCm, GPU drivers, or PyTorch itself.

Follow the
[official AMD installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
for a PyTorch build compatible with the host ROCm release and GPU architecture.
Then install LM7 without replacing that build:

```bash
python -m pip install -e ".[dev]"
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
