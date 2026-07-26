# Google TPU support with OpenXLA

LM7 has initial single-process inference support for Google Cloud TPU through
[PyTorch/XLA](https://docs.pytorch.org/xla/master/), the PyTorch frontend for
the OpenXLA compiler and PJRT runtime.

Use a TPU VM and a version-matched PyTorch/PyTorch-XLA environment. The current
LM7 extra follows the latest stable PyTorch/XLA 2.9 pair:

```bash
python3 -m venv .venv-tpu
source .venv-tpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,openxla]"
```

PyTorch/XLA versions must match PyTorch. Follow the
[official installation instructions](https://github.com/pytorch/xla#installation)
if the current TPU VM image requires a different pair.

## Verify the runtime

```bash
python - <<'PY'
import torch_xla
import torch_xla.runtime as xr

print("PyTorch/XLA:", torch_xla.__version__)
print("PJRT device:", xr.device_type())
print("addressable devices:", xr.addressable_device_count())
print("attributes:", xr.global_runtime_device_attributes())
PY
```

`PJRT device` must report `TPU`.

## Compile and test

```bash
python examples/tpu_mlp.py
python -m pytest tests/test_tpu_integration.py -q
```

The public API is:

```python
compiled = lm7.compile(
    model.eval(),
    target="tpu",
    backend="openxla",
    transfers="automatic",
    fallback="error",
)
output = compiled(cpu_input)
```

Importing PyTorch/XLA initializes its runtime. LM7 detects addressable TPU
devices, moves the model and inputs to the XLA device, compiles with
`torch.compile(..., backend="openxla")`, and synchronizes the first execution so
compiler failures remain inside LM7's fallback boundary. OpenXLA execution uses
`torch.no_grad()` because PyTorch/XLA tracing requires tensor version counters
that `torch.inference_mode()` disables.

## Scope

This initial adapter covers one Python process and the TPU devices addressable
to that process. SPMD sharding, multi-host execution, TPU benchmarking,
persistent XLA executables, and physical TPU CI are not implemented.
