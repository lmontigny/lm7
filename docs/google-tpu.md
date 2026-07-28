# Google TPU support with OpenXLA

LM7 has initial single-process inference support for Google Cloud TPU through
[PyTorch/XLA](https://docs.pytorch.org/xla/master/), the PyTorch frontend for
the OpenXLA compiler and PJRT runtime.

Use a TPU VM and a version-matched PyTorch/PyTorch-XLA environment. The current
LM7 extra follows the latest stable PyTorch/XLA 2.9 pair:

```bash
uv venv --python 3.12 .venv-tpu
uv pip install --python .venv-tpu/bin/python -e ".[dev,openxla]"
source .venv-tpu/bin/activate
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
to that process. SPMD sharding, multi-host execution, persistent XLA
executables, and physical TPU CI are not implemented.

## Benchmark

On a TPU VM, compare eager XLA execution and the OpenXLA `torch.compile`
backend with the local MLP benchmark:

```bash
python benchmarks/tpu.py \
  --backend eager openxla \
  --dtype bfloat16 \
  --batch-size 8 \
  --warmup 5 \
  --repeats 30 \
  --output artifacts/benchmarks/tpu-mlp-bf16-b8.json
```

The report includes first-call compile cost, median and p95 latency,
throughput, PyTorch/XLA runtime metadata, and the number of addressable TPU
devices visible to the process. Results are descriptive for the current TPU VM
and are not portable CI thresholds.
