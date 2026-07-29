# ONNX Runtime backend

LM7 can convert a `torch.export.ExportedProgram` to ONNX and execute it through
an explicit ONNX Runtime execution provider. The backend supports both lazy
`lm7.compile` use and persistent `lm7.export` artifacts.

The initial validated targets are:

- CPU through `CPUExecutionProvider`.
- NVIDIA GPU through `CUDAExecutionProvider`.

ONNX Runtime supports many other execution providers, but LM7 does not label
them supported until the matching hardware and package have been exercised.

## Installation

Choose exactly one runtime wheel. The packages expose the same Python module and
must not be installed together.

For CPU:

```bash
uv pip install -e ".[onnxruntime]"
```

For the CUDA 13 PyTorch build used by this repository:

```bash
uv pip install -e ".[onnxruntime-gpu]"
```

The GPU extra uses ONNX Runtime 1.27 or newer, whose PyPI GPU wheel targets CUDA
13. Importing PyTorch before ONNX Runtime preloads the CUDA and cuDNN libraries
shipped with PyTorch; LM7 already imports PyTorch as part of its normal startup.
Other CUDA major versions need a compatible ONNX Runtime wheel.

See the official [installation matrix](https://onnxruntime.ai/docs/install/) and
[CUDA execution-provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## Lazy compilation

```python
compiled = lm7.compile(
    model.eval(),
    target="cpu",
    backend="onnxruntime",
    fallback="error",
)
result = compiled(example)
```

For NVIDIA:

```python
compiled = lm7.compile(
    model.eval(),
    target="nvidia:sm89",
    backend="onnxruntime",
    fallback="error",
)
```

The first call captures the graph on the CPU, converts it with PyTorch's
`torch.export`-based ONNX exporter, validates the model with ONNX, creates an
`InferenceSession`, and executes it. The NVIDIA default disables CPU execution-
provider fallback, so unsupported CUDA graphs fail rather than being reported as
GPU runs while silently placing nodes on the CPU.

The backend has priority 70, below Inductor, AOTInductor, TensorRT, and OpenVINO.
It therefore does not replace the established automatic choices.

## Persistent artifacts

```python
artifact = lm7.export(
    model.eval(),
    args=(example,),
    target="cpu",
    backend="onnxruntime",
    output="model-onnx.lm7",
)

reloaded = lm7.load_artifact("model-onnx.lm7")
result = reloaded(example)
```

The artifact contains `compiled_model.onnx`, its checksum, the source `.pt2`
program, and runtime settings in the manifest. Loading verifies both payloads
before creating the session.

Bounded dynamic shapes captured through `ShapeProfile` are retained by the ONNX
exporter. The integration suite exercises one artifact at batch sizes 1, 5, and
8 after a batch range of 1–8 was captured.

## Options

```python
compiled = lm7.compile(
    model,
    target="nvidia:sm89",
    backend="onnxruntime",
    options={
        "provider": "CUDAExecutionProvider",
        "provider_options": {"device_id": "0"},
        "disable_cpu_fallback": True,
        "opset_version": 20,
        "optimize": True,
    },
)
```

- `provider` selects one provider reported by
  `onnxruntime.get_available_providers()`.
- `provider_options` is passed to the provider when the session is created.
- `disable_cpu_fallback` defaults to `True` for non-CPU providers and `False`
  for `CPUExecutionProvider`.
- `opset_version` selects the ONNX opset; the default lets the installed PyTorch
  exporter choose.
- `optimize` controls the PyTorch ONNX exporter's graph optimization pass.

ONNX Runtime execution providers are ordered and can normally partition a graph
across providers. LM7 intentionally configures one requested provider and uses
ONNX Runtime's `session.disable_cpu_ep_fallback` option for strict accelerator
runs. See the official [execution-provider overview](https://onnxruntime.ai/docs/execution-providers/).

## Validated coverage and limitations

- FP32 MLP execution is validated on CPU and an NVIDIA GeForce RTX 4070 SUPER.
- CUDA validation used ONNX Runtime GPU 1.27 with PyTorch 2.13 CUDA 13 and CPU
  fallback disabled.
- A fixed-shape SmolLM2-135M logits graph exported to a 542 MB embedded-weight
  ONNX file. ONNX Runtime CPU matched the eager next-token result and had maximum
  full-logit error below `6e-5`.
- Bounded dynamic batch execution is validated for a small MLP. Dynamic causal-
  LM sequence lengths and KV-cache decode graphs are not yet claimed.
- Inputs must be tensors. Outputs are returned as a tensor or a flat tuple of
  tensors on the CPU, even when the execution provider is CUDA; this initial
  adapter uses NumPy feeds and outputs rather than ORT I/O binding.
- Weights are embedded in one ONNX file so LM7 can checksum one payload. Models
  whose serialized ONNX protobuf exceeds the 2 GiB limit need external-data
  packaging, which is future work.
- Operator coverage is bounded by both PyTorch's ONNX exporter and the selected
  ONNX Runtime execution provider. `fallback="error"` exposes failures directly.

PyTorch documents the current exporter at
[`torch.onnx`](https://docs.pytorch.org/docs/stable/onnx.html), and ONNX Runtime's
Python session API is documented in its
[API summary](https://onnxruntime.ai/docs/api/python/api_summary.html).
