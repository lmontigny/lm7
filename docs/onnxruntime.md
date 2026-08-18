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

A model whose weights approach protobuf's 2 GiB message ceiling gets them in a
`compiled_model.onnx.data` sidecar instead, carried as a second payload with its
own checksum — the same shape as the OpenVINO backend's `.bin`. The graph
references the sidecar by relative name, so it has to stay beside the `.onnx`,
which is what an artifact directory gives it. It is verified on load for the
reason OpenVINO's is: ONNX Runtime reads it implicitly while building the
session, so corruption there would otherwise surface as wrong numbers rather
than as an error.

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
        "external_data": "auto",
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
- `external_data` decides whether weights are written beside the graph rather
  than embedded in it. `"auto"`, the default, embeds them until they approach
  protobuf's 2 GiB ceiling. `True` and `False` ask for one or the other, and are
  requests rather than instructions: the exporter keeps tensors under roughly a
  kilobyte inline whatever it was told, and writes a sidecar above the ceiling
  whatever it was told. The manifest and `compiled.artifact.metadata` record
  what happened, not what was asked for.

ONNX Runtime execution providers are ordered and can normally partition a graph
across providers. LM7 intentionally configures one requested provider and uses
ONNX Runtime's `session.disable_cpu_ep_fallback` option for strict accelerator
runs. See the official [execution-provider overview](https://onnxruntime.ai/docs/execution-providers/).

## Validated coverage and limitations

- FP32 MLP execution is validated on CPU and an NVIDIA GeForce RTX 4070 SUPER.
- CUDA validation used ONNX Runtime GPU 1.27 or newer with PyTorch 2.13 CUDA 13
  and CPU fallback disabled. The I/O binding below was run on 1.29.
- A fixed-shape SmolLM2-135M logits graph exported to a 542 MB embedded-weight
  ONNX file. ONNX Runtime CPU matched the eager next-token result and had maximum
  full-logit error below `6e-5`.
- Bounded dynamic batch execution is validated for a small MLP. Dynamic causal-
  LM sequence lengths and KV-cache decode graphs are not yet claimed.
- Inputs must be tensors, and may already sit on the session's device. Outputs
  come back on the device that produced them, because the adapter binds torch
  storage through ONNX Runtime's I/O binding rather than feeding NumPy. Checked
  on an RTX 4070 SUPER (Ada `sm89`, 12 GiB) through `CUDAExecutionProvider` with
  CPU fallback disabled, for FP32 and bfloat16.
- **What the binding is worth depends on the size of the fetch, and it is not
  always positive.** [`benchmarks/onnx_io_binding.py`](../benchmarks/onnx_io_binding.py)
  runs both strategies against the same `InferenceSession`, so the only
  difference is how tensors reach and leave it. On the same 4070 SUPER, batch 8:

  | output | NumPy feeds | I/O binding | |
  | --- | --- | --- | --- |
  | 10 features (0.3 KiB) | **0.333 ms** | 0.410 ms | 0.81x |
  | 1000 features (31 KiB) | **0.346 ms** | 0.475 ms | 0.73x |
  | 32000 features (1000 KiB) | 0.679 ms | **0.568 ms** | 1.19x |
  | 128000 features (4000 KiB) | 1.976 ms | **1.043 ms** | 1.89x |

  The crossover sits between 31 KiB and 1000 KiB of output, and the direction
  holds at p10, the median and p90 rather than only on average. Below it the
  binding's fixed setup costs more than the copies it removes, so a small
  classifier is measurably *worse* off — the reason to keep the binding anyway is
  that it is what makes outputs stay on the device at all, which is a
  correctness property rather than a speed one.
- **The same measurement on SmolLM2-135M is inconclusive**, and is recorded that
  way rather than rounded into the table above. Its run-to-run spread on this box
  is much wider than the difference being measured — p10 to p90 covers roughly
  19–46 ms against a median gap near 10 ms — and I/O binding wins at p10 while
  losing at the median. A real causal LM needs a quieter machine or far more
  samples before anything can be claimed about it.
- Weights move to a `compiled_model.onnx.data` sidecar when they approach
  protobuf's 2 GiB message ceiling, and the artifact carries it as a second
  payload with its own checksum. `external_data="auto"` decides by measuring the
  exported program's weights, counting a tied weight once.
- The largest export run is **Llama-3.2-1B-Instruct at FP32**, 4.60 GiB of
  weights, as a fixed-shape logits graph on an x86-64 host: a 2.04 MiB graph
  beside a 4.60 GiB sidecar, 66 s to convert, 11.2 GiB peak RSS. Reloaded through
  `CPUExecutionProvider` it matched eager to a maximum full-logit error of
  `2.1e-5` and produced the same next token. Before the sidecar existed the same
  export failed in the ONNX checker with protobuf's `Failed to serialize proto` —
  the checker is now handed the path rather than a resolved in-memory model, so
  it never has to hold one.
- Operator coverage is bounded by both PyTorch's ONNX exporter and the selected
  ONNX Runtime execution provider. `fallback="error"` exposes failures directly.

PyTorch documents the current exporter at
[`torch.onnx`](https://docs.pytorch.org/docs/stable/onnx.html), and ONNX Runtime's
Python session API is documented in its
[API summary](https://onnxruntime.ai/docs/api/python/api_summary.html).
