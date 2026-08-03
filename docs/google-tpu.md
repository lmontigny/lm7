# Google TPU support with OpenXLA

LM7 has initial single-process inference support for Google Cloud TPU through
[PyTorch/XLA](https://docs.pytorch.org/xla/master/), the PyTorch frontend for
the OpenXLA compiler and PJRT runtime.

Validated on a **TPU v6e (Trillium), one chip**, with torch 2.9.1+cpu,
torch_xla 2.9.0 and libtpu 0.0.21. Everything below is measured on that host
unless it says otherwise; numbers are descriptive for one TPU generation and are
not portable CI thresholds.

Use a TPU VM and a version-matched PyTorch/PyTorch-XLA environment. The current
LM7 extra follows the latest stable PyTorch/XLA 2.9 pair:

```bash
uv venv --python 3.12 .venv-tpu
uv pip install --python .venv-tpu/bin/python "torch==2.9.*" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-tpu/bin/python -e ".[dev,openxla]"
```

Take PyTorch from the CPU index first. A TPU VM has no GPU, but the default
PyPI `torch` wheel is the CUDA build, so a plain resolve drags in the whole
NVIDIA wheel set for nothing.

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

`PJRT device` must report `TPU`. On the validated host this prints one
addressable device and `{'coords': [0, 0, 0], 'core_on_chip': 0, 'num_cores': 1,
'name': 'TPU:0'}`. `PJRT_DEVICE` does not need to be set: torch_xla finds
`libtpu.so` and the device and sets it, logging that it has done so.

`tpu-info`, installed with the `openxla` extra, reports the generation and
per-chip HBM.

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

## fp32 accuracy: matmul precision

**This is the thing to know before trusting a TPU result.** XLA lowers an fp32
matmul to bf16 passes on TPU unless told otherwise, so a model that agrees with
CPU eager to 1e-6 on every other target agrees to about 1e-3 here. Measured
against CPU eager, max absolute difference:

| `mat_mul_precision` | bare 8x16 @ 16x32 | 3-layer MLP | what it does |
| --- | --- | --- | --- |
| `default` | 3.6e-02 | 1.7e-03 | one bf16 pass |
| `high` | 1.6e-04 | 1.2e-05 | 3 passes |
| `highest` | 1.9e-06 | 4.5e-08 | 6 passes, true fp32 |

GELU is exact to 4.8e-07 under all three, so the divergence is entirely the
matmul. LM7 exposes the choice rather than making it:

```python
compiled = lm7.compile(
    model.eval(),
    target="tpu",
    backend="openxla",
    options={"mat_mul_precision": "highest"},
)
```

Two constraints come with it, both from XLA rather than from LM7:

- **It is process-global and read once**, while XLA lowers the first
  computation. Set it before anything else touches the TPU in that process.
- **Setting it late is not an error in torch_xla.** `set_mat_mul_precision`
  updates what `get_mat_mul_precision` reports while the numerics stay on the
  old setting, so the getter cannot confirm its own effect. LM7 will not pass
  that silence on: if a computation has already run, the option raises
  `CompilationError` instead of being quietly ignored.

It costs both compile time and latency, and how much latency depends on whether
the workload is large enough to be doing MXU work. The 1024-4096-1024 MLP in
fp32:

| Batch | `default` | `high` | `highest` | Compile, `default` -> `highest` |
| --- | --- | --- | --- | --- |
| 8 | 0.335 ms | 0.357 ms | 0.356 ms | 2.3 s -> 4.7 s |
| 4096 | 2.360 ms | 2.557 ms | 2.713 ms | 3.4 s -> 15.2 s |

At batch 8 it is roughly free, because the chip is waiting on memory rather than
saturating the MXU, and extra passes cost nothing you were not already paying
for. At batch 4096 `highest` is **15% slower**, which is the number to plan
with. Compile time is the harsher end: 4.5x for `highest` at batch 4096.

## One process per chip

A TPU chip is claimed by a single process, and *probing* the runtime claims it —
`xr.device_type()` is enough. A second process then gets:

```text
RuntimeError: TPU initialization failed: open(/dev/vfio/0): Device or resource
busy; Couldn't open iommu group /dev/vfio/0
```

Two consequences. Nothing else may be holding the TPU when you run LM7 — a
stray notebook kernel is the usual culprit. And LM7's TPU tests cannot isolate
cases in subprocesses the way the ExecuTorch and OpenVINO suites do, because the
pytest process that can see the TPU is by construction one whose children
cannot. The matmul-precision test therefore runs first in its module and skips
in a full-suite run, where the StableHLO module has already executed on XLA:

```bash
python -m pytest tests/test_tpu_integration.py -q   # exercises the setting
python -m pytest -q                                 # skips it, with the reason
```

## StableHLO artifacts execute here too

`lm7.export(..., backend="stablehlo")` produces a target-independent payload,
and loading it back into a torch callable goes through torch_xla — which on a
TPU VM means the TPU, even for an artifact exported with `target="cpu"`. The
execution device is a property of the host, not of the artifact. That is by
design, but it means artifact numerics shift by the same amount as the table
above, so `tests/test_stablehlo_integration.py` picks its tolerance from the
device that will actually run it. See
[StableHLO and PJRT](stablehlo-pjrt-evaluation.md).

## Scope

This adapter covers one Python process and the TPU devices addressable to that
process. SPMD sharding, multi-host execution, persistent XLA executables, and
physical TPU CI are not implemented. The validated host has a **single**
addressable chip, so the sharding and multi-host paths are not merely
unimplemented but untested — nothing here says what happens on a v6e-8.

## Benchmark

On a TPU VM, compare eager XLA execution and the OpenXLA `torch.compile`
backend with the local MLP benchmark:

```bash
python benchmarks/tpu.py \
  --model smollm2 \
  --backend eager openxla \
  --dtype bfloat16 \
  --batch-size 1 \
  --warmup 5 \
  --repeats 30 \
  --output artifacts/benchmarks/tpu-v6e-smollm2-bf16-b1.json
```

`--model` takes `mlp` or a Hugging Face causal LM (`smollm2`, `llama32-1b`,
`deepseek-coder-1.3b`). The report includes first-call compile cost, median and
p95 latency, throughput, the TPU generation, PyTorch/XLA runtime metadata, and
the number of addressable TPU devices visible to the process.

On the validated host, bf16:

| Workload | Batch | `eager` median | `openxla` median | `openxla` compile | Winner |
| --- | --- | --- | --- | --- | --- |
| MLP 1024-4096-1024 | 8 | **0.152 ms** | 0.345 ms | 614 ms | eager, by 2.3x |
| SmolLM2-135M | 1 | 32.312 ms | **1.213 ms** | 14.5 s | openxla, by **27x** |
| Llama-3.2-1B | 1 | 17.644 ms | **2.557 ms** | 8.4 s | openxla, by 6.9x |

**The MLP is the outlier, and it was the only workload this harness could build
until now.** On three layers there is nothing for dynamo to add — PyTorch/XLA's
lazy-tensor eager path already traces and fuses the whole graph, and the guards
cost more than they save. On a real model the ranking inverts hard, because
eager re-traces the graph on the host *every call* while `openxla` compiles once
and dispatches a cached executable.

That also explains the shape of the eager column: SmolLM2-135M is slower in
eager than Llama-3.2-1B despite being 7x smaller, because eager's cost tracks op
count rather than parameter count, and SmolLM2 has 30 layers to Llama's 16.
Under `openxla`, where the tracing is amortised, the order is the one the
parameter counts predict.

### Timing an accelerator that answers early

These numbers are the second set measured. The first were wrong, in a way worth
recording because it is not specific to LM7.

`torch_xla.sync(wait=True)` flushes the pending lazy graph and returns when the
work has been *dispatched*. Its `wait` flag is about the lazy-tensor barrier, not
about the chip finishing. Timing `(a @ w) @ w` with the result held live, so
nothing is eliminated as dead code:

| Batch | With `sync(wait=True)` only | Implied |
| --- | --- | --- |
| 512 | 0.098 ms | 352 TFLOP/s |
| 4096 | 0.079 ms | 3,463 TFLOP/s |
| 16384 | 0.076 ms | 14,538 TFLOP/s |

Constant time for 32x the work, on a chip that peaks near 918 TFLOP/s bf16. The
barrier that actually blocks is `torch_xla.core.xla_model.wait_device_ops()`:

| Batch | With `wait_device_ops()` | Implied |
| --- | --- | --- |
| 512 | 0.269 ms | 128 TFLOP/s |
| 4096 | 0.573 ms | 480 TFLOP/s |
| 16384 | 1.723 ms | 638 TFLOP/s |

638 TFLOP/s is about 69% of peak for a large matmul, which is a believable place
to land. `lm7.benchmark` now issues both. If you time TPU work yourself, issue
both — a benchmark loop that only calls `sync()` measures your host.
