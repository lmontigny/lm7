# CPU inference

LM7 supports CPU inference without an additional runtime:

- `eager` executes with standard PyTorch.
- `inductor` uses `torch.compile` and generates optimized CPU kernels.
- `aot_inductor` creates the current persistent CPU `.pt2` prototype.

Install a CPU-capable PyTorch build and LM7:

```bash
uv venv --python 3.12
uv pip install torch --torch-backend=cpu
uv pip install -e ".[dev]"
source .venv/bin/activate
```

`--torch-backend=cpu` pins the CPU-only wheel index, which keeps the download
small on a machine that also has a GPU. With pip instead of uv, the equivalent is
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Validate CPU and GPU locally

The correctness example runs identical weights and inputs through CPU
TorchInductor and, when available, NVIDIA or Apple Silicon TorchInductor:

```bash
python examples/local_targets.py
python examples/local_targets.py --require-nvidia
python examples/local_targets.py --require-apple
```

Run the real CPU integration test directly:

```bash
python -m pytest tests/test_cpu_integration.py -q
```

Compare first-call compilation cost and steady-state inference:

```bash
python benchmarks/local.py \
  --target cpu nvidia \
  --backend eager inductor \
  --batch-size 8 \
  --warmup 5 \
  --repeats 30
```

The benchmark uses FP32 on both targets so the numbers are directly
comparable. Production GPU inference will often use FP16 or BF16 instead.

## Is OpenVINO required?

No. TorchInductor already provides the generic CPU compiler path used by LM7,
and PyTorch can generate optimized C++ CPU kernels without OpenVINO.

OpenVINO is now available as an **opt-in** backend for Intel CPU deployment and
for OpenVINO IR artifacts:

```bash
uv pip install -e ".[openvino]"
```

```python
model = lm7.compile(model, target="cpu", backend="openvino")
```

It is not an automatic choice. It ranks below Inductor and AOTInductor, so
`backend="auto"` never selects it — the evaluation established a latency win on
Intel but not broad operator coverage, and pulling a large optional runtime into
generic CPU support is not needed for correctness.

Reach for it when you want an artifact that runs without PyTorch, or when you
are deploying to Intel hardware specifically. See the
[OpenVINO evaluation](openvino-evaluation.md) for the measurements behind that
and for the backend's documented limits.

## Shrinking a model on CPU

`lm7 model run` can quantize a Hugging Face causal LM's weights to INT8 on CPU,
which is the one weight-only mode measured off NVIDIA:

```bash
uv pip install -e ".[hf,torchao]"
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct --target cpu --quantize int8
```

That cuts SmolLM2-135M from 513 MiB to 210 MiB with the same next token on every
prompt tried. Compute stays FP32, because x86-64 without AVX-512 has no native
BF16 path.

The footprint saving is reliable; the latency effect is not, and depends on both
model size and CPU. On an AVX2-only part INT8 was at parity for SmolLM2-135M and
2.6x slower for Llama-3.2-1B — INT8 GEMM wants VNNI, which that machine lacks.
Measure it for your model rather than assuming. See
[quantization](quantization.md).

For an artifact to ship to an edge device rather than a process to run here, the
ExecuTorch backend has a separate calibrated INT8 export flow — see
[ExecuTorch](executorch.md).
