# CPU inference

LM7 supports CPU inference without an additional runtime:

- `eager` executes with standard PyTorch.
- `inductor` uses `torch.compile` and generates optimized CPU kernels.
- `aot_inductor` creates the current persistent CPU `.pt2` prototype.

Install a CPU-capable PyTorch build and LM7:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

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

OpenVINO may be useful as a future optional backend when LM7 specifically
targets Intel CPU, GPU, or NPU deployment, needs OpenVINO IR artifacts, or wants
to measure OpenVINO quantization and serving behavior. It should be evaluated
against eager and Inductor on representative models before becoming an
automatic choice. Adding it to generic CPU support would otherwise introduce a
large optional runtime and another model-coverage surface without being needed
for correctness.
