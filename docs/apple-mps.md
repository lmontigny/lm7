# Apple Silicon GPU support

LM7 has initial support for local Apple Silicon GPUs through PyTorch's Metal
Performance Shaders (MPS) backend and TorchInductor. It does not install
Xcode, macOS, or PyTorch itself.

Install a PyTorch build for macOS, which includes MPS support by default,
then install LM7:

```bash
python -m pip install torch
python -m pip install -e ".[dev]"
```

## Verify the runtime

```bash
python - <<'PY'
import torch

print("built:", torch.backends.mps.is_built())
print("available:", torch.backends.mps.is_available())
PY
```

## Compile and test

```bash
python examples/mac_mlp.py
python -m pytest tests/test_mac_integration.py -q
```

The equivalent API is:

```python
compiled = lm7.compile(
    model.eval(),
    target="apple",
    backend="inductor",
    transfers="automatic",
    fallback="error",
)
output = compiled(cpu_input)
```

`target="auto"` also selects the local Apple GPU (reported as `apple:metal`)
when no other accelerator is detected.

## Benchmark

```bash
python benchmarks/local.py \
  --target cpu apple \
  --backend eager inductor
python benchmarks/gpu.py \
  --target apple \
  --model mlp \
  --backend eager inductor \
  --dtype float16
```

`benchmarks/gpu.py` also accepts `--model smollm2`/`--model lfm25` with the
`hf` extra installed, and `--compile-mode reduce-overhead`/`max-autotune` for
the Inductor backend; `max-autotune` prints an informational
`Not enough SMs to use max_autotune_gemm mode` warning from Inductor's CUDA
heuristics but still compiles and runs correctly on MPS. `peak_memory_bytes`
is always `null` for the `apple` vendor: PyTorch's `torch.mps` module has no
CUDA-equivalent peak-tracking API, only current allocation.

Representative local run (`mlp`, batch size 8, float16, M4):

```text
     eager  first=  577 ms  median=1.68 ms  p95=1.91 ms  throughput= 4769 samples/s
  inductor  first=  678 ms  median=0.78 ms  p95=0.96 ms  throughput=10313 samples/s
```

## Hugging Face models

The [Hugging Face causal-LM path](../README.md#hugging-face-models) runs on
Apple Silicon through `target="auto"` or `target="apple"`:

```bash
python -m pip install -e ".[dev,hf]"
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct --target apple --backend inductor
python examples/hf_causal_lm.py --model hf://LiquidAI/LFM2.5-230M --target apple
LM7_RUN_HF_TESTS=1 python -m pytest tests/test_hf_integration.py -q
```

MPS float16 matmul reductions accumulate in a different order than CUDA and
produce a wider tail of outlier logits. Validated locally on SmolLM2-135M
(max absolute logit diff about 0.20) and LFM2.5-230M (about 0.03); both keep
matching next-token predictions and cosine similarity above 0.9999 against
eager. `tests/test_hf_integration.py` uses a wider `atol` on the `apple`
vendor to account for this. TorchAO INT8/FP8 weight-only quantization
remains NVIDIA-only.

The initial integration covers local single-GPU inference. It does not yet
provide Apple-specific AOT packages, quantization, or CI on physical Apple
hardware; TorchInductor coverage for MPS is newer and less mature than CUDA,
so validate compiled output against eager for your own models.
