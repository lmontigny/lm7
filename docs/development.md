# Development and testing

This guide contains the detailed validation commands kept out of the main
user-facing README.

## Environment

LM7 currently targets Linux. Create a development environment:

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
source .venv/bin/activate
```

Run the portable checks:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## CPU compilation

The basic example exercises lazy compilation and fallback:

```bash
python examples/basic_mlp.py
```

The real AOTInductor path also needs a C++ compiler:

```bash
c++ --version
python examples/aot_mlp.py
python examples/aot_mlp.py --output artifacts/model.lm7
python examples/aot_mlp.py --load artifacts/model.lm7
```

`tests/test_aot_inductor.py` uses mocked compiler APIs. The example above
invokes the real local toolchain.

## NVIDIA CUDA

Use a CUDA-enabled PyTorch installation:

```bash
nvidia-smi
python -m pytest tests/test_nvidia_integration.py -q
python examples/cuda_mlp.py --target nvidia
```

LM7 detects the compute capability, moves the model and CPU inputs when
`transfers="automatic"`, and compares TorchInductor output with eager CUDA.

## TensorRT

Torch-TensorRT releases require matching PyTorch and CUDA versions. A separate
environment avoids changing an existing PyTorch installation:

```bash
uv venv --python 3.12 .venv-trt
uv pip install --python .venv-trt/bin/python -e ".[dev,tensorrt]"
.venv-trt/bin/python -m pytest tests/test_tensorrt_backend.py tests/test_tensorrt_integration.py -q
```

`--python` targets the alternate environment without activating it, so the
TensorRT install cannot leak into an already-working PyTorch environment.

The TensorRT backend is explicit and experimental:

```python
compiled = lm7.compile(
    model,
    target="nvidia",
    backend="tensorrt",
    fallback="error",
)
```

Engine construction occurs on the first call and can take tens of seconds.
The [RTX 4070 evaluation](nvidia-tensorrt-evaluation.md) records a real
fixed-shape win for SmolLM2 alongside MLP regressions, which is why TensorRT
remains opt-in rather than replacing Inductor as the automatic NVIDIA backend.

Run the opt-in Hugging Face accuracy regression with:

```bash
LM7_RUN_HF_TESTS=1 .venv-trt/bin/python -m pytest \
  tests/test_tensorrt_integration.py -q
```

## Hugging Face integration

These tests are opt-in because they download weights and require a CUDA or
MPS GPU (`resolve_target("auto")` picks whichever is local); TorchAO
quantization tests additionally require CUDA specifically:

```bash
uv pip install -e ".[dev,hf]"
LM7_RUN_HF_TESTS=1 python -m pytest tests/test_hf_integration.py -q
```

The initial INT8 path is validated only for SmolLM2-135M. The LFM2.5 hybrid
architecture remains available in BF16/FP16, but LM7 rejects INT8 for it after
local full-logit validation showed unacceptable divergence.

The examples use `HuggingFaceTB/SmolLM2-135M-Instruct`, `LiquidAI/LFM2.5-230M`,
`unsloth/Llama-3.2-1B-Instruct` (an ungated mirror of Meta's
Llama-3.2-1B-Instruct, which itself requires accepting a license and an
authenticated token), and `Qwen/Qwen3.5-0.8B`. Qwen3.5 is also a hybrid
linear-attention/convolution architecture like LFM2.5, and its Inductor
compilation is noticeably slower (roughly a minute for the first call
locally) than the other three models.

Exercise the user-facing compiled model command on the local GPU:

```bash
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target nvidia \
  --backend inductor
```

Validate the optional TorchAO INT8 and FP8 weight-only paths:

```bash
uv pip install -e ".[dev,hf,torchao]"
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target nvidia \
  --backend inductor \
  --dtype bfloat16 \
  --quantization int8-weight-only
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target nvidia \
  --backend inductor \
  --dtype bfloat16 \
  --quantization fp8-weight-only
LM7_RUN_HF_TESTS=1 python -m pytest tests/test_hf_integration.py -q
```

FP8 requires NVIDIA Ada (`sm89`), Hopper (`sm90`), or newer. NVFP4 requires
Blackwell (`sm100+`) in the current TorchAO execution path and is therefore not
available on the local RTX 4070. LM7 quantizes only MLP linear weights in the
validated FP8 path because applying FP8 to every linear changed the predicted
next token.

## Compiler IR and generated code

The AOT export API can retain indexed compiler debug files:

```python
artifact = lm7.export(
    model,
    args=(example_input,),
    target="cpu",
    backend="aot_inductor",
    output="model-debug.lm7",
    debug=True,
)

for path in artifact.debug_files():
    print(path)
```

LM7 requests exported graphs, FX graphs, pre/post-fusion Inductor IR, generated
source, and lower-level PTX, assembly, CUBIN, or HSACO when the selected
toolchain emits them. Debug output can reveal model structure and generated
code; treat it as sensitive development data.

For NVIDIA JIT compilation:

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python examples/cuda_mlp.py \
  --target nvidia \
  --debug-dir artifacts/cuda-debug
find artifacts/cuda-debug -type f | sort
```

## GPU benchmarks

The benchmark harness reports first-call cost, median and p95 latency,
throughput, peak allocated GPU memory, and environment metadata:

```bash
python benchmarks/gpu.py \
  --model mlp \
  --backend eager inductor \
  --dtype float16 \
  --batch-size 8 \
  --warmup 5 \
  --repeats 30 \
  --output artifacts/benchmarks/mlp-fp16-b8.json
```

With Hugging Face dependencies installed, use `--model smollm2`,
`--model lfm25`, `--model llama32-1b`, or `--model qwen35-0.8b`. Llama and
Qwen use the same causal-LM benchmark path as the smaller validation models;
Qwen's hybrid architecture can make first-call Inductor compilation much
slower than the steady-state latency. Add `tensorrt` to `--backend` in a
Torch-TensorRT environment. Use `--compile-mode reduce-overhead` or
`--compile-mode max-autotune` for Inductor.

Benchmark results are descriptive for the current machine. Portable CI does
not enforce hardware-specific timing thresholds.
