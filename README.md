# LM7

[![CI](https://github.com/lmontigny/lm7/actions/workflows/ci.yml/badge.svg)](https://github.com/lmontigny/lm7/actions/workflows/ci.yml)

LM7 is a small, PyTorch-first compiler orchestration layer for local inference.
Give it an `nn.Module`; LM7 detects the machine, selects a compatible backend,
moves inputs when requested, and returns a normal callable module.

```python
import torch
import lm7

model = torch.nn.Linear(16, 4).eval()
model = lm7.compile(model, target="auto")
output = model(torch.randn(2, 16))
```

> [!WARNING]
> LM7 is an early inference-only prototype. Model coverage and compiled-artifact
> compatibility are not yet stable.

## Install

LM7 currently targets Linux with Python 3.10 or newer and PyTorch 2.x. Start
with an environment containing the PyTorch build appropriate for your hardware:

```bash
git clone https://github.com/lmontigny/lm7.git
cd lm7
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

LM7 does not install GPU drivers, CUDA/ROCm toolchains, or platform C++
compilers.

Check the installation and see what LM7 can use locally:

```bash
lm7 doctor
lm7 targets
lm7 backends
lm7 explain --target auto
```

Add `--json` to any command for machine-readable output, for example
`lm7 doctor --json`. The same CLI is available as `python -m lm7`.

Hardware setup details:

- [CPU inference](docs/cpu.md)
- [NVIDIA and development testing](docs/development.md#nvidia-cuda)
- [AMD GPUs with ROCm](docs/amd-rocm.md)
- [Apple Silicon GPUs with Metal (MPS)](docs/apple-mps.md)
- [Google TPUs with PyTorch/XLA and OpenXLA](docs/google-tpu.md)

## Use LM7

### Let LM7 choose

`target="auto"` prefers a detected GPU or accelerator and otherwise uses the
CPU. Compilation is lazy: the first call selects a backend and compiles the
input shape.

```python
compiled = lm7.compile(
    model.eval(),
    target="auto",
    transfers="automatic",
    fallback="warn",
)
result = compiled(example_input)

print(compiled.target)
print(compiled.selected_backend)
```

`fallback="warn"` falls back to eager PyTorch if compilation fails. Use
`fallback="error"` when a compiler failure must stop execution.

### Choose hardware or a backend

Hardware targets and compiler backends are separate:

```python
lm7.compile(model, target="cpu")
lm7.compile(model, target="nvidia")
lm7.compile(model, target="nvidia:sm89")
lm7.compile(model, target="amd:gfx942")
lm7.compile(model, target="apple")
lm7.compile(model, target="tpu")

lm7.compile(model, target="nvidia", backend="inductor")
lm7.compile(model, target="nvidia", backend="tensorrt")
lm7.compile(model, target="tpu", backend="openxla")
```

| Backend | Availability | Purpose |
| --- | --- | --- |
| `eager` | Any detected PyTorch device | Reference execution and fallback |
| `inductor` | PyTorch with `torch.compile` | Default JIT compiler |
| `aot_inductor` | CPU prototype | Persistent ahead-of-time `.pt2` package |
| `tensorrt` | Optional NVIDIA prototype | Torch-TensorRT JIT engine |
| `openxla` | Optional Google TPU prototype | PyTorch/XLA and OpenXLA JIT compiler |

TensorRT must be installed in a version-matched environment and selected
explicitly:

```bash
python -m pip install -e ".[tensorrt]"
```

The current extra installs Torch-TensorRT 2.12.1 and its compatible PyTorch
2.12/CUDA 13 stack.

On a Google TPU VM, install the version-matched PyTorch/XLA runtime:

```bash
python -m pip install -e ".[openxla]"
```

### Inspect the decision

```python
print(lm7.detect_targets())
print(lm7.backends())
print(lm7.explain(model, target="auto"))
```

The environment variables `LM7_TARGET`, `LM7_BACKEND`, `LM7_FALLBACK`, and
`LM7_CACHE_DIR` provide defaults. Explicit function arguments take precedence.

## Export and deploy

Create a source artifact with `torch.export`:

```python
artifact = lm7.export(
    model,
    args=(example_input,),
    target="cpu",
    output="model.lm7",
)

loaded = lm7.load_artifact("model.lm7")
output = loaded(example_input)
```

An `.lm7` artifact is a directory containing a versioned manifest, checksums,
and a PyTorch `.pt2` program. Use `backend="aot_inductor"` to build the current
CPU AOT prototype. Compiled artifacts remain specific to compatible PyTorch,
runtime, and hardware versions.

Applications targeting several machines can combine artifacts:

```python
bundle = lm7.create_bundle(
    ["build/cpu.lm7", "build/nvidia.lm7"],
    output="model.bundle.lm7",
)

deployed = lm7.load_bundle("model.bundle.lm7").load(target="auto")
```

## Hugging Face models

Install the optional dependencies:

```bash
python -m pip install -e ".[hf]"
```

Then try either compact, ungated test model:

```bash
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompt "The capital of France is" \
  --target auto \
  --backend auto
lm7 model run hf://LiquidAI/LFM2.5-230M \
  --prompt "The capital of France is" \
  --target nvidia \
  --backend inductor
```

The command downloads through the normal Hugging Face cache, compiles a
causal-LM forward pass, and reports the selected target and backend, first-call
and steady-call time, and predicted next token. Add `--json` for structured
output. The current
JIT result is process-local; this command does not yet package weights,
tokenizers, or a persistent GPU executable.

For experimental NVIDIA INT8 or FP8 weight-only inference, install TorchAO and
select quantization explicitly:

```bash
python -m pip install -e ".[hf,torchao]"
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target nvidia \
  --backend inductor \
  --dtype bfloat16 \
  --quantization int8-weight-only
```

Use `--quantization fp8-weight-only` instead on NVIDIA Ada (`sm89`), Hopper
(`sm90`), or newer GPUs. LM7 uses TorchAO's version 2 weight-only
configurations and reports model storage before and after quantization,
quantization time, steady-call latency, and peak GPU memory. Model storage
measures weights and buffers; peak GPU memory also includes compiler
workspaces, kernels, activations, and temporary allocations, so it will not
fall by the same percentage.

These paths are initially NVIDIA-only and validated for SmolLM2-135M. LM7
leaves `lm_head` in BF16. The FP8 policy quantizes MLP linear weights only:
local validation reduced stored model bytes by about 29% while preserving the
next token. The same local RTX 4070 SUPER run measured about 195 MiB peak
allocated GPU memory, versus about 279 MiB for BF16. Exact latency and memory
depend on the model, prompt shape, compiler cache, and GPU. Quantizing every
linear reduced storage further but changed the output, so LM7 does not use that
policy. LM7 rejects unvalidated model IDs, and LiquidAI LFM2.5 remains
supported without quantization. Low-bit inference remains opt-in because small
models may still be slower on other shapes and hardware.

NVFP4 is not exposed yet: TorchAO's execution kernels require NVIDIA Blackwell
(`sm100+`), while an RTX 4070 is Ada (`sm89`). FP8 is the supported low-bit
floating-point path on that GPU.

The Python example additionally validates logits and deterministic generation:

```bash
python examples/hf_causal_lm.py \
  --model hf://HuggingFaceTB/SmolLM2-135M-Instruct
python examples/hf_causal_lm.py \
  --model hf://LiquidAI/LFM2.5-230M
```

Model weights stay in the normal Hugging Face cache and are not added to this
repository.

## Test local CPU and NVIDIA

Validate identical model weights and inputs on CPU TorchInductor and the local
NVIDIA GPU:

```bash
python examples/local_targets.py --require-nvidia
```

Compare first-call cost and steady-state latency:

```bash
python benchmarks/local.py \
  --target cpu nvidia \
  --backend eager inductor
```

OpenVINO is not required for CPU execution. LM7 already uses eager PyTorch,
TorchInductor, and the AOTInductor CPU prototype. OpenVINO remains a possible
future optional backend for Intel-specific CPU, GPU, or NPU deployment.

## Examples

```bash
python examples/basic_mlp.py
python examples/aot_mlp.py
python examples/cuda_mlp.py --target nvidia
python examples/rocm_mlp.py
python examples/mac_mlp.py
python examples/tpu_mlp.py
python benchmarks/gpu.py --target auto --model mlp --backend eager inductor
```

See [development and testing](docs/development.md) for environment checks, GPU
integration tests, compiler IR output, and benchmarks. See
[architecture](docs/architecture.md) for the backend and artifact design.

## Current limitations

- Inference only; training and backward compilation are unsupported.
- Only local PyTorch devices are detected.
- JIT compiled callables and TensorRT engines are process-local.
- AOTInductor is validated only for CPU and uses Beta PyTorch APIs.
- AMD ROCm, Apple Silicon (MPS), and OpenXLA TPU support are initial
  single-process integrations without physical-hardware CI.
- Quantization, distributed inference, remote hardware, and a stable compiled
  artifact ABI are future work.

## License

LM7 is licensed under the [BSD 3-Clause License](LICENSE).
