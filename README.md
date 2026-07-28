# LM7

[![CI](https://github.com/lmontigny/lm7/actions/workflows/ci.yml/badge.svg)](https://github.com/lmontigny/lm7/actions/workflows/ci.yml)

LM7 is a small, PyTorch-first compiler orchestration layer for local inference.
Give it a PyTorch `nn.Module` and LM7 picks the best local CPU, GPU, or
accelerator backend for the machine it runs on — you keep writing ordinary
PyTorch and LM7 decides *where* and *how* it runs.

```python
import torch
import lm7

model = torch.nn.Linear(16, 4).eval()
model = lm7.compile(model, target="auto")
output = model(torch.randn(2, 16))
```

`lm7.compile` returns a normal callable module, so the rest of your code does
not change.

> [!WARNING]
> LM7 is an early inference-only prototype. Model coverage and compiled-artifact
> compatibility are not yet stable.

## How it works

- **Target vs. backend are separate.** A *target* is where the model runs
  (`cpu`, `nvidia`, `apple`, `tpu`, …); a *backend* is the compiler used to get
  there (`eager`, `inductor`, `tensorrt`, …). Pin either, or let LM7 choose.
- **Detection is automatic.** `target="auto"` prefers a detected GPU or
  accelerator and otherwise uses the CPU.
- **Compilation is lazy and per-input-shape.** Nothing compiles until the first
  call; the first call is slow, later calls are fast.
- **Fallback is safe by default.** If a backend fails to compile, LM7 falls
  back to PyTorch eager and warns. Use `fallback="error"` to stop instead.

The tasks below follow the usual path: install, detect the hardware, compile a
local model, run a Hugging Face model, and export an artifact.

## 1. Install

LM7 needs Python 3.10 or newer and a PyTorch build that matches the target
machine. It does **not** install GPU drivers, CUDA/ROCm toolchains, Xcode,
PyTorch/XLA, or platform C++ compilers.

```bash
git clone https://github.com/lmontigny/lm7.git
cd lm7
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .          # add ".[dev]" for pytest + ruff
```

Per-hardware setup: [CPU](docs/cpu.md) ·
[NVIDIA](docs/development.md#nvidia-cuda) · [AMD ROCm](docs/amd-rocm.md) ·
[Apple Silicon (MPS)](docs/apple-mps.md) · [Google TPU](docs/google-tpu.md).

## 2. Detect the hardware

See what LM7 can use locally, and why it would choose a given backend:

```bash
lm7 doctor                    # environment and install check
lm7 targets                   # detected hardware targets
lm7 backends                  # registered compiler backends
lm7 explain --target auto     # which backend LM7 would pick, and why
```

Add `--json` to any command for machine-readable output. The same CLI is
available as `python -m lm7`.

## 3. Compile a local model

`target="auto"` selects hardware for you; the first real call compiles that
input signature:

```python
compiled = lm7.compile(model.eval(), target="auto")
result = compiled(example_input)

print(compiled.target, compiled.selected_backend)
```

Pin the hardware target, the backend, or both:

```python
lm7.compile(model, target="cpu")
lm7.compile(model, target="nvidia:sm89")
lm7.compile(model, target="amd:gfx942")
lm7.compile(model, target="apple")
lm7.compile(model, target="nvidia", backend="tensorrt")
lm7.compile(model, target="tpu", backend="openxla")
```

| Backend | Availability | Purpose |
| --- | --- | --- |
| `eager` | Any detected PyTorch device | Reference execution and fallback |
| `inductor` | PyTorch with `torch.compile` | Default JIT compiler |
| `aot_inductor` | CPU/Apple prototype | Persistent ahead-of-time `.pt2` package |
| `tensorrt` | Optional NVIDIA prototype | Torch-TensorRT JIT engine |
| `openxla` | Optional Google TPU prototype | PyTorch/XLA and OpenXLA JIT compiler |

`tensorrt` and `openxla` need version-matched extras and must be selected
explicitly: `pip install -e ".[tensorrt]"` (Torch-TensorRT 2.12.1 / PyTorch
2.12 / CUDA 13) or, on a TPU VM, `pip install -e ".[openxla]"`.

The environment variables `LM7_TARGET`, `LM7_BACKEND`, `LM7_FALLBACK`, and
`LM7_CACHE_DIR` set defaults; explicit function arguments take precedence.

## 4. Run a Hugging Face model

Install the optional dependencies, then run a causal-LM forward pass from a
Hugging Face URI:

```bash
python -m pip install -e ".[hf]"
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompt "The capital of France is" --target auto --backend auto
```

Validated compact, ungated models: `HuggingFaceTB/SmolLM2-135M-Instruct`,
`LiquidAI/LFM2.5-230M`, `unsloth/Llama-3.2-1B-Instruct` (an ungated mirror of
Meta's Llama-3.2-1B-Instruct), and `Qwen/Qwen3.5-0.8B` (hybrid architecture; its
first-call compilation is noticeably slower, about a minute locally). All four
also compile on a local Apple Silicon GPU, with slightly wider float16
tolerance on MPS than CUDA.

The command downloads through the normal Hugging Face cache, compiles the
forward pass, and reports the selected target and backend, first-call and
steady-call time, and the predicted next token (`--json` for structured
output). The JIT result is process-local: weights, tokenizers, and a persistent
GPU executable are not yet packaged.

The Python example additionally validates logits and deterministic generation:

```bash
python examples/hf_causal_lm.py --model hf://HuggingFaceTB/SmolLM2-135M-Instruct
```

**Experimental low-bit inference (NVIDIA only).** Install TorchAO and select
quantization explicitly:

```bash
python -m pip install -e ".[hf,torchao]"
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target nvidia --backend inductor --dtype bfloat16 \
  --quantization int8-weight-only
```

Use `--quantization fp8-weight-only` on NVIDIA Ada (`sm89`), Hopper (`sm90`), or
newer GPUs. LM7 quantizes MLP linear weights only, leaves `lm_head` in BF16, and
reports model storage, quantization time, latency, and peak GPU memory. This
path is validated for SmolLM2-135M and rejects unvalidated model IDs; it stays
opt-in because small models may be slower on other shapes and hardware.

## 5. Export an artifact

Capture a model with `torch.export` and reload it in another process:

```python
artifact = lm7.export(model, args=(example_input,), target="cpu", output="model.lm7")

loaded = lm7.load_artifact("model.lm7")
output = loaded(example_input)
```

An `.lm7` artifact is a directory with a versioned manifest, checksums, and a
PyTorch `.pt2` program. Use `backend="aot_inductor"` for the persistent CPU/Apple
AOT prototype. Artifacts stay specific to compatible PyTorch, runtime, and
hardware versions — they are not a stable cross-version ABI.

Combine per-target artifacts into one bundle and select at load time, from
Python or the CLI:

```python
lm7.create_bundle(["build/cpu.lm7", "build/nvidia.lm7"], output="model.bundle.lm7")
deployed = lm7.load_bundle("model.bundle.lm7").load(target="auto")
```

```bash
lm7 bundle create model.bundle.lm7 build/cpu.lm7 build/nvidia.lm7
lm7 bundle inspect model.bundle.lm7      # add --json for structured output
```

## Examples and more

```bash
python examples/basic_mlp.py                 # CPU
python examples/cuda_mlp.py --target nvidia   # NVIDIA
python examples/mac_mlp.py                    # Apple Silicon
python examples/local_targets.py --require-nvidia   # CPU vs NVIDIA parity
python benchmarks/local.py --target cpu nvidia --backend eager inductor
```

More examples live in [`examples/`](examples), and benchmarks in
[`benchmarks/`](benchmarks). See [development and testing](docs/development.md)
for environment checks, GPU integration tests, and compiler IR output, and
[architecture](docs/architecture.md) for the backend and artifact design.

## Current limitations

- Inference only; training and backward compilation are unsupported.
- Only local PyTorch devices are detected.
- JIT compiled callables and TensorRT engines are process-local.
- AOTInductor is validated only for CPU and Apple Silicon (MPS) and uses Beta
  PyTorch APIs.
- AMD ROCm, Apple Silicon (MPS), and OpenXLA TPU support are initial
  single-process integrations without physical-hardware CI.
- Quantization, distributed inference, remote hardware, and a stable compiled
  artifact ABI are future work.

## License

LM7 is licensed under the [BSD 3-Clause License](LICENSE).
