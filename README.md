# LM7

[![CI](https://github.com/lmontigny/lm7/actions/workflows/ci.yml/badge.svg)](https://github.com/lmontigny/lm7/actions/workflows/ci.yml)

LM7 is a small, PyTorch-first compiler orchestration layer for local inference.
Hand it a model you already run in PyTorch — a full pretrained network, a
Hugging Face causal LM, a vision model, or a single layer — and you get back a
normal callable. Name the hardware yourself, or let LM7 detect what the machine
has.

```python
import lm7
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct").eval()

model = lm7.compile(model, target="auto")  # LM7 detects the best local device
model = lm7.compile(model, target="nvidia:sm89")  # or pin it exactly
```

Either way, `lm7.compile` returns a normal callable module, so the rest of your
code does not change — no device juggling, no per-vendor branches, no manual
compile cache. LM7 resolves the target, picks a compatible compiler, moves the
inputs, compiles once per input shape, and falls back to plain PyTorch if a
backend cannot handle the model.

**LM7 does not write kernels or a compiler of its own.** Each vendor already
ships a good one, and LM7 drives it — so the same call site reaches a different
vendor toolchain depending only on the target you ask for:

```python
lm7.compile(model, target="cpu")  # TorchInductor CPU kernels
lm7.compile(model, target="apple")  # Metal, via MPS
lm7.compile(model, target="tpu")  # PyTorch/XLA and OpenXLA
lm7.compile(model, target="nvidia", backend="tensorrt")  # Torch-TensorRT
```

You do not install or learn five toolchains to try a second device; you change a
string, and `lm7 doctor` tells you what is missing if anything is. The corollary
is that LM7's reach is bounded by what those vendor toolchains already support —
adding hardware means wiring up its compiler, not writing one, which is what the
evaluation plans under [supported hardware](#supported-hardware) work through.

### What this replaces

Retargeting by hand means a branch per vendor, because the probe, the device
string, and the compiler call are all different:

```python
if torch.cuda.is_available():
    # True for NVIDIA *and* AMD ROCm; torch.version.hip is what tells them apart
    device = torch.device("cuda", 0)
    compiled = torch.compile(model.to(device))
    # ...unless you want TensorRT, which is a different import and backend:
    #   import torch_tensorrt; torch.compile(model, backend="tensorrt")
elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
    device = torch.device("xpu", 0)
    compiled = torch.compile(model.to(device))
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # note: no ordinal, unlike the others
    compiled = torch.compile(model.to(device))
elif tpu_runtime_is_really_a_tpu():  # importable torch_xla is not enough
    import torch_xla

    device = torch_xla.device(0)
    compiled = torch.compile(model.to(device), backend="openxla")
else:
    compiled = model  # eager fallback

inputs = move_every_tensor(inputs, device)  # yours to write
```

And the branch is the easy part. You also own the traps: `torch.compile` is lazy,
so a compile failure surfaces on the *first call* and a fallback has to wrap that
call rather than the compile; TPU needs `torch.no_grad()` rather than
`torch.inference_mode()`, because XLA tracing depends on the tensor version
counters `inference_mode` removes; and each new input shape recompiles, so
caching is on you.

LM7 is that ladder, written once and tested:

```python
compiled = lm7.compile(model, target="auto")  # or target="apple", "tpu", ...
```

> [!WARNING]
> LM7 is an early inference-only prototype. Model coverage and compiled-artifact
> compatibility are not yet stable.

## How it works

- **Target vs. backend are separate.** A *target* is where the model runs
  (`cpu`, `nvidia`, `apple`, `tpu`, …); a *backend* is the vendor compiler used
  to get there (`eager`, `inductor`, `tensorrt`, …). Pin either, or let LM7
  choose. This split is what makes hardware swappable: the same module and the
  same call site work on any target that has a compatible backend.
- **Detection is automatic.** `target="auto"` prefers a detected GPU or
  accelerator and otherwise uses the CPU.
- **JIT compilation is lazy and per-input-shape.** With a JIT backend nothing
  compiles until the first call; that call is slow and later calls are fast. AOT
  backends move that cost out of the process entirely — see
  [JIT vs. AOT](#jit-vs-aot).
- **Fallback is safe by default.** If a backend fails to compile, LM7 falls
  back to PyTorch eager and warns. Use `fallback="error"` to stop instead.

## Supported hardware

`lm7 targets` reports what is actually present on the current machine. LM7
detects local PyTorch devices only — it does not install drivers or vendor
toolchains, and it never reaches remote hardware.

| Target | Detected via | Default backend | Status |
| --- | --- | --- | --- |
| `cpu`, `cpu:arm64` | always present, listed last | `inductor` | Supported, covered by CI |
| `nvidia`, `nvidia:sm89` | `torch.cuda` on a CUDA build | `inductor` (or `tensorrt`) | Supported, no physical-GPU CI |
| `amd`, `amd:gfx942` | `torch.cuda` with `torch.version.hip` | `inductor` | Initial integration, no CI |
| `apple` | `torch.backends.mps` | `inductor` | Initial integration, wider float16 tolerance |
| `intel` | `torch.xpu` | `inductor` | Detected and plannable, least exercised |
| `tpu` | a real PJRT `TPU` runtime | `openxla` | Initial integration, TPU VM only |

Targets accept an optional qualifier — an architecture (`nvidia:sm89`,
`amd:gfx942`), a device model, or an ordinal for multi-GPU hosts.
`target="auto"` prefers the first detected GPU or accelerator and falls back to
CPU.

**Under evaluation, not yet selectable.** Each has a written plan and a
measurement harness, and none is registered as a backend or reachable from
automatic planning:

| Hardware | Vendor compiler | Plan | Harness |
| --- | --- | --- | --- |
| Intel CPU, GPU, NPU | OpenVINO | [plan](docs/openvino-evaluation.md) | `benchmarks/openvino_eval.py` |
| AMD GPU | MIGraphX | [plan](docs/amd-migraphx.md) | `benchmarks/migraphx.py` |
| Qualcomm Hexagon NPU | Hexagon-MLIR | [plan](docs/qualcomm-hexagon.md) | `benchmarks/hexagon.py` |

The OpenVINO evaluation is furthest along: on an Apple M4 Pro at matched FP32
precision, OpenVINO IR artifacts beat eager PyTorch on real models (1.3x on
ResNet-18, 2.2x on SmolLM2 prefill) while losing on a trivial MLP, and the
`torch.compile` path was consistently worse than the IR path. Those numbers come
from an ARM host, so the Intel measurements that would actually decide the
question are still outstanding.

`aws:trainium` parses into a `TargetSpec` with `remote=True`, but nothing
detects or executes it. Hexagon would need the same remote handling, which is
why it is a plan rather than a backend.

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

| Backend | Underlying compiler | Model | Targets | Priority |
| --- | --- | --- | --- | --- |
| `inductor` | TorchInductor (`torch.compile`) | JIT | cpu, nvidia, amd, intel, apple | 100 |
| `openxla` | PyTorch/XLA + OpenXLA | JIT | tpu | 100 |
| `aot_inductor` | AOTInductor (`.pt2` package) | **AOT** | cpu, apple | 90 |
| `tensorrt` | Torch-TensorRT | JIT | nvidia | 90 |
| `eager` | none — plain PyTorch | none | any detected device | 0 |

With `backend="auto"` LM7 picks the highest-priority backend that reports support
for the resolved target, so CPU, NVIDIA, AMD, Intel, and Apple default to
`inductor` and TPU defaults to `openxla`. `eager` wins only when nothing else
supports the target, or when a compile fails and `fallback="warn"` takes over.

`tensorrt` and `openxla` need version-matched extras and must be selected
explicitly: `pip install -e ".[tensorrt]"` (Torch-TensorRT 2.12.1 / PyTorch
2.12 / CUDA 13) or, on a TPU VM, `pip install -e ".[openxla]"`.

The environment variables `LM7_TARGET`, `LM7_BACKEND`, `LM7_FALLBACK`, and
`LM7_CACHE_DIR` set defaults; explicit function arguments take precedence.

### JIT vs. AOT

The difference is *when* compilation happens and *whether the result outlives
the process*.

**JIT** (`inductor`, `tensorrt`, `openxla`) compiles inside your process, on the
first call, once per input signature:

```python
compiled = lm7.compile(model, target="cpu")  # returns immediately, compiles nothing
out = compiled(example_input)  # first call: compiles, then runs
out = compiled(example_input)  # subsequent calls: fast
out = compiled(other_shape)  # new signature: compiles again
```

Nothing is written that another process can use. Restarting pays the compile
cost again, and TensorRT engines and JIT callables cannot be shipped.

**AOT** happens up front, through `lm7.export`, and writes an `.lm7` directory
another process can load. There are two levels of it, which is worth being
precise about:

```python
# Capture only: a portable ExportedProgram. PyTorch still generates kernels
# when it runs, so this removes tracing, not compilation.
lm7.export(model, args=(example_input,), target="cpu", output="model.lm7")

# Capture and compile: a persistent AOTInductor package with kernels baked in.
lm7.export(
    model, args=(example_input,), target="cpu", backend="aot_inductor", output="model-aot.lm7"
)

loaded = lm7.load_artifact("model-aot.lm7")  # another process, nothing to compile
out = loaded(example_input)
```

Use JIT while iterating locally, and `aot_inductor` when you want the compile
cost paid once at build time instead of on every process start. Two caveats: AOT
fixes the input signature captured at export time, and `.lm7` artifacts are tied
to compatible PyTorch, runtime, and hardware versions rather than being a stable
cross-version ABI.

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

Quantization is opt-in and covered in its own section below.

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

## Quantization

Quantization stores or computes a tensor in fewer bits than the model was
trained in. There are two halves to it, and **LM7 currently implements only the
first**:

- **Weight quantization** shrinks the stored parameters. Weights are converted
  once, up front, and dequantized on the fly during the matmul, which keeps
  arithmetic in a higher precision. The win is memory footprint and bandwidth,
  and it needs no calibration data.
- **Activation quantization** also narrows the tensors flowing between layers, so
  the matmul itself runs in low precision. That is where the larger speedups
  come from, but it needs calibration or a quantization-aware recipe to pick
  per-tensor scales, and it costs accuracy more readily. **LM7 does not do
  this today.**

Everything below is therefore *weight-only*: activations, and the accumulation
inside each matmul, stay in BF16.

### Supported data types

| `--quantization` | Weight storage | Compute dtype | Requires |
| --- | --- | --- | --- |
| `none` (default) | as loaded | FP32 / FP16 / BF16 | nothing |
| `int8-weight-only` | INT8 | BF16 | NVIDIA GPU |
| `fp8-weight-only` | FP8 | BF16 | NVIDIA Ada (`sm89`), Hopper (`sm90`), or newer |

Both modes force BF16 compute: `--dtype` must be `auto` or `bfloat16`, and
`auto` resolves to BF16 whenever quantization is on. `--backend` must be `auto`
or `inductor`. Anything else raises `UnsupportedModelError` rather than silently
degrading.

Which layers get converted differs between the two, which matters for the
memory saving you should expect:

- `int8-weight-only` converts **every `nn.Linear` except `lm_head`**, including
  the attention projections.
- `fp8-weight-only` converts **only the MLP linears** (`.mlp.` in the module
  path), leaving attention and `lm_head` in BF16.

### TorchAO

The conversion itself is [TorchAO](https://github.com/pytorch/ao)'s, not LM7's.
LM7 pins `torchao==0.17.0` and calls `torchao.quantization.quantize_()` with
`Int8WeightOnlyConfig` or `Float8WeightOnlyConfig`, passing a filter that selects
the modules above. The quantized model is then compiled and run through the
normal `inductor` path, so quantization composes with the rest of LM7 rather than
being a separate execution path.

```bash
python -m pip install -e ".[hf,torchao]"
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target nvidia --backend inductor --dtype bfloat16 \
  --quantization int8-weight-only
```

The run reports model storage bytes, quantization time, first-call and
steady-state latency, and peak GPU memory, so the footprint/latency trade is
visible rather than assumed.

> [!NOTE]
> This path is validated for exactly one model,
> `HuggingFaceTB/SmolLM2-135M-Instruct`, and rejects every other model id. It is
> NVIDIA-only and stays opt-in because weight-only quantization can be *slower*
> at small batch sizes, where dequantization overhead outweighs the bandwidth
> saved. Treat it as a measurement tool, not a default.

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
- Only local PyTorch devices are detected, and only the hardware listed under
  [supported hardware](#supported-hardware).
- JIT compiled callables and TensorRT engines are process-local; only
  `aot_inductor` and `lm7.export` produce something another process can load.
- AOTInductor is validated only for CPU and Apple Silicon (MPS) and uses Beta
  PyTorch APIs.
- AMD ROCm, Apple Silicon (MPS), Intel XPU, and OpenXLA TPU support are initial
  single-process integrations without physical-hardware CI.
- Intel OpenVINO, AMD MIGraphX, and Qualcomm Hexagon are evaluation plans with
  measurement harnesses, not usable backends.
- Quantization, distributed inference, remote hardware, and a stable compiled
  artifact ABI are future work.

## License

LM7 is licensed under the [BSD 3-Clause License](LICENSE).
