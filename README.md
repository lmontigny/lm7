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
lm7.compile(model, target="cpu")  # TorchInductor, C++/OpenMP kernels
lm7.compile(model, target="nvidia")  # TorchInductor, Triton kernels + cuBLAS/cuDNN
lm7.compile(model, target="nvidia", backend="tensorrt")  # Torch-TensorRT instead
lm7.compile(model, target="apple")  # TorchInductor, Metal via MPS
lm7.compile(model, target="tpu")  # PyTorch/XLA and OpenXLA
lm7.compile(model, target="tenstorrent")  # tt-xla, tt-mlir, tt-metal
```

You do not install or learn five toolchains to try a second device; you change a
string, and `lm7 doctor` tells you what is missing if anything is. The corollary
is that LM7's reach is bounded by what those vendor toolchains already support —
adding hardware means wiring up its compiler, not writing one, which is what the
evaluation plans under [supported hardware](#supported-hardware) work through.

For the per-vendor code this replaces — the detection branches, the device-string
inconsistencies, and the behaviour you would otherwise have to know about — see
[what LM7 replaces](docs/what-this-replaces.md).

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

| Vendor | Hardware | `target` | Backends | Status |
| --- | --- | --- | --- | --- |
| Intel, AMD, Arm, Apple | CPU (x86-64, ARM64) | `cpu` | `inductor`, `aot_inductor`, `openvino`, `onnxruntime`, `eager` | Supported |
| NVIDIA | GPU | `nvidia` | `inductor`, `aot_inductor`, `tensorrt`, `onnxruntime`, `iree_vulkan`, `eager` | Supported; ORT/Vulkan experimental |
| AMD | GPU (ROCm/Vulkan) | `amd` | `inductor`, `iree_vulkan`, `eager` | Supported; Vulkan AOT experimental |
| Apple | GPU (Metal) | `apple` | `inductor`, `aot_inductor`, `eager` | Supported |
| Intel | GPU (XPU/Vulkan) | `intel` | `inductor`, `iree_vulkan`, `eager` | Supported; Vulkan AOT experimental |
| Google | TPU | `tpu` | `openxla`, `eager` | Supported |
| Tenstorrent | Wormhole, Blackhole | `tenstorrent` | `tenstorrent`, `eager` | Supported |
| Android, iOS, embedded | CPU (ARM64, x86-64) | `cpu` | `executorch` | Export only, [ExecuTorch](docs/executorch.md) |
| Intel | NPU | — | — | Not supported, [OpenVINO plan](docs/openvino-evaluation.md) |
| Qualcomm | Hexagon NPU | — | — | Not supported, [Hexagon plan](docs/qualcomm-hexagon.md) |
| AWS | Trainium | `aws:trainium` | — | Parses only, never executed |

Any x86-64 or ARM64 CPU runs through the `cpu` target, Intel and AMD included —
and that is the only path with CI coverage. A vendor listed twice has an
*additional* accelerator; it does not mean its CPU is unsupported.

[OpenVINO](docs/openvino-evaluation.md) is a registered backend for the `cpu`
target, but it is opt-in: it ranks below Inductor and AOTInductor, so
`backend="auto"` never selects it. Ask for it with `backend="openvino"`. It
compiles to Intel's IR format, which runs in a process without PyTorch installed
— as does `stablehlo` below, on any hardware with a PJRT plugin.

[IREE Vulkan](docs/iree-vulkan.md) is an explicit, export-only backend for
NVIDIA, AMD, and Intel GPUs. It lowers fixed-shape tensor graphs to a persistent
VMFB containing Vulkan/SPIR-V code. It is not considered by `backend="auto"`,
and full Hugging Face model coverage is not claimed yet.

[ONNX Runtime](docs/onnxruntime.md) is an opt-in backend for CPU and NVIDIA. It
uses PyTorch's `torch.export`-based ONNX exporter and an explicit execution
provider, and supports both lazy compilation and persistent `.onnx` artifacts.
Fixed-shape SmolLM2 logits, bounded dynamic MLP batches, CPU, and CUDA have been
validated locally.

[MIGraphX](docs/amd-migraphx.md) on AMD GPU is still under evaluation — it has a
benchmark harness but no registered backend.

[Tenstorrent](docs/tenstorrent.md) is the one vendor here whose whole stack is
open source, and LM7 drives it end to end: `torch.compile(..., backend="tt")`
goes through [tt-xla](https://github.com/tenstorrent/tt-xla)'s PJRT plugin to
StableHLO, [tt-mlir](https://github.com/tenstorrent/tt-mlir), and
[tt-metal](https://github.com/tenstorrent/tt-metal). The plugin ships from
Tenstorrent's own package index, so it needs an explicit `--extra-index-url` and
a matching tt-kmd driver.

Backends are listed highest priority first, so the leftmost is what
`backend="auto"` picks and `eager` is the fallback. `tensorrt` and `openxla` also
need their optional extra installed — see [backends](#3-compile-a-local-model)
for what each one compiles with.

Run `lm7 targets` to see what is actually present on your machine. LM7 detects
local PyTorch devices only, and installs no drivers or vendor toolchains.

Add a qualifier to pin an architecture, model, or ordinal — `nvidia:sm89`,
`amd:gfx942`, `cpu:arm64`. `target="auto"` takes the first detected GPU or
accelerator and falls back to CPU.

The GPU and TPU integrations are early and have no physical-hardware CI.
NVIDIA Inductor, quantization, and TensorRT have been exercised on a local Ada
GPU; see the [TensorRT evaluation](docs/nvidia-tensorrt-evaluation.md) for the
measured backend trade-offs.

The tasks below follow the usual path: install, detect the hardware, compile a
local model, run a Hugging Face model, and export an artifact.

## 1. Install

LM7 needs Python 3.10 or newer and a PyTorch build that matches the target
machine. It does **not** install GPU drivers, CUDA/ROCm toolchains, Xcode,
PyTorch/XLA, or platform C++ compilers.

```bash
git clone https://github.com/lmontigny/lm7.git
cd lm7
uv venv --python 3.12
uv pip install -e .            # add ".[dev]" for pytest + ruff
```

`uv` also picks the right PyTorch wheel for the machine, which matters here more
than usual — CPU, CUDA, and ROCm builds come from different indexes:

```bash
uv pip install torch --torch-backend=auto
```

Then either activate the environment (`source .venv/bin/activate`) or prefix
commands with `uv run`. Without `uv`, the standard tools work unchanged:
`python3 -m venv .venv && source .venv/bin/activate && python -m pip install -e .`.

Per-hardware setup: [CPU](docs/cpu.md) ·
[NVIDIA](docs/development.md#nvidia-cuda) · [AMD ROCm](docs/amd-rocm.md) ·
[Apple Silicon (MPS)](docs/apple-mps.md) · [Google TPU](docs/google-tpu.md) ·
[Tenstorrent](docs/tenstorrent.md).

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
lm7.compile(model, target="tenstorrent:blackhole")
```

| Backend | Underlying compiler | Generates | Model | Targets | Priority |
| --- | --- | --- | --- | --- | --- |
| `inductor` | TorchInductor (`torch.compile`) | Triton kernels on GPU, C++/OpenMP on CPU, plus vendor library calls | JIT | cpu, nvidia, amd, intel, apple | 100 |
| `openxla` | PyTorch/XLA + OpenXLA | XLA HLO fusions, target IR | JIT | tpu | 100 |
| `tenstorrent` | tt-xla + tt-mlir + tt-metal | StableHLO, then a TT-NN flatbuffer | JIT | tenstorrent | 100 |
| `aot_inductor` | AOTInductor | persistent `.pt2` package | **AOT** | cpu, apple, nvidia | 90 |
| `tensorrt` | Torch-TensorRT | TensorRT engine | JIT | nvidia | 90 |
| `openvino` | Intel OpenVINO | persistent IR (`.xml` + `.bin`) | **AOT** | cpu (Intel) | 80 |
| `onnxruntime` | PyTorch ONNX exporter + ONNX Runtime | persistent `.onnx` model | JIT + **AOT** | cpu, nvidia | 70 |
| `iree_vulkan` | IREE Vulkan HAL | persistent VMFB with SPIR-V | **AOT export only** | nvidia, amd, intel | explicit |
| `stablehlo` | PyTorch/XLA + OpenXLA | portable StableHLO for any PJRT plugin | **AOT**, export only | any | — |
| `executorch` | ExecuTorch + XNNPACK | `.pte` for phones and embedded CPUs | **AOT**, export only | cpu | — |
| `eager` | none — plain PyTorch | nothing | none | any detected device | 0 |

On NVIDIA both GPU paths are available and `inductor` is the default: TorchInductor
schedules and fuses the graph, then emits **Triton** kernels and calls into cuBLAS
and cuDNN where those win. `tensorrt` is the opt-in alternative and is
deliberately lower priority, because TensorRT's engine build is slower and its
model coverage is narrower. On a local RTX 4070 SUPER, TensorRT beat Inductor
1.76x on a fixed-shape SmolLM2 FP16 forward pass but lost on two small MLP
workloads and took 56 seconds for the SmolLM2 first call. See the
[evaluation](docs/nvidia-tensorrt-evaluation.md). LM7 never invokes Triton
itself — TorchInductor owns kernel generation and selection.

With `backend="auto"` LM7 picks the highest-priority backend that reports support
for the resolved target, so CPU, NVIDIA, AMD, Intel, and Apple default to
`inductor`, TPU defaults to `openxla`, and Tenstorrent defaults to
`tenstorrent`. `eager` wins only when nothing else supports the target, or when
a compile fails and `fallback="warn"` takes over.

`tensorrt`, `openxla`, `openvino`, `onnxruntime`, and `iree_vulkan` need extras
and must be selected explicitly: `uv pip install -e ".[tensorrt]"`
(Torch-TensorRT 2.12.1 / PyTorch 2.12 / CUDA 13),
`uv pip install -e ".[openvino]"` on an Intel CPU, or choose exactly one of
`".[onnxruntime]"` for CPU and `".[onnxruntime-gpu]"` for CUDA 13. Use
`uv pip install -e ".[iree-vulkan]"` for Vulkan AOT or, on a TPU VM,
`uv pip install -e ".[openxla]"`. The two ONNX Runtime wheels expose the same
module and must not be installed together. The Tenstorrent plugin is not on PyPI:

```bash
uv pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
```

The environment variables `LM7_TARGET`, `LM7_BACKEND`, `LM7_FALLBACK`, and
`LM7_CACHE_DIR` set defaults; explicit function arguments take precedence.

### JIT vs. AOT

The difference is *when* compilation happens and *whether the result outlives the
process*.

- **JIT** (`inductor`, `tensorrt`, `onnxruntime`, `openxla`) compiles inside your
  process, on the first call, once per input signature. Nothing is written that another
  process can use, so restarting recompiles.
- **AOT** (`lm7.export`) compiles up front and writes an `.lm7` directory another
  process loads with no compile step. Use `backend="aot_inductor"` to bake in
  kernels; the default `backend="export"` captures a portable `ExportedProgram`
  but still generates kernels at run time.

Use JIT while iterating locally, and AOT when you want the compile cost paid once
at build time instead of on every process start.

See [JIT vs. AOT](docs/jit-vs-aot.md) for the two export levels, bundles, and the
caveats — an AOT artifact is fixed to its captured input signature unless a
dimension was captured as dynamic, and artifacts are not a stable ABI.

## 4. Run a Hugging Face model

Install the optional dependencies, then run a causal-LM forward pass from a
Hugging Face URI:

```bash
uv pip install -e ".[hf]"
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

For greedy token generation, use the static KV-cache path:

```bash
lm7 model generate hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompt "The capital of France is" --max-new-tokens 32 --target nvidia
```

Generation runs the prompt prefill eagerly, then uses one Inductor-compiled,
fixed-shape single-token decode graph against a static KV cache. The first run
includes compilation; the reported steady run reuses that graph for every
decoded token. This path is currently greedy-only and requires a Transformers
causal LM that supports static caching. See
[compiled Hugging Face generation](docs/huggingface-generation.md).

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
PyTorch `.pt2` program. Use `backend="aot_inductor"` for the persistent CPU,
Apple, and NVIDIA AOT prototype, or `backend="openvino"` on Intel CPU to add
OpenVINO IR (`compiled_model.xml` + `.bin`). `backend="onnxruntime"` adds an ONNX
model, and `backend="iree_vulkan"` adds a Vulkan VMFB for an NVIDIA, AMD, or
Intel GPU. Artifacts stay specific to compatible compiler, runtime, and hardware
versions — they are not a stable cross-version ABI.

The ONNX Runtime path writes `compiled_model.onnx` and records its execution
provider in the manifest:

```python
artifact = lm7.export(
    model,
    args=(example_input,),
    target="cpu",
    backend="onnxruntime",
    output="model-onnx.lm7",
)
```

Use `target="nvidia:sm89"` with the GPU wheel to select
`CUDAExecutionProvider`; LM7 disables CPU provider fallback by default for that
target. See the [ONNX Runtime guide](docs/onnxruntime.md) for dynamic shapes,
provider options, CUDA compatibility, SmolLM2 coverage, and the initial 2 GiB
embedded-weight limit.

The IREE Vulkan path compiles without a local GPU and loads the runtime lazily:

```python
artifact = lm7.export(
    model,
    args=(example_input,),
    target="nvidia:sm89",
    backend="iree_vulkan",
    output="model-vulkan.lm7",
)
```

It currently requires fixed shapes and tensor-only I/O. See the
[IREE Vulkan guide](docs/iree-vulkan.md), including the validated WSL compile →
native Windows RTX 4070 SUPER execution workflow and why WebGPU is separate.

`backend="stablehlo"` writes a third kind of payload: portable StableHLO plus
one `.npy` per weight, executed by a PJRT plugin chosen at load time. The same
artifact bytes have been measured on a CPU and an NVIDIA sm89 plugin with no
PyTorch installed, so unlike every other compiled payload it is not bound to the
device it was exported for. It needs PyTorch/XLA to lower, which pins PyTorch, so
install it into its own environment:

```bash
uv pip install -e ".[stablehlo]"   # torch_xla, ABI-tied to a matching PyTorch
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct model.lm7 \
  --target cpu --backend stablehlo
```

`backend="executorch"` is the edge path: it writes a `.pte`, the file Android
and iOS load, which the ExecuTorch C++ runtime executes with no Python and no
PyTorch on the device. Like `stablehlo` it is not bound to the machine that
built it — the XNNPACK delegate covers ARM64 and x86-64 alike, which is also why
it is the one mobile route this project can validate on ordinary CI:

```bash
python3 -m venv .venv-et && .venv-et/bin/python -m pip install -e ".[executorch]"
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct model.lm7 \
  --target cpu --backend executorch
```

XNNPACK is a CPU delegate, so `--target cpu` is required; reaching phone NPUs
(Core ML, Qualcomm QNN, MediaTek) means adding those delegates, each needing a
macOS host or a vendor SDK. See [ExecuTorch](docs/executorch.md) for the measured
lowering cost, the partition-coverage field in the manifest, and why artifact
size makes quantization matter more here than elsewhere.

An NVIDIA AOTInductor artifact packages native GPU code, while ONNX Runtime
packages a provider-neutral ONNX graph. `inductor` and `tensorrt` compile on the
first call and keep nothing. Packaging AOTInductor needs a CUDA toolkit, which
the PyTorch CUDA wheel does not include, so install the extra first:

```bash
uv pip install -e ".[cuda-aot]"
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct model.lm7 \
  --target nvidia --backend aot_inductor
```

A Hugging Face model can be exported without writing any PyTorch, straight from
the CLI:

```bash
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct model.lm7 \
  --target cpu --backend aot_inductor
```

The example inputs come from tokenizing `--prompt`, so by default the artifact is
pinned to that many tokens. `--dynamic-seq` captures the sequence length as a
bounded dynamic dimension instead, and one artifact then serves every prompt
length inside those bounds:

```bash
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct model.lm7 \
  --target nvidia --backend aot_inductor --dynamic-seq        # bounds from the model config
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct model.lm7 \
  --target nvidia --backend aot_inductor --dynamic-seq 1:512  # or set them
```

The bounds are recorded in the manifest and enforced on every call, so a prompt
outside them raises rather than producing quietly wrong output. A dynamic capture
uses the model's eager attention path — see [JIT vs. AOT](docs/jit-vs-aot.md) for
why, and for the batch dimension, which stays fixed.

LM7 captures a logits-only graph either way, because a causal LM's
`CausalLMOutputWithPast` return value cannot be deserialized by
`torch.export.load`; the reloaded artifact takes `input_ids` and
`attention_mask` and returns a logits tensor.

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

Quantization stores or computes tensors in fewer bits than the model was trained
in. It has two halves, and **LM7 implements only the first**:

- **Weight quantization** shrinks the stored parameters and dequantizes inside
  the matmul, so arithmetic stays in higher precision. Saves memory and
  bandwidth; needs no calibration.
- **Activation quantization** also narrows the tensors between layers so the
  matmul itself runs low-precision — bigger speedups, but it needs calibration
  and costs accuracy. **Not implemented.**

So everything LM7 offers is weight-only, with BF16 compute:

| `--quantization` | Weight storage | Compute | Requires |
| --- | --- | --- | --- |
| `none` (default) | as loaded | FP32 / FP16 / BF16 | nothing |
| `int8-weight-only` | INT8 | BF16 | NVIDIA GPU |
| `fp8-weight-only` | FP8 | BF16 | NVIDIA Ada (`sm89`) or newer |

```bash
uv pip install -e ".[hf,torchao]"
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target nvidia --backend inductor --dtype bfloat16 \
  --quantization int8-weight-only
```

The conversion is [TorchAO](https://github.com/pytorch/ao)'s — LM7 pins
`torchao==0.17.0` and calls `quantize_()` with a module filter, then runs the
result through its normal `inductor` path. It is NVIDIA-only, validated for
`SmolLM2-135M-Instruct` alone, and can be *slower* at small batch sizes, so it
stays opt-in.

See [quantization](docs/quantization.md) for which layers each mode converts, the
validation gates, and the full caveats.

## Examples and more

```bash
python examples/basic_mlp.py                 # CPU
python examples/cuda_mlp.py --target nvidia   # NVIDIA
python examples/mac_mlp.py                    # Apple Silicon
python examples/tenstorrent_mlp.py            # Tenstorrent Wormhole / Blackhole
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
  `aot_inductor`, `openvino`, `onnxruntime`, `iree_vulkan`, and `lm7.export`
  produce something another process can load.
- Exported causal-LM artifacts capture a prefill-only graph. `--dynamic-seq`
  makes the sequence length variable within recorded bounds; the batch dimension
  stays fixed, and a KV-cache decode loop is not captured.
- AOTInductor is validated for CPU, Apple Silicon (MPS), and NVIDIA GPU, and uses
  Beta PyTorch APIs. On NVIDIA it packages against a CUDA toolkit the PyTorch
  wheel does not ship — install `".[cuda-aot]"`, and see
  [NVIDIA AOT](docs/development.md#nvidia-aot-inductor) for the WSL linker caveat.
- AMD ROCm, Apple Silicon (MPS), Intel XPU, OpenXLA TPU, and Tenstorrent support
  are initial single-process integrations without physical-hardware CI.
- ExecuTorch is export-only and XNNPACK-only, so the edge story is CPU: phone
  NPUs (Core ML, Qualcomm QNN, MediaTek, Exynos) are not wired up, artifacts are
  unquantized and static-shape, and validation is host x86-64 rather than a real
  phone. LM7 writes the `.pte`; deploying it into an app is ExecuTorch's tooling.
- Tenstorrent is JIT-only and single-card: the compiled flatbuffer does not
  outlive the process, multi-card sharding is not exposed, and model coverage is
  bounded by what tt-mlir lowers — see [Tenstorrent](docs/tenstorrent.md).
- OpenVINO is validated for Intel CPU only, and rejects bfloat16 models because
  its runtime exchanges tensors through NumPy. It returns tensors or tuples, so
  models whose forward returns a dataclass need a wrapper.
- IREE Vulkan is export-only and experimental: fixed shapes, tensor-only I/O,
  and FP32 MLP execution are the validated scope. Full causal LMs, dynamic
  sequences, KV caches, and WebGPU/browser execution remain future work; see
  the [IREE Vulkan guide](docs/iree-vulkan.md).
- ONNX Runtime is validated for CPU and NVIDIA CUDA. It returns CPU tensors even
  after CUDA execution because the initial adapter uses NumPy rather than I/O
  binding. Tensor-only inputs and flat outputs are supported; external-data ONNX
  packaging for models above the 2 GiB protobuf limit remains future work. See
  the [ONNX Runtime guide](docs/onnxruntime.md).
- AMD MIGraphX and Qualcomm Hexagon are evaluation plans with measurement
  harnesses, not usable backends. Replacing the `stablehlo` lowering with
  torch-mlir, which would unpin it from a matching PyTorch, is evaluated in
  [torch-mlir lowering](docs/torch-mlir-lowering-evaluation.md) and not adopted.
- Quantization is weight-only, NVIDIA-only, and validated per (model, mode)
  pair — see [quantization](docs/quantization.md) for the list and the
  measurements behind it.
- Distributed inference, remote hardware, and a stable compiled artifact ABI are
  future work.

## License

LM7 is licensed under the [BSD 3-Clause License](LICENSE).
