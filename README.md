# LM7

[![CI](https://github.com/lmontigny/lm7/actions/workflows/ci.yml/badge.svg)](https://github.com/lmontigny/lm7/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Keep your PyTorch model. Change the hardware target, not your application.**

LM7 is a PyTorch-first compiler orchestration layer for local inference. It
keeps the normal `nn.Module` interface while detecting hardware, selecting an
available compiler, moving inputs, caching compiled variants, and handling
controlled fallback.

```python
import lm7

compiled = lm7.compile(model.eval(), target="auto")
output = compiled(example_input)
```

> [!WARNING]
> **LM7 is an early, inference-only prototype.** Model coverage and
> compiled-artifact compatibility are not stable. CPU and Apple are the only
> targets with continuous integration coverage; other targets are tested by
> hand, export-tested, mock-tested, or parse-only as documented below. See
> [limitations](docs/limitations.md) before depending on it.

## Why not just `torch.compile`?

On CPU, NVIDIA, AMD, Intel GPU, and Apple Silicon, LM7 often *does* use
`torch.compile` with TorchInductor underneath. LM7 is not another compiler and
does not replace Inductor, TensorRT, OpenXLA, OpenVINO, or the other toolchains
it integrates.

`torch.compile(model)` is a good answer when you already know the hardware and
compiler you want to use. LM7 is for the case where the hardware can change.

The missing layer is everything around the compiler call:

- detecting NVIDIA, AMD ROCm, Intel XPU, Apple MPS, TPU, and other accelerators;
- normalizing their different device semantics;
- selecting an available compiler for the resolved target;
- moving nested inputs and caching variants by input signature;
- handling first-call compilation failures and controlled fallback;
- explaining backend selection and managing artifacts across compiler stacks.

Some targets are not an Inductor call at all: TPU uses PyTorch/XLA and OpenXLA,
Intel NPU uses OpenVINO, and other accelerators bring their own compiler and
runtime.

```python
lm7.compile(model, target="auto")
lm7.compile(model, target="nvidia", backend="tensorrt")
lm7.compile(model, target="amd")
lm7.compile(model, target="tpu")
lm7.compile(model, target="intel:npu")
```

**PyTorch and hardware vendors provide the compilers. LM7 provides the
vendor-neutral orchestration layer between them.**

If one `torch.compile(model)` call already covers your machine and deployment
needs, you probably do not need LM7. If your PyTorch application needs to
**survive a change of hardware**, LM7 is intended to make that change boring.

See [what LM7 replaces](docs/what-this-replaces.md) for the concrete per-vendor
code and behavior it centralizes.

## When should I use LM7?

Use LM7 when your PyTorch application needs to run on hardware you do not want
to hard-code: a developer laptop today, NVIDIA tomorrow, a TPU or another
accelerator later.

LM7 is useful when you want to:

- ship the same PyTorch application to machines with different hardware;
- compare compiler backends without rewriting application code;
- detect hardware and compiler availability at runtime;
- build, inspect, and load artifacts through one interface.

If you have one fixed target and already know the vendor stack you want, use
that stack directly. A fixed NVIDIA deployment running vLLM, for example, does
not need LM7.

## How it works

LM7 sits between one PyTorch model and the vendor toolchains that compile it.
`lm7.compile()` returns a normal callable; `lm7.export()` writes a versioned
`.lm7` artifact. Both accept the same target vocabulary.

![LM7 in five layers: the PyTorch model, the LM7 orchestrator, vendor backends, their lowering and runtime layers, and hardware](docs/figures/lm7-architecture.png)

- **Targets and backends are separate.** A target says where the model runs
  (`cpu`, `nvidia`, `apple`, `tpu`); a backend says which compiler gets it
  there (`inductor`, `tensorrt`, `openxla`).
- **Detection is automatic.** `target="auto"` prefers a detected accelerator and
  otherwise uses CPU.
- **Compilation is lazy and signature-aware.** The first call compiles; compiled
  variants are cached by input signature.
- **Fallback is controlled.** A failed backend can warn and fall back to eager,
  or `fallback="error"` can stop immediately.
- **Selection is inspectable.** `lm7 explain --target auto` reports which backend
  would be selected and why.

## Validated on real hardware

> [!NOTE]
> **NVIDIA:** RTX 4070 SUPER · H100 · RTX PRO 6000 Blackwell
>
> **CPU:** AMD EPYC x86-64 · Arm Neoverse N2/N3
>
> **Apple:** M3 Pro · M4 · M4 Pro
>
> **Google:** TPU v6e
>
> **Qualcomm:** Snapdragon 8 Elite

These machines have executed LM7 paths on physical hardware. Only CPU and Apple
MPS run in CI; the remaining machines were exercised manually. Exact parts,
backends, workloads, and known gaps are recorded in
[tested hardware](docs/tested-hardware.md).

## Integrated targets

Integration means that LM7 has target and backend code for a toolchain. It does
**not** mean every row has run on physical hardware.

| Vendor | Hardware | `target` | Integrated backends |
| --- | --- | --- | --- |
| Intel, AMD, Arm, Apple | CPU (x86-64, ARM64) | `cpu` | Inductor, AOTInductor, OpenVINO, ONNX Runtime, eager; explicit/export integrations |
| NVIDIA | GPU | `nvidia` | Inductor, AOTInductor, TensorRT, ONNX Runtime, eager, IREE Vulkan export |
| AMD | GPU (ROCm/Vulkan) | `amd` | Inductor, eager, IREE Vulkan export |
| Apple | GPU (Metal) | `apple` | Inductor, AOTInductor, eager, Core ML export |
| Intel | GPU (XPU/Vulkan) | `intel` | Inductor, eager, IREE Vulkan export |
| Arm | GPU (Mali/Vulkan) | `arm`, `arm:mali-g715` | IREE Vulkan export; never executed on device |
| Intel | NPU | `intel:npu` | OpenVINO; mock-tested |
| Google | TPU | `tpu` | OpenXLA, eager |
| Tenstorrent | Wormhole, Blackhole | `tenstorrent` | tt-xla/tt-mlir/tt-metal, eager; mock-tested |
| Mobile/embedded | CPU | `cpu` | ExecuTorch export |
| Qualcomm | Snapdragon 8 Elite HTP | `qualcomm:sm8750` | QNN export |
| AWS | Trainium | `aws:trainium` | Parse only; never executed |

AMD ROCm GPU, Intel XPU, Tenstorrent, Intel NPU, and Trainium have not run
through LM7 on real hardware. See [tested hardware](docs/tested-hardware.md) and
[limitations](docs/limitations.md#hardware-validation) for the evidence behind
each row.

## Quick start

LM7 requires Python 3.10+ and a PyTorch build matching the target machine. It
does **not** install GPU drivers, CUDA or ROCm, Xcode, PyTorch/XLA, or vendor
toolchains.

```bash
git clone https://github.com/lmontigny/lm7.git
cd lm7
uv venv --python 3.12
uv pip install -e .
uv pip install torch --torch-backend=auto
```

You still install the driver and compiler/runtime required by your hardware.
LM7 removes the per-vendor application glue and tells you what is missing
through `lm7 doctor`.

```bash
lm7 doctor
lm7 targets
lm7 backends
lm7 explain --target auto
```

Then compile a model without hard-coding its device:

```python
compiled = lm7.compile(model.eval(), target="auto")
result = compiled(example_input)

print(compiled.target, compiled.selected_backend)
```

Per-hardware setup: [CPU](docs/cpu.md) · [NVIDIA](docs/development.md#nvidia-cuda) ·
[AMD ROCm](docs/amd-rocm.md) · [Apple Silicon](docs/apple-mps.md) ·
[Google TPU](docs/google-tpu.md) · [Tenstorrent](docs/tenstorrent.md).

## Compiler and backend overview

| Backend | Compiler/runtime | Mode | Targets |
| --- | --- | --- | --- |
| `inductor` | TorchInductor | JIT | CPU, NVIDIA, AMD, Intel GPU, Apple |
| `openxla` | PyTorch/XLA + OpenXLA | JIT | TPU |
| `tenstorrent` | tt-xla + tt-mlir + tt-metal | JIT | Tenstorrent |
| `aot_inductor` | AOTInductor | AOT | CPU, NVIDIA, Apple |
| `tensorrt` | Torch-TensorRT | JIT/AOT | NVIDIA |
| `openvino` | OpenVINO | AOT | Intel CPU/NPU |
| `onnxruntime` | ONNX Runtime | JIT/AOT | CPU, NVIDIA |
| `iree_vulkan` | IREE Vulkan | Export | NVIDIA, AMD, Intel, Arm GPU |
| `executorch` | ExecuTorch | Export | CPU/mobile |
| `qnn` | ExecuTorch + Qualcomm QNN | Export | Snapdragon 8 Elite |
| `coreml` | ExecuTorch + Core ML | Export | Apple |
| `stablehlo` | OpenXLA StableHLO + PJRT | Export | Any PJRT target |
| `litert`, `tvm`, `eager` | LiteRT, TVM, plain PyTorch | Export/JIT/eager | See backend guides |

With `backend="auto"`, LM7 chooses the highest-priority installed backend for
the resolved target. Export-only integrations are never selected by
`lm7.compile`. See [JIT vs. AOT](docs/jit-vs-aot.md), the
[architecture guide](docs/architecture.md), and the
[documentation index](docs/README.md) for details and optional dependencies.

## Common workflows

Run or generate with a Hugging Face model:

```bash
uv pip install -e ".[hf]"
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompt "The capital of France is" --target auto
lm7 model generate hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompt "The capital of France is" --max-new-tokens 32 --target nvidia
```

See [model compatibility](docs/model-compatibility.md) and
[compiled generation](docs/huggingface-generation.md).

Serve a model for local validation:

```bash
uv pip install -e ".[serve,hf]"
lm7 model serve hf://HuggingFaceTB/SmolLM2-135M-Instruct --target auto
```

This is a single-user validation server, not a production serving engine. For
production NVIDIA serving, `--backend vllm` hands execution to vLLM. See
[serving](docs/serving.md).

Export and reload a versioned artifact:

```python
lm7.export(model, args=(example_input,), target="cpu", output="model.lm7")
loaded = lm7.load_artifact("model.lm7")
output = loaded(example_input)
```

Choose an export backend for AOTInductor, TensorRT, OpenVINO, ONNX Runtime,
IREE, ExecuTorch, QNN, Core ML, or StableHLO payloads. See
[JIT vs. AOT](docs/jit-vs-aot.md) and
[artifact inspection](docs/artifact-inspection.md).

Quantize a validated Hugging Face model:

```bash
uv pip install -e ".[hf,torchao]"
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target cpu --quantize int8
```

Quantization is admitted per model and mode rather than promised universally.
See [quantization](docs/quantization.md) for measurements and accuracy checks.

Examples and reproducible benchmark harnesses live in
[`examples/`](examples) and [`benchmarks/`](benchmarks).

## Design thesis

Projects such as ZML and Roofline.ai pursue the same broader goal of making ML
workloads portable across heterogeneous hardware.

LM7 makes a narrower architectural bet: hardware portability does not
necessarily require another compiler or runtime. Existing stacks such as
TorchInductor, TensorRT, OpenVINO, OpenXLA, and ExecuTorch already contain
substantial hardware-specific work. LM7 provides a neutral layer above them
instead of replacing them.

## Documentation

Start with:

- [Limitations](docs/limitations.md) — what LM7 does not do, per backend.
- [Tested hardware](docs/tested-hardware.md) — physical machines and exact gaps.
- [Architecture](docs/architecture.md) — targets, backends, planner, artifacts.
- [What LM7 replaces](docs/what-this-replaces.md) — per-vendor application glue.
- [JIT vs. AOT](docs/jit-vs-aot.md) — compilation timing and artifact rules.
- [Development and testing](docs/development.md) — running the suite and GPU tests.

The [documentation index](docs/README.md) links every hardware and backend guide.

## Contributing

Issues and pull requests are welcome. Run the checks before opening one:

```bash
uv pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Hardware-specific tests skip automatically when the device or toolchain is
absent. See [development and testing](docs/development.md).

## License

LM7 is licensed under the [BSD 3-Clause License](LICENSE).
