# LM7

[![CI](https://github.com/lmontigny/lm7/actions/workflows/ci.yml/badge.svg)](https://github.com/lmontigny/lm7/actions/workflows/ci.yml)

LM7 is an early PyTorch-first prototype for running the same inference model on
different local hardware through one stable API.

```python
import torch
import lm7

model = torch.nn.Linear(16, 4).eval()
model = lm7.compile(model, target="auto")
output = model(torch.randn(2, 16))
```

> [!WARNING]
> LM7 is an early prototype. It is inference-only, does not support every
> PyTorch model, and does not promise a stable compiled-artifact ABI yet.

## Installation

LM7 requires Python 3.10 or newer and PyTorch 2.x. Accelerator toolchains and
platform C++ compilers are not installed by LM7.

### Linux and macOS

```bash
git clone https://github.com/lmontigny/lm7.git
cd lm7
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Confirm the environment before running the AOT test:

```bash
uname -a
c++ --version
```

## Test locally

### 1. Run the fast CPU checks

These checks do not require a GPU or native compiler:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

The complete suite uses mocks for compiler-specific behavior, so it should pass
on a normal CPU-only development machine.

### 2. Test lazy compilation and fallback

```bash
python examples/basic_mlp.py
```

The example prints a `torch.Size([2, 4])` result and the planner explanation.
LM7 prefers JIT Inductor on CPU when `torch.compile` exists. If the machine
lacks a compatible C++ compiler, `fallback="warn"` emits a warning and executes
with eager PyTorch instead. That warning is expected; it is not a model failure.

Inspect the current machine and selection policy directly:

```bash
python -c "import lm7; print(lm7.detect_targets())"
python -c "import lm7; print(lm7.backends())"
python -c "import lm7; print(lm7.explain(target='auto'))"
```

### 3. Test source-artifact export

Source artifacts use `torch.export` and do not require a C++ compiler:

```bash
python -m pytest tests/test_exporting.py -q
```

The public API is:

```python
import torch
import lm7

model = torch.nn.Linear(16, 4).eval()
example_input = torch.randn(2, 16)

artifact = lm7.export(
    model,
    args=(example_input,),
    target="cpu",
    output="model.lm7",
)

loaded = lm7.load_artifact("model.lm7")
torch.testing.assert_close(loaded(example_input), model(example_input))
```

An artifact is a directory containing `manifest.json` and
`exported_program.pt2`. Loading validates the format version and SHA-256 payload
checksum. LM7 never overwrites an existing output path implicitly.

For bounded dynamic dimensions, attach a named shape profile. Input names match
the model's `forward` parameters:

```python
profile = lm7.ShapeProfile({"input": {0: lm7.DynamicDimension("batch", min=1, max=32)}})
artifact = lm7.export(
    model,
    args=(example_input,),
    target="cpu",
    output="dynamic-model.lm7",
    shape_profile=profile,
)

artifact(torch.randn(8, 16))  # accepted
```

The profile is stored in `manifest.json` and checked when the artifact is
called. Inputs outside the declared bounds fail with a clear `ValueError`.

### 4. Test real CPU AOTInductor compilation

AOTInductor requires PyTorch's Beta package APIs and a working platform C++
toolchain.

On Linux, first verify that a compiler is visible:

```bash
c++ --version
python examples/aot_mlp.py
```

To retain the artifact and verify that another Python process can load it:

```bash
python examples/aot_mlp.py --output artifacts/model.lm7
python examples/aot_mlp.py --load artifacts/model.lm7
```

If `c++` is unavailable, the smoke test should exit nonzero, explain that the
compiler is unavailable, and leave no partial artifact behind.

`tests/test_aot_inductor.py` verifies LM7's orchestration with mocked compiler
APIs. `examples/aot_mlp.py` is the end-to-end test that invokes the real local
toolchain.

### 5. Inspect compiler IR and generated code

The AOT example enables `debug=True`. A successful artifact contains a `debug/`
directory:

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

LM7 requests and indexes:

- Exported graph and graph signature
- FX graphs before and after Inductor transformations
- Inductor IR before and after fusion
- Generated C++, CUDA, Python, or Triton source
- PTX, assembly, CUBIN, or HSACO when the target and toolchain emit them

Every indexed file has a SHA-256 checksum in `manifest.json`. CPU compilation
normally emits C++ rather than PTX. Debug output can expose model structure and
generated code, so treat it as sensitive development data.

## Targets and diagnostics

Hardware targets and compiler backends are separate:

```python
lm7.compile(model, target="cpu")
lm7.compile(model, target="nvidia:h100")
lm7.compile(model, target="amd:gfx942")

print(lm7.detect_targets())
print(lm7.backends())
print(lm7.explain(model, target="auto"))
```

Explicit function arguments override `LM7_TARGET`, `LM7_BACKEND`,
`LM7_FALLBACK`, and `LM7_CACHE_DIR`.

## Current backends

| Backend | Status | Notes |
| --- | --- | --- |
| `eager` | Supported | Reference and fallback execution on detected PyTorch devices |
| `inductor` | When `torch.compile` exists | JIT compilation through public `torch.compile` |
| `aot_inductor` | CPU prototype | Ahead-of-time `.pt2` package; requires PyTorch package APIs and C++ toolchain |

Auto planning prefers Inductor when it reports support. If compilation fails,
`fallback="warn"` warns and uses eager execution. Set `fallback="error"` for
strict behavior.

## Current limitations

- Inference only; training and backward compilation are unsupported.
- Only local PyTorch devices are detected.
- JIT compiled callables are cached only in memory.
- AOTInductor is validated only for CPU.
- PyTorch's AOTInductor package APIs are Beta.
- Compiled artifacts require compatible PyTorch, target architecture, and
  platform runtime versions.
- Cache identity hashes graph and state metadata, not full weight contents.
- Remote hardware, vendor compiler adapters, stable dynamic-shape profiles,
  quantization, and distributed inference are future work.

See [the architecture notes](docs/architecture.md) for extension points.

## Development commands

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python examples/basic_mlp.py
python examples/aot_mlp.py
```
