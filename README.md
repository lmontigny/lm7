# LM7

LM7 is an early PyTorch-first prototype for running the same inference model on
different local hardware through one stable API.

```python
import torch
import lm7

model = torch.nn.Linear(16, 4).eval()
model = lm7.compile(model, target="auto")
output = model(torch.randn(2, 16))
```

## Exported artifacts

LM7 can capture a model with the public `torch.export` API and save a versioned
source artifact:

```python
artifact = lm7.export(
    model,
    args=(torch.randn(2, 16),),
    target="cpu",
    output="model.lm7",
)

loaded = lm7.load_artifact("model.lm7")
output = loaded.module()(torch.randn(2, 16))
```

An artifact is a directory containing `manifest.json` and
`exported_program.pt2`. Loading validates the format version and SHA-256 payload
checksum. Existing output paths are never overwritten implicitly.

To compile and package a CPU model ahead of time with AOTInductor:

```python
artifact = lm7.export(
    model,
    args=(torch.randn(2, 16),),
    target="cpu",
    backend="aot_inductor",
    output="model.lm7",
)

loaded = lm7.load_artifact("model.lm7")
output = loaded(torch.randn(2, 16))
```

This path requires PyTorch's Beta AOTInductor package APIs and a working
platform C++ compiler. The compiled package is target- and PyTorch-version
specific.

Pass `debug=True` to include compiler intermediates in the artifact:

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

LM7 always records the exported graph and asks Inductor for its FX graphs,
pre/post-fusion IR, and generated output code. The manifest indexes every file
that PyTorch actually emits. Final assembly, PTX, or accelerator binaries are
target- and toolchain-dependent; they are included and classified when present,
but are not guaranteed. Debug artifacts can reveal model structure and generated
code and should be handled as sensitive development data.

## Installation

```bash
python -m pip install -e ".[dev]"
```

LM7 requires Python 3.10+ and PyTorch 2.x. It does not install accelerator
toolchains.

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
| `eager` | Supported | Reference/fallback execution on detected PyTorch devices |
| `inductor` | Supported when `torch.compile` exists | JIT compilation through public `torch.compile` |
| `aot_inductor` | CPU prototype when package APIs and a C++ compiler exist | Ahead-of-time `.pt2` package |

Auto planning prefers Inductor when it reports support. If compilation fails,
`fallback="warn"` emits a warning and uses eager execution. Set
`fallback="error"` for strict behavior.

## Current limitations

LM7 is not production-ready and does not promise full PyTorch model coverage.
Only local PyTorch devices are detected. Compiled callables are cached only in
memory. Exported source artifacts and CPU AOTInductor packages are persistent.
Cache identity deliberately hashes graph and state metadata rather than all
weight contents. AOTInductor's Python package APIs are Beta, and compiled
packages are not a stable cross-version ABI. Remote hardware, non-CPU AOT
validation, vendor compiler adapters, production dynamic-shape profiles,
quantization, training, and distributed inference are future work.

See [the architecture notes](docs/architecture.md) for extension points.

## Development

```bash
pytest
ruff check .
ruff format --check .
python examples/basic_mlp.py
```
