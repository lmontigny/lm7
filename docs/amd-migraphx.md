# AMD MIGraphX evaluation plan

LM7's current AMD path uses a ROCm-enabled PyTorch build with TorchInductor.
MIGraphX should be evaluated as an optional AMD backend before it is added to
automatic planning.

## Candidate integration paths

- Torch-MIGraphX: preferred first path because it keeps LM7 close to the
  existing PyTorch module workflow.
- ONNX Runtime with the MIGraphX execution provider: useful if a model already
  exports cleanly to ONNX or if runtime packaging is the priority.
- MIGraphX driver: useful for offline validation, operator inspection, and
  performance tests against generated ONNX artifacts.

## Acceptance criteria

Evaluate MIGraphX against the existing AMD Inductor path on the same host,
ROCm version, GPU architecture, dtype, batch size, and prompt shape.

- Correctness: compiled logits must match eager ROCm within the same tolerance
  policy used by AMD integration tests.
- Coverage: start with `mlp`, SmolLM2, LFM2.5, Llama 3.2 1B, and Qwen3.5 0.8B.
- Latency: report first-call compile time, median latency, p95 latency, and
  throughput.
- Memory: report peak allocated GPU memory when the runtime exposes it.
- Packaging: document whether the compiled result can be serialized and loaded
  in a fresh process.
- Failure behavior: backend import, unsupported op, and compile failures must
  return actionable LM7 errors and preserve eager fallback semantics.

## First implementation slice

Do not add a registered backend until the local evaluation shows a clear use
case. `benchmarks/migraphx.py` is that first slice: it runs a model through
eager (the correctness reference), TorchInductor, and a manually installed
Torch-MIGraphX `torch.compile` path under one measurement harness, reporting
first-call cost, median and p95 latency, throughput, peak GPU memory, and the
maximum absolute difference from eager. It does not register an LM7 backend. A
later backend PR can wrap the winning path behind `backend="migraphx"` with
lower automatic priority than `inductor` until model coverage is proven.

It now covers the causal-LM shapes from the acceptance criteria too (`--model
smollm2`, `lfm25`, `llama32-1b`, `qwen35-0.8b`), not just `mlp` — see
[Validation commands](#validation-commands).

## What "install Torch-MIGraphX" actually means

Unlike `apache-tvm` (see [the TVM evaluation](tvm.md)), `torch_migraphx` on
PyPI is a pure-Python wheel (`py3-none-any`) with no native library bundled.
Two things have to be true before `pip install torch_migraphx` does anything
useful, confirmed by inspecting the wheel rather than assumed:

- **The native `migraphx` Python bindings must already be importable**
  (`import migraphx`), and they are not on PyPI at all. They come from ROCm's
  own package manager — `sudo apt install migraphx` on Ubuntu once ROCm is
  installed — or from AMD's Docker images. `torch_migraphx/dynamo/lower_dynamo.py`
  imports it directly; without it, `import torch_migraphx` still succeeds, but
  the `migraphx_backend` dynamo backend fails the first time it actually lowers
  a graph, not at import time.
- **The wheel JIT-compiles a small C++ extension on first import**
  (`torch_migraphx/_C.py` calls `torch.utils.cpp_extension.load(...)` if a
  prebuilt `_torch_migraphx` module is not already present). That needs a C++
  toolchain and headers matching the installed PyTorch build. If it fails, set
  `TORCH_MIGRAPHX_VERBOSE_BUILD=1` before re-importing to see the real
  compiler error instead of a generic one.

So on the rented host, verify in this order: `rocminfo` sees the GPU → `apt
list --installed | grep migraphx` (or the Docker image already has it) →
`python -c "import migraphx"` succeeds → *then* `pip install torch_migraphx`.
Confirmed current (checked June 2026): PyPI has one release, `torch_migraphx
1.1`, uploaded April 2026, Linux-only
([source](https://pypi.org/project/torch-migraphx/)); the project is
[ROCm/torch_migraphx](https://github.com/ROCm/torch_migraphx) and registers a
real `torch._dynamo.register_backend(name="migraphx")` — an AOTAutograd-based
FX lowering to MIGraphX, not PyTorch's Relay-era dead-backend problem TVM
has. It also has its own `save_compiled`/`load_compiled` args
(`torch.compile(model, backend="migraphx", options={"save_compiled": path})`)
that persist a compiled graph via `torch.save`/`torch.load` — worth checking
whether that maps to an AOT export path, the way `tvm`'s did.

## Validation commands

On a freshly rented AMD GPU, in this order:

```bash
# 1. Confirm the GPU and ROCm are visible before installing anything LM7-side.
rocminfo | grep gfx

# 2. Confirm the native migraphx bindings are already importable -- see
#    "What 'install Torch-MIGraphX' actually means" above. If this fails,
#    `sudo apt install migraphx` (Ubuntu, ROCm already installed) before
#    going further; pip cannot supply this half.
python -c "import migraphx; print(migraphx.__file__)"

# 3. Baseline the existing eager and Inductor AMD paths through LM7.
uv pip install -e ".[dev,hf]"
python benchmarks/gpu.py --target amd --model mlp --backend eager inductor
python benchmarks/gpu.py --target amd --model smollm2 --backend eager inductor

# 4. Install Torch-MIGraphX itself. First import JIT-compiles a C++ extension
#    (see above) -- if it fails, re-run with TORCH_MIGRAPHX_VERBOSE_BUILD=1.
python -m pip install torch_migraphx
python -c "import torch_migraphx; print(torch_migraphx.__file__)"

# 5. Run the side-by-side evaluation: mlp first (cheap, fast feedback), then
#    the causal-LM shapes from the acceptance criteria.
python benchmarks/migraphx.py --model mlp --dtype float16 --batch-size 8 \
  --output artifacts/benchmarks/migraphx-mlp-fp16-b8.json
python benchmarks/migraphx.py --model smollm2 --dtype float16 \
  --output artifacts/benchmarks/migraphx-smollm2-fp16.json
python benchmarks/migraphx.py --model llama32-1b --dtype float16 \
  --output artifacts/benchmarks/migraphx-llama32-1b-fp16.json
```

Without Torch-MIGraphX installed, step 5 still reports the eager and Inductor
paths and marks `migraphx` unavailable rather than failing the run — useful as
a sanity check before step 4, or if step 2/4 cannot be made to work on the
rented host.

Record the exact ROCm, PyTorch, Torch-MIGraphX, GPU, and driver versions
beside the results — `benchmarks/gpu.py`'s JSON output already captures most
of this via `lm7 doctor`; `migraphx.py`'s does not yet, so note it by hand.
Per the acceptance criteria, `lfm25` and `qwen35-0.8b` are the two shapes still
unexercised here even after the above.

## References

- [AMD MIGraphX documentation](https://rocmdocs.amd.com/projects/AMDMIGraphX/en/latest/index.html)
- [Install MIGraphX for Radeon GPUs](https://rocmdocs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-migraphx.html)
