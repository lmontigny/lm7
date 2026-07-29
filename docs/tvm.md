# Apache TVM support

LM7 has an initial [Apache TVM](https://github.com/apache/tvm) backend for CPU,
compiling through TVM's Relax IR.

```bash
python3 -m venv .venv-tvm
.venv-tvm/bin/python -m pip install -e ".[dev,tvm]"
```

```python
compiled = lm7.compile(model.eval(), target="cpu", backend="tvm", fallback="error")
output = compiled(example_input)
```

> [!WARNING]
> This backend is **explicit-only and far slower than Inductor** — 167x slower
> than eager on the benchmark below. It exists so TVM's IR and toolchain are
> reachable from LM7, not because it is a good way to run a model today. Read
> [performance](#performance) before using it for anything.

## What LM7 does *not* use, and why

**Not `torch.compile(backend="tvm")`.** PyTorch ships a built-in `tvm` dynamo
backend, and it is broken against any current TVM. It imports `tvm.relay`,
`tvm.contrib.graph_executor`, `tvm.auto_scheduler`, and `tvm.meta_schedule`;
TVM removed all four in the Relax migration. Worse, its `ImportError` handler
reports:

```
ImportError: Please install apache-tvm to use the tvm backend.
```

even when TVM *is* installed, so the message sends you in a circle. Verified
against TVM 0.25.0.post1 and PyTorch 2.13.

**Not TVM's `relax_dynamo()`.** That is the supported replacement and it works,
but it converts through `from_fx`, whose operator table rejects `embedding`:

```
AssertionError: Unsupported function types ['embedding']
```

No causal LM compiles through it.

**LM7 uses `torch.export` + `from_exported_program`.** The exported-program
frontend has a wider operator table than `from_fx` — it lowers `embedding`, and
it converts SmolLM2-135M without complaint. LM7 captures the model with
`torch.export`, converts to Relax, builds for `llvm`, and runs on the Relax VM.
Compilation happens in-process on the first call per input signature, so this is
a JIT backend in LM7's sense: nothing it produces outlives the process.

## Performance

`python benchmarks/tvm_relax.py`, MLP 1024→4096→1024, batch 8, float32, x86-64
CPU, TVM 0.25.0.post1:

| Backend | First call | Steady median | vs eager |
| --- | --- | --- | --- |
| `eager` | 1060 ms | **2.14 ms** | 1.0x |
| `inductor` | 1857 ms | **2.29 ms** | 0.94x |
| `tvm` | 1314 ms | **357 ms** | **0.006x** |

Numerically it is fine — 1.9e-6 against eager on the MLP, 1.2e-4 on SmolLM2,
which picks the same top token. It is simply slow.

The reason is structural. TVM's historic advantage was autotuning, and the
`apache-tvm` wheel no longer ships it: `auto_scheduler`, `meta_schedule`, and
`dlight` are all absent from the installed package. What remains is Relax plus
generic TOPI schedules. Relax's CPU library dispatch
(`tvm.relax.backend.cpu_generic.library_dispatch_passes`) covers only sampling
and sort/scan — there is **no BLAS or oneDNN dispatch for matmul**, so a linear
layer becomes generated loop code with no tuning behind it.

That is why `supports()` reports priority 0. It ties with `eager`, and
`backend="auto"` breaks ties by name, so `eager` wins and TVM is never selected
implicitly. You have to ask for it.

### Autotuning

`tvm.relax.pipeline.static_shape_tuning_pipeline` still exists and is the
documented way to recover performance. LM7 does not wire it up, and it did not
survive evaluation here:

- It has three undeclared dependencies, each surfacing only after installing the
  previous one: `psutil`, then `cloudpickle`, then `xgboost`.
- 400 trials on a *two-layer* MLP ran past ten minutes without finishing.

Even a good result would not change the recommendation for a JIT backend:
`inductor` reaches 2.29 ms in under two seconds of compilation.

## Scope

- **CPU (`llvm`) only.** TVM's CUDA codegen needs a CUDA toolkit it can
  discover; `find_cuda_path()` does not accept the pip `nvidia/cu13` layout LM7
  uses elsewhere, and the wheel's CUDA device API only registers once the CUDA
  runtime libraries are on `LD_LIBRARY_PATH`. Reaching an Ada GPU took both, and
  even then Relax has no cuBLAS dispatch. Not wired up.
- **Positional tensor inputs only.** The Relax VM entry point takes a flat
  positional list, so a model must be called as `model(a, b)`, not with kwargs.
- **JIT only.** No `.lm7` artifact; `lm7.export` does not accept `backend="tvm"`.
- **No autotuning, no quantization, no dynamic shapes.** Each input signature
  compiles its own module.
- **Weights are baked in.** LM7 converts with `keep_params_as_input=False`, so
  the built module takes only the call's real arguments.

Set a different TVM target with `options={"target": "llvm -mcpu=..."}` if you
want to try architecture-specific codegen.

## Is TVM abandoned?

No, and it is worth being precise, because the broken PyTorch integration
invites that conclusion. As of July 2026 `apache/tvm` is not archived, was
pushed to the same day this was written, has over 100 commits in the preceding
90 days, and released v0.25.0 in June 2026. The project is healthy.

What has rotted is the *PyTorch-facing* integration: PyTorch still targets the
Relay API that TVM deleted, and the Relax replacement's `from_fx` path does not
yet cover the operators a transformer needs. This backend routes around the
second problem and cannot route around the first.

## Validate

```bash
.venv-tvm/bin/python -m pytest tests/test_tvm_integration.py -q
.venv-tvm/bin/python benchmarks/tvm_relax.py
```

The integration suite includes an `embedding` model specifically, because that
is the operator separating the working path from the one LM7 rejected.

## References

- [Apache TVM](https://github.com/apache/tvm)
- [Relax PyTorch frontend](https://tvm.apache.org/docs/reference/api/python/relax/frontend.html)
- [`torch/_dynamo/backends/tvm.py`](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/backends/tvm.py)
  — the built-in backend this document explains around
