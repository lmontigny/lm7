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

TVM also has an AOT export path — see [AOT export](#aot-export) below:

```python
artifact = lm7.export(
    model.eval(), args=(example_input,), target="cpu", backend="tvm", output="model.lm7"
)
reloaded = lm7.load_artifact("model.lm7")
output = reloaded(example_input)
```

> [!WARNING]
> This backend is **explicit-only and far slower than Inductor** — 62-200x
> slower than eager, depending on host, on the benchmarks below. It exists so
> TVM's IR and toolchain are reachable from LM7, not because it is a good way
> to run a model today. Read [performance](#performance) before using it for
> anything.

> [!NOTE]
> `pip install -e ".[tvm]"` used to fail on every platform with `undefined
> symbol` from `libtvm_runtime`. The published `apache-tvm==0.25.0.post1`
> wheel bundles a runtime built against `apache-tvm-ffi==0.1.12`, but declares
> only `apache-tvm-ffi>=0.1.12`, so an unpinned install resolves the newest
> `tvm-ffi` (0.1.13, which changed the `Stringify` ABI) and the runtime cannot
> dlopen. The `tvm` extra now pins `apache-tvm-ffi==0.1.12` to match the
> wheel, which is what makes the install command above work at all.

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

### BYOC (DNNL / Arm Compute Library)

TVM's BYOC mechanism can offload matmul to a real library instead of generic
TOPI loop code, which is the actual fix for the missing-BLAS problem above.
It does not help here: the published `apache-tvm` wheel's Relax
`backend.contrib` only ships `example_npu` and `nnapi`, and
`tvm.support.libinfo()` reports no `USE_DNNL`, `USE_CBLAS`, or Arm Compute
Library build flag at all -- those codegens are not compiled into the
distributed wheel on any platform. Reaching them means building TVM from
source against oneDNN or ACL, which breaks the pip-install model every other
LM7 backend follows and was out of scope here. Not wired up.

### Apple Silicon (arm64 macOS)

`apache-tvm` publishes a native `macosx_11_0_arm64` wheel, and the backend
works unmodified on it — same `torch.export` → `from_exported_program` →
`llvm` build → Relax VM path as x86-64, once the `apache-tvm-ffi` pin above is
in place. Verified on an Apple M3 Pro, TVM 0.25.0.post1, PyTorch 2.13:

`python benchmarks/tvm_relax.py`, same MLP as above:

| Backend | First call | Steady median | vs eager |
| --- | --- | --- | --- |
| `eager` | 573 ms | **1.40 ms** | 1.0x |
| `inductor` | 1211 ms | **1.45 ms** | 0.96x |
| `tvm` | 491 ms | **87.3 ms** | **0.016x** |

Same shape as x86-64 — untuned Relax loses badly to both eager and Inductor —
though the gap is narrower here (62x vs 167x slower than eager), consistent
with Apple Silicon's CPU being a smaller relative win for `llvm` codegen than
for whatever BLAS Inductor's path reaches on x86.

It also compiles a real model correctly on this hardware: `HuggingFaceTB/
SmolLM2-135M-Instruct`, target `cpu`, `backend="tvm"`, called positionally
with `(input_ids, attention_mask)` — logits match eager within 5.1e-5 max
absolute difference and pick the same top token ("Paris" for "The capital of
France is"). This exercises `embedding` plus a full transformer block, not
just the MLP above.

Architecture-specific codegen (`options={"target": {"kind": "llvm", "mcpu":
"apple-m3"}}`) does not close the gap either: 80.3 ms vs 82.4 ms median on the
MLP above, indistinguishable from noise. Consistent with the BYOC finding —
the bottleneck is the missing library dispatch for matmul, not instruction
selection, so tuning what LLVM targets does not move it.

### Linux aarch64 (Neoverse servers)

The other Arm, and the one a Graviton or Axion deployment is. `apache-tvm`
publishes an aarch64 Linux wheel, the backend works unmodified, and the whole
integration suite passes — 15 tests on an Arm Neoverse N3 (GCP
`n4a-standard-8`), TVM 0.25.0.post1, PyTorch 2.13.0+cpu. TVM reports its LLVM
target as `aarch64-conda-linux-gnu`.

`python benchmarks/tvm_relax.py`, same MLP as above:

| Backend | First call | Steady median | vs eager |
| --- | --- | --- | --- |
| `eager` | 941 ms | **1.380 ms** | 1.0x |
| `inductor` | 2260 ms | **1.504 ms** | 0.92x |
| `tvm` | 1045 ms | **275.3 ms** | **0.005x** |

Same shape again, and worse than either of the others: untuned Relax is **200x
slower than eager** here, against 62x on Apple Silicon and 167x on x86-64. The
ordering of the three hosts is not something this measurement explains, and one
untuned part is thin evidence for a claim about Arm codegen — but the direction
is consistent everywhere, and it is the reason `tvm` ranks below Inductor and
`backend="auto"` never selects it.

Note the `inductor` row loses to `eager` here. That is not a TVM fact; it is
this workload being 97–99% GEMM on a part where fusion has nothing to win — see
[CPU inference](cpu.md#latency-on-a-neoverse-n3).

## Scope

- **CPU (`llvm`) only.** TVM's CUDA codegen needs a CUDA toolkit it can
  discover; `find_cuda_path()` does not accept the pip `nvidia/cu13` layout LM7
  uses elsewhere, and the wheel's CUDA device API only registers once the CUDA
  runtime libraries are on `LD_LIBRARY_PATH`. Reaching an Ada GPU took both, and
  even then Relax has no cuBLAS dispatch. Not wired up.
- **Positional tensor inputs only.** The Relax VM entry point takes a flat
  positional list, so a model must be called as `model(a, b)`, not with kwargs.
- **No autotuning, no quantization, no dynamic shapes.** Each input signature
  compiles its own module.
- **Weights are baked in.** LM7 converts with `keep_params_as_input=False`, so
  the built module takes only the call's real arguments.

Tensors cross into TVM and back over **DLPack**, so the exchange is zero-copy
and needs no NumPy — unlike the OpenVINO adapter, which round-trips through
NumPy arrays.

Set a different TVM target with `options={"target": {"kind": "llvm", "mcpu":
"..."}}` if you want to try architecture-specific codegen. TVM 0.25 dropped
the CLI-string target form (`"llvm -mcpu=..."`) entirely — passing it raises
`ValueError: CLI target string form ... is no longer supported`, wrapped in
LM7's `CompilationError`. Use the JSON-dict form shown above instead.

## AOT export

```python
artifact = lm7.export(
    model.eval(), args=(example_input,), target="cpu", backend="tvm", output="model.lm7"
)
reloaded = lm7.load_artifact("model.lm7")
output = reloaded(example_input)
```

This writes `compiled_model.tvm.so` — TVM's own `Executable.export_library()`
output — into the `.lm7` artifact directory. Reloading it, either through the
`artifact` returned by `export()` or later through `load_artifact()`, uses
only `tvm.runtime.load_module()` plus `relax.VirtualMachine` — no
`torch.export` and no Relax PyTorch frontend import, unlike the JIT path's
`load()`. That is the actual point of exporting ahead of time: the process
that runs the model does not need the compiler.

It does not run any faster: the `.so` holds the same untuned Relax codegen
measured above, so exporting only removes the `torch.export` + `relax.build`
cost from every process start, not the per-call cost shown in
[performance](#performance). Same constraints as the JIT path apply —
positional inputs only, static shapes only (`dynamic_shapes` and
`shape_profile` raise `BackendUnavailableError`), weights baked in — plus one
new one: **the library is bound to the exporting host's CPU architecture**.
TVM's LLVM codegen targets the host's triple (`arm64-apple-darwin...` on this
project's own validation host) and any `mcpu` given in `options`, so a `.so`
built on arm64 does not reload on x86-64, or vice versa.

LM7 refuses to load an artifact whose recorded architecture does not match
the local host, the same protection AOTInductor and TensorRT get for GPU
architectures.

> [!NOTE]
> This used to require `target="cpu:arm64"` spelled out, because an unqualified
> `target="cpu"` recorded no architecture and the check is silent without one —
> so the export most people write was the one the guard could not protect. That
> was fixed: an architecture-bound backend now resolves the host's architecture
> even from a bare vendor. `target="cpu"` on a Neoverse host records
> `cpu:aarch64`, and the guard fires. Portable backends still record the vendor
> alone, because the architecture is part of an artifact's identity inside a
> bundle.

Verified end to end on Linux aarch64 rather than from fixtures. A `target="cpu"`
export on an Arm Neoverse N3 writes `compiled_model.tvm.so` — *"ELF 64-bit LSB
shared object, ARM aarch64"* — records `architecture: aarch64` in its manifest,
reloads on the same host agreeing with eager to `2.4e-07`, and is refused when
the host reports `x86_64`:

```
its tvm payload was built for cpu:aarch64, but this machine is cpu:x86_64.
Kernels are compiled per architecture, so it cannot run here.
```

That is the claim this section makes — that the `.so` carries the exporting
host's triple — demonstrated with an artifact built on one architecture rather
than inferred from the codegen target.

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
