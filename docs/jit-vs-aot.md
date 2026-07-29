# JIT vs. AOT

Two things distinguish LM7's compilation paths: *when* compilation happens, and
*whether the result outlives the process*.

## JIT: compiled in-process, on the first call

`inductor`, `tensorrt`, and `openxla` are all JIT backends. `lm7.compile()` does
no work — it validates options and returns a `CompiledModule`. Compilation
happens on the first call with a given input signature.

```python
compiled = lm7.compile(model, target="cpu")  # returns immediately, compiles nothing
print(compiled.selected_backend)  # None - nothing has been chosen yet

out = compiled(example_input)  # first call: resolves target, compiles, then runs
print(compiled.selected_backend)  # "inductor"

out = compiled(example_input)  # subsequent calls: fast
out = compiled(other_shape)  # new signature: compiles a second variant
```

LM7 compiles **one variant per input signature** and caches it, so a model called
with several batch sizes pays several compilations. The cache key covers model
identity, input signature, target, and backend.

Nothing is written that another process can use. Restarting pays the compile cost
again, and neither TensorRT engines nor `torch.compile` callables can be shipped
to another machine.

One consequence worth knowing: because `torch.compile` is lazy, a *compilation*
failure surfaces from the first call rather than from `lm7.compile`. LM7 runs a
warmup call inside its own error boundary so `fallback="warn"` still works — see
[what LM7 replaces](what-this-replaces.md).

## AOT: compiled up front, into a file

AOT goes through `lm7.export`, which writes an `.lm7` directory. There are two
levels of it, and conflating them is easy:

```python
# Level 1 - capture only (the default, backend="export").
# Writes a portable ExportedProgram. PyTorch still generates kernels when it
# runs, so this removes tracing, not compilation.
lm7.export(model, args=(example_input,), target="cpu", output="model.lm7")

# Level 2 - capture and compile. A persistent AOTInductor package with kernels
# already generated.
lm7.export(
    model, args=(example_input,), target="cpu", backend="aot_inductor", output="model-aot.lm7"
)

# Level 3 - capture and compile to a vendor runtime's own format. On Intel CPU,
# OpenVINO IR, which does not need PyTorch to execute.
lm7.export(
    model, args=(example_input,), target="cpu", backend="openvino", output="model-ov.lm7"
)

loaded = lm7.load_artifact("model-aot.lm7")  # another process, nothing to compile
out = loaded(example_input)
```

An `.lm7` artifact is a directory holding a versioned JSON manifest, checksums,
and a PyTorch `.pt2` program. `load_artifact()` validates the manifest schema
version and the checksums before loading anything.

`aot_inductor` is validated for CPU and Apple Silicon only, and uses Beta PyTorch
APIs. On other targets, export still works at level 1.

`openvino` is validated for Intel CPU only. It adds `compiled_model.xml` and
`compiled_model.bin` next to the `.pt2`, and both are checksummed. That IR pair
is the only LM7 payload a machine can run **without PyTorch installed** — load it
straight through `openvino.Core().compile_model()`, no `lm7` and no `torch`:

```python
import numpy, openvino

model = openvino.Core().compile_model("model-ov.lm7/compiled_model.xml", "CPU")
result = model([numpy.random.rand(8, 16).astype("float32")])[model.outputs[0]]
```

## Bundles

Several single-target artifacts can be combined, with the choice made at load
time:

```python
lm7.create_bundle(["build/cpu.lm7", "build/nvidia.lm7"], output="model.bundle.lm7")
deployed = lm7.load_bundle("model.bundle.lm7").load(target="auto")
```

## Which to use

Use JIT while iterating locally: there is nothing to manage, and the only cost is
a slow first call per shape.

Use `aot_inductor` when you want the compile cost paid once at build time rather
than on every process start — a short-lived process, a cold-start-sensitive
service, or a machine where you would rather not ship a compiler.

Two caveats either way:

- **AOT fixes the input signature** captured at export time. A JIT path adapts to
  new shapes by recompiling; an artifact does not.
- **Artifacts are not a stable ABI.** An `.lm7` directory is tied to compatible
  PyTorch, runtime, and hardware versions. Treat it as a build output to
  regenerate, not a long-lived binary format.

## Related

- [Backends table](../README.md#3-compile-a-local-model) — which backends are JIT and which are AOT
- [What LM7 replaces](what-this-replaces.md) — the lazy-compilation traps LM7 handles
- [Architecture](architecture.md) — the compile flow end to end
