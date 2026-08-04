# JIT vs. AOT

Keep two independent questions separate: *when a backend does code generation*,
and *what the LM7 API returns*. `lm7.compile()` returns an in-process callable;
`lm7.export()` writes persistent files. JIT and AOT describe backend timing, so
they are related to those APIs but are not synonyms for them.

## Compile: a lazy in-process callable

`inductor`, `openxla`, and `tenstorrent` are JIT backends, and so is
`tensorrt` when reached through `lm7.compile` — it also has an AOT export
path, below.
`lm7.compile()` does no work — it validates options and returns a
`CompiledModule`. Compilation happens on the first call with a given input
signature.

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

The API remains lazy even when the selected backend uses an ahead-of-time
compiler internally. For example, `aot_inductor`, `openvino`, and `tensorrt`
can build a package or engine on the first call and load it immediately. LM7
still returns a callable, not a user-owned artifact; restarting invokes the
backend again.

No persistent output is promised by `lm7.compile()`. Transient compiler files
may exist, but the callable and its LM7 shape cache cannot be deployed.

One consequence worth knowing: because `torch.compile` is lazy, a *compilation*
failure surfaces from the first call rather than from `lm7.compile`. LM7 runs a
warmup call inside its own error boundary so `fallback="warn"` still works — see
[what LM7 replaces](what-this-replaces.md).

## Export: persistent files, optionally AOT-lowered

`lm7.export()` is an artifact API, not a synonym for AOT compilation. It always
writes an `.lm7` directory at build time, but there are multiple lowering
levels, and conflating them is easy:

```python
# Level 1 - capture only (the default, backend="export").
# Writes a portable ExportedProgram. PyTorch still generates kernels when it
# runs, so this removes tracing, not compilation.
lm7.export(model, args=(example_input,), target="cpu", output="model.lm7")

# Level 2 - capture and compile. A persistent AOTInductor package with kernels
# already generated. Swap target="nvidia" to bake in CUDA kernels instead.
lm7.export(
    model, args=(example_input,), target="cpu", backend="aot_inductor", output="model-aot.lm7"
)

# Level 3 - capture and compile to a vendor runtime's own format. On Intel CPU,
# OpenVINO IR, which does not need PyTorch to execute.
lm7.export(model, args=(example_input,), target="cpu", backend="openvino", output="model-ov.lm7")

loaded = lm7.load_artifact("model-aot.lm7")  # another process, nothing to compile
out = loaded(example_input)
```

An `.lm7` artifact is a directory holding a versioned JSON manifest, checksums,
and a PyTorch `.pt2` program. `load_artifact()` validates the manifest schema
version and the checksums before loading anything. What that reload costs in a
process that never compiled the model, and what it refuses to load on, is
measured in
[AOTInductor artifact compatibility](aot-artifact-compatibility.md).

`aot_inductor` is validated for CPU, Apple Silicon, and NVIDIA GPU, and uses Beta
PyTorch APIs. On other targets, export still works at level 1.

On NVIDIA it reaches the GPU without a compiler in the process, as does a
`tensorrt` artifact; plain `inductor` compiles on the first call and keeps
nothing. Packaging one needs a CUDA toolkit, because AOTInductor compiles a C++
wrapper against the CUDA headers that the PyTorch CUDA wheel does not ship;
`uv pip install -e ".[cuda-aot]"` supplies them, and LM7 finds them without a
`CUDA_HOME`. Loading the artifact needs no toolkit, so a deployment host needs
the CUDA runtime and a compatible GPU only. See
[NVIDIA AOT Inductor](development.md#nvidia-aot-inductor) for the WSL caveat and
the CUDA 12 case.

`openvino` is validated for Intel CPU only. It adds `compiled_model.xml` and
`compiled_model.bin` next to the `.pt2`, and both are checksummed. That IR pair
is the only LM7 payload a machine can run **without PyTorch installed** — load it
straight through `openvino.Core().compile_model()`, no `lm7` and no `torch`:

```python
import numpy, openvino

model = openvino.Core().compile_model("model-ov.lm7/compiled_model.xml", "CPU")
result = model([numpy.random.rand(8, 16).astype("float32")])[model.outputs[0]]
```

`executorch` is the level-3 payload for phones and embedded devices. It adds a
`compiled_model.pte`, which the ExecuTorch C++ runtime executes on Android, iOS,
or the build host with no Python and no PyTorch. Unlike the payloads above it is
not bound to the machine that produced it, because the XNNPACK delegate covers
ARM64 and x86-64 alike. It is CPU-only and export-only — see
[ExecuTorch](executorch.md).

## From the CLI

A Hugging Face model can be exported without writing PyTorch at all:

```bash
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct model.lm7 \
  --target cpu --backend aot_inductor
```

Two things to know about the captured graph. The shape comes from tokenizing
`--prompt`, so by default the artifact is fixed to that many tokens. And LM7
captures a **logits-only** graph: a causal LM returns a `CausalLMOutputWithPast`
dataclass, `torch.export` records it in the output pytree, and
`torch.export.load` then fails with "Deserializing
transformers.modeling_outputs.CausalLMOutputWithPast in pytree is not
registered" — the artifact would save but never reload. The reloaded artifact
takes `input_ids` and `attention_mask` and returns one logits tensor.

### A dynamic sequence length

A one-prompt-length artifact is rarely what you want from a causal LM.
`--dynamic-seq` (or `dynamic_sequence=` from Python) captures the sequence
dimension as a bounded `torch.export.Dim`, so a single artifact serves every
length in range:

```bash
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct model.lm7 \
  --target nvidia --backend aot_inductor --dynamic-seq 1:512
```

```python
lm7.huggingface.export_hf_model(
    "hf://HuggingFaceTB/SmolLM2-135M-Instruct",
    output="model.lm7",
    target="nvidia",
    backend="aot_inductor",
    dynamic_sequence=True,  # or an explicit (min, max)
)
```

Measured on an RTX 4070 SUPER (sm89), one packaged SmolLM2-135M-Instruct
artifact answered 1-, 5-, and 17-token prompts in a fresh process, matching eager
on every one.

Four things constrain it:

- **Bounds are part of the artifact.** They are written to the manifest and
  checked on every call, so an out-of-range prompt raises instead of silently
  producing wrong output. `dynamic_sequence=True` derives them from the model
  config; pass `(min, max)` to set them.
- **The example prompt must sit inside the bounds.** `torch.export` traces the
  example input, so a 5-token prompt cannot capture a `min=8` artifact.
- **Dynamic capture uses eager attention.** The default attention path builds its
  mask in blocks, which makes `torch.export` emit a `sequence % 8` guard it
  cannot prove over a range and fail with "Not all values of sequence ... satisfy
  the generated guard". A fixed export keeps the faster default.
- **Only the sequence dimension is dynamic.** Batch stays at the captured size,
  and the graph is still prefill-only (`use_cache=False`), so this is not a
  KV-cache decode loop.


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
  new shapes by recompiling; an artifact does not, unless a dimension was
  captured as dynamic — see [a dynamic sequence length](#a-dynamic-sequence-length).
- **Artifacts are not a stable ABI.** An `.lm7` directory is tied to compatible
  PyTorch, runtime, and hardware versions. Treat it as a build output to
  regenerate, not a long-lived binary format.

## Related

- [Backends table](../README.md#backends) — which backends are JIT and which are AOT
- [What LM7 replaces](what-this-replaces.md) — the lazy-compilation traps LM7 handles
- [Architecture](architecture.md) — the compile flow end to end
