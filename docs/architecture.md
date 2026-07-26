# Architecture

LM7 separates a hardware `TargetSpec` from a compiler/runtime backend. The public
API returns a lazy `CompiledModule`; its first call detects the target, asks the
deterministic planner to select a registered backend, compiles one variant per
input signature, and executes it through a common artifact interface.

Backends implement the small protocol in `lm7.backends.base`. A backend probes
availability without compiling, reports target support and priority, compiles a
request, and loads its artifact. LM7 0.1 includes executable eager and JIT
TorchInductor adapters plus an optional NVIDIA Torch-TensorRT adapter.
Persistent compiler artifact serialization and third-party entry-point
discovery are intentionally deferred.

The TensorRT adapter lazily imports Torch-TensorRT, which registers the public
`tensorrt` `torch.compile` backend. It is deliberately lower priority than
Inductor in automatic planning while model and dynamic-shape coverage remain
experimental. TensorRT engine construction happens on the first call and the
resulting callable is process-local.

## Source artifacts

`lm7.export()` captures an `nn.Module` through `torch.export` or accepts an
existing `ExportedProgram`. It writes an `.lm7` directory atomically with a
versioned JSON manifest and the public PyTorch `.pt2` serialization. Load-time
validation checks the manifest schema version and program checksum before calling
`torch.export.load`.

The manifest cache key covers the exported graph structure, parameter names,
shapes and dtypes, representative input signature, target, PyTorch version, and
LM7 version. It intentionally does not hash full parameter contents yet. This
format is an early source-artifact contract, not a stable cross-version binary
ABI or a compiled AOT package.

## AOTInductor packages

The `aot_inductor` backend consumes the same `ExportedProgram` and uses
PyTorch's Beta `aoti_compile_and_package` API to create `compiled_model.pt2`.
LM7 records the backend and PyTorch versions, runtime target, and compiled
payload checksum in the manifest. `load_artifact()` validates both source and
compiled payloads before using `aoti_load_package`.

LM7 0.1 validates this path only for CPU. Packages require a compatible PyTorch
runtime and target architecture and do not provide a stable cross-version ABI.

### Compiler debug artifacts

`lm7.export(..., debug=True)` stores source capture details under `debug/` and
enables PyTorch's Inductor trace for FX graphs, pre/post-fusion IR, and generated
output code. The artifact manifest contains a normalized index with a pipeline
level, kind, and relative path for each emitted file.

PTX, assembly, CUBIN, and other low-level outputs are indexed when the selected
target and toolchain emit them. LM7 does not synthesize or claim unavailable
levels. The current CPU-only AOT path normally emits C++ source rather than PTX.
Debug output may contain model structure and generated code and is disabled by
default.

The eager backend is both the reference implementation and the fallback. Only
backend compilation failures trigger fallback; exceptions from model execution
are returned to the caller unchanged.
