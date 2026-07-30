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

The optional `openxla` adapter uses PyTorch/XLA to register the OpenXLA
`torch.compile` backend. TPU discovery is lazy and accepts only a PJRT runtime
whose device type is `TPU`; an installed XLA CPU runtime is not reported as TPU
hardware. LM7 uses `torch.no_grad()` for TPU execution because PyTorch/XLA
tracing requires tensor version counters that `torch.inference_mode()` removes.
See [Google TPU support](google-tpu.md) for setup and current scope.

The optional `tenstorrent` adapter reaches Tenstorrent Wormhole and Blackhole
cards through the same PyTorch/XLA seam, but with tt-xla's PJRT plugin instead
of libtpu: it registers the `tt` `torch.compile` backend, which lowers the FX
graph to StableHLO and hands it to tt-mlir and tt-metal. PJRT serves one device
type per process, so discovery selects `TT` only when the plugin is installed
and `PJRT_DEVICE` is unset or `TT`, and never reassigns a runtime that has
already come up as `TPU`. It shares the TPU path's `torch.no_grad()` execution
for the same reason. See [Tenstorrent support](tenstorrent.md).

The `intel:npu` target is the one place where the target/backend split carries
real weight. It is the first `TargetSpec` with no torch device behind it: its
`kind` is `"npu"`, `torch_device()` maps it to the host CPU, and the OpenVINO
NPU plugin owns both the compilation and the transfer. Detection asks the
OpenVINO runtime rather than torch, `inductor` and `eager` decline the kind
instead of silently lowering for the CPU, and `target="auto"` skips it because
it is neither a `gpu` nor an `accelerator`. See [Intel NPU](intel-npu.md).

The optional `executorch` adapter is export-only and targets edge devices. It
lowers an ExportedProgram through ExecuTorch's XNNPACK partitioner to a `.pte`,
which the ExecuTorch C++ runtime executes on Android, iOS, or the build host
with no PyTorch present. Its `supports()` always reports unsupported, so
`lm7.compile` never selects it — a phone is not reachable from the calling
process, and the artifact is the deliverable. Because XNNPACK spans ARM64 and
x86-64, export *and* execution are validated on ordinary CI. See
[ExecuTorch support](executorch.md).

The optional `tvm` adapter compiles through Apache TVM's Relax IR for CPU. It
does not use PyTorch's built-in `tvm` dynamo backend, which still imports the
Relay API TVM deleted, nor TVM's `relax_dynamo()`, whose `from_fx` translator
rejects `embedding` and so cannot lower a causal LM. Instead it captures with
`torch.export` and converts with `from_exported_program`, then builds and runs
on the Relax VM. It reports priority 0, tying with `eager` so that automatic
planning never selects it — TVM's untuned codegen measured far slower than
Inductor. See [Apache TVM support](tvm.md).

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

LM7 0.1 validates this path for CPU, Apple Silicon (MPS), and NVIDIA GPU.
Packages require a compatible PyTorch runtime and target architecture and do not
provide a stable cross-version ABI.

A CUDA target adds one build-time requirement the others do not have: the
wrapper is compiled against the CUDA headers, which the PyTorch CUDA wheel does
not ship. The backend resolves a toolkit before packaging — an explicit
`CUDA_HOME`/`CUDA_PATH` first, then the `nvidia/cu<major>` tree in site-packages,
then `nvcc` on `PATH`, then `/usr/local/cuda` — and reports an unavailable
backend when none of them is complete, rather than failing inside the C++
compiler. Loading a package needs no toolkit.

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
