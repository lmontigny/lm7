# Architecture

LM7 separates a hardware `TargetSpec` from a compiler/runtime backend. The public
API returns a lazy `CompiledModule`; its first call detects the target, asks the
deterministic planner to select a registered backend, compiles one variant per
input signature, and executes it through a common artifact interface.

Backends implement the small protocol in `lm7.backends.base`. A backend probes
availability without compiling, reports target support and priority, compiles a
request, and loads its artifact. LM7 0.1 includes executable eager and JIT
TorchInductor adapters. Persistent compiler artifact serialization and third-party
entry-point discovery are intentionally deferred.

The eager backend is both the reference implementation and the fallback. Only
backend compilation failures trigger fallback; exceptions from model execution
are returned to the caller unchanged.
