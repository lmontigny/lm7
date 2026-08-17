# ExecuTorch Core ML for Apple Silicon

LM7's `coreml` backend exports an ExecuTorch `.pte` through Apple's Core ML
delegate. Unlike [`qnn`](qnn.md), it is not deployment-only: Core ML is part
of macOS/iOS itself, so the same `.pte` compiles and executes on the Mac that
built it, through `executorch.runtime.Runtime` — no phone, no vendor SDK
download, no separate device runtime. That is also why this backend requires
macOS: the runtime bridge (`ETCoreMLModelManager`) is Objective-C++, unlike
XNNPACK's portable C++.

```text
PyTorch module -> torch.export -> ExecuTorch CoreMLPartitioner -> .pte -> Core ML (ANE/GPU/CPU)
```

```python
artifact = lm7.export(
    model.eval(),
    args=(example_input,),
    target="apple",
    backend="coreml",
    output="model.lm7",
)
output = artifact(example_input)  # runs through Core ML immediately

reloaded = lm7.load_artifact("model.lm7")
output = reloaded(example_input)
```

## Install

Core ML support ships inside the base `executorch` package — no separate
extra beyond `.[executorch]`, and `coremltools` comes along as its dependency:

```bash
python3 -m venv .venv-et
.venv-et/bin/python -m pip install -e ".[dev,executorch]"
```

Let the extra resolve PyTorch — the `executorch` wheel declares the torch its
prebuilt runtime extension is ABI-linked against, and pinning one yourself
afterwards is what [docs/executorch.md](executorch.md#install) warns about.
macOS has no CUDA build to avoid, so the extra alone is the whole install here.

coremltools reported "Torch 2.13.0 has not been tested with coremltools" at
install time (tested up to 2.7.0) — a warning, not a failure; every test in
this backend's suite passed against it, on an M-series Mac under macOS 26 with
ExecuTorch 1.4.0 and coremltools 9.0.

## Options

```python
options = {"compute_unit": "all", "compute_precision": "float16"}  # both shown are the defaults
```

- `compute_unit`: `"all"` (ANE + GPU + CPU, Core ML's own choice), `"cpu_only"`,
  `"cpu_and_gpu"`, or `"cpu_and_ne"`.
- `compute_precision`: `"float16"` (Core ML's default) or `"float32"`.

## Scope

- **macOS only.** `probe()` checks `sys.platform == "darwin"`; Linux reports
  the backend unavailable rather than half-succeeding at export and failing
  at load.
- **`target="apple"` only.** Exporting for `cpu` (Intel or Arm) raises
  `BackendUnavailableError`; only LM7's Apple GPU target reaches this backend
  today, even though Core ML itself can run on any Mac.
- **Export-only**, like every ExecuTorch delegate. `lm7.compile()` never
  selects it.
- **Positional tensor inputs only**, static shapes only — matches the other
  ExecuTorch delegates.
- **Delegation must be non-zero.** Like `qnn`, a lowering that delegates no
  call sites raises rather than silently shipping an artifact that only runs
  portable kernels.
- **Not device-bound.** The `.pte` embeds an *uncompiled* Core ML model spec;
  whichever Mac loads it compiles it locally with Apple's own compiler (the
  first load is the slow one — that step is cached after). This is why the
  manifest records `"device_bound": false`, unlike `qnn`'s SoC-pinned
  artifacts.

## Validated

Real ExecuTorch + coremltools, on an Apple M3 Pro, ExecuTorch 1.3.1:

- Export, immediate execution, and reload-then-execute all match eager —
  `tests/test_coreml_integration.py`.
- `embedding` compiles and executes correctly, the same operator that
  separates a working PyTorch→Core ML path from a broken one elsewhere in
  this project (see [TVM](tvm.md)).
- `compute_unit="cpu_only", compute_precision="float32"` matches eager to
  1e-4 — no ANE/GPU float16 rounding in that path. The default
  (`all`/`float16`) measured ~3e-4 to ~4e-4 max absolute difference on the
  same MLP, well inside float16 precision.
- Delegation is complete for both models tested: 1 delegate call site covers
  the whole graph, same shape as the XNNPACK numbers in
  [executorch.md](executorch.md).

Not yet tried: a real causal LM (only a small MLP and an embedding+linear
model so far), INT8/PT2E quantization (executorch's XNNPACK quantizer is
XNNPACK-specific — Core ML has its own separate quantization path in
coremltools, not wired up here), and `minimum_deployment_target` (the export
warns that the default spec version "will not run on all versions of
iOS/macOS" without it).

For physical iPhone validation, use the staged runbook in
[iOS device testing](ios-device-testing.md). The first target is an AWS Device
Farm run on the device AWS lists as "Apple iPhone 12", using a tiny deterministic
Core ML `.pte` before moving to language models.

## References

- [ExecuTorch Core ML backend](https://docs.pytorch.org/executorch/stable/backends-coreml.html)
- [coremltools](https://apple.github.io/coremltools/docs-guides/)
- [ExecuTorch Core ML source](https://github.com/pytorch/executorch/tree/release/1.3/backends/apple/coreml)
