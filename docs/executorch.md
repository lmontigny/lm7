# Edge and mobile support with ExecuTorch

LM7 exports models for phones and embedded devices through
[ExecuTorch](https://github.com/pytorch/executorch), PyTorch's on-device
runtime. `lm7.export(..., backend="executorch")` writes a `.pte` — the file
Android and iOS actually load — into the `.lm7` artifact.

```
nn.Module → torch.export → [optional PT2E INT8] → XNNPACK lowering → .pte → phone
```

This is an **export-only** backend. `lm7.compile()` never selects it, because a
phone is not a device the calling process can reach; see [scope](#scope).

## Why this is the edge path LM7 could actually build

Vendor-specific mobile delegates need their platform SDK before they can lower a
graph. ExecuTorch's XNNPACK delegate needs neither, while LM7's separate
[QNN backend](qnn.md) explicitly gates on Qualcomm AI Engine Direct SDK. It targets ARM64 *and*
x86-64, so the same lowering that produces a phone artifact runs on an ordinary
Linux CI box, and `tests/test_executorch_integration.py` validates numerics
against eager on every run.

That is also the honest limit of what is claimed for this backend: XNNPACK is a
**CPU** delegate. Qualcomm HTP export is available through `backend="qnn"` and
`target="qualcomm:sm8750"`; other vendor delegates remain separate work.

## Install

ExecuTorch's prebuilt runtime extension is linked against a specific libtorch,
so it needs an environment built around the matching PyTorch rather than the
latest one. Installing it beside a CUDA PyTorch will fail at import with an
undefined `c10` symbol.

```bash
python3 -m venv .venv-et
.venv-et/bin/python -m pip install -e ".[dev,executorch]"
.venv-et/bin/python -m pip install "torch==2.12.*" --index-url https://download.pytorch.org/whl/cpu
source .venv-et/bin/activate
```

Activate the environment rather than calling `.venv-et/bin/python` directly.
ExecuTorch shells out to `flatc` during serialization and resolves it from
`PATH`; the wheel ships one in its `bin/` directory, which only lands on `PATH`
when the environment is active. LM7 works around this — `compile_exported` adds
the wheel's `bin/` for the duration of a lowering — but `lm7 backends` reports
the situation either way.

## Export

```bash
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct model.lm7 \
  --target cpu --backend executorch --quantize int8
```

```python
artifact = lm7.export(
    model.eval(),
    args=(example_input,),
    target="cpu",
    backend="executorch",
    output="model.lm7",
    options={"quantization": "int8"},
)
output = artifact(example_input)  # runs through the ExecuTorch runtime
```

`target` must be `cpu`. XNNPACK is a CPU delegate, and a `.pte` built for it
runs on Android and iOS CPUs as well as the host — so the CPU target is not a
limitation of the artifact, only a description of the delegate.

Reload it like any other LM7 artifact:

```python
loaded = lm7.load_artifact("model.lm7")
output = loaded(example_input)
```

The reloaded callable takes **positional tensors only**. An ExecuTorch method
has a flat input list with no keyword names, so capture with
`args=(input_ids, attention_mask)` rather than kwargs.

## INT8 quantization

Pass `--quantize int8` to `lm7 model export`, or
`options={"quantization": "int8"}` to `lm7.export`. LM7 uses ExecuTorch's
XNNPACK PT2E post-training quantizer: activations use symmetric per-tensor INT8
and weights use symmetric per-channel INT8. The captured example input is run
once as the calibration sample before conversion and XNNPACK lowering.

Calibration data matters. For `model export`, choose a representative prompt;
for the Python API, pass a representative tensor batch. One sample is a useful
first export, not an accuracy guarantee. Compare the quantized model on the
dataset and shapes that matter before shipping it.

INT8 currently requires a fixed capture shape. LM7 also fails the export when
the quantizer inserts no quantized operators, instead of silently producing a
float artifact. Unsupported operators can remain as portable ExecuTorch
kernels, so inspect both `quantized_ops` and delegate coverage in the manifest.

The `.pte` is the deployable, quantized payload. LM7 still retains the original
float `exported_program.pt2` for reproducibility and debugging, so total `.lm7`
directory size does not shrink in proportion to the `.pte`.

## What the manifest records

```json
"runtime_requirements": {
  "executorch": "1.3.1",
  "delegate": "xnnpack",
  "delegated_calls": 155,
  "total_calls": 1970,
  "quantization": "int8",
  "quantized_ops": 42,
  "calibration_samples": 1,
  "device_bound": false
}
```

Two fields are worth reading before shipping an artifact.

`device_bound` is `false`. Every other compiled LM7 payload is pinned to the
machine that produced it; a `.pte` carries its program and its weights and the
delegate spans ARM64 and x86-64, so the same bytes run on the build host and the
phone. This has been checked on a Snapdragon 8 Elite rather than assumed:
[android-device-testing.md](android-device-testing.md) reports the device
agreeing with the host x86-64 runtime to 5e-07 on the same `.pte`.

`delegated_calls` / `total_calls` is **partition coverage**, and it is usually
not 1.0. Operators XNNPACK does not implement stay on ExecuTorch's portable
kernels, which are correct but unoptimized. A low ratio on a model you care
about is the signal to look at what did not partition, not a failure.

## Measured on this project

An `x86-64` Linux host, ExecuTorch 1.3.1, PyTorch 2.12.1+cpu, float32.

| | 3-layer MLP | SmolLM2-135M prefill |
| --- | --- | --- |
| Total export | 4.3 s | 178 s |
| `.pte` size | 8 KB | 622 MiB |
| `.lm7` artifact size | 12 KB | 1139 MiB |
| Delegate coverage | whole graph | 155 of 1970 calls |
| Max logit difference vs eager | 1e-4 | 1.4e-4 |
| Steady latency | 0.14 ms | 106 ms |

Three things to take from the SmolLM2 column.

Lowering is slow — minutes, not seconds. That is fine for a build step and
unacceptable in a request path, which is the whole argument for AOT.

The artifact is nearly twice the `.pte`, because LM7 always writes
`exported_program.pt2` beside the compiled payload. Only the `.pte` ships to a
device; the `.pt2` is what makes the artifact reloadable as a torch program.

And 622 MiB is float32 weights written verbatim. The new INT8 path addresses
that deployment cost, but the SmolLM2 numbers above remain the float baseline;
full-model INT8 accuracy and latency have not yet been characterized.

A separate 256→512→64 MLP check reduced the deployable `.pte` from 660,480 to
174,976 bytes (73.5%) and had 0.0082 maximum absolute error against eager FP32.
That verifies the storage and runtime path; it is not a language-model quality
claim.

## Scope

- **Export only.** `lm7.compile()` will not select this backend, and asking for
  it raises. The artifact is the deliverable.
- **XNNPACK only.** This backend does not change delegates. Qualcomm HTP uses the
  separate [QNN backend](qnn.md); Core ML, MediaTek, Vulkan, Arm Ethos-U, and
  Exynos remain unwired here.
- **INT8 PTQ only.** There is no INT4, QAT, multi-sample calibration API, or
  backend-specific accuracy gate yet. Quantization support depends on the
  operator patterns recognized by ExecuTorch's XNNPACK quantizer.
- **Static shapes.** Whatever `torch.export` captured is what the `.pte` accepts;
  INT8 export explicitly rejects dynamic sequence capture.
- **No on-device execution or deployment.** LM7 writes the artifact; getting it
  onto a phone and into an app is ExecuTorch's Android and iOS tooling.
- **No physical-device CI.** CI validation is host XNNPACK on x86-64. A `.pte`
  has been run on a real ARM64 phone and agreed with the host — see
  [android-device-testing.md](android-device-testing.md) — but that is a manual
  gate on one device, not continuous coverage.

## Validate an artifact

```bash
python -m pytest tests/test_executorch_integration.py -q
```

The third test loads the `.pte` in a fresh interpreter that never imports LM7,
which is the property that makes the artifact worth producing.

To take the same artifact all the way to a phone, see
[android-device-testing.md](android-device-testing.md).

## References

- [ExecuTorch](https://github.com/pytorch/executorch) and its
  [documentation](https://docs.pytorch.org/executorch/stable/)
- [XNNPACK backend](https://docs.pytorch.org/executorch/stable/backends-xnnpack.html)
- [Backend overview](https://github.com/pytorch/executorch/blob/main/docs/source/backends-overview.md)
  — the full delegate list and which platform each needs
- [Runtime Python API](https://docs.pytorch.org/executorch/stable/runtime-python-api-reference.html)
