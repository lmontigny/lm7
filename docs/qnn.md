# ExecuTorch QNN for Snapdragon HTP

LM7's `qnn` backend exports a device-bound ExecuTorch `.pte` for the Hexagon
HTP in Snapdragon 8 Elite (`SM8750`, HTP v79). It is an AOT deployment path:
the compiler runs on an x86-64 Linux host with Qualcomm AI Engine Direct SDK,
while the resulting program runs in an Android ExecuTorch runtime built with
the QNN backend.

```text
PyTorch module -> torch.export -> ExecuTorch QnnPartitioner -> QNN context -> .pte
```

This backend does not use `torch.compile`, open a device connection, deploy the
artifact, or run it through LM7's host Python process.

## Initial scope

- Target: `qualcomm:sm8750` only.
- Backend: QNN HTP only; not QNN GPU or LPAI.
- Precision: FP16 only.
- Inputs: positional tensors only.
- Shapes: static only.
- Output: `compiled_model.pte` inside a normal LM7 artifact.
- Delegation: export fails when QNN takes zero call sites.

Quantized `8a8w`, `16a8w`, `16a4w`, and blockwise modes need calibration and
accuracy validation and are intentionally deferred to a separate change.

## Host setup

Use the same PyTorch/ExecuTorch release pair described in
[executorch.md](executorch.md). QNN additionally requires an ExecuTorch source
checkout with its Qualcomm Python modules and Qualcomm AI Engine Direct SDK.
For ExecuTorch 1.3.1, use the QNN version required by that release; SDK and
device runtime versions must match.

```bash
export EXECUTORCH_ROOT=/path/to/executorch
export QNN_SDK_ROOT=/path/to/qairt/version
source "$QNN_SDK_ROOT/bin/envsetup.sh"
export PYTHONPATH="$EXECUTORCH_ROOT/..:$PYTHONPATH"

lm7 backends
```

`QNN_SDK_ROOT` must point at the directory containing `QNN_README.txt`.
Sourcing `envsetup.sh` supplies the host libraries used during QNN lowering.
LM7 reports the backend unavailable when ExecuTorch, the SDK root, Qualcomm
Python modules, or ExecuTorch's FlatBuffers serializer cannot be resolved.

The SDK is not downloaded or redistributed by LM7.

## Export

Python API:

```python
artifact = lm7.export(
    model.eval(),
    args=(example_input,),
    target="qualcomm:sm8750",
    backend="qnn",
    output="model-sm8750.lm7",
    options={"precision": "fp16"},
)
```

Hugging Face CLI:

```bash
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  model-sm8750.lm7 \
  --target qualcomm:sm8750 \
  --backend qnn \
  --dtype float16
```

The CLI example describes the supported interface, not validated full-model
operator coverage. Start with a small graph and inspect delegation before
attempting a language model.

## Artifact contract

QNN compilation is SoC- and SDK-bound. The manifest records that boundary:

```json
"runtime_requirements": {
  "delegate": "qnn",
  "backend": "htp",
  "soc_model": "SM8750",
  "htp_arch": "v79",
  "vtcm_mb": 8,
  "precision": "fp16",
  "qnn_sdk": "2.37.0",
  "delegated_calls": 1,
  "total_calls": 1,
  "device_bound": true
}
```

The manifest also lists the matching runtime payload names, including the QNN
ExecuTorch backend, QNN System/HTP libraries, v79 stub and skeleton, and HTP
prepare library. Use ExecuTorch's Qualcomm build/deployment scripts to package
them; do not mix a `.pte` built with one QNN SDK version with another runtime.

Inspect the deployment contract and verify every payload without installing or
initializing QNN:

```bash
lm7 artifact inspect model-sm8750.lm7
lm7 artifact inspect model-sm8750.lm7 --json
```

See [Inspecting LM7 artifacts](artifact-inspection.md) for output fields and
exit statuses.

Loading the LM7 artifact verifies checksums and exposes the source
`ExportedProgram`, but calling it raises an explicit deployment-only error. A
QNN `.pte` must run through an Android ExecuTorch runtime linked with the QNN
backend and the matching Qualcomm libraries.

## Validation boundary

Unit tests exercise target parsing, dependency diagnostics, compiler-spec
construction, HTP selection, strict delegation coverage, manifest metadata,
checksum validation, and host-execution refusal with mocked ExecuTorch QNN
modules.

This PR does not claim a successful real QNN lowering or device execution,
because Qualcomm AI Engine Direct SDK is not installed in the test environment.
Those require a separately provisioned SDK and compatible device runtime.

## References

- [ExecuTorch Qualcomm AI Engine backend](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)
- [ExecuTorch Android backends](https://docs.pytorch.org/executorch/stable/android-backends.html)
- [ExecuTorch Qualcomm source](https://github.com/pytorch/executorch/tree/release/1.3/backends/qualcomm)
