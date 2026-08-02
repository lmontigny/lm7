# Inspecting LM7 artifacts

Use `lm7 artifact inspect` to check an artifact before copying it to a device
or loading any optional runtime:

```bash
lm7 artifact inspect model.lm7
lm7 artifact inspect model.lm7 --json
```

Inspection reads `manifest.json` and hashes every declared source, compiled,
and weights payload. It does not load the PyTorch program or initialize the
artifact backend, so QNN, ExecuTorch, TensorRT, and other deployment artifacts
can be inspected on a machine where their runtime is absent.

The report includes the backend and target, compiled payload, portability,
runtime requirements, delegate coverage, deployment constraints, and checksum
status. QNN reports additionally show the SoC, HTP architecture, precision,
QNN SDK version, and required runtime libraries.

```text
Backend:          qnn
Target:           qualcomm:sm8750
Payload:          compiled_model.pte
Device-bound:     yes
Precision:        fp16
Delegation:       5 / 7 calls
QNN SDK:          2.37.0
QNN SoC:          SM8750
HTP architecture: v79
Host executable:  no
Deployment:       requires Android SM8750 HTP runtime and QNN SDK 2.37.0
Checksums:        valid
```

The command exits with status `1` when a declared payload is missing, lacks a
checksum, or does not match its checksum. Invalid paths and malformed or
unsupported manifests exit with status `2`, like other LM7 CLI usage errors.

Delegate coverage below 50% emits a warning. This is an inspection heuristic,
not a correctness failure: undelegated calls may run through portable kernels,
depending on the runtime.

Python callers can consume the same structured result without parsing CLI
output:

```python
inspection = lm7.inspect_artifact("model.lm7")
if not inspection.valid:
    print(inspection.errors)
```
