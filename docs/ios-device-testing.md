# iOS device testing

LM7's Apple phone path is an export path, not an in-process compile path. The
artifact to validate is an ExecuTorch `.pte`, optionally delegated to Core ML:

```text
PyTorch module
  -> torch.export
  -> ExecuTorch CoreMLPartitioner
  -> compiled_model.pte
  -> iOS app
  -> ExecuTorch runtime
  -> Core ML delegate
  -> Core ML chooses ANE / GPU / CPU
```

There is also a portable CPU path:

```text
PyTorch module
  -> torch.export
  -> ExecuTorch XNNPACK
  -> compiled_model.pte
  -> iOS app
  -> ExecuTorch runtime
  -> XNNPACK / portable CPU kernels
```

The Core ML path is the one to validate first for an iPhone support claim. The
XNNPACK path is the CPU fallback.

## Support tiers

Keep the claim tied to the tier that has actually run.

| Tier | What runs | What it proves | What it does not prove |
| --- | --- | --- | --- |
| Export | `lm7.export(..., target="apple", backend="coreml")` on macOS | LM7 can produce an ExecuTorch/Core ML `.pte` and manifest | iOS app integration or real-device execution |
| Simulator smoke | Minimal iOS app or XCTest loads the `.pte` in the iOS simulator | Framework linkage, resource packaging, and basic runtime call shape | iPhone hardware, ANE/GPU dispatch, or latency |
| Real iPhone | The same app/test runs on a physical iPhone or device cloud | The artifact executes on iOS hardware and matches a host reference | Which Core ML compute unit ran the whole graph, unless separately instrumented |
| Acceleration evidence | Real device, compare `all` against CPU-only where available | A practical speed signal for Core ML dispatch | A definitive ANE-only claim; Core ML may split work across engines |

HN-safe wording before real-device evidence:

> LM7 exports Apple-targeted ExecuTorch/Core ML artifacts today. They are
> validated on macOS; iOS simulator and real-iPhone validation are the next
> device gates.

HN-safe wording after the first AWS Device Farm run:

> LM7 exports an ExecuTorch/Core ML `.pte` that has run on a real iPhone 12
> through AWS Device Farm, matching a host reference on a fixed test input.

Do not say "uses the Neural Engine" until there is instrumentation or a measured
compute-unit comparison supporting that claim.

## Export the first artifact

Start with a tiny deterministic model, not a causal LM. A
`Linear -> GELU -> Linear` MLP or embedding+linear model is enough to prove the
runtime path and keeps the app fixture small.

```python
from pathlib import Path

import torch
import lm7

torch.manual_seed(0)

model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.GELU(),
    torch.nn.Linear(8, 2),
).eval()
example = torch.randn(3, 4)

artifact = lm7.export(
    model,
    args=(example,),
    target="apple",
    backend="coreml",
    output="artifacts/ios/coreml-mlp.lm7",
    options={
        "compute_unit": "all",
        "compute_precision": "float16",
    },
)

expected = model(example).detach()
Path("artifacts/ios/input.pt").parent.mkdir(parents=True, exist_ok=True)
torch.save({"input": example, "expected": expected}, "artifacts/ios/golden.pt")

print(artifact.path / "compiled_model.pte")
print((artifact(example) - expected).abs().max())
```

Then inspect the artifact:

```bash
lm7 artifact inspect artifacts/ios/coreml-mlp.lm7
```

The file the iOS app needs is:

```text
artifacts/ios/coreml-mlp.lm7/compiled_model.pte
```

The `golden.pt` file is a host-side convenience. For the app, convert the input
and expected tensors to a simple JSON, binary float array, or XCTest fixture.

## Simulator smoke

Use the simulator only as an integration gate. It is useful because ExecuTorch
publishes Apple `.xcframework` targets that are compatible with iOS devices and
simulators, but it is not an iPhone hardware result.

The minimal simulator test should:

1. Build an iOS test app or XCTest target.
2. Bundle `compiled_model.pte`.
3. Link ExecuTorch runtime plus the backend used by the artifact:
   `executorch`, `backend_coreml`, and the required kernels for Core ML; or
   `backend_xnnpack` for the CPU fallback.
4. Force-load static registration if needed. ExecuTorch documents `-all_load`
   or `-force_load` when errors mention unregistered kernels or backends.
5. Load the `.pte`.
6. Feed the fixed input tensor.
7. Compare against the golden output with float16-style tolerance.

Suggested tolerance for the first Core ML smoke:

```text
max_abs_diff <= 1e-2
```

Tighten it after the first real output is known. The existing macOS Core ML
integration tests saw much smaller differences on simple models, but an iPhone
run should record its own number.

## AWS Device Farm run

AWS Device Farm is the first real-device target for this repo. Select the device
AWS lists as:

```text
Apple iPhone 12
```

Record the exact values in the PR or follow-up doc:

| Field | Value |
| --- | --- |
| Provider | AWS Device Farm |
| Device name | Apple iPhone 12 |
| iOS version | TODO |
| Minimum OS expectation | iOS 16 or newer for the ExecuTorch Core ML backend |
| App build | TODO |
| ExecuTorch version | TODO |
| Backend | `coreml` first; optionally `executorch` / XNNPACK fallback |
| LM7 commit | TODO |
| Model | tiny MLP fixture first |
| Input shape | `3x4` |
| Compute options | `compute_unit=all`, `compute_precision=float16` |
| Max abs diff vs host | TODO |
| Latency | Optional; record only if measured inside the app |
| Result | TODO |

For the first pass, use an interactive remote access session if that is faster
than wiring full XCTest automation. Device Farm supports installing an uploaded
`.ipa` in a remote session, which is enough for a manual validation screenshot
and device log. The stronger follow-up is an automated XCTest that exits
non-zero when the diff exceeds tolerance.

Minimum successful evidence:

- device log line naming the artifact and backend;
- device log line with input shape and output shape;
- max absolute difference against the host golden output;
- pass/fail line;
- screenshot or Device Farm run URL;
- exact iOS version and iPhone model.

Example result block to paste back into this doc:

```text
Provider: AWS Device Farm
Device: Apple iPhone 12
iOS: TODO
Artifact: coreml-mlp.lm7/compiled_model.pte
Backend: ExecuTorch Core ML
Input: float32[3, 4]
Output: float32[3, 2]
Max abs diff vs host eager: TODO
Result: PASS
```

## Acceleration evidence

Core ML's `all` compute setting allows the OS to use available compute units,
including the Neural Engine. That does not mean the whole model ran on ANE.
Core ML may split work across CPU, GPU, and ANE, and there is no LM7-level API
that reports the exact processor for each op.

For a practical first signal, build or export variants that let the app compare:

- Core ML `all`, via `options={"compute_unit": "all"}`
- Core ML CPU-only, via `options={"compute_unit": "cpu_only"}`
- XNNPACK CPU fallback, via `backend="executorch", target="cpu"`

Report latency deltas as evidence of acceleration. Do not report "ANE support"
unless the app is instrumented through Xcode/Instruments or another accepted
device-side method.

## Current gaps

- **No checked-in iOS app harness yet.** The repo has Android device testing
  docs, but not the equivalent iOS XCTest app.
- **No physical iPhone result yet.** Core ML export/reload has been validated on
  macOS; iPhone execution needs the AWS Device Farm run.
- **`minimum_deployment_target` is not wired through LM7's Core ML options.**
  Core ML supports deployment targets, and this should be exposed before making
  broad iOS-version compatibility claims.
- **No language-model iPhone claim.** Start with a small deterministic model.
  A causal LM adds tokenizer, memory, artifact-size, and operator-coverage
  variables that obscure the device-runtime question.

## References

- ExecuTorch iOS runtime and framework integration:
  <https://docs.pytorch.org/executorch/stable/using-executorch-ios.html>
- ExecuTorch iOS backends:
  <https://docs.pytorch.org/executorch/stable/ios-backends.html>
- ExecuTorch Core ML backend:
  <https://docs.pytorch.org/executorch/stable/backends-coreml.html>
- Apple Core ML compute units:
  <https://developer.apple.com/documentation/coreml/mlcomputeunits>
- AWS Device Farm:
  <https://docs.aws.amazon.com/devicefarm/>
