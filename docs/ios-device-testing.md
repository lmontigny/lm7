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

## How users use LM7 on iPhone

LM7 is not installed by end users on the phone. It is a developer-side export
tool. The phone only sees the exported model artifact and the iOS runtime that
loads it:

```text
Developer trains or builds a PyTorch model
  -> developer runs lm7.export(..., target="apple", backend="coreml")
  -> LM7 writes an ExecuTorch/Core ML compiled_model.pte
  -> iOS developer bundles compiled_model.pte in the app
  -> user installs and opens the app
  -> app runs local inference through ExecuTorch and Core ML
```

From the user's point of view there is no Python, no LM7 CLI, and no model
conversion on device. If the app bundles the `.pte`, inference can run locally
without a network request.

The iOS app side is intentionally small:

```swift
let module = Module(filePath: compiledModelPath)
try module.load("forward")
let output = try module.forward(input)
```

ExecuTorch loads the `.pte`, dispatches the delegated segment to Core ML, and
Core ML chooses the actual iPhone compute plan.

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

The `golden.pt` file is a host-side convenience. The app reads `golden.json`
instead, so it needs no torch reader on device.

`compute_unit="all"` means LM7 asks the Core ML path to allow every available
Core ML compute unit: CPU, GPU, and Apple Neural Engine where available. It
does not force a Neural Engine-only run. On a physical iPhone, Core ML decides
how to place the graph across CPU, GPU, and ANE based on the model, operators,
shapes, precision, hardware, and iOS version. On the simulator, the same setting
exercises the Core ML software path on the Mac and does not prove iPhone ANE
usage.

## The harness

`tools/ios_runner` is a minimal SwiftUI app that bundles the `.pte` and
`golden.json`, runs `forward` at launch, and reports `max_abs_diff` against the
golden output both on screen and through `NSLog` with an `LM7VALIDATOR` prefix.
It links ExecuTorch `executorch`, `backend_coreml`, and `kernels_optimized`
under `-all_load`, without which the backend and kernel registrations are
stripped from the static libraries.

```bash
./tools/ios_runner/build.sh project   # generate the Xcode project
./tools/ios_runner/build.sh ipa       # unsigned .ipa for a device cloud
```

The SPM branch must match the `executorch` pip version used to export the
artifact (`swiftpm-1.3.1` against 1.3.1); resolution pulls about 1.5 GB.

Two build notes that are not obvious:

- **No signing identity is needed for a device-cloud `.ipa`.** `build.sh ipa`
  builds with `CODE_SIGNING_ALLOWED=NO` and zips the `.app` under `Payload/`.
  AWS Device Farm re-signs apps for its public fleet and accepts the upload.
  Opening the generated project in Xcode is unaffected, so local device runs
  still use ordinary automatic signing.
- **`build.sh ipa` uses `-target`, not `-scheme`.** A machine with Xcode but
  without the downloaded iOS platform has no build destinations and fails with
  "iOS `<ver>` is not installed" even though the SDK is present. `-target`
  skips destination resolution. Local Xcode use — simulator or a connected
  iPhone — does need the platform: `xcodebuild -downloadPlatform iOS`.

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

### Result, 2026-08-17

`tools/ios_runner` on the iPhone 17 Pro simulator, iOS 26.5, hosted on an Apple
Silicon Mac:

```text
Artifact: coreml-mlp.lm7/compiled_model.pte
Backend: ExecuTorch Core ML
Input: float32[3, 4]
Output: float32[3, 2]
Max abs diff vs host eager: 3.051162e-04
Result: PASS
```

Host eager against the same artifact through `lm7.export`'s reloaded module was
`2.967e-04`, so the simulator agrees with the host to within the float16
rounding the Core ML path already introduces.

This clears the **Simulator smoke** tier and nothing above it. The simulator
executes Core ML against the *Mac's* hardware, so this says nothing about an
iPhone's Neural Engine, an A-series part, or latency.

## AWS Device Farm run

AWS Device Farm is the first real-device target for this repo. Select the device
AWS lists as:

```text
Apple iPhone 12
```

Record the exact values in the PR or follow-up doc:

| Field | Value |
| --- | --- |
| Provider | AWS Device Farm, `us-west-2`, public fleet |
| Device name | Apple iPhone 12, `iPhone13,2`, `D53gAP` |
| SoC | chip id `0x8101` (A14), `arm64e`, `ProductionSOC: true` |
| iOS version | 26.6, build `23G71` |
| Minimum OS expectation | iOS 16 or newer for the ExecuTorch Core ML backend |
| App build | `tools/ios_runner`, unsigned `.ipa`, `com.lm7.validator` |
| ExecuTorch version | 1.3.1 (pip and `swiftpm-1.3.1`) |
| Backend | `coreml` first; optionally `executorch` / XNNPACK fallback |
| LM7 commit | `53eb6cf` |
| Model | tiny MLP fixture first |
| Input shape | `3x4` |
| Compute options | `compute_unit=all`, `compute_precision=float16` |
| Max abs diff vs host | TODO — never read back off the device |
| Latency | Optional; record only if measured inside the app |
| Result | TODO |

Device identity above was read from the device itself over Appium
(`mobile: deviceInfo`), not from the console label.

Prefer a scheduled run over an interactive remote access session. Remote access
sessions expire on their own (about 40 minutes, observed) and meter while idle,
and their only programmatic control channel is
`endpoints.remoteDriverEndpoint`, a WebDriver URL — there is no SSH and no adb
into a Device Farm device. A scheduled run instead leaves device logs and video
as durable artifacts:

```bash
aws devicefarm create-device-pool --project-arn <proj> --name lm7-iphone12 \
  --rules '[{"attribute":"ARN","operator":"IN","value":"[\"<device-arn>\"]"}]'
aws devicefarm schedule-run --project-arn <proj> --app-arn <app> \
  --device-pool-arn <pool> --name lm7-coreml-mlp-validation \
  --test 'type=BUILTIN_FUZZ,parameters={event_count=200,throttle=200}'
```

Pin the pool to a single device ARN so a rerun cannot silently land on a
different phone.

**A `PASSED` run does not mean the model ran.** `BUILTIN_FUZZ` reports `PASSED`
whenever nothing crashed, and it does not report whether it ever launched the
app under test. A first attempt here used `event_count=1`, on the reasoning
that the harness does its work at launch — the run passed, and the syslog shows
the app installed at 13:05:57 and uninstalled at 13:08:50 with no launch event
and no `LM7Validator` process in between. Nothing was validated.

So always confirm from the device log, never from the run result:

```bash
./tools/ios_runner/fetch_devicefarm_result.sh <run-arn>
```

Absence of `LM7VALIDATOR` lines means no result, whatever the run says. Because
a fuzz run cannot be told to launch a specific app deterministically, the
reliable options are an Appium session that launches the bundle id explicitly,
or an XCTest target that exits non-zero when the diff exceeds tolerance. The
latter is the right end state.

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

- **No physical iPhone result yet.** Core ML export/reload has been validated on
  macOS; iPhone execution needs the AWS Device Farm run. The app has been built,
  uploaded, and launched on an iPhone 12 through Device Farm, but no
  `max_abs_diff` has been read back off the device, so nothing here is a device
  result.
- **The harness is a launch-time check, not an XCTest.** `tools/ios_runner`
  reports its verdict through `NSLog` and the screen. It cannot fail a CI job;
  an XCTest target that exits non-zero above tolerance is the stronger
  follow-up.
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
