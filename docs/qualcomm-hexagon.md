# Qualcomm Hexagon NPU evaluation plan

> [!NOTE]
> LM7 now reaches Snapdragon devices through their **CPU**, via
> [ExecuTorch](executorch.md)'s XNNPACK delegate. That path needed no Qualcomm
> SDK and no device, which is why it shipped first. This plan remains the route
> to the Hexagon **NPU** specifically. ExecuTorch also has a Qualcomm QNN
> delegate, which is likely a cheaper way in than hexagon-mlir and should be
> compared before this plan is executed.

LM7 has no Qualcomm NPU path today. [Hexagon-MLIR](https://github.com/qualcomm/hexagon-mlir)
is Qualcomm's open-source compiler toolchain for running Triton kernels and
PyTorch models on Hexagon NPUs, and it should be evaluated before a Qualcomm
target or backend is added to automatic planning.

This document records what the integration would cost, what has to be true
before it is worth building, and how to measure it. `benchmarks/hexagon.py` is
the measurement harness described below.

## Why this is not a normal backend

Every backend LM7 registers today compiles for a device that PyTorch can already
see: `torch.cuda`, `torch.xpu`, `torch.backends.mps`, or a PJRT TPU. Hexagon has
none of that.

- There is no `torch.hexagon` device and no dynamo backend. The PyTorch path is
  `torch_mlir.fx.export_and_import(...)` to Linalg-on-tensors IR, then
  `TorchMLIRHexagonLauncher.run_torch_mlir(...)`, which lowers the IR to object
  code, links a `.so`, and executes it.
- Execution happens **off-host**. The compiler runs on an x86 Linux host; the
  kernel runs on an adb-reachable Hexagon device (`ANDROID_HOST`,
  `ANDROID_SERIAL`) or on the Hexagon simulator (`RUN_ON_SIM=1`).

That collides with LM7's local-inference-only charter in three specific places,
and a future backend would have to answer all three:

- `TargetSpec` would need `remote=True`, as `aws:trainium` already parses. No
  other registered backend sets it.
- `transfers="automatic"` has nothing to transfer to. Inputs stay CPU tensors
  and the launcher moves them to the device itself, so `CompiledModule`'s device
  preparation and `InputDeviceError` checks do not apply.
- The compiled result is a cross-compiled `.so` for a specific
  `HEXAGON_ARCH_VERSION`, not a process-local callable. It is closer to an
  `aot_inductor` artifact than to an `inductor` one, and its portability across
  SDK and device versions is an open question.

## Prerequisites

Hexagon-MLIR is built from source. There is no wheel. Per its user guide:

- Ubuntu 22.04 on x86_64, Python 3.11.
- LLVM built from source at the revision in `triton/cmake/llvm-hash.txt`, with
  `Hexagon` in `LLVM_TARGETS_TO_BUILD`.
- Hexagon SDK 6.4.0.2, Hexagon Tools 19.0.02, and Hexagon Kernel Library
  (HexKL) 1.0.0, all downloaded from Qualcomm's Software Center.
- Triton and triton-shared with Qualcomm patches applied
  (`ci/setup_submodules.sh`).
- `HEXAGON_ARCH_VERSION` set to a tested architecture: 73, 75, 79, or 81. HexKL
  is documented as not valid for v81.
- Either a Hexagon device (`ANDROID_HOST` and `ANDROID_SERIAL`) or the
  simulator (`RUN_ON_SIM=1`). Qualcomm directs device-access requests to
  hexagon-mlir.support@qti.qualcomm.com.

`scripts/build_hexagon_mlir.sh` in the hexagon-mlir repository automates most of
this. Expect a multi-hour first build.

Nothing in this plan can be validated on macOS, on arm64, or without the
Qualcomm SDK downloads. The harness is written so that its host CPU paths still
run anywhere and the Hexagon paths report themselves unavailable.

## Candidate integration paths

- **torch-mlir to `TorchMLIRHexagonLauncher`**: the preferred first path and the
  one the harness implements. It is the only PyTorch-native route, it keeps the
  input an `nn.Module`, and it is what Qualcomm's own tutorials exercise.
- **Qualcomm AI Engine Direct (QNN) via ONNX**: worth comparing if runtime
  packaging or Android app deployment becomes the priority. It is a separate,
  largely closed stack and does not reuse the torch-mlir work.
- **`linalg-hexagon-opt` and `linalg-hexagon-translate` CLIs**: offline lowering
  and assembly inspection. Requires no device, so it is the cheapest way to see
  whether a model's operators lower at all before chasing hardware access.

## Acceptance criteria

Evaluate against eager CPU on the same host, dtype, batch size, and prompt
shape, with the Hexagon architecture version recorded alongside every result.

- **Correctness**: compiled logits must match eager CPU within the tolerance
  policy used for LM7's low-precision paths. Qualcomm's own GPT-2 tutorial uses
  `atol=0.03` for float16, which is the harness default.
- **Coverage**: start with `mlp` and GPT-2 trimmed to two layers, matching the
  tutorial. Full-size causal LMs are not assumed to fit a mobile NPU; extending
  coverage to the SmolLM2 and Llama 3.2 1B shapes used elsewhere in LM7 is a
  later step, gated on the two-layer case passing.
- **Compile cost**: report wall-clock time from Linalg IR to a linked, deployed,
  executed `.so`.
- **Device latency**: see the limitation below. Steady-state per-inference
  latency is not measurable through the launcher's top-level API today.
- **Memory**: not reported by the harness. The launcher's Light Weight
  Profiling (`enableLWP`) output is the likely source for TCM and VTCM usage.
- **Packaging**: document whether the generated `.so` can be reloaded in a fresh
  process and what it is pinned to (SDK version, tools version, arch version).
- **Failure behavior**: a missing toolchain, an unsupported operator, and a
  device that cannot be reached must each produce an actionable message. If this
  ever becomes a backend, those must map onto `BackendUnavailableError` and
  `CompilationError` and must preserve eager fallback semantics.

### Known measurement limitation

`run_torch_mlir()` is a single entry point that lowers, links, deploys, and
executes. Every Python-level call repeats all of it, so timing a loop of calls
measures compilation, not inference. On-device repetition is expressed by the
launcher's `iterations` argument instead.

The harness therefore reports `compile_and_run_ms` for Hexagon paths and emits
`latency_median_ms`, `latency_p95_ms`, and `samples_per_second` as `null` with a
reason, rather than publishing a number that means something else. Getting true
steady-state latency requires either the launcher's profiler output or splitting
`compile_torch_mlir()` from `execute_kernel()`. Both are internal APIs that the
repository's own TODO comments mark as in flux, so the harness does not depend
on them yet.

## First implementation slice

Do not register an LM7 backend until an evaluation on real hardware shows a
clear use case. `benchmarks/hexagon.py` is the first slice. It runs one workload
through four paths under a single harness:

| Path | Runs on | Reports |
| --- | --- | --- |
| `eager` | host CPU | correctness reference, full latency |
| `inductor` | host CPU | full latency, host baseline |
| `hexagon` | NPU over adb | `compile_and_run_ms`, accuracy vs eager |
| `hexagon-sim` | Hexagon simulator | accuracy vs eager only |

`hexagon-sim` exists so correctness can be checked without device access. Its
timings are meaningless, so it reports none.

A later backend PR could wrap the winning path behind `backend="hexagon"` with a
`qualcomm` vendor target, lower automatic priority than `inductor`, and
`remote=True`. That PR should not be written before this evaluation runs.

## Validation commands

The host paths need no Qualcomm toolchain and establish the CPU baseline:

```bash
uv pip install -e ".[dev,hf]"
python benchmarks/hexagon.py --model mlp --path eager inductor
python benchmarks/hexagon.py --model gpt2 --path eager inductor
```

`--model gpt2` downloads GPT-2 small and keeps the first two blocks, so
`transformers` reports the discarded blocks as `UNEXPECTED` weights on load.
That output is expected and means the trim took effect.

Inside a built hexagon-mlir environment, add the device-free correctness check:

```bash
source scripts/set_local_env.sh          # from the hexagon-mlir repository
export HEXAGON_ARCH_VERSION=75
python benchmarks/hexagon.py \
  --model gpt2 --path eager hexagon-sim \
  --dtype float16 \
  --output artifacts/benchmarks/hexagon-gpt2-fp16-sim.json
```

With a device attached, run the full comparison:

```bash
export ANDROID_HOST=<hostname> ANDROID_SERIAL=<device>
python benchmarks/hexagon.py \
  --model gpt2 --path eager inductor hexagon \
  --dtype float16 --iterations 10 \
  --output artifacts/benchmarks/hexagon-gpt2-fp16-v75.json
```

Record the exact Hexagon SDK, Hexagon Tools, HexKL, LLVM, Triton, torch-mlir,
PyTorch, device, and `HEXAGON_ARCH_VERSION` beside any published result. None of
these numbers are comparable across toolchain versions.

## References

- [Hexagon-MLIR repository](https://github.com/qualcomm/hexagon-mlir)
- [Hexagon-MLIR user guide](https://github.com/qualcomm/hexagon-mlir/blob/main/docs/user-guide.md)
- [torch-mlir tutorials](https://github.com/qualcomm/hexagon-mlir/tree/main/docs/tutorials/torch-mlir),
  including the [GPT-2 walkthrough](https://github.com/qualcomm/hexagon-mlir/blob/main/docs/tutorials/torch-mlir/gpt2.md)
  this harness mirrors
- [torch-mlir](https://github.com/llvm/torch-mlir)
- [Hexagon SDK](https://softwarecenter.qualcomm.com/catalog/item/Hexagon_SDK)
