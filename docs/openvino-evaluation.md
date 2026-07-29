# OpenVINO evaluation plan

LM7's generic CPU path already works through PyTorch eager and TorchInductor.
OpenVINO should be evaluated as an optional Intel deployment backend before it
is considered for automatic planning.

## Candidate integration paths

- `torch.compile` backend: import `openvino.torch` and compile PyTorch modules
  with `backend="openvino"`. This best matches LM7's lazy backend protocol.
  `openvino` is the only dynamo backend OpenVINO registers today; the older
  TorchScript-based `openvino_ts` name no longer exists. Device and plugin
  configuration are passed through `torch.compile(options={"device": "CPU",
  "config": {...}, "model_caching": True, "cache_dir": ...})`.
- OpenVINO IR artifacts: convert a PyTorch module or `ExportedProgram` with
  `openvino.convert_model()`, save IR with `openvino.save_model()`, and load it
  through `openvino.Core().compile_model()`. This best matches LM7's artifact
  and deployment goals.

## Acceptance criteria

Compare OpenVINO against eager and TorchInductor on the same Intel host.

- Correctness: outputs must match eager PyTorch within a documented tolerance.
- Coverage: start with `mlp`, common TorchVision models, and at least one small
  Hugging Face encoder or causal-LM shape.
- First inference: measure conversion/compile time separately from steady-state
  inference.
- Deployment: verify whether IR artifacts load in a fresh Python process
  without importing PyTorch.
- Hardware: test CPU first, then Intel GPU or NPU only when the host runtime
  exposes those devices.
- Fallback: unsupported operators must produce actionable errors or fall back
  according to LM7's configured fallback policy.

## Behaviour that invalidates a naive comparison

These were confirmed by reading the OpenVINO sources and reproducing each one.
Any OpenVINO measurement that does not control for them is not comparable to
eager or Inductor.

- **Warmup.** OpenVINO's CPU plugin needs tens of calls to reach steady state,
  far more than eager or Inductor. A 5-call warmup left its median 4-5x too
  high, enough to invert the ranking against Inductor and to produce the wrong
  conclusion entirely. Use at least 30 warmup calls and check that the first
  half of the timed samples matches the second half before trusting a median.
- **Compile accounting.** Inductor and the OpenVINO dynamo backend compile
  lazily inside the first call; the IR path converts and compiles up front. A
  first-call number is therefore not comparable across paths — compare
  build + first call.
- **Non-AOT dynamo path breaks on CNNs.** `torch.compile(backend="openvino")`
  without `aot_autograd` fails on TorchVision models under recent PyTorch with
  `AssertionError: sources must not be empty for symbol sN`, raised from
  dynamo's guard machinery rather than caught by the backend. Passing
  `options={"aot_autograd": True}` avoids it.
- **Exported IR is dynamically shaped.** `torch.export` leaves batch and
  spatial dims symbolic, so `convert_model` yields IR like `[?,3,?,?]` while the
  dynamo path pins static shapes before compiling. Reshape to the intended
  shapes for a like-for-like comparison. On this host static versus dynamic made
  no measurable difference once warmup was correct, but the two paths are not
  otherwise comparable.
- **Reduced default precision.** The CPU plugin's `INFERENCE_PRECISION_HINT`
  does not default to FP32; it is FP16 on ARM hosts and BF16 on x86 hosts with
  AMX. An FP32 model therefore runs in reduced precision and looks both faster
  and less accurate than eager. Set
  `config={"INFERENCE_PRECISION_HINT": "f32"}` for a like-for-like comparison,
  and report the plugin default alongside it.
- **Silent compile-time fallback.** `fx_openvino` wraps the whole compilation in
  `try/except` and returns `torch._inductor.compile_fx(...)` on any exception,
  so a run labelled `openvino` can quietly be TorchInductor. It is not enough to
  check that `torch.compile(backend="openvino")` returned without raising.
- **Silent runtime fallback.** `OpenVINOGraphModule.__call__` catches runtime
  exceptions per partition, latches `perm_fallback = True`, and executes that
  subgraph in eager PyTorch from then on.
- **Verifying OpenVINO actually ran.** Read
  `openvino.frontend.pytorch.torchdynamo.execute.compiled_cache` (one entry per
  executed OpenVINO partition), `partitioned_modules`, and each
  `OpenVINOGraphModule.perm_fallback`. Zero compiled models after a call means
  the backend fell back.
- **IR weight compression.** `openvino.save_model()` defaults to
  `compress_to_fp16=True`, which shows up as FP16-level error on an otherwise
  FP32 comparison. Pass `compress_to_fp16=False` unless the compression is the
  thing being measured.
- **No bfloat16 on the dynamo path.** The FX decoder's dtype map covers
  float32/float16/float64/int/bool but not bfloat16, and the executor
  round-trips tensors through NumPy, which has no bfloat16 dtype. A bfloat16
  model only ever measures a silent fallback.
- **Output buffer aliasing.** The executor infers with `share_outputs=True`, so
  returned tensors can alias OpenVINO's output buffer and be overwritten by the
  next inference. Clone any tensor kept for a correctness comparison.

## First implementation slice

`benchmarks/openvino_eval.py` runs one model through eager (the correctness
reference), TorchInductor, OpenVINO `torch.compile`, and a saved-and-reloaded
OpenVINO IR artifact under a single harness, and writes JSON with environment
metadata, conversion/compile time, median and p95 latency, throughput, and error
against eager. It records `ov_partitions`, `ov_compiled_models`, and
`ov_runtime_fallbacks` per run so a silent fallback is visible in the results
rather than reported as an OpenVINO win.

It is named `openvino_eval.py` rather than `openvino.py` because running
`python benchmarks/openvino.py` would put `benchmarks/` first on `sys.path` and
shadow the real `openvino` package.

`--model` covers `mlp`, the TorchVision CNNs `resnet18`, `resnet50`, and
`mobilenet_v3_small` (random weights, so the run stays offline), and the same
causal-LM ids as `benchmarks/gpu.py`: `smollm2`, `lfm25`, `llama32-1b`, and
`qwen35-0.8b`. Causal LMs are wrapped so the harness sees positional tensors in
and one logits tensor out, with `use_cache=False`, which measures a single
prefill forward pass rather than a decode loop.

## The registered backend

The evaluation met its bar on Intel CPU, so `backend="openvino"` now exists as a
registered backend in `src/lm7/backends/openvino.py`.

```python
model = lm7.compile(model, target="cpu", backend="openvino")
```

It implements **the IR path only**, because that is the path the measurements
justify: `openvino_ir` beat eager on every workload on both hosts and beat
Inductor on all six Intel workloads, while the `torch.compile` path lost to
eager on two and never beat Inductor. LM7 exports with `torch.export`, converts
with `convert_model`, saves IR with `save_model`, and executes through
`Core().compile_model()`.

It is **opt-in**. `supports()` reports priority 80, below Inductor (100) and
AOTInductor (90), so `backend="auto"` never selects it. The latency case is
made; broad operator coverage is not, which is what would justify raising it.

The pitfalls in the section above are encoded as behaviour rather than left to
the caller:

| Pitfall | What the backend does |
| --- | --- |
| Reduced default precision | Pins `INFERENCE_PRECISION_HINT` to `f32`; override with `options={"inference_precision": ...}` |
| FP16 weight compression | Passes `compress_to_fp16=False` |
| Dynamically shaped export | Reshapes IR to the example input shapes; disable with `options={"static_shapes": False}` |
| Output buffer aliasing | Clones every returned tensor |
| No bfloat16 | Rejects bfloat16 models in `supports()` with an actionable reason |
| Silent device fallback | Raises if the requested device is absent from `Core().available_devices` instead of quietly using CPU |

Two limits worth knowing. The callable returns a tensor or a tuple of tensors,
so a model whose `forward` returns a dataclass (a Hugging Face `ModelOutput`,
for example) needs the same wrapper the benchmark harness uses. And only the
`cpu` and `intel` target vendors are accepted; Intel GPU and NPU are untested
because no such device was available.

### Artifacts

`lm7.export(..., backend="openvino")` writes the IR into an `.lm7` artifact
alongside the `ExportedProgram`, with both `compiled_model.xml` and
`compiled_model.bin` checksummed in the manifest:

```python
lm7.export(model, args=(example,), target="cpu", backend="openvino", output="model.lm7")
```

`load_artifact()` verifies both files and returns a callable backed by the IR.
In a bundle the OpenVINO entry ranks at 80, below `aot_inductor` and above a bare
`export`.

This is the payload that answers the evaluation's deployment criterion: the IR
runs on a machine with no PyTorch installed, through
`openvino.Core().compile_model()` alone.

`tests/test_openvino_integration.py` covers eager parity, the planner ranking,
the bfloat16 and device guards, FP32 weight preservation, the export round trip,
weight-checksum validation, and loading the saved IR in a subprocess that never
imports `torch`.

## Validation commands

Baseline the existing CPU paths through LM7:

```bash
uv pip install -e ".[dev,hf]"
python benchmarks/local.py --target cpu --backend eager inductor
```

Then run the side-by-side evaluation. Without OpenVINO installed it still
reports eager and Inductor and marks the OpenVINO paths unavailable:

```bash
uv pip install openvino torchvision
python benchmarks/openvino_eval.py \
  --model resnet18 \
  --path eager inductor openvino openvino_ir \
  --dtype float32 \
  --batch-size 4 \
  --inference-precision f32 \
  --output artifacts/benchmarks/openvino-resnet18-fp32-b4.json
```

Cover the causal-LM shapes the acceptance criteria ask for:

```bash
for model in smollm2 lfm25 llama32-1b qwen35-0.8b; do
  python benchmarks/openvino_eval.py --model "$model" --batch-size 1 \
    --output "artifacts/benchmarks/openvino-$model-fp32-b1.json"
done
```

Use `--device GPU` or `--device NPU` only on a host whose runtime exposes them
(`lm7 targets` and `openvino.Core().available_devices` both report what is
present), `--inference-precision default` to measure the plugin's own default
instead of FP32, and `--require-all` to fail rather than skip when a requested
path is unavailable. `--no-aot-autograd` and `--no-static-ir` reproduce the two
pitfalls above.

Record the OpenVINO, PyTorch, CPU, GPU/NPU, operating system, and driver
versions with the results; the JSON `environment` block captures most of this
automatically.

## Status

Measured on two hosts. The Intel run is the one that bears on the decision; the
Apple run established the harness and is kept because it is what the pitfalls
above were found on.

### Intel CPU (the decision-relevant host)

Intel Core i7-8086K (Coffee Lake, 6 cores / 6 threads, AVX2, **no AVX-512 and no
AMX**), Linux on WSL2, `openvino` 2026.2.1, PyTorch 2.13, FP32,
`--inference-precision f32`, 60 warmup calls. Every path was within the FP32
tolerance, and every run reported `ov_compiled_models == 1` with
`ov_runtime_fallbacks == 0`, so no result here is a disguised Inductor or eager
run.

Median latency in ms, lower is better:

| workload | eager | inductor | openvino | openvino_ir | IR vs eager |
| --- | --- | --- | --- | --- | --- |
| `mlp`, batch 4 | 2.52 | 2.67 | 5.24 | **1.78** | 1.4x |
| `mobilenet_v3_small`, batch 4 | 16.79 | 10.08 | 13.93 | **4.12** | 4.1x |
| `resnet18`, batch 4 | 81.63 | 66.55 | 68.80 | **50.52** | 1.6x |
| `resnet50`, batch 4 | 231.78 | 168.82 | 151.12 | **105.25** | 2.2x |
| `smollm2`, batch 1, 5 tokens | 50.02 | 36.62 | 52.43 | **22.80** | 2.2x |
| `lfm25`, batch 1, 5 tokens | 66.49 | 58.77 | 61.99 | **34.75** | 1.9x |

- **`openvino_ir` won every workload measured, on both hosts.** On Intel it beat
  eager by 1.4-4.1x and beat Inductor on all six. This is a stronger and more
  uniform result than the Apple run, and it is the central finding: the artifact
  path is where OpenVINO's value is, and it is also the path that fits
  `exporting.py` and `bundles.py`.
- **The `torch.compile` path is not worth adopting on this evidence.** It was
  slower than eager on `mlp` (5.24 vs 2.52) and on SmolLM2 (52.43 vs 50.02), and
  never beat Inductor on any workload. Both hosts agree.
- **The reduced-precision pitfall does not fire on pre-AMX Intel.** The doc
  above warns that `INFERENCE_PRECISION_HINT` defaults to BF16 on x86 with AMX.
  This CPU reports `optimization_capabilities: ['FP32', 'INT8', 'BIN',
  'EXPORT_IMPORT']` — no BF16 entry — and an `inference_precision_hint` that is
  already `float32`. So on Coffee Lake the default and `f32` are the same thing,
  and the accuracy trap the Apple host hit is AMX-gated rather than x86-wide.
  It will reappear on Sapphire Rapids or newer, so the flag stays necessary.
- **`mlp` never reached steady state**, even at 60 warmup calls: the harness
  still reported an early/late median ratio above its threshold on some runs.
  Its 1.4x is therefore the softest number in the table. Every other workload
  reported steady-state cleanly.
- Time to first inference still favours the JIT paths, and by a wider margin
  than on Apple: `openvino_ir` needs 20.3 s for SmolLM2 (5.5 s export, 11.9 s
  convert, 0.8 s save, 1.8 s compile) and writes a 540 MB FP32 IR, against
  43.8 s for Inductor — though for `resnet50` the IR path is *fastest* to first
  inference at 3.96 s against Inductor's 7.52 s. That cost is paid once per
  artifact rather than per process, which is the point of the IR path.

### Apple M4 Pro (harness reference)

`openvino` 2026.2.1, PyTorch 2.13, FP32, `--inference-precision f32`, 30 warmup
calls, all paths within the FP32 tolerance and all reported steady-state. This
is an ARM host, so it says nothing about Intel deployment speed.

| workload | eager | inductor | openvino | openvino_ir |
| --- | --- | --- | --- | --- |
| `mlp`, batch 4 | 0.86 | 1.02 | 2.44 | 1.16 |
| `resnet18`, batch 4 | 17.8 | 18.8 | 17.1 | **13.8** |
| `smollm2`, batch 1, 5 tokens | 17.4 | 17.3 | 29.8 | **7.8** |

- **An earlier version of this document reported OpenVINO as uniformly slower.
  That was a measurement error**, not a property of OpenVINO: the 5-call warmup
  in the first version of the harness left OpenVINO's median 4-5x too high. The
  numbers above use 30 warmup calls and a steady-state check.
- At the plugin default precision, both OpenVINO paths differed from eager by
  ~1.5e-2, far outside the FP32 tolerance; `--inference-precision f32` dropped
  that to ~9e-7 (`torch.compile`) and ~2e-7 (IR), confirming reduced precision
  as the cause. See the Intel note above for why this is AMX-gated.
- `--device NPU` on a host without an NPU reproduced the silent compile-time
  fallback: identical accuracy and eager-level latency, caught only by
  `ov_compiled_models == 0`.
- The deployment criterion holds: the saved IR loaded through
  `Core().compile_model()` and inferred correctly in a fresh process with
  `torch` never imported, which no LM7 artifact path currently offers.

Remaining work:

- Run `llama32-1b` and `qwen35-0.8b`, the two causal-LM ids still uncovered. The
  Intel host used here is a WSL2 VM capped at 15 GB of RAM, which is tight for a
  1B-class FP32 model held resident during IR conversion; raise the cap before
  attempting them.
- Measure Intel GPU and NPU. Neither exists on this host —
  `Core().available_devices` reports `['CPU']` only — so the `--device GPU` and
  `--device NPU` criteria remain untested on real hardware.
- Decide whether a decode loop with a KV cache is in scope. The current harness
  measures a single `use_cache=False` prefill, and a stateful decode path is a
  materially different OpenVINO integration.
- Decide whether the IR path graduates into a registered backend. The
  performance case is now made on Intel; what is not yet established is
  operator coverage across a wider model set and the artifact-lifecycle work in
  `bundles.py`.

## References

- [OpenVINO running inference](https://docs.openvino.ai/2025/openvino-workflow/running-inference.html)
- [PyTorch deployment with torch.compile](https://docs.openvino.ai/2025/openvino-workflow/torch-compile.html)
- [OpenVINO model preparation](https://docs.openvino.ai/2025/openvino-workflow/model-preparation.html)
- [Convert to OpenVINO IR](https://docs.openvino.ai/2025/openvino-workflow/model-preparation/convert-model-to-ir.html)
- [Precision control on CPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html)

The fallback and precision behaviour above comes from the OpenVINO sources
rather than the documentation:
`src/bindings/python/src/openvino/frontend/pytorch/torchdynamo/{backend,execute,compile,backend_utils}.py`
in [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino).
