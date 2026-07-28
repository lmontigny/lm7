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

Only add `backend="openvino"` after the evaluation shows a clear advantage for
Intel CPU, GPU, NPU, or IR-based deployment. Keep it lower priority than
Inductor until model coverage and artifact behavior are proven.

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

Measured on an Apple M4 Pro, `openvino` 2026.2.1, PyTorch 2.13, FP32,
`--inference-precision f32`, 30 warmup calls, all paths within the FP32
tolerance and all reported steady-state. This is an ARM host, so it says nothing
about Intel deployment speed; it establishes the harness and the shape of the
result.

Median latency in ms, lower is better:

| workload | eager | inductor | openvino | openvino_ir |
| --- | --- | --- | --- | --- |
| `mlp`, batch 4 | 0.86 | 1.02 | 2.44 | 1.16 |
| `resnet18`, batch 4 | 17.8 | 18.8 | 17.1 | **13.8** |
| `smollm2`, batch 1, 5 tokens | 17.4 | 17.3 | 29.8 | **7.8** |

- **The IR path is the promising one, not `torch.compile`.** `openvino_ir` was
  1.3x faster than eager on `resnet18` and 2.2x faster on SmolLM2 prefill, while
  the dynamo backend was at best level with eager and clearly worse on the two
  larger models. If OpenVINO earns a place in LM7, the evidence points at the
  artifact path, which is also the one that fits `exporting.py` and `bundles.py`.
- **OpenVINO loses on trivial graphs.** On `mlp` it is slower than eager because
  the per-call NumPy round-trip and infer-request overhead dominate three
  layers of work. Model size, not just vendor, decides whether it helps.
- **An earlier version of this document reported OpenVINO as uniformly slower.
  That was a measurement error**, not a property of OpenVINO: the 5-call warmup
  in the first version of the harness left OpenVINO's median 4-5x too high. The
  numbers above use 30 warmup calls and a steady-state check.
- Time to first inference (build plus first call) still favours the JIT paths for
  large models: `openvino_ir` needs ~7.1 s for SmolLM2 (1.3 s export, 4.1 s
  convert, 1.3 s save, 3.0 s compile) and writes a 540 MB FP32 IR, against
  ~2.5 s for Inductor. That cost is paid once per artifact rather than per
  process, which is the point of the IR path.
- At the plugin default precision, both OpenVINO paths differed from eager by
  ~1.5e-2, far outside the FP32 tolerance; `--inference-precision f32` dropped
  that to ~9e-7 (`torch.compile`) and ~2e-7 (IR), confirming reduced precision
  as the cause.
- `--device NPU` on a host without an NPU reproduced the silent compile-time
  fallback: identical accuracy and eager-level latency, caught only by
  `ov_compiled_models == 0`.
- The deployment criterion holds: the saved IR loaded through
  `Core().compile_model()` and inferred correctly in a fresh process with
  `torch` never imported, which no LM7 artifact path currently offers.

Remaining work:

- Run the same matrix on an Intel CPU host, which is the only place the latency
  numbers mean anything for the decision, then on Intel GPU and NPU.
- Extend coverage to `resnet50`, `mobilenet_v3_small`, and the remaining
  causal-LM ids, which the script already accepts but which have not been run.
- Decide whether a decode loop with a KV cache is in scope. The current harness
  measures a single `use_cache=False` prefill, and a stateful decode path is a
  materially different OpenVINO integration.

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
