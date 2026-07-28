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

Only add `backend="openvino"` after the evaluation shows a clear advantage for
Intel CPU, GPU, NPU, or IR-based deployment. Keep it lower priority than
Inductor until model coverage and artifact behavior are proven.

## Validation commands

Baseline the existing CPU paths through LM7:

```bash
python -m pip install -e ".[dev,hf]"
python benchmarks/local.py --target cpu --backend eager inductor
```

Then run the side-by-side evaluation. Without OpenVINO installed it still
reports eager and Inductor and marks the OpenVINO paths unavailable:

```bash
python -m pip install openvino
python benchmarks/openvino_eval.py \
  --path eager inductor openvino openvino_ir \
  --dtype float32 \
  --batch-size 8 \
  --inference-precision f32 \
  --output artifacts/benchmarks/openvino-mlp-fp32-b8.json
```

Use `--device GPU` or `--device NPU` only on a host whose runtime exposes them
(`lm7 targets` and `openvino.Core().available_devices` both report what is
present), `--inference-precision default` to measure the plugin's own default
instead of FP32, and `--require-all` to fail rather than skip when a requested
path is unavailable.

Record the OpenVINO, PyTorch, CPU, GPU/NPU, operating system, and driver
versions with the results; the JSON `environment` block captures most of this
automatically.

## Status

First measurement recorded on an Apple M4 Pro (`openvino` 2026.2.1, PyTorch
2.13, `mlp`, FP32, batch 8) — an ARM host, so it is a smoke test of the harness
rather than evidence about Intel deployment:

- At the plugin default precision, both OpenVINO paths differed from eager by
  ~1.5e-2, far outside the FP32 tolerance. With
  `--inference-precision f32` the difference dropped to ~9e-7
  (`torch.compile`) and ~2e-7 (IR), confirming reduced precision as the cause.
- At equal precision OpenVINO was slower than both eager and Inductor on this
  workload (~2.7 ms vs ~1.1 ms median), and `openvino_ir` avoided Inductor's
  ~1 s first-call compile.
- `--device NPU` on a host without an NPU reproduced the silent compile-time
  fallback: identical accuracy and Inductor-level latency, caught only by
  `ov_compiled_models == 0`.
- The deployment criterion holds: the saved IR loaded through
  `Core().compile_model()` and inferred correctly in a fresh process with
  `torch` never imported, which no LM7 artifact path currently offers.

The remaining work is the coverage in the acceptance criteria — TorchVision and
causal-LM shapes — measured on an Intel CPU, GPU, and NPU host.

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
