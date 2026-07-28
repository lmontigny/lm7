# OpenVINO evaluation plan

LM7's generic CPU path already works through PyTorch eager and TorchInductor.
OpenVINO should be evaluated as an optional Intel deployment backend before it
is considered for automatic planning.

## Candidate integration paths

- `torch.compile` backend: import `openvino.torch` and compile PyTorch modules
  with `backend="openvino"` or `backend="openvino_ts"`. This best matches
  LM7's lazy backend protocol.
- OpenVINO IR artifacts: convert a PyTorch module or `ExportedProgram` with
  `openvino.convert_model()`, save IR, and load it through OpenVINO Runtime.
  This best matches LM7's artifact and deployment goals.

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

## First implementation slice

Add an evaluation script before adding a registered backend. The script should
run the same model through eager, Inductor, OpenVINO `torch.compile`, and
OpenVINO IR when available, then write JSON with environment metadata, compile
time, median latency, p95 latency, and output error statistics.

Only add `backend="openvino"` after the evaluation shows a clear advantage for
Intel CPU, GPU, NPU, or IR-based deployment. Keep it lower priority than
Inductor until model coverage and artifact behavior are proven.

## Validation commands

```bash
python -m pip install -e ".[dev,hf]"
python benchmarks/local.py --target cpu --backend eager inductor
```

After installing OpenVINO in a separate environment, run equivalent OpenVINO
compile and IR-conversion measurements and record the OpenVINO, PyTorch, CPU,
GPU/NPU, operating system, and driver versions with the results.

## References

- [OpenVINO running inference](https://docs.openvino.ai/2024/openvino-workflow/running-inference.html)
- [OpenVINO PyTorch deployment with torch.compile](https://docs.openvino.ai/2023.3/pytorch_2_0_torch_compile.html)
- [OpenVINO model preparation](https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html)
- [Convert to OpenVINO IR](https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-to-ir.html)
