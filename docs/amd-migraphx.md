# AMD MIGraphX evaluation plan

LM7's current AMD path uses a ROCm-enabled PyTorch build with TorchInductor.
MIGraphX should be evaluated as an optional AMD backend before it is added to
automatic planning.

## Candidate integration paths

- Torch-MIGraphX: preferred first path because it keeps LM7 close to the
  existing PyTorch module workflow.
- ONNX Runtime with the MIGraphX execution provider: useful if a model already
  exports cleanly to ONNX or if runtime packaging is the priority.
- MIGraphX driver: useful for offline validation, operator inspection, and
  performance tests against generated ONNX artifacts.

## Acceptance criteria

Evaluate MIGraphX against the existing AMD Inductor path on the same host,
ROCm version, GPU architecture, dtype, batch size, and prompt shape.

- Correctness: compiled logits must match eager ROCm within the same tolerance
  policy used by AMD integration tests.
- Coverage: start with `mlp`, SmolLM2, LFM2.5, Llama 3.2 1B, and Qwen3.5 0.8B.
- Latency: report first-call compile time, median latency, p95 latency, and
  throughput.
- Memory: report peak allocated GPU memory when the runtime exposes it.
- Packaging: document whether the compiled result can be serialized and loaded
  in a fresh process.
- Failure behavior: backend import, unsupported op, and compile failures must
  return actionable LM7 errors and preserve eager fallback semantics.

## First implementation slice

Do not add a registered backend until the local evaluation shows a clear use
case. `benchmarks/migraphx.py` is that first slice: it runs a model through
eager (the correctness reference), TorchInductor, and a manually installed
Torch-MIGraphX `torch.compile` path under one measurement harness, reporting
first-call cost, median and p95 latency, throughput, peak GPU memory, and the
maximum absolute difference from eager. It does not register an LM7 backend. A
later backend PR can wrap the winning path behind `backend="migraphx"` with
lower automatic priority than `inductor` until model coverage is proven.

## Validation commands

Baseline the existing eager and Inductor AMD paths through LM7:

```bash
python -m pip install -e ".[dev,hf]"
python benchmarks/gpu.py --target amd --model mlp --backend eager inductor
python benchmarks/gpu.py --target amd --model smollm2 --backend eager inductor
```

Then run the side-by-side evaluation. Without Torch-MIGraphX installed it still
reports the eager and Inductor paths and marks `migraphx` unavailable:

```bash
python benchmarks/migraphx.py \
  --path eager inductor migraphx \
  --dtype float16 \
  --batch-size 8 \
  --output artifacts/benchmarks/migraphx-mlp-fp16-b8.json
```

After installing Torch-MIGraphX in a matching ROCm environment, record the
exact ROCm, PyTorch, Torch-MIGraphX, GPU, and driver versions beside the
results. The script currently covers the `mlp` workload; extending it to the
SmolLM2, LFM2.5, Llama 3.2 1B, and Qwen3.5 0.8B causal-LM shapes from the
acceptance criteria is the natural next step.

## References

- [AMD MIGraphX documentation](https://rocmdocs.amd.com/projects/AMDMIGraphX/en/latest/index.html)
- [Install MIGraphX for Radeon GPUs](https://rocmdocs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-migraphx.html)
