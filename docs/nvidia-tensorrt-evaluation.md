# TensorRT evaluation on NVIDIA Ada

This evaluation answers a deliberately narrow question: should LM7 prefer
TensorRT over TorchInductor on the locally available NVIDIA GPU?

The answer is **no as a general default**. TensorRT won the steady-state causal
LM workload, but lost the small MLP workloads and had the largest first-call
cost. It remains an explicit, experimental backend that users should benchmark
for their fixed model and input shape.

## Host and software

- NVIDIA GeForce RTX 4070 SUPER, Ada `sm89`, 12 GiB
- Intel Core i7-8086K, WSL2 Linux
- NVIDIA driver 595.71
- Python 3.12.3
- PyTorch 2.12.1+cu130
- Torch-TensorRT 2.12.1
- TensorRT 10.16.1.11

The TensorRT dependencies were installed in the isolated `.venv-trt`
environment documented in [development.md](development.md#tensorrt). The real
backend and CUDA integration tests passed without LM7 fallback:

```text
6 passed in 34.86s
```

## Method

All rows use LM7's `benchmarks/gpu.py` harness with FP16, fixed shapes,
`fallback="error"`, CUDA synchronization around every timed call, 10 warmups
and 50 measured calls for the MLPs, and 5 warmups and 20 measured calls for
SmolLM2. Lower latency is better.

```bash
.venv-trt/bin/python benchmarks/gpu.py \
  --model smollm2 \
  --backend eager inductor tensorrt \
  --target nvidia \
  --dtype float16 \
  --batch-size 1 \
  --warmup 5 \
  --repeats 20
```

The SmolLM2 TensorRT output was also compared against eager CUDA through the
opt-in integration test. Last-token logits had 0.99994 cosine similarity and
0.140625 p99 absolute error; top-1 and the top-5 token set agreed. Full-logit
elementwise parity was not claimed: the maximum absolute difference was
0.3203125 in FP16.

## Results

| workload | backend | first call | median | p95 | throughput | peak memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MLP, batch 1 | eager | 1.62 s | **0.330 ms** | **0.873 ms** | **3,032/s** | 24 MiB |
| MLP, batch 1 | Inductor | 4.30 s | 0.462 ms | 1.043 ms | 2,164/s | 72 MiB |
| MLP, batch 1 | TensorRT | 26.49 s | 0.676 ms | 1.347 ms | 1,479/s | 24 MiB |
| MLP, batch 8 | eager | 1.52 s | **0.399 ms** | **0.714 ms** | **20,038/s** | 24 MiB |
| MLP, batch 8 | Inductor | 4.26 s | 0.829 ms | 1.523 ms | 9,653/s | 72 MiB |
| MLP, batch 8 | TensorRT | 27.22 s | 1.600 ms | 2.769 ms | 5,000/s | 24 MiB |
| SmolLM2-135M, batch 1 | eager | 1.08 s | 52.129 ms | 73.225 ms | 19.18/s | 265 MiB |
| SmolLM2-135M, batch 1 | Inductor | 42.73 s | 29.774 ms | 41.062 ms | 33.59/s | **313 MiB** |
| SmolLM2-135M, batch 1 | TensorRT | 56.38 s | **16.942 ms** | **33.974 ms** | **59.02/s** | 467 MiB |

On SmolLM2, TensorRT was 1.76x faster than Inductor and 3.08x faster than eager
at steady state. That win cost an additional 13.65 seconds of first-call time
and 154 MiB of peak allocated GPU memory relative to Inductor. At the measured
median latency, approximately 1,064 calls are needed to recover TensorRT's
extra first-call cost relative to Inductor.

The MLP result points in the opposite direction: eager won, and TensorRT was
2.0x to 4.0x slower depending on batch size. A backend-wide priority change
would therefore improve one measured workload while regressing two others.

## Decision

Keep the current ranking:

1. Inductor, priority 100
2. TensorRT, priority 90 and explicit selection only
3. eager fallback

Use TensorRT when the model, precision, shapes, and expected request count are
stable enough to benchmark and amortize engine construction. Inductor remains
the safer automatic choice because it has broader PyTorch operator coverage and
the local evidence does not establish a universal TensorRT advantage.

## Serializing the engine

The costly half of TensorRT is the engine build, and the JIT path pays it in
every process. `lm7.export(..., backend="tensorrt")` writes the engine into the
`.lm7` artifact instead, through `torch_tensorrt.save`, so a second process
loads it rather than rebuilding it:

```bash
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct out.lm7 \
  --target nvidia --backend tensorrt --dtype float16
```

Measured on the same Ada (`sm89`) host as the table above, SmolLM2-135M at
5 tokens, FP16:

| step | time |
| --- | ---: |
| `lm7 model export --backend tensorrt` (build + save) | 49.0 s |
| `lm7.load_artifact()` + first call, fresh process | **10.6 s** |
| ...of which the engine itself | 6.1 s |
| ...of which SHA-256 of the 345 MB payload | 0.7 s |

The remainder is reading the `exported_program.pt2` that LM7 keeps beside every
artifact. Time-to-first-inference drops 4.6x; against the 56.38 s first call in
the JIT table above, the saving is the whole build.

The artifact is the least portable payload LM7 writes, and the manifest says so
in `runtime_requirements`: `device_bound: true`, the compute capability, the
GPU name, and the TensorRT and Torch-TensorRT versions. An engine built for
`sm89` will not load on `sm90`, and a TensorRT upgrade invalidates it. Loading
one whose provenance does not match fails with an actionable error rather than
silently rebuilding.

Two constraints on this path:

- **Static shapes only.** `dynamic_shapes=` and `shape_profile=` are rejected.
  A dynamically shaped engine needs min/opt/max `Input` specs rather than
  example tensors, and choosing those well is a separate evaluation.
- **No `enabled_precisions`.** Torch-TensorRT 2.12 turns explicit typing on by
  default and then rejects that option; precision comes from the exported
  graph's own dtypes, so export the model in the dtype you want. LM7 turns the
  resulting assertion into an error that says this.

## Limits

- These are single-host descriptive measurements, not CI thresholds.
- The serialization numbers above are one run each, not a distribution.
- Only FP16 and fixed input shapes were measured.
- SmolLM2 is a prefill forward pass with `use_cache=False`, not token-by-token
  generation with a KV cache.
- LFM2.5, Llama 3.2 1B, and Qwen3.5 were not validated through TensorRT.
- Successful Torch-TensorRT compilation may partition unsupported operations
  back to PyTorch; this evaluation establishes end-to-end behavior, not
  complete TensorRT operator coverage.
