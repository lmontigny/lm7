# Validating the TensorRT path on Blackwell

[nvidia-tensorrt-evaluation.md](nvidia-tensorrt-evaluation.md) asked whether LM7
should prefer TensorRT over Inductor on one Ada GPU, for one model, at one shape.
This page asks a wider question on `sm120`: across four model families, four
precisions, several batch sizes, dynamic sequence length, and artifact reload —
**does it work, where does it win, and where does it fail?**

The short version:

- **It works.** 106 cells and 32 artifact reloads on Blackwell, zero failures,
  no code changes.
- **It wins where it engages**, by 1.4x to 4.4x over Inductor, and most at small
  batch. The serialized artifact is the product; the JIT path is not.
- **It fails quietly, four different ways.** Every one of them reports success.
  That is the finding worth your attention, and the rest of this page is mostly
  about it.

## Host and software

- NVIDIA RTX PRO 6000 Blackwell Server Edition, `sm120`, 96 GiB, driver 580.126.20
- 16 vCPU, 124 GB RAM, Lightning Studio
- Python 3.12.3, PyTorch 2.12.1+cu130, Torch-TensorRT 2.12.1, TensorRT 10.16.1.11
- transformers 5.14.1, torchvision 0.27.1+cu130, torchao 0.18.0

An Ada `sm89` control (RTX 4070 SUPER, same software) ran the precision grid and
the FP8 probes. Every defect below reproduces on both, so none of them is about
Blackwell.

All rows come from [`benchmarks/tensorrt_matrix.py`](../benchmarks/tensorrt_matrix.py),
one cell per process, 5 warmups and 20 measured calls, CUDA-synchronized, inputs
already on the device. Batch 8 and sequence 32 unless stated.

```bash
python benchmarks/tensorrt_matrix.py --plan core | while read -r args; do
  python benchmarks/tensorrt_matrix.py $args --results-dir artifacts/trt --skip-existing
done
python benchmarks/tensorrt_matrix.py --summarize artifacts/trt
```

## The measurement that changed the conclusion

Every cell records `tensorrt_engaged`, and it exists because the obvious success
criteria are all satisfiable without TensorRT doing anything.

The partitioner declines any subgraph below `min_block_size` (default **5**) and
returns the graph **unconverted, without raising**. The result compiles, runs,
matches eager exactly, is faster than eager, serializes, reloads, and contains no
TensorRT. The harness therefore counts engine builds by wrapping
`TRTInterpreter.run`, and counts engines and leftover PyTorch operators in the
graph it is about to call.

Read `engines` as the claim and everything else as the evidence for it.

## Precision grid

Median latency, batch 8, sequence 32. `engines` is what the artifact actually
contains.

| model | dtype | eager | inductor | TRT JIT | TRT export | JIT engines | export engines |
| --- | --- | ---: | ---: | ---: | ---: | :--: | :--: |
| mlp | float32 | 0.072 | 0.102 | 0.146 | 0.098 | 1 | **0** |
| mlp | float16 | 0.054 | 0.096 | 0.143 | 0.099 | 1 | **0** |
| mlp | bfloat16 | 0.062 | 0.092 | 0.151 | 0.086 | 1 | **0** |
| resnet18 | float32 | 1.167 | 1.106 | 1.487 | **1.010** | 1 | 1 |
| resnet18 | float16 | 1.252 | 0.802 | 0.844 | **0.373** | 1 | 1 |
| resnet18 | bfloat16 | 1.243 | 0.822 | 0.893 | **0.475** | 1 | 1 |
| bert | float32 | 3.261 | 3.176 | 2.087 | **1.126** | 1 | 1 |
| bert | float16 | 2.761 | 1.737 | 1.747 | **0.794** | 1 | 1 |
| bert | bfloat16 | 2.622 | 1.740 | 1.716 | **0.789** | 1 | 1 |
| smollm2 | float32 | 15.325 | 6.427 | 3.754 | **2.197** | 1 | 1 |
| smollm2 | float16 | 11.729 | 5.636 | 2.965 | **1.555** | 1 | 1 |
| smollm2 | bfloat16 | 10.944 | 5.529 | 2.971 | **1.526** | 1 | 1 |

FP32, FP16 and BF16 all work, on all four model families. No precision needed
special handling, and none of them failed.

Build cost, for the same rows:

| model | inductor | TRT JIT | TRT export | artifact |
| --- | ---: | ---: | ---: | ---: |
| mlp fp16 | 1.8 s | 3.2 s | 0.3 s | 34 MB |
| resnet18 fp16 | 4.5 s | 52.4 s | 13.5 s | 61 MB |
| bert fp16 | 5.6 s | 20.1 s | 27.5 s | 513 MB |
| smollm2 fp16 | 15.1 s | 27.3 s | 34.8 s | 711 MB |

## Where it wins

The serialized engine, at small batch, on real models. Speedup is
`tensorrt-export` against `inductor`, FP16:

| model | batch 1 | batch 8 | batch 32 |
| --- | ---: | ---: | ---: |
| bert | **4.37x** | 2.19x | 1.36x |
| smollm2 | **4.32x** | 3.62x | 2.49x |
| resnet18 | **3.09x** | 2.15x | 1.37x |
| mlp | 1.13x | 0.98x | 1.09x |

The trend is the useful part: **the advantage is largest at batch 1 and decays
monotonically**. TensorRT's win here is mostly the elimination of per-launch
overhead, and larger batches amortize that overhead for everyone. A serving
workload at batch 1 is close to the best case; a throughput workload at batch 32
gets a third of the benefit.

Reload is cheap, which is what makes the artifact worth building:

| artifact | build | load | first call | time to first inference |
| --- | ---: | ---: | ---: | ---: |
| resnet18 fp16 | 13.5 s | 0.48 s | 3.2 ms | **0.48 s** |
| bert fp16 | 27.5 s | 2.49 s | 6.9 ms | **2.50 s** |
| smollm2 fp16 | 34.8 s | 4.95 s | 11.2 ms | **4.96 s** |
| smollm2 fp32 | 49.3 s | 7.90 s | 8.2 ms | **7.90 s** |

All 32 artifacts reloaded in a fresh process and ran. Reload is dominated by
reading the `exported_program.pt2` LM7 keeps beside the engine, not by the engine
— which is why FP32 costs roughly twice FP16.

## Where it fails

Four distinct failures. **None of them raises**, and three of them report
latency that looks like a success.

### 1. `lm7.export(backend="tensorrt")` can write an artifact with no TensorRT in it

The MLP produces this at every precision and every batch size:

| cell | build | engines in artifact | fallback ops | max abs diff |
| --- | ---: | :--: | ---: | ---: |
| `mlp fp16 tensorrt-export` | 0.3 s | **0** | 3 | 0.00e+00 |
| `mlp fp16 tensorrt-export`, `min_block_size=1` | 9.4 s | 1 | 1 | 3.91e-03 |

The first artifact's graph is `aten.linear`, `aten.gelu`, `aten.linear`. Its
manifest says `backend: tensorrt`, carries a `backend_version`, and records
`device_bound: true` with the compute capability — an artifact tied to this GPU
that contains nothing that needs one. The 0.3 s build and the exactly-zero
difference from eager are the only signals, and both are easy to read as good
news.

The mechanism is `torch.export`'s less-decomposed graph: three ATen nodes, below
the partitioner's floor of five. The JIT path clears the floor on the same model
because AOTAutograd decomposes further, which is why one path converts and the
other does not. Confirmed directly:

```python
ep = torch.export.export(model, (x,))
torch_tensorrt.dynamo.compile(ep, arg_inputs=[x])                    # 0 engines
torch_tensorrt.dynamo.compile(ep, arg_inputs=[x], min_block_size=1)  # 1 engine
```

`options={"min_block_size": 1}` reaches the partitioner through
`lm7.export(..., options=...)` and fixes it. On models that already convert it
changes almost nothing — BERT, ResNet and SmolLM2 all land within noise of their
default builds — so it is close to a free correction.

### 2. The JIT path computes the wrong answer on BERT

Same model, same library, same device, same inputs:

| path | max abs diff vs eager | cosine |
| --- | ---: | ---: |
| `torch_tensorrt.dynamo.compile(exported_program)` | 0.0097 | 1.0000 |
| `torch.compile(model, backend="tensorrt")` | **8.69** | 0.855 |

It is deterministic across calls, it reproduces at FP32, FP16 and BF16, it
reproduces on Ada `sm89` and Blackwell `sm120`, and it survives
`min_block_size=1`. Every position of every row is wrong, so it is not precision
drift — the first hidden vector reads `[-0.246, 0.299, -0.132, ...]` in eager and
`[-0.787, 0.315, 0.187, ...]` through the JIT backend.

Minimal reproduction, batch 1, sequence 7, FP32, no LM7 involved:

```python
import torch, torch_tensorrt, transformers

bert = transformers.AutoModel.from_pretrained("bert-base-uncased").eval().cuda()
ids = torch.tensor([[101, 1996, 3007, 1997, 2605, 2003, 102]], device="cuda")
mask = torch.ones_like(ids)

with torch.no_grad():
    reference = bert(input_ids=ids, attention_mask=mask).last_hidden_state
    actual = torch.compile(bert, backend="tensorrt")(
        input_ids=ids, attention_mask=mask
    ).last_hidden_state
print((actual - reference).abs().max())   # ~8.7
```

This is a Torch-TensorRT defect rather than an LM7 one. LM7's exposure is that
`lm7.compile(backend="tensorrt")` hands it to a caller with no indication, and
the greedy-token check that guards the causal-LM paths would not catch it —
BERT has no next token to agree on.

### 3. Dynamic sequence length: the artifact ignores the request

`lm7.export(backend="tensorrt")` rejects `dynamic_shapes=` and `shape_profile=`
outright, which is honest. It does **not** reject `options={"dynamic": True}`,
which flows to `dynamo.compile` and does nothing. The artifact is shape-locked
either way, and identical to the static control:

| built with | seq 32 | seq 16 | seq 64 | seq 128 |
| --- | :--: | :--: | :--: | :--: |
| `tensorrt-export`, static | 0.814 ms | `Guard failed` | `Guard failed` | `Guard failed` |
| `tensorrt-export`, `dynamic=True` | 0.831 ms | `Guard failed` | `Guard failed` | `Guard failed` |

`AssertionError: Guard failed: args_0.size()[1] == 32`. A clear error at call
time, and no way to get a dynamic engine out of the export path.

The JIT path does support it, and the contrast between its two modes matters
more than either number:

| BERT fp16, JIT | seq 32 | seq 16 | seq 64 | seq 128 |
| --- | ---: | ---: | ---: | ---: |
| static, first call at each length | 2.0 ms | **17,683 ms** | **14,965 ms** | **13,958 ms** |
| static, steady | 1.758 | 1.784 | 1.925 | 2.369 |
| `dynamic=True`, first call | 5.7 ms | 20.2 ms | 15.6 ms | 16.4 ms |
| `dynamic=True`, steady | 4.915 | 4.871 | 5.044 | 5.599 |

A statically compiled TensorRT module handed a new sequence length **silently
rebuilds an entire engine** — one recompile and one new engine per shape, a
14-to-18 second stall inside an ordinary inference call. A server with variable
prompt lengths would stall on every new one until it had seen them all.

`dynamic=True` fixes that: four lengths, zero recompiles, zero new engines. It
costs 2.8x steady-state latency (4.9 ms against 1.76 ms), which is a real price
but a knowable one. Inductor with `dynamic=True` absorbs the same four lengths at
1.85–1.94 ms with no rebuild, within 7% of its own static build.

### 4. FP8 is not reachable through this stack

TensorRT 10.16 has `FP8` and `FP4` data types. Torch-TensorRT 2.12.1 registers
230 converters and **none of them is `aten._scaled_mm`**, the operator PyTorch
issues for an FP8 matmul. A four-layer `torch._scaled_mm` stack — deliberately
big enough to clear `min_block_size` — converts to **zero engines** on both
cards, at FP16 and BF16, through both paths, while reporting success.

The library says so at import, and it is easy to miss:

```text
Unable to import quantization op. Please install modelopt library ... to add
support for compiling quantized models
```

The supported FP8 route is Q/DQ nodes inserted by `nvidia-modelopt`, which is not
installed and which LM7 does not use — LM7's `fp8` is torchao. Through torchao,
on SmolLM2 with 90 quantized layers, BF16:

| mode | path | engines | fallback ops | median |
| --- | --- | ---: | ---: | ---: |
| none | `tensorrt-export` | 1 | 1 | **1.526 ms** |
| `fp8` | `tensorrt-export` | 1 | 1 | 1.532 ms |
| `fp8` | `tensorrt` JIT | **0** | — | 20.569 ms |
| `fp8-dynamic` | `tensorrt-export` | **91** | 752 | 20.768 ms |
| `fp8-dynamic` | `tensorrt` JIT | **0** | — | 42.885 ms |

Weight-only `fp8` exports to a single engine and matches the unquantized latency,
because the weight is dequantized before the matmul and TensorRT never sees an
FP8 operation. `fp8-dynamic` quantizes activations, and the graph **shatters into
91 engines around 752 PyTorch operators** — 13.6x slower than the unquantized
artifact. Both JIT rows produce no engine at all.

So: FP8 storage and FP8 arithmetic are different questions, and this stack
answers the second one no. That is consistent with
[nvidia-blackwell.md](nvidia-blackwell.md#native-is-not-the-same-as-used), where
`fp4: native` also turned out to mean "the silicon could, if something asked it
to".

## What to do with this

- **Prefer `lm7.export(backend="tensorrt")` to `lm7.compile(backend="tensorrt")`.**
  The artifact is 1.5x to 2.3x faster than the same engine reached through JIT on
  every model and precision measured here, and the JIT path is the one with the
  BERT defect and the hidden per-shape rebuild.
- **Pass `options={"min_block_size": 1}` when exporting**, unless you have
  checked that your model converts without it. It costs little on models that
  already convert and is the difference between an engine and a mislabelled
  PyTorch graph on models that do not.
- **Check that an engine exists** rather than that the export succeeded. The
  graph of `lm7.load_artifact(path).module()` should contain `_run_on_acc` or
  `execute_engine` nodes.
- **Export one artifact per sequence length**, or use `lm7.compile` with
  `options={"dynamic": True}` and accept ~2.8x steady-state latency.
- **Do not expect FP8 arithmetic** from this pairing of Torch-TensorRT and
  TensorRT, on any NVIDIA generation.

Inductor stays the automatic default. No Inductor cell diverged from eager the way
the TensorRT JIT path did on BERT, it needs no engine-presence check, it handles
dynamic shapes for a 7% penalty rather than 2.8x, and it builds in a fraction of
the time. TensorRT is worth reaching for deliberately,
for a fixed model at a fixed shape and precision, once you have confirmed it is
actually running.

## Limits

- One card per generation, one run per cell. These are descriptive measurements,
  not CI thresholds, and nothing here is covered by CI, which remains CPU-only.
- Steady-state reload latencies vary run to run by more than the differences
  between some adjacent rows; the Ada control measured the same artifacts twice
  and disagreed with itself by up to 2.7x. Engine and fallback-operator counts
  are stable and are what the failure claims rest on. Treat single latency
  comparisons within ~20% as unresolved.
- `max_abs_diff` is against eager in the same dtype, so it measures the backend
  and not the precision. It does not establish accuracy on any downstream task.
- The causal-LM cells are prefill with `use_cache=False`, not token-by-token
  generation with a KV cache.
- `aot_inductor` is implemented in the harness but was not run here; the
  artifact-to-artifact comparison against AOTInductor is the obvious extension.
- The FP8 conclusion is specific to Torch-TensorRT 2.12.1 without
  `nvidia-modelopt`. Installing modelopt was out of scope and would plausibly
  change it.
