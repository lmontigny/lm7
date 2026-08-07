# TorchInductor compile options

LM7 forwards TorchInductor controls through the `options` argument to
`lm7.compile` and `lm7.benchmark`. LM7's top-level `mode` has a different
meaning: it selects LM7's `"lazy"` or `"eager"` lifecycle. Put the PyTorch
compile mode under `options["compile_mode"]`.

```python
compiled = lm7.compile(
    model,
    target="nvidia",
    backend="inductor",
    options={"compile_mode": "max-autotune"},
)
```

Compilation still happens on the first real call. A more aggressive preset can
make that call much slower, so always compare first-call cost, steady latency,
and peak memory for the shapes used in production.

## Mode presets

The public `torch.compile` API documents four modes:

| `compile_mode` | Main effect | CUDA Graphs on GPU | When to try it |
| --- | --- | --- | --- |
| `default` | Balances compile time and runtime | Not enabled by this preset | Start here. |
| `reduce-overhead` | Reduces Python launch overhead | Enabled | Small batches and repeated fixed-shape calls. |
| `max-autotune` | Benchmarks more Triton/template matmul and convolution choices | Enabled | Maximum fixed-shape throughput when a long first compile is acceptable. |
| `max-autotune-no-cudagraphs` | Performs the same aggressive autotuning | Disabled | Changing shapes/state, CUDA Graph incompatibility, or excess graph memory. |

`max-autotune-no-cudagraphs` still uses TorchDynamo/FX graph capture and
TorchInductor code generation. It disables only CUDA Graph capture and replay.

PyTorch 2.13's internal `torch._inductor.list_mode_options()` also reports a
`lite` preset. It is not part of the documented public `torch.compile` mode
contract, so LM7 does not recommend depending on it.

## CUDA Graph tradeoffs

CUDA Graphs replay a recorded sequence of GPU work with much lower CPU launch
overhead. They work best when tensor addresses and shapes remain stable. They
can be skipped or become counterproductive when a model mutates inputs, uses
dynamic shapes, performs CPU synchronization, or needs the extra cached
workspace memory for something else.

Use the CUDA Graph preset first for a fixed-shape inference loop:

```python
options = {"compile_mode": "max-autotune"}
```

Keep autotuning but remove CUDA Graphs when diagnosing graph breaks, unexpected
memory use, or stateful/dynamic workloads:

```python
options = {"compile_mode": "max-autotune-no-cudagraphs"}
```

`TORCH_LOGS=perf_hints` explains why CUDA Graph capture was skipped.

### What LM7 reports

Two of the four preset names say nothing about CUDA Graphs and one of those two
enables them, so LM7 records the answer on the compiled artifact rather than
leaving it to be inferred:

```python
compiled = lm7.compile(
    model, target="nvidia", backend="inductor", options={"compile_mode": "reduce-overhead"}
)
compiled(example)  # compilation is lazy
compiled.artifact.metadata
# {'compiled': True, 'compile_mode': 'reduce-overhead',
#  'cudagraphs': True, 'cudagraph_skips': 0, 'cudagraphs_active': True}
```

`cudagraphs` is what the configuration asked for; `cudagraph_skips` is how many
times Inductor declined during that compile; `cudagraphs_active` is the
conjunction, which is the field to read. Both zero means the preset never asked.
`lm7.backends.inductor.cudagraphs_requested(mode, options)` answers the first
question without compiling anything.

### Measured behaviour

A two-layer BF16 MLP (width 2048), batch 64, on an RTX PRO 6000 Blackwell
(`sm120`), through [`benchmarks/cudagraphs.py`](../benchmarks/cudagraphs.py):

| preset | requests | captured | reuses graph | steady latency | steady peak memory |
| --- | --- | --- | --- | --- | --- |
| `default` | no | — | yes | 0.0694 ms | 68.2 MB |
| `reduce-overhead` | **yes** | **yes** | yes | 0.0843 ms | **33.8 MB** |
| `max-autotune` | **yes** | **yes** | yes | 0.0827 ms | **33.8 MB** |
| `max-autotune-no-cudagraphs` | no | — | yes | 0.0602 ms | 68.2 MB |

**CUDA Graphs halved steady-state memory** — 33.8 MB against 68.2 MB — because
replay reuses one pool of buffers instead of reallocating intermediates per
call. That is the clearest benefit in this measurement, and the opposite of the
"needs the extra cached workspace memory" caution above, which describes capture
rather than replay.

**CUDA Graphs were slower here**, by 21% against `default` and 37% against
`max-autotune-no-cudagraphs`, reproducibly across runs. Read that with its
caveat: the timing loop synchronizes after every call, which is precisely the
pattern that denies CUDA Graphs their advantage. What they remove is CPU launch
overhead, and a loop that blocks on the GPU after each iteration never lets that
overhead overlap with anything. A generation loop that queues many steps before
synchronizing is the shape where they pay; this benchmark is not that shape.

**A changed input shape recompiles once, not once per shape.** Feeding batch 16,
then 32, then 128 to a graph compiled at 64 produced exactly one new Dynamo
frame — on the first change — and none afterwards. PyTorch marks the dimension
dynamic after the first mismatch and serves every later size from that one
graph. Identical across all four presets, with no CUDA Graph skips on the
dynamic graph.

**Repeated calls do not grow memory.** Peak device memory is identical across
two consecutive post-compilation rounds for every preset.

> [!NOTE]
> Measuring that last point needs two rounds that are *both* past compilation. An
> earlier revision of the benchmark compared the first round against the second
> and reported stability, but the first round includes the compile workspace, so
> it passed for a reason unrelated to whether replay leaks.

### Stateful models break the cache, not the capture

The expectation is that a KV-cache-style loop prevents CUDA Graph capture. On a
module that writes into a persistent buffer and advances a Python counter each
call, that is not what happens:

| preset | CUDA Graph skips | Dynamo frames for 20 calls |
| --- | --- | --- |
| all four | **0** | **9** |

Capture is never refused. What breaks is graph reuse: `self.position += 1` is
Python state that Dynamo guards on, so a new value means a new graph, and 20
calls cost 9 compilations before the recompile limit stops it. The cost is real
and it is mode-independent — turning CUDA Graphs off does not help, because they
were never the problem.

The practical rule: keep step counters in tensors or pass them as arguments, and
check `cudagraph_skips` and the frame count separately. A model can be capturing
CUDA Graphs perfectly while recompiling on every call.

### A stateful model may not want the warmup call either

LM7's Inductor backend compiles by *calling* the compiled artifact once, so that
a compilation failure raises `CompilationError` inside the backend where
`fallback` can act on it. For a stateless forward that extra execution is
invisible; for a model that mutates something it is not.

`options={"warmup": False}` declines it:

```python
compiled = lm7.compile(
    model, target="nvidia", backend="inductor", options={"warmup": False}
)
```

The trade is explicit. `torch.compile` then stays lazy, so compilation happens on
the caller's own first call and a failure surfaces there rather than as a
`CompilationError` the planner can fall back from — and `cudagraph_skips` and
`cudagraphs_active` come back `None` in the artifact metadata, because nothing
has run yet and neither answer is known.

`lm7.compile_generation` is the caller this exists for: its decode graph writes
into a KV cache that advances once per execution, so an unasked-for warmup spends
a cache slot and, at a long enough prompt, indexes past the end of the buffer.
See [prefill and KV-cache decode](kv-cache-decode.md#compiling-a-graph-that-writes-into-a-cache).

## Individual Inductor options

Instead of a preset, pass individual backend options. LM7 removes its own
`dynamic`, `fullgraph` and `warmup` keys and forwards every remaining key through
`torch.compile(options=...)`.

```python
compiled = lm7.compile(
    model,
    target="nvidia",
    backend="inductor",
    options={
        "dynamic": False,
        "fullgraph": True,
        "max_autotune": True,
        "triton.cudagraphs": False,
        "shape_padding": True,
    },
)
```

Useful options exposed by current PyTorch releases include:

| Option | Purpose |
| --- | --- |
| `max_autotune` | Profile candidate matmul implementations. |
| `max_autotune_gemm` | Restrict aggressive autotuning to GEMM choices. |
| `max_autotune_pointwise` | Autotune eligible pointwise kernels. |
| `triton.cudagraphs` | Explicitly enable or disable CUDA Graphs. |
| `shape_padding` | Pad matrix shapes for better tensor-core alignment. |
| `epilogue_fusion` | Fuse pointwise epilogues; most useful with autotuning. |
| `trace.enabled` | Write Inductor debug traces. |
| `trace.graph_diagram` | Include a post-fusion graph diagram in a trace. |

Option names are owned by PyTorch and may change between releases. Inspect the
installed build rather than assuming every key exists:

```python
print(torch._inductor.list_mode_options())
print(torch._inductor.list_options())
```

Do not combine `compile_mode` with individual backend options: PyTorch treats a
mode as a complete option preset and rejects receiving both. `dynamic` and
`fullgraph` are top-level `torch.compile` controls, so LM7 permits them alongside
either a preset or individual options.

## Benchmark the choice

The GPU benchmark accepts every documented public preset:

```bash
python benchmarks/gpu.py --target nvidia --model smollm2 \
  --backend inductor --compile-mode max-autotune

python benchmarks/gpu.py --target nvidia --model smollm2 \
  --backend inductor --compile-mode max-autotune-no-cudagraphs
```

Run each configuration in a fresh process. Inductor caches compiled code, and a
warm cache can otherwise make the second configuration appear to compile faster.

A short fresh-process smoke check on this project's RTX 4070, PyTorch 2.13.0
with CUDA 13.0, used the FP16 MLP at batch size 8 with one warmup and three
measured calls:

| Preset | First call | Median | Peak allocated |
| --- | ---: | ---: | ---: |
| `max-autotune` | 7.96 s | 0.464 ms | 24.2 MiB |
| `max-autotune-no-cudagraphs` | 11.24 s | 0.810 ms | 72.3 MiB |

This is a path check, not a stable benchmark: three calls are too few for a
performance claim. PyTorch also printed `Not enough SMs to use
max_autotune_gemm mode` for both runs. The preset remained valid, but that
warning shows why selecting `max-autotune` does not guarantee every autotuner
activates on every GPU.

## References

- [PyTorch `torch.compile` API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch compile troubleshooting](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html)
