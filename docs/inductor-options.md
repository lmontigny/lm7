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

## Individual Inductor options

Instead of a preset, pass individual backend options. LM7 removes its own
`dynamic` and `fullgraph` keys and forwards every remaining key through
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
