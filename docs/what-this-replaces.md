# What LM7 replaces

LM7's whole value is that `lm7.compile(model, target=...)` is one line. This
document is the code you would otherwise write, and the behaviour you would
otherwise have to know about, so the trade is concrete rather than asserted.

Everything below is what LM7 already does internally — the per-vendor branches
live in `detection.py`, `planner.py`, and `backends/`.

## Detection is not one check

There is no single "which accelerator do I have" call in PyTorch. Each vendor has
its own probe, and two of them are easy to get wrong:

```python
torch.cuda.is_available()  # True for NVIDIA *and* AMD ROCm builds
torch.version.hip  # non-None is what actually distinguishes ROCm
torch.xpu.is_available()  # Intel; torch.xpu may not exist on older builds
torch.backends.mps.is_available()  # Apple
```

TPU is the worst case: an importable `torch_xla` proves nothing, because the PJRT
runtime may be pointed at CPU or GPU. You have to ask the runtime what device
type it actually has before believing there is a TPU. LM7 does this in
`_detect_tpu_targets()`.

## Device strings are not uniform

| Vendor | `torch.device` | Note |
| --- | --- | --- |
| NVIDIA | `cuda:N` | |
| AMD ROCm | `cuda:N` | same string as NVIDIA |
| Intel | `xpu:N` | |
| Apple | `mps` | **no ordinal** |
| TPU | `xla:N` | |
| CPU | `cpu` | |

AMD reusing `cuda` and Apple rejecting an ordinal are both things you find out by
hitting them. LM7 centralises the mapping in `torch_device()`.

## The compiler call differs per target

```python
torch.compile(model)  # CPU, NVIDIA, AMD, Intel, Apple - TorchInductor
torch.compile(model, backend="tensorrt")  # NVIDIA, after `import torch_tensorrt`
torch.compile(model, backend="openxla")  # TPU, after `import torch_xla`
torch._inductor.aoti_compile_and_package(...)  # persistent AOT package
```

Each also has its own import that must not be attempted when the runtime is
absent, or detection itself starts failing on machines that were fine before.
LM7's optional backends import their heavy dependency inside `probe()`/
`supports()` for exactly this reason.

## The whole ladder

```python
if torch.cuda.is_available():
    # True for NVIDIA *and* AMD ROCm; torch.version.hip is what tells them apart
    device = torch.device("cuda", 0)
    compiled = torch.compile(model.to(device))
    # ...unless you want TensorRT, which is a different import and backend:
    #   import torch_tensorrt; torch.compile(model, backend="tensorrt")
elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
    device = torch.device("xpu", 0)
    compiled = torch.compile(model.to(device))
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # note: no ordinal, unlike the others
    compiled = torch.compile(model.to(device))
elif tpu_runtime_is_really_a_tpu():  # importable torch_xla is not enough
    import torch_xla

    device = torch_xla.device(0)
    compiled = torch.compile(model.to(device), backend="openxla")
else:
    compiled = model  # eager fallback

inputs = move_every_tensor(inputs, device)  # yours to write, including nesting
```

## The branch is the easy part

The behaviour you inherit is harder than the dispatch:

- **`torch.compile` is lazy.** It returns a wrapper immediately and compiles on
  the first call, so a compilation failure surfaces from the *call*, not from
  `torch.compile`. A fallback therefore has to wrap the first call. Every LM7
  backend runs a warmup call inside its own error boundary for this reason.
- **TPU needs `torch.no_grad()`, not `torch.inference_mode()`.** PyTorch/XLA
  tracing depends on tensor version counters, which `inference_mode` removes.
  Using the faster-sounding context manager breaks XLA specifically, and nothing
  warns you.
- **Every new input shape recompiles.** Caching compiled variants per input
  signature is on you, as is deciding what counts as the same signature.
- **Moving inputs means walking nested structures.** Args, kwargs, tuples, and
  dicts all have to be traversed — a causal LM takes `input_ids` and
  `attention_mask` as kwargs, not a single positional tensor.
- **Vendor toolchains have their own semantics.** The
  [OpenVINO evaluation](openvino-evaluation.md) found silent fallback to
  Inductor on any
  compile exception, a CPU plugin that does not default to FP32, and a warmup
  profile long enough to invert a latency ranking. Each additional toolchain
  brings its own version of that list.

## What you write instead

```python
compiled = lm7.compile(model, target="auto")  # or "cpu", "nvidia", "apple", "tpu"
```

LM7 resolves the target, picks a compatible backend by priority, moves the
inputs, compiles once per input signature, caches the result, and falls back to
eager with a warning if a backend cannot handle the model.

The honest limit: LM7's reach is bounded by the vendor toolchains it has wired
up. Adding hardware means integrating that vendor's compiler — not writing one —
which is what the evaluation plans linked from the README work through.
