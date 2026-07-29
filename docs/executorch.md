# Edge and mobile support with ExecuTorch

LM7 exports models for phones and embedded devices through
[ExecuTorch](https://github.com/pytorch/executorch), PyTorch's on-device
runtime. `lm7.export(..., backend="executorch")` writes a `.pte` — the file
Android and iOS actually load — into the `.lm7` artifact.

```
nn.Module → torch.export → to_edge_transform_and_lower(XnnpackPartitioner) → .pte → phone
```

This is an **export-only** backend. `lm7.compile()` never selects it, because a
phone is not a device the calling process can reach; see [scope](#scope).

## Why this is the edge path LM7 could actually build

Every other mobile route needs hardware or a vendor SDK before you can tell
whether it works at all — the [Hexagon plan](qualcomm-hexagon.md) is blocked on
exactly that. ExecuTorch's XNNPACK delegate needs neither. It targets ARM64 *and*
x86-64, so the same lowering that produces a phone artifact runs on an ordinary
Linux CI box, and `tests/test_executorch_integration.py` validates numerics
against eager on every run.

That is also the honest limit of what is claimed here: XNNPACK is a **CPU**
delegate. Reaching the NPUs — Core ML, Qualcomm QNN, MediaTek, Samsung Exynos —
means adding those delegates, and each needs a macOS host or a vendor SDK.

## Install

ExecuTorch's prebuilt runtime extension is linked against a specific libtorch,
so it needs an environment built around the matching PyTorch rather than the
latest one. Installing it beside a CUDA PyTorch will fail at import with an
undefined `c10` symbol.

```bash
python3 -m venv .venv-et
.venv-et/bin/python -m pip install -e ".[dev,executorch]"
.venv-et/bin/python -m pip install "torch==2.12.*" --index-url https://download.pytorch.org/whl/cpu
source .venv-et/bin/activate
```

Activate the environment rather than calling `.venv-et/bin/python` directly.
ExecuTorch shells out to `flatc` during serialization and resolves it from
`PATH`; the wheel ships one in its `bin/` directory, which only lands on `PATH`
when the environment is active. LM7 works around this — `compile_exported` adds
the wheel's `bin/` for the duration of a lowering — but `lm7 backends` reports
the situation either way.

## Export

```bash
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct model.lm7 \
  --target cpu --backend executorch
```

```python
artifact = lm7.export(
    model.eval(),
    args=(example_input,),
    target="cpu",
    backend="executorch",
    output="model.lm7",
)
output = artifact(example_input)  # runs through the ExecuTorch runtime
```

`target` must be `cpu`. XNNPACK is a CPU delegate, and a `.pte` built for it
runs on Android and iOS CPUs as well as the host — so the CPU target is not a
limitation of the artifact, only a description of the delegate.

Reload it like any other LM7 artifact:

```python
loaded = lm7.load_artifact("model.lm7")
output = loaded(example_input)
```

The reloaded callable takes **positional tensors only**. An ExecuTorch method
has a flat input list with no keyword names, so capture with
`args=(input_ids, attention_mask)` rather than kwargs.

## What the manifest records

```json
"runtime_requirements": {
  "executorch": "1.3.1",
  "delegate": "xnnpack",
  "delegated_calls": 155,
  "total_calls": 1970,
  "device_bound": false
}
```

Two fields are worth reading before shipping an artifact.

`device_bound` is `false`. Every other compiled LM7 payload is pinned to the
machine that produced it; a `.pte` carries its program and its weights and the
delegate spans ARM64 and x86-64, so the same bytes run on the build host and the
phone.

`delegated_calls` / `total_calls` is **partition coverage**, and it is usually
not 1.0. Operators XNNPACK does not implement stay on ExecuTorch's portable
kernels, which are correct but unoptimized. A low ratio on a model you care
about is the signal to look at what did not partition, not a failure.

## Measured on this project

An `x86-64` Linux host, ExecuTorch 1.3.1, PyTorch 2.12.1+cpu, float32.

| | 3-layer MLP | SmolLM2-135M prefill |
| --- | --- | --- |
| Total export | 4.3 s | 178 s |
| `.pte` size | 8 KB | 622 MiB |
| `.lm7` artifact size | 12 KB | 1139 MiB |
| Delegate coverage | whole graph | 155 of 1970 calls |
| Max logit difference vs eager | 1e-4 | 1.4e-4 |
| Steady latency | 0.14 ms | 106 ms |

Three things to take from the SmolLM2 column.

Lowering is slow — minutes, not seconds. That is fine for a build step and
unacceptable in a request path, which is the whole argument for AOT.

The artifact is nearly twice the `.pte`, because LM7 always writes
`exported_program.pt2` beside the compiled payload. Only the `.pte` ships to a
device; the `.pt2` is what makes the artifact reloadable as a torch program.

And 622 MiB is float32 weights written verbatim. A 135M-parameter model has no
business being that large on a phone — that is what quantization is for, and
LM7 does not do it on this path yet.

## Scope

- **Export only.** `lm7.compile()` will not select this backend, and asking for
  it raises. The artifact is the deliverable.
- **XNNPACK only.** Core ML, QNN, MediaTek, Vulkan, Arm Ethos-U, and Exynos are
  ExecuTorch delegates LM7 does not wire up.
- **No quantization.** LM7's quantization is NVIDIA-only, and ExecuTorch has its
  own quantization flow that is not integrated here. This matters more for edge
  than anywhere else — see the artifact size above.
- **Static shapes.** Whatever `torch.export` captured is what the `.pte` accepts.
- **No on-device execution or deployment.** LM7 writes the artifact; getting it
  onto a phone and into an app is ExecuTorch's Android and iOS tooling.
- **No physical-device CI.** Validation is host XNNPACK. The delegate is the same
  one used on ARM, but "runs correctly on x86-64" is not "runs correctly on a
  Pixel", and this project has not measured the latter.

## Validate an artifact

```bash
python -m pytest tests/test_executorch_integration.py -q
```

The third test loads the `.pte` in a fresh interpreter that never imports LM7,
which is the property that makes the artifact worth producing.

## References

- [ExecuTorch](https://github.com/pytorch/executorch) and its
  [documentation](https://docs.pytorch.org/executorch/stable/)
- [XNNPACK backend](https://docs.pytorch.org/executorch/stable/backends-xnnpack.html)
- [Backend overview](https://github.com/pytorch/executorch/blob/main/docs/source/backends-overview.md)
  — the full delegate list and which platform each needs
- [Runtime Python API](https://docs.pytorch.org/executorch/stable/runtime-python-api-reference.html)
