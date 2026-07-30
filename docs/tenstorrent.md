# Tenstorrent support with tt-xla

LM7 has initial single-process inference support for Tenstorrent Wormhole and
Blackhole cards through [tt-xla](https://github.com/tenstorrent/tt-xla), the
PJRT plugin that connects PyTorch/XLA to Tenstorrent's open-source compiler
stack.

The whole path is open source, which is why it is a backend rather than an
evaluation plan: `torch.compile(..., backend="tt")` captures an FX graph,
tt-xla lowers it to StableHLO, [tt-mlir](https://github.com/tenstorrent/tt-mlir)
compiles that to a flatbuffer, and [tt-metal](https://github.com/tenstorrent/tt-metal)
executes it on the device through TT-NN.

```
nn.Module → torch.compile(backend="tt") → FX → StableHLO → tt-mlir → tt-metal/TT-NN → Wormhole / Blackhole
```

> [!NOTE]
> [tt-torch](https://github.com/tenstorrent/tt-torch), Tenstorrent's earlier
> torch-mlir frontend, is deprecated in favour of tt-xla. LM7 wires up tt-xla
> only.

## Install

tt-xla ships from Tenstorrent's own package index, not PyPI, so the extra needs
an explicit `--extra-index-url`. Their wheels target Ubuntu and CPython 3.12.

```bash
uv venv --python 3.12 .venv-tt
uv pip install --python .venv-tt/bin/python -e ".[dev]"
uv pip install --python .venv-tt/bin/python pjrt-plugin-tt \
  --extra-index-url https://pypi.eng.aws.tenstorrent.com/
source .venv-tt/bin/activate
tt-forge-install
```

`pjrt-plugin-tt` carries `pjrt_plugin_tt.so` plus the tt-mlir and tt-metal
payload, and installs two thin wrappers: `torch_plugin_tt` for PyTorch/XLA and
`jax_plugin_tt` for JAX. LM7 uses `torch_plugin_tt`, which registers the plugin
through torch-xla's `torch_xla.plugins` entry point. The matching `torch-xla`
and `torch` versions come with it, so install this into a dedicated environment
rather than on top of an existing CUDA or ROCm PyTorch.

The host also needs the [tt-kmd](https://github.com/tenstorrent/tt-kmd) kernel
driver and firmware new enough for the installed tt-metal. Follow Tenstorrent's
[getting started guide](https://docs.tenstorrent.com/tt-xla/getting_started.html)
for the driver, firmware, and HugePages setup; LM7 installs none of it.

## Verify the runtime

```bash
ls /dev/tenstorrent      # one node per card, published by tt-kmd
tt-smi                   # firmware and board health

python - <<'PY'
import torch_xla
import torch_xla.runtime as xr

xr.set_device_type("TT")
print("PyTorch/XLA:", torch_xla.__version__)
print("PJRT device:", xr.device_type())
print("addressable devices:", xr.addressable_device_count())
print("attributes:", xr.global_runtime_device_attributes())
PY
```

`PJRT device` must report `TT`. Then check what LM7 sees:

```bash
lm7 targets
lm7 backends
lm7 explain --target tenstorrent
```

## Compile and test

```bash
python examples/tenstorrent_mlp.py
python -m pytest tests/test_tenstorrent_integration.py -q
```

The public API is:

```python
compiled = lm7.compile(
    model.eval(),
    target="tenstorrent",
    backend="tenstorrent",
    transfers="automatic",
    fallback="error",
)
output = compiled(cpu_input)
```

`backend="auto"` also selects `tenstorrent` for a Tenstorrent target, since it
is the only backend that supports one.

LM7 detects addressable Tenstorrent devices, moves the model and inputs to the
XLA device, compiles with `torch.compile(..., backend="tt")`, and synchronizes
the first execution so compiler failures stay inside LM7's fallback boundary.
Execution uses `torch.no_grad()` rather than `torch.inference_mode()`, because
PyTorch/XLA tracing needs the tensor version counters that inference mode
disables — the same reason the OpenXLA TPU path does.

## Targets

| Target | Selects |
| --- | --- |
| `tenstorrent` | the first detected card |
| `tenstorrent:wormhole` | Wormhole silicon (n150, n300, TT-LoudBox, Galaxy) |
| `tenstorrent:blackhole` | Blackhole silicon (p100, p150, p300) |
| `tenstorrent:n300` | a board model, matched against the reported device name |

The architecture comes from the PJRT runtime's `device_kind` attribute, so a
qualifier that the runtime does not report raises `TargetNotFoundError` rather
than silently compiling for the wrong card.

## Device selection

PJRT serves one device type per process. LM7 therefore activates `TT` only when
`torch_plugin_tt` is installed and `PJRT_DEVICE` is unset or already `TT`, and
it never reassigns a runtime that has come up as `TPU`. Set `PJRT_DEVICE=TT`
explicitly if something else in the process reaches PyTorch/XLA first.

## Scope

This initial adapter covers one Python process and the cards addressable to it.
Not implemented:

- Multi-card and multi-host sharding. tt-xla supports it through PJRT; LM7 does
  not expose it, so a multi-card box is used as `addressable_device_count`
  separate single devices.
- Persistent artifacts. This is a JIT backend — the compiled flatbuffer lives in
  the process and a restart recompiles. `lm7.export` and `.lm7` bundles do not
  cover Tenstorrent.
- Quantization, which reaches only NVIDIA GPUs and CPU in LM7 today.
- Physical-hardware CI. Nothing in this path is exercised by GitHub Actions, and
  the numbers below have not been measured on a card by this project.

Model coverage is bounded by what tt-mlir lowers. An unsupported operator
surfaces as a `CompilationError` from the compile stage, and the default
`fallback="warn"` then drops to PyTorch eager on CPU.

## Benchmark

```bash
python benchmarks/tenstorrent.py \
  --backend eager tenstorrent \
  --dtype bfloat16 \
  --batch-size 8 \
  --warmup 5 \
  --repeats 30 \
  --output artifacts/benchmarks/tenstorrent-mlp-bf16.json
```

`eager` here is PyTorch/XLA's lazy-tensor execution on the same card, so the
comparison isolates what the tt-mlir compile path adds over op-by-op dispatch.
Record the `pjrt-plugin-tt`, `torch-xla`, tt-metal, firmware, and board versions
next to any published result; none of these numbers are comparable across
toolchain versions.

## References

- [tt-xla](https://github.com/tenstorrent/tt-xla) and its
  [documentation](https://docs.tenstorrent.com/tt-xla/)
- [tt-mlir](https://github.com/tenstorrent/tt-mlir) — the MLIR compiler
- [tt-metal](https://github.com/tenstorrent/tt-metal) — TT-NN and the
  TT-Metalium kernel programming model
- [tt-forge](https://github.com/tenstorrent/tt-forge) — the umbrella project and
  frontend index
- [tt-kmd](https://github.com/tenstorrent/tt-kmd) — the kernel-mode driver
