# zentorch (AMD ZenDNN) support

LM7 has a [zentorch](https://pypi.org/project/zentorch/) backend for AMD CPUs.
zentorch is AMD's own PyTorch extension, built on ZenDNN, and it plugs in as a
`torch.compile` backend — so LM7 drives it exactly the way it drives
TorchInductor, and writes no kernels of its own.

It is the AMD-CPU counterpart to what `openvino` is for Intel CPUs: a vendor's
own CPU compiler, reachable by name.

```bash
uv pip install -e ".[zentorch]"
```

```python
compiled = lm7.compile(model.eval(), target="cpu", backend="zentorch", fallback="error")
output = compiled(example_input)
```

```bash
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct --target cpu --backend zentorch
```

> [!NOTE]
> **Explicit-only.** It ranks below `inductor`, so `backend="auto"` never selects
> it. The measurements below are why: zentorch wins on one workload, ties on
> another, and loses on a third. That is not a strong enough result to change
> what every CPU user gets by default, and it is a good enough one to make the
> backend worth having.

## Installing

zentorch publishes **x86-64 Linux wheels only**, and its version tracks the
PyTorch it was built against — `zentorch 2.13.x` pairs with `torch 2.13`. A
mismatched pair is the usual failure, so a separate environment is the safe way
to try it:

```bash
uv venv --python 3.12 .venv-zen
uv pip install --python .venv-zen/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-zen/bin/python -e ".[dev,zentorch]"
.venv-zen/bin/python -m pytest tests/test_zentorch_integration.py -q
```

`lm7 backends` reports what it found:

```
zentorch: available, version 2.13.0.0
  zentorch provides a ZenDNN torch.compile backend.
```

## Performance

Measured on an **AMD EPYC 7B13** (Zen 3, 8 physical cores, 16 logical, AVX2),
`torch 2.13.0+cpu`, `zentorch 2.13.0.0`, FP32, median of 20 after 5 warmup
iterations, one process per pair:

| workload | eager | inductor | zentorch | zentorch vs inductor |
| --- | --- | --- | --- | --- |
| SmolLM2-135M prefill | 40.45 ms | 34.67 ms | **31.28 ms** | **1.11x faster** |
| Llama-3.2-1B prefill | 357.15 ms | 385.30 ms | 379.98 ms | 1.01x — within noise |
| 4-layer transformer block | 128.5 ms | **118.2 ms** | 126.2 ms | 1.07x *slower* |

All three produced the same greedy next token as eager, and the causal-LM rows
agreed on `' Paris'`.

Read the rows carefully, because they do not agree:

- **SmolLM2-135M is a real win.** The distributions do not overlap — zentorch's
  p95 is 32.67 ms against Inductor's *best* sample of 34.02 ms — so the 1.11x is
  not sampling luck.
- **Llama-3.2-1B is a wash.** The medians differ by 1.4%, but Inductor's spread
  is wide (min 294.76 ms, p95 405.23 ms) and its minimum beats zentorch's. On
  this workload the two are indistinguishable, and a single-sample measurement of
  it will report whichever one it happened to catch. An earlier one-shot run
  showed zentorch 1.13x ahead here; median-of-20 does not support that.
- **A synthetic transformer block goes the other way.** Inductor wins by 1.07x
  on a hand-built 4-layer block with no embedding or vocabulary projection.

Compilation cost is a tie: 6.93 s against 6.93 s for SmolLM2, 5.28 s against
5.29 s for Llama-1B. Neither backend is meaningfully cheaper to compile.

**This is one AMD part, and an old one for this purpose.** Zen 3 has no
AVX-512, no BF16, and no VNNI, which is where ZenDNN's larger wins are supposed
to live. A Genoa or Turin EPYC could land very differently, in either direction,
and nothing here measures BF16 or INT8 — the formats zentorch's own materials
emphasise most. Treat the table as "what happened on Zen 3 at FP32", not as a
verdict on ZenDNN.

## Numerics

zentorch matched Inductor exactly on the synthetic block — a maximum absolute
difference of 1.19e-06 against eager, the same figure Inductor produced — and
agreed on the greedy next token for both causal LMs. The integration test
asserts parity with eager at `rtol=1e-4, atol=1e-4`.

## Scope and limits

- **CPU targets only.** `amd` in an LM7 target string means the ROCm *GPU*, which
  shares nothing with this extension; the backend declines it explicitly rather
  than compiling for the CPU behind your back.
- **Any x86-64 CPU is accepted, not just AMD.** zentorch is tuned for EPYC, but
  LM7 does not refuse to run it elsewhere. Nothing here has measured it on an
  Intel part.
- **JIT-only.** Like `inductor`, the compiled callable does not outlive the
  process. There is no zentorch artifact, so `lm7.export` and `lm7 model export`
  do not offer it.
- **`dynamic` is the only supported option.** Inductor's `options` are inductor
  config keys and mean nothing to another Dynamo backend, so passing them raises
  instead of being silently dropped.
- **No quantization path.** `--quantize` routes through TorchAO or NNCF, neither
  of which knows about ZenDNN. ZenDNN's own INT8/BF16 paths are not wired up.
- **No CI.** No AMD EPYC runner exists, so `tests/test_zentorch_integration.py`
  skips everywhere except a host with the package installed.
