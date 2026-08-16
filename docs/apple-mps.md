# Apple Silicon GPU support

LM7 has initial support for local Apple Silicon GPUs through PyTorch's Metal
Performance Shaders (MPS) backend and TorchInductor. It does not install
Xcode, macOS, or PyTorch itself.

Install a PyTorch build for macOS, which includes MPS support by default,
then install LM7:

```bash
uv pip install torch
uv pip install -e ".[dev]"
```

## Verify the runtime

```bash
python - <<'PY'
import torch

print("built:", torch.backends.mps.is_built())
print("available:", torch.backends.mps.is_available())
PY
```

## Compile and test

```bash
python examples/mac_mlp.py
python -m pytest tests/test_mac_integration.py -q
python examples/local_targets.py --require-apple
```

The equivalent API is:

```python
compiled = lm7.compile(
    model.eval(),
    target="apple",
    backend="inductor",
    transfers="automatic",
    fallback="error",
)
output = compiled(cpu_input)
```

`target="auto"` also selects the local Apple GPU (reported as `apple:metal`)
when no other accelerator is detected.

Both files compare `inductor`/`aot_inductor` against eager MPS with a loosened
tolerance (`rtol=0.05, atol=0.25`), not `torch.testing.assert_close`'s tight
float32 default. That default holds on an M3 Pro but not on GitHub's `macos-26`
CI runner, which measured a 0.149 max absolute difference on this exact
Linear→GELU→Linear model — most likely Inductor's Metal GELU codegen using a
different approximation than eager MPS's kernel, on a GPU generation CI does
not pin. Not yet root-caused on a specific chip; loosened deliberately rather
than tightened back down without knowing which one produced it.

## Persistent AOT packages

`aot_inductor` also supports the `apple` vendor, producing a persistent
`.pt2` package that reloads in a fresh process without recompiling:

```python
artifact = lm7.export(
    model.eval(),
    args=(example_input,),
    target="apple",
    backend="aot_inductor",
    output="model.lm7",
)
loaded = lm7.load_artifact("model.lm7")
output = loaded(example_input.to("mps"))
```

`lm7.compile(model, target="apple", backend="aot_inductor")` works the same
way through the lazy runtime API. Both paths move the model and example
inputs to the resolved device before `torch.export.export()` — the same
device placement `eager`/`inductor` already perform — since AOTInductor's
generated code is captured against whatever device the traced tensors sit
on. See `tests/test_mac_integration.py` for a same-process and
cross-process reload example.

## Benchmark

```bash
python benchmarks/local.py \
  --target cpu apple \
  --backend eager inductor
python benchmarks/gpu.py \
  --target apple \
  --model mlp \
  --backend eager inductor \
  --dtype float16
```

`benchmarks/gpu.py` also accepts `--model smollm2`, `--model lfm25`,
`--model llama32-1b`, and `--model qwen35-0.8b` with the `hf` extra installed,
plus `--compile-mode reduce-overhead`/`max-autotune` for the Inductor backend;
`max-autotune` prints an informational
`Not enough SMs to use max_autotune_gemm mode` warning from Inductor's CUDA
heuristics but still compiles and runs correctly on MPS. `peak_memory_bytes`
is always `null` for the `apple` vendor: PyTorch's `torch.mps` module has no
CUDA-equivalent peak-tracking API, only current allocation.

Representative local run (`mlp`, batch size 8, float16, M4):

```text
     eager  first=  577 ms  median=1.68 ms  p95=1.91 ms  throughput= 4769 samples/s
  inductor  first=  678 ms  median=0.78 ms  p95=0.96 ms  throughput=10313 samples/s
```

## What compiling buys, measured on an M3 Pro

The benchmark above is a hand-built MLP. This is the same question asked of a
causal LM, through `benchmarks/generation_paths.py` — four ways of generating the
same tokens in **one** harness, sharing one tokenized prompt, one token budget,
one timing function and a freshly loaded copy of the same checkpoint per arm.
One script rather than four because two harnesses in this repo disagree by 2.3x
on the same card purely from building inputs differently.

```bash
python benchmarks/generation_paths.py --target apple --model smollm2-135m
python benchmarks/generation_paths.py --target apple --model llama-3.2-1b
```

Apple M3 Pro (14-core GPU, 18 GB unified memory), macOS 26.5.2, torch 2.13.0,
transformers 5.15.0, `apple:metal`, float16, 64 new tokens, median of 3:

| Arm | SmolLM2-135M | vs eager | Llama-3.2-1B | vs eager |
| --- | --- | --- | --- | --- |
| `eager` — `model.generate()` | 19.01 ms/token | 1.00x | 26.87 ms/token | 1.00x |
| `hf-static` — `+ StaticCache + CompileConfig` | 18.21 ms/token | 1.04x | 29.51 ms/token | 0.91x |
| `hf-forced` — the same, past the device allowlist | 10.17 ms/token | 1.87x | 25.56 ms/token | 1.05x |
| `lm7` — `lm7.compile_generation()` | **7.08 ms/token** | **2.69x** | **21.90 ms/token** | **1.23x** |

All four arms produced byte-identical text in both runs, and `steady_frames` was
0 — no token triggered a recompile. JSON in `artifacts/` on the machine that ran
it; the harness writes it with `--output`.

Four things are worth reading off that table.

**`hf-static` is the arm you would otherwise write, and on Apple it does not
compile at all.** Transformers gates compiled generation on a hardcoded device
allowlist — `["cuda", "xpu", "neuron", "tpu"]` in `generation/utils.py` — so
asking for it on MPS logs `unable to meet the criteria for compilation` and
decodes eagerly. 1.04x and 0.91x is the sound of nothing happening.

**The hardware was capable the whole time.** `hf-forced` flips the private
`_compile_all_devices` escape hatch, which is the only way to separate "this
device cannot compile" from "this device is not on the list", and reaches 1.87x
on SmolLM2. So the gap the first point measures is a gating gap, not a Metal
one.

**LM7 is ahead of even the forced arm** — 1.44x on SmolLM2 and 1.17x on
Llama-3.2-1B — because `compile_generation` compiles prefill and decode as
separate graphs against one static cache, rather than compiling the generate
loop as a whole. See [prefill and KV-cache decode](kv-cache-decode.md).

**The speedup shrinks as the model grows, and 2.69x does not transfer.**
SmolLM2-135M is 30 layers of small matmuls, so it is launch-bound and compiling
removes the launch overhead; by 1B enough of the time is GEMM that there is less
overhead left to remove. This is the same shape as the H100 finding that
[compiling prefill stops paying at about 2,048
tokens](kv-cache-decode.md#compiling-prefill-stops-paying-at-about-2048-tokens).
Read the small number as "what a launch-bound model gets", not as LM7's number.

### What LM7 costs over calling `torch.compile` yourself

The table above compares LM7 against what a reader would otherwise write. This
is the other direction: what the orchestration layer costs over the toolchain it
is orchestrating. `benchmarks/gpu.py` has arms for it — `torch-compile` calls
PyTorch directly with LM7 nowhere in the call path, and `inductor-placed` is LM7
with `transfers="explicit"` — timed by `benchmark_callable`, the same loop,
warmup and statistics the ordinary arms get.

```bash
python benchmarks/gpu.py --target apple --model smollm2 --dtype float16 \
  --backend torch-compile inductor-placed inductor --repeats 100
```

Same M3 Pro, float16, batch 1. Every arm compiles through Inductor with
`mode=None`, so the generated code is the same and the difference is dispatch:

| | `torch.compile` | LM7, inputs placed | LM7, default transfers |
| --- | --- | --- | --- |
| SmolLM2-135M forward (7 runs) | 7.88 ms | 7.91 ms — **1.03x** (0.90–1.11) | 8.30 ms — **1.07x** (0.97–1.15) |
| MLP 1024→4096→1024, batch 1 (5 runs) | 0.44 ms | 0.47 ms — **1.06x** (+0.03 ms) | 0.68 ms — **1.44x** (+0.21 ms) |

Ratios are computed within each process, where the arms run seconds apart, and
the parenthesised spread is across whole runs.

The [figure in the README](../README.md#and-what-does-the-layer-itself-cost) plots
one of those runs at 300 repeats rather than summarizing it, because whether the
two distributions overlap is the actual question and a pair of medians cannot
answer it. Regenerate it from a fresh run with:

```bash
python benchmarks/gpu.py --target apple --model smollm2 --dtype float16 \
  --backend torch-compile inductor-placed inductor \
  --repeats 300 --record-latencies --output artifacts/overhead-hist-smollm2.json
python docs/figures/overhead.py
```

`--record-latencies` keeps every per-call measurement in the report instead of
only the median and p95; it is off by default so that a 300-repeat arm does not
put 300 floats into every JSON that gets quoted in a doc. That single run has the
two medians 0.21 ms apart (2.6%), which is inside the run-to-run spread above —
so read the figure as "these overlap", not as a measurement of 2.6%. **On the real model both ranges
contain 1.0**: LM7's overhead there is smaller than what this machine can
resolve, and individual runs put LM7 ahead of the direct call as often as a few
percent behind.

What the MLP row adds is the shape of the cost, because a 0.44 ms workload
resolves what a 7.9 ms one hides. LM7 charges a **fixed** amount per call, not a
proportional one, in two parts:

- **~0.03 ms of dispatch** — the input signature, the compiled-variant lookup,
  and entering the inference context. At the edge of measurability even here,
  and one run of five came out negative.
- **~0.21 ms of input transfer**, under the default `transfers="automatic"`,
  which copies inputs to the device on every call. This is the whole of the
  1.44x, and it is consistent run to run (+0.159 to +0.251 ms).

So the honest summary is that the layer costs a fixed fraction of a millisecond
per call. On a model doing real work that is a few percent at most; on a
microbenchmark small enough to be dominated by dispatch it is 44%, which is a
statement about the microbenchmark. The transfer half is opt-out — pass
`transfers="explicit"` and place inputs yourself, which is what `inductor-placed`
does and what any code already holding device tensors would do.

> Two caveats on these numbers. They are one chip, and MPS run-to-run jitter on
> this machine is large enough (p95 routinely 1.5-2x the median) that the
> conclusion is "a few percent, below the noise floor" rather than a specific
> percentage. And the comparison is Inductor against Inductor: it says nothing
> about backends whose vendor path LM7 wraps more thickly.

> **LFM2.5-230M could not be measured on this path.** Its hybrid
> linear-attention/convolution cache leaves its conv states unallocated by
> `early_initialization`, so they are created lazily inside the first
> inference-mode call and the *second* sequence raises `Inplace update to
> inference tensor outside InferenceMode`. Confirmed on this machine and filed
> as [#192](https://github.com/lmontigny/lm7/issues/192); Qwen3.5-0.8B shares
> the architecture and is likely affected but was not checked. Both models run
> fine through `lm7.compile` and `lm7 model run`, which is a different path.

## Hugging Face models

The [Hugging Face causal-LM path](../README.md#4-run-a-hugging-face-model) runs on
Apple Silicon through `target="auto"` or `target="apple"`:

```bash
uv pip install -e ".[dev,hf]"
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct --target apple --backend inductor
python examples/hf_causal_lm.py --model hf://LiquidAI/LFM2.5-230M --target apple
python examples/hf_causal_lm.py --model hf://unsloth/Llama-3.2-1B-Instruct --target apple
python examples/hf_causal_lm.py --model hf://Qwen/Qwen3.5-0.8B --target apple
LM7_RUN_HF_TESTS=1 python -m pytest tests/test_hf_integration.py -q
```

`unsloth/Llama-3.2-1B-Instruct` is an ungated mirror of Meta's
Llama-3.2-1B-Instruct; the original `meta-llama` repository is gated behind
an accepted license and an authenticated Hugging Face token.

MPS float16 matmul reductions accumulate in a different order than CUDA and
produce a wider tail of outlier logits. Validated locally on SmolLM2-135M
(max absolute logit diff about 0.20), LFM2.5-230M (about 0.03),
Llama-3.2-1B-Instruct (about 0.03), and Qwen3.5-0.8B (about 0.02); all four
keep matching next-token predictions and cosine similarity above 0.9999
against eager. `tests/test_hf_integration.py` uses a wider `atol` on the
`apple` vendor to account for this. TorchAO weight-only quantization reaches
NVIDIA GPUs and CPU only — there is no Apple/MPS quantization path.

Qwen3.5 (like LFM2.5) uses a hybrid linear-attention/convolution
architecture rather than plain attention. It runs correctly through both
`inductor` and `aot_inductor` on Apple Silicon, but the generated C++ is
much larger for this architecture: first-call compilation took roughly a
minute for `inductor` and several minutes for `aot_inductor` locally
(clang genuinely compiling a large generated wrapper, not hung), versus
single-digit seconds for SmolLM2/LFM2.5/Llama.

The initial integration covers local single-GPU inference. It does not yet
provide TorchAO quantization or CI on physical Apple hardware; TorchInductor
coverage for MPS is newer and less mature than CUDA, so validate compiled
output against eager for your own models.
