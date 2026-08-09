# Compiled Hugging Face generation

`lm7 model generate` runs token-by-token causal-LM generation, reusing one
compiled decode graph rather than compiling a new one per token:

```bash
lm7 model generate hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --target nvidia \
  --backend inductor
```

The command reports two timings. The first generation includes Inductor
compilation. The steady generation reuses the compiled graph in the same
process. `--json` includes the generated token IDs, decoded text, cache type,
and peak GPU memory.

> [!IMPORTANT]
> **Transformers owns this compilation, not LM7.** Asking for
> `cache_implementation="static"` is enough to make `model.generate` compile the
> decode step by itself, and `CompileConfig()`'s own defaults are already
> `backend="inductor", mode="reduce-overhead"` — exactly what LM7 passes. Pinning
> the config keeps LM7 unaffected if that default ever changes, but it does not
> *cause* the speedup. What this command adds over calling Transformers directly
> is target resolution, the per-vendor dtype policy, the timing and peak-memory
> reporting, and the `max_new_tokens >= 2` guard.

## The speedup is real, and it is the compiler's

Measured on `HuggingFaceTB/SmolLM2-135M-Instruct`, 16 greedy tokens, RTX 4070
SUPER (Ada, `sm89`), float16:

| Generation call | Time |
| --- | --- |
| static cache, compiled (second call) | **117 ms** |
| static cache, `disable_compile=True` | 1142 ms |
| default dynamic cache | 890 ms |

So compiling the decode loop is worth roughly 8x — but note the baseline: a static
cache on its own is *slower* than the default dynamic cache. The static cache is
what makes the decode step compilable; it is not itself the win.

Decomposed on `deepseek-ai/deepseek-coder-1.3b-instruct` (same host, 16 tokens,
against its own uncompiled static-cache baseline of 937 ms) by swapping the dynamo
backend inside `CompileConfig`:

| Dynamo backend | Steady | Speedup | Same tokens? |
| --- | --- | --- | --- |
| `eager` | 910 ms | 1.03x | yes |
| `cudagraphs` | 453 ms | 2.01x | yes |
| `inductor`, `reduce-overhead` | 240 ms | **4.35x** | yes |
| `tensorrt` | 332 ms | 3.17x | **no — see below** |
| `openvino` | — | — | crashes |

`eager` is the control: dynamo capture plus a static cache, with no compiler,
buys nothing. `cudagraphs` alone gets half the win, which places roughly half of
Inductor's advantage in launch-overhead elimination and half in codegen.

## Execution boundary

Autoregressive generation has two materially different workloads:

1. **Prefill** processes the entire prompt and populates the KV cache. Prompt
   lengths vary, so LM7 leaves this call eager.
2. **Decode** consumes one token at a time. Transformers allocates a static KV
   cache and invokes one Inductor-compiled, fixed-shape decode graph for every
   subsequent token.

This boundary follows Transformers' compiled-generation contract. It avoids a
shape-specialized prefill graph for every prompt length and prevents the cache
growth that would otherwise trigger decode recompilation.

The first generated token comes directly from the prefill logits. The compiled
decode graph starts with the second token, so the command requires
`--max-new-tokens 2` or greater.

## Why TensorRT is not offered here

TensorRT is the reason `--backend` is restricted rather than opened up. Given the
decode loop it compiles successfully — 57 s engine build, one graph, no graph
breaks — and runs 3.17x faster than the uncompiled baseline. It also generates
**different text**, silently:

```text
reference (uncompiled)  Paris.\n\n## How to use\n\n1. Click on the "
tensorrt                Paris.\n---\n\n\n\n\n\n\n\n\n\n\n\n
```

The two agree for three tokens, diverge on the fourth, and TensorRT then
degenerates into repeating a single newline token — the signature of a KV cache
that is not being updated across steps. Nothing raises, and the output is
fluent enough at a glance to pass casual inspection.

Worth being precise about what this does and does not say. The same backend
handles a *prefill* forward pass for this model correctly, within the accuracy
bars in [deepseek.md](deepseek.md) — so this is not "TensorRT is broken", it is
"a stateful decode loop is a different problem from a pure forward pass, and
per-backend prefill validation does not transfer to it." That is exactly why the
generation path admits one backend it has measured end to end.

The `openvino` dynamo backend does not get that far: it raises
`IndexError: list index out of range` while building a dynamo guard.

## Compilation is silently skipped off CUDA

Transformers only compiles the decode step when the model sits on one of
`cuda`, `xpu`, `neuron`, or `tpu` (`_valid_auto_compile_criteria` in
`transformers/generation/utils.py`). Everywhere else it logs

```text
You have set `compile_config`, but we are unable to meet the criteria for
compilation. Compilation will be skipped.
```

and decodes eagerly. Because LM7 maps `apple` to `mps`, `tpu` and `tenstorrent`
to `xla`, and `intel:npu` to `cpu`, **generation is compiled only on NVIDIA, AMD,
and Intel GPU targets.** Everywhere else the command still works and still
produces correct tokens; it just runs eager, and now reports `backend: eager`
rather than claiming a compiler that never ran.

Note the trap in that list: `tpu` appears in Transformers' criteria, but the
torch device type for a TPU under PyTorch/XLA is `xla`, not `tpu`. The entry is
for the older `torch_xla` TPU device naming, and a Cloud TPU reached the way LM7
reaches it does not match it. TPU decode runs eager.

LM7 no longer sends a `compile_config` it knows will be refused. Passing one
anyway is not free: it produces that warning on every non-compiling target,
which reads like an LM7 fault rather than an upstream gate.

### Forcing it is worth it on Apple Silicon, and not on CPU

The gate is a **hardcoded device allowlist, not a capability check**:

```python
valid_hardware = self.device.type in ["cuda", "xpu", "neuron", "tpu"] or bool(
    generation_config.compile_config is not None
    and generation_config.compile_config._compile_all_devices
)
```

So the honest question is whether a skipped target *cannot* compile or merely
*is not allowed to*, and the private `_compile_all_devices` escape hatch answers
it. The two answers are not the same:

| | forced compile vs eager |
| --- | --- |
| CPU, DeepSeek-Coder-1.3B | 1.06x, after a 42.8 s build — a rounding error |
| CPU, SmolLM2-135M | 1.01–1.04x |
| **Apple Silicon (MPS), SmolLM2-135M** | **1.76–1.79x** |

MPS compiles perfectly well and gets meaningfully faster for it. It is excluded
by a list, not by a limitation. An earlier revision of this page concluded
"forcing the issue is not worth it" from the CPU number alone; that generalized
one target's result to every skipped target, and the MPS measurement below
corrects it.

## Four ways to generate the same tokens

`benchmarks/generation_paths.py` runs all four arms in one harness — same
checkpoint, same tokenized prompt, same token budget, same timing function — and
compares their output text, so an arm that is "faster" because it decoded
something else is reported rather than believed.

```bash
python benchmarks/generation_paths.py --target apple --repeats 9
python benchmarks/generation_paths.py --target cpu --output artifacts/paths.json
```

Apple M-series, SmolLM2-135M-Instruct, 64 new tokens, `--repeats 9`, total wall
clock ÷ tokens, compile cost excluded by a warmup call. Ranges span **three
independent runs**, because a laptop is not a quiet machine. Every arm produced
**identical text**:

| arm | `apple:metal` (fp16) | | `cpu:arm64` (fp32) | |
| --- | --- | --- | --- | --- |
| `eager` — `model.generate()` | 18.4–19.0 ms/tok | 1.00x | 15.1–15.3 ms/tok | 1.00x |
| `hf-static` — `+ StaticCache + CompileConfig` | 17.8–18.0 | 1.03–1.06x | 17.0–17.2 | 0.88–0.89x |
| `hf-forced` — `+ _compile_all_devices` | 10.5–10.8 | 1.76–1.79x | 14.6–15.0 | 1.01–1.04x |
| `lm7` — `compile_generation()` | **6.5–6.8** | **2.77–2.93x** | 17.6–20.2 | **0.75–0.86x** |

> Ranges rather than single numbers on purpose. At `--repeats 3` the eager arm
> wandered between 13.9 and 19.5 ms/token while the compiled arms held to within
> 0.3, so the *ratio* moved by a third on the strength of the baseline alone.
> Nine repeats and three runs is what it took for these to stop moving; a
> one-shot number here would have been fiction.

Three things worth reading off this table:

- **`hf-static` is the eager arm in disguise** on both targets — and *slower*
  than plain eager on CPU, because it pays for a static cache and gets no
  compilation for it. A server that configures Transformers exactly as
  documented still decodes eagerly here, and the only sign is one
  `warning_once` in the logs.
- **`compile_generation` beats even the forced arm on MPS**, ~6.6 against ~10.6
  ms/token. Both compile the decode step; LM7 additionally compiles prefill as
  its own fixed-shape graph and drives the loop itself instead of going through
  `generate`. Which of those accounts for the gap is **not isolated here** — it
  is a measured difference, not an explained one.
- **On CPU, LM7 loses**, consistently, across every run: 0.75–0.86x. At 135M on
  this CPU there is nothing for Inductor to reclaim — `hf-forced` says the same
  at 1.01–1.04x — and LM7's fully materialized static cache costs what it costs.
  Compiling is not free, and this is the case where it does not pay.

Measured on Apple M-series with torch 2.13.0 / transformers 5.14.1. One model at
one size on two targets: nothing here says what happens at 8B, on CUDA, or under
concurrency. The CPU arm in particular is a 135M result and should not be read
as "compiling never pays on CPU".

## Current support

- Greedy decoding (`do_sample=False`)
- Batch size one, as produced by the CLI tokenizer call
- Static KV cache
- Inductor, and only on targets Transformers will compile (see above);
  `--backend auto` selects it
- Causal-LM classes whose Transformers implementation supports
  `cache_implementation="static"` and compiled decoding

This is not a promise that every Hub model works unchanged. Custom model code,
unsupported operators, dynamic cache implementations, multimodal processors,
or model-specific generation inputs can still require integration work.

## Local NVIDIA validation

The path was validated on an NVIDIA GeForce RTX 4070 (Ada, sm89) with
`HuggingFaceTB/SmolLM2-135M-Instruct`, float16, and Transformers 5.14.1. For the
prompt `The capital of France is` and four greedy tokens, it produced:

```text
 Paris. Paris is
```

PyTorch Dynamo reported one unique compiled graph for the decode loop. The same
tokens matched ordinary `model.generate` with a static cache.

`deepseek-ai/deepseek-coder-1.3b-instruct` was validated on the same host: 32
greedy tokens in 389 ms steady after a 34.5 s first call, one decode graph, no
graph breaks, and text identical to the uncompiled static-cache baseline. Its
per-backend numbers are in the tables above.

`tests/test_hf_integration.py::test_compiled_decode_generates_the_same_tokens_as_eager_decode`
keeps this honest under `LM7_RUN_HF_TESTS=1`: it generates once compiled and once
with `disable_compile=True` and requires identical token IDs plus exactly one
decode graph. That comparison is what caught the TensorRT divergence below, and
nothing else in the suite would have — every other Hugging Face test checks a
prefill forward pass.

## What actually triggers another compile

The obvious worry about a fixed-shape decode graph is that varying request shapes
recompile it, which would make a serving process pay ~10-30 s per new shape.
Measured on the RTX 4070 SUPER, one process, cumulative unique Dynamo graphs:

**Prompt length is free.** With `max_new_tokens` held at 16, SmolLM2 served
5-, 50-, and 200-token prompts on one graph, at a flat ~300 ms:

| Prompt tokens | Time | Graphs |
| --- | --- | --- |
| 5 (cold) | 12,824 ms | 1 |
| 5 | 268 ms | 1 |
| 50 | 365 ms | 1 |
| 200 | 306 ms | 1 |

**Generation length costs exactly one extra compile, once.** The second *distinct*
`max_new_tokens` recompiles, and every value after that is free — Dynamo
specializes on the first shape, then recompiles the dimension as dynamic, and the
dynamic graph covers the rest. Two graphs is the ceiling, not two per value:

| `max_new_tokens` in call order | 16 | 32 | 16 | 48 | 64 | 96 | 7 | 123 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SmolLM2-135M, graphs | 1 | **2** | 2 | 2 | 2 | 2 | 2 | 2 |
| DeepSeek-Coder-1.3B, graphs | 1 | **2** | 2 | 2 | 2 | 2 | 2 | 2 |

Which means a process that warms up on two different generation lengths will not
compile again, whatever it is asked for afterwards. Compile times vary widely run
to run — the same SmolLM2 second compile was observed between 11 s and 68 s — so
treat them as an order of magnitude, not a benchmark.

Handing `generate` one pre-allocated `StaticCache` of a fixed bucket length
instead of letting it size a cache per call does collapse this to a **single**
graph. It was measured and **not adopted**: at `max_cache_len=256` the one compile
cost 40.7 s against 25.1 s for the two it replaces, so it trades a larger upfront
cost for a marginal gain in predictability. Worth revisiting if a future workload
cannot tolerate a recompile mid-stream.

## Limits

- Compilation is process-local. Restarting the command recompiles the decode
  graph.
- Changing model, dtype, device, or batch shape can require another graph. For
  prompt and generation length, see the measurements above.
- Sampling controls, beam search, quantized generation, and persistent AOT
  generation artifacts are not exposed yet.
- The static cache reserves space for the generation length, so larger
  `--max-new-tokens` values consume more memory.
- `model export` remains a prefill/logits artifact; it does not package this
  stateful decode loop.
- No backend other than Inductor is exercised for decode. TensorRT produces
  wrong tokens here and OpenVINO crashes, so the restriction is a measured
  result rather than a placeholder.
