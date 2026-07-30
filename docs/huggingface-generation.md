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

Forcing the issue is not worth it. With Transformers' private
`_compile_all_devices` flag set, CPU decode for DeepSeek-Coder-1.3B compiled in
42.8 s and then ran 1.06x faster than eager — a rounding error for a 43-second
build, which is presumably why the upstream gate exists.

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

## Limits

- Compilation is process-local. Restarting the command recompiles the decode
  graph.
- Changing model, dtype, device, batch shape, or generation/cache shape can
  require another graph.
- Sampling controls, beam search, quantized generation, and persistent AOT
  generation artifacts are not exposed yet.
- The static cache reserves space for the generation length, so larger
  `--max-new-tokens` values consume more memory.
- `model export` remains a prefill/logits artifact; it does not package this
  stateful decode loop.
- No backend other than Inductor is exercised for decode. TensorRT produces
  wrong tokens here and OpenVINO crashes, so the restriction is a measured
  result rather than a placeholder.
