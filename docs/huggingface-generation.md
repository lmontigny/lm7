# Compiled Hugging Face generation

`lm7 model generate` adds token-by-token causal-LM generation without compiling
a new graph for each token:

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

## Current support

- Greedy decoding (`do_sample=False`)
- Batch size one, as produced by the CLI tokenizer call
- Static KV cache
- Inductor (`--backend auto` selects Inductor for this command)
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
