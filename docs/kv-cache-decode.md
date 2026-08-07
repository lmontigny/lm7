# Compiled prefill and KV-cache decode

`lm7.compile()` compiles a forward pass. Serving a language model needs two
compiled things, not one, because generation is two workloads sharing a set of
weights:

```text
prefill   prompt tokens          -> next-token logits + a filled KV cache
decode    one token + that cache -> next-token logits + one more cache entry
```

`lm7.compile_generation()` returns a runner that compiles them separately
against one static KV cache allocated on the target device, and counts every
Dynamo frame, graph break and recompilation each phase costs.

```python
import lm7

runner = lm7.compile_generation(
    model,
    target="nvidia",
    max_batch_size=1,
    max_sequence_length=8192,
    compile_mode="reduce-overhead",   # ask Inductor for CUDA Graphs
)

state = runner.prefill(input_ids)
token, state = runner.decode(state.next_token, state)
```

`runner.generate(input_ids, max_new_tokens=N)` runs that loop greedily and
returns the tokens with the two phases timed apart.

> [!NOTE]
> This is a different thing from [`lm7 model
> generate`](huggingface-generation.md), and both are worth having. That command
> hands the whole loop to `model.generate` and lets **Transformers** decide what
> to compile; this API takes the loop apart and lets **LM7** own the compile
> boundary, the cache lifetime, and the reporting. The CLI is the shorter path to
> text out; this is the one you can measure and build a server on.

## What LM7 owns here, and what it does not

LM7 writes no kernels and no cache. The cache is Transformers' `StaticCache` and
the model is whatever causal LM was passed in — anything whose forward accepts
`past_key_values` and `cache_position`. What LM7 adds is:

- **two graphs instead of one**, so the step that runs a thousand times is not
  sharing a Dynamo cache entry with the one that runs once;
- **the cache allocated up front on the target device**, materialized rather than
  left to allocate itself inside the first traced call;
- **a count of what compiled, per phase**, so "it does not recompile per token"
  is a number rather than a claim;
- **backend selection through the same planner as `lm7.compile`**, so `eager`,
  `inductor` and Inductor-with-CUDA-Graphs are one argument apart.

Quantization is deliberately not an argument: the runner takes the model as
given, so a model quantized before the call decodes quantized, through exactly
the gates in [quantization.md](quantization.md).

## Measured on an H100

<!-- NUMBERS -->

## Compiling a graph that writes into a cache

<!-- WARMUP FINDING -->

## What recompiles

<!-- RECOMPILE -->

## Limits

<!-- LIMITS -->
