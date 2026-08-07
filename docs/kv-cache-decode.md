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
- **backend selection through the same planner as `lm7.compile`**, so eager,
  Inductor, and Inductor-with-CUDA-Graphs are one argument apart.

Quantization is deliberately not an argument: the runner takes the model as
given, so a model quantized before the call decodes quantized, through exactly
the gates in [quantization.md](quantization.md).

## What the runner reports

```python
runner.counters      # {"prefill": {...}, "decode": {...}, "steady": {...}}
runner.cudagraphs    # per phase: requested, skips, and whether capture happened
runner.cache_bytes   # KV bytes allocated on the device
runner.cache_sequence_length   # what the cache itself thinks it holds
```

`counters["steady"]` is the one that matters. It accumulates over every call that
produced an answer rather than compiled one, so a nonzero `frames` there means a
*token* triggered a compile. `cache_sequence_length` is the cross-check on
correctness: if it ever disagrees with `state.sequence_length`, the positions and
the cache have desynchronized and every token after that is computed against the
wrong thing — see [below](#compiling-a-graph-that-writes-into-a-cache).

## Measured on an H100

All of the following is one **NVIDIA H100 80GB HBM3** (`sm90`, Hopper, driver
580.173.02, Intel Xeon Platinum 8470 host) running `unsloth/Llama-3.2-1B-Instruct`
in bfloat16 with `torch 2.13.0+cu130` and `transformers 5.14.1`, 100 decode steps
per configuration, prompts of tiled English. 60 configurations: four execution
arms across prompt lengths 512–8192 and batch sizes 1, 4 and 8. The command that
produced it is at the [end of this page](#reproducing-this).

The four arms:

| Arm | What it is |
| --- | --- |
| `eager` | the same two-graph structure and the same static cache, with no compiler under it |
| `inductor` | both graphs through TorchInductor, no CUDA Graphs |
| `cudagraphs` | both graphs through Inductor with `compile_mode="reduce-overhead"` |
| `decode-only` | as `cudagraphs`, but the prompt pass left eager (`compile_prefill=False`) |

### At one shape

512-token prompt, batch 1:

| Arm | Prefill | Decode | Throughput | Decode vs eager |
| --- | --- | --- | --- | --- |
| `eager` | 12.3 ms | 13.72 ms/token | 73 tok/s | 1.00x |
| `inductor` | 5.4 ms | 4.45 ms/token | 225 tok/s | 3.08x |
| `cudagraphs` | 15.7 ms | **1.77 ms/token** | **565 tok/s** | **7.75x** |
| `decode-only` | 12.0 ms | 1.79 ms/token | 558 tok/s | 7.66x |

Inductor's codegen is worth 3.1x and CUDA Graphs another 2.5x on top. That split
matters: more than half of what compiling buys a decode step is not better
kernels, it is not launching them one at a time from Python.

### Eager decode does not measure the GPU

Per-token decode latency, `eager` arm, milliseconds:

| Prompt | batch 1 | batch 4 | batch 8 |
| --- | --- | --- | --- |
| 512 | 13.72 | 13.11 | 14.33 |
| 1024 | 12.43 | 12.95 | 12.81 |
| 2048 | 12.32 | 13.23 | 11.56 |
| 4096 | 12.65 | 11.53 | 12.53 |
| 8192 | 12.49 | 11.65 | 15.60 |

That table is flat. Eight times the batch and sixteen times the context change
nothing, because a decode step for a 1B model on an H100 does so little work that
the wall clock is Python and kernel launches almost end to end. Only the last
cell, 8192 tokens of context at batch 8, has enough real work to poke above the
overhead.

The same grid with CUDA Graphs:

| Prompt | batch 1 | batch 4 | batch 8 |
| --- | --- | --- | --- |
| 512 | 1.77 | 2.15 | 2.24 |
| 1024 | 1.84 | 2.33 | 2.66 |
| 2048 | 1.94 | 2.81 | 3.49 |
| 4096 | 2.12 | 3.80 | 5.12 |
| 8192 | 2.54 | 5.76 | 8.37 |

Now it scales, because the overhead that was hiding the work is gone. Which also
means the *speedup* falls as the work grows — 7.75x at 512 tokens and batch 1,
1.86x at 8192 and batch 8:

| Prompt | batch 1 | batch 4 | batch 8 |
| --- | --- | --- | --- |
| 512 | 7.75x | 6.09x | 6.40x |
| 1024 | 6.76x | 5.56x | 4.81x |
| 2048 | 6.36x | 4.71x | 3.31x |
| 4096 | 5.96x | 3.04x | 2.45x |
| 8192 | 4.91x | 2.02x | 1.86x |

Decode throughput, `cudagraphs`, tokens/second across the batch:

| Prompt | batch 1 | batch 4 | batch 8 |
| --- | --- | --- | --- |
| 512 | 565 | 1860 | 3571 |
| 1024 | 544 | 1717 | 3005 |
| 2048 | 516 | 1424 | 2294 |
| 4096 | 471 | 1054 | 1561 |
| 8192 | 393 | 694 | 956 |

### Compiling prefill stops paying at about 2,048 tokens

This is the finding that justifies `compile_prefill=False` existing. Prefill
milliseconds, eager against Inductor, with the ratio:

| Prompt | batch | eager | inductor | Inductor vs eager |
| --- | --- | --- | --- | --- |
| 512 | 1 | 12.3 | 5.4 | **2.26x faster** |
| 1024 | 1 | 12.6 | 6.7 | 1.87x faster |
| 2048 | 1 | 13.3 | 13.0 | 1.02x — a tie |
| 4096 | 1 | 24.7 | 31.6 | 0.78x — slower |
| 8192 | 1 | 51.4 | 91.6 | **0.56x — slower** |
| 8192 | 8 | 392.0 | 718.3 | 0.55x — slower |

A short prompt is small enough that launch overhead dominates it too, and
Inductor removes that. A long one is a large batched matmul against a
memory-bound attention, which eager already dispatches to the same cuBLAS and
SDPA kernels — so all compiling adds is a less favourable attention lowering.

`decode-only` is the arm that takes both halves of that: eager prefill (391.2 ms
at 8192×8, matching the eager arm's 392.0) with the compiled decode step
(8.38 ms/token against `cudagraphs`' 8.37). Across the whole sweep it is within
1% of the fully compiled arm on decode, and never worse than eager on prefill.

**So the useful default on this hardware is to compile the decode step and leave
prefill alone** — which is the boundary Transformers' own compiled generation
draws, now with a number attached rather than a convention.

### Cold start

What a serving process pays before its first token, measured after CUDA
initialization (the very first configuration in a process also pays ~10 s of
context setup, which is not attributable to any arm):

| Arm | First prompt | First tokens | Steady decode |
| --- | --- | --- | --- |
| `eager` | 473–579 ms | 94–108 ms | 11.5–15.6 ms/token |
| `inductor` | 13.6–14.6 s | 12.7–13.0 s | 3.8–8.7 ms/token |
| `cudagraphs` | 13.8–14.5 s | 11.7–13.5 s | 1.8–8.4 ms/token |
| `decode-only` | 415–447 ms | 2.5–3.6 s | 1.8–8.4 ms/token |

`decode-only` again: compiling one graph instead of two costs about a quarter of
the cold start for the same steady decode. Note that compiling the prefill graph
does not merely add its own compile time — it also makes the *decode* compile
about four times more expensive, 12–13 s against 2.5–3.6 s, which is a cost of
having two `reduce-overhead` graphs in one CUDA Graph tree rather than one.

### Nothing recompiled, anywhere

Across all 60 configurations — every arm, every prompt length, every batch size:

```text
configurations that recompiled during decode: 0
steady graph breaks:                          0 in every arm
```

Each compiled arm reports exactly one Dynamo frame for prefill and one for
decode, zero graph breaks in either, and zero of everything in `steady`. That is
the claim this whole path exists to be able to make, and it is a counter rather
than an argument.

### A thousand tokens, not a hundred

100 steps is short enough that a slow leak would hide in it. A separate run of
1,000 decode steps over 16 configurations says it does not: zero recompiles
again, and
`cache_sequence_length` equal to prompt + 1,000 in every one — the positions and
the cache stayed in step for the whole run.

Per-token latency over 1,000 steps against the same configuration over 100:

| Prompt | batch | `eager` | `cudagraphs` | `cudagraphs` change |
| --- | --- | --- | --- | --- |
| 512 | 1 | 11.60 | 1.89 | 1.07x |
| 512 | 8 | 11.50 | 2.95 | 1.32x |
| 4096 | 1 | 11.92 | 2.14 | 1.01x |
| 4096 | 8 | 12.71 | 5.87 | 1.15x |

The compiled arms get slightly *slower* per token over a longer run and the
eager arm slightly faster, and both are the same effect seen from opposite
sides. Decoding 1,000 tokens after a 512-token prompt nearly triples the context,
so each step reads more KV — visible once launch overhead is gone (1.32x) and
buried under it when it is not. After a 4,096-token prompt the same 1,000 tokens
grow the context by only a quarter, and the latency grows by 15%.

So the per-token cost tracks how much context there is, which is the arithmetic,
rather than creeping upward with how long the loop has been running, which would
be a leak.

### What "the same tokens" means in bfloat16

49 of the 60 configurations produced the same first sixteen tokens as the eager
arm. The other eleven are all `inductor`, `cudagraphs` or `decode-only` at 4096
or 8192 tokens of prompt — and they are a property of the number format, not of
the decode path. Measured directly, on an RTX 4070 SUPER (Ada, `sm89`):

| Model, dtype | What disagrees |
| --- | --- |
| SmolLM2-135M, bf16 | Transformers' **own** dynamic and static caches, by token 5 |
| Llama-3.2-1B, bf16 | eager and Inductor, by token 13 (the two caches agree) |
| SmolLM2-135M, fp32 | nothing — every arm, both caches, and `model.generate` agree exactly |

Greedy decoding puts an `argmax` on top of the logits, so any rounding difference
that crosses a near-tie becomes a different word and then a different sentence.
The logits underneath are the quantity that survives. Driving an eager runner and
a compiled one down the *same* forced token sequence, so that every step's inputs
are identical, and comparing the largest logit difference against the logit
scale:

| dtype | Compiled against eager |
| --- | --- |
| float32 | 1.3e-06 — floating-point noise |
| bfloat16 | 2.1e-02 — four orders of magnitude larger |

Both numbers are stable across the eight steps measured. That gap is the whole
story: in float32 there is nothing for an argmax to trip over, and in bfloat16
there is, eventually.

So the correctness gate in `tests/test_generation_integration.py` runs in
**float32**, where token equality with `model.generate` is exact and meaningful,
and the logits test carries the two tolerances above. The benchmark's own
cross-arm token column is a sanity check, not a gate — and it prompts with real
prose, because on random token ids the next-token distribution is flat enough
that nearly every configuration disagrees for the same reason.

## Compiling a graph that writes into a cache

Two things had to change before a stateful graph could go through LM7 at all, and
both fail the same way: fluent, wrong output, with nothing raised.

### A backend that compiles by executing consumes a cache slot

LM7's Inductor backend compiles on first call and then warms the fresh artifact
up by **calling** it, so that a compilation failure lands inside the backend's
error boundary where `fallback` can act on it. For a stateless forward that extra
execution is invisible.

It is not invisible for a graph that writes into a KV cache. Transformers'
`StaticLayer.update` does not write at the `cache_position` it is handed — that
argument steers the causal mask and the rotary embedding. It writes at the
layer's own `cumulative_length` and advances it by the number of tokens it was
given:

```python
cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
self.cumulative_length.add_(kv_length)
```

So a warmup call spends cache slots the caller never asked for. With a short
prompt the run continues and simply produces different text. With a 512-token
prompt against a 533-slot cache the second write indexes past the end of the
buffer, and the failure surfaces as a wall of

```text
Assertion `index out of bounds: 0 <= tmp40 < 533` failed.
```

from inside a Triton kernel, followed by a `CUDA error: device-side assert
triggered` raised by whatever CUDA call came next — in one run, `empty_cache()`.

The fix is a new Inductor backend option, `warmup: False`, which declines that
call. It is a real trade: without a warmup, `torch.compile` stays lazy, so a
compilation failure surfaces at the call site rather than as a `CompilationError`
the planner can fall back from. A caller that mutates state is the one who should
make that trade, so it is an option rather than a change of default.

`tests/test_generation.py::test_the_inductor_path_advances_the_cache_exactly_once_per_call`
pins it in the portable suite by faking `torch.compile` to the identity, which
keeps the backend warmup and removes the seconds of real compilation.

### A prompt-length attention mask describes the wrong thing

Transformers builds the decode step's mask against the **whole** static cache, so
a mask that covers only the prompt is wrong the moment a token is decoded.
Measured on SmolLM2-135M, a left-padded batch of two prompts, 24 greedy tokens,
with `model.generate` as the reference:

| Mask passed to the decode step | Result |
| --- | --- |
| none | the padded row diverges immediately |
| the prompt's own `(batch, prompt)` mask | both rows diverge, into repeated newlines |
| a `(batch, max_sequence_length)` mask | both rows match `model.generate` exactly |

None of the three raises, and all three produce fluent text. So `prefill` takes
the prompt's own mask and widens it itself; `decode` takes no mask at all and
marks one more slot attendable per token. Cache length is a fixed shape, so
growing the mask is a write rather than a new graph.

Pass a mask only for a padded batch. `None` means "every position is real", which
is correct for equal-length prompts and is what the benchmarks measure.

## What recompiles

<!-- RECOMPILE -->

## Reproducing this

```bash
python benchmarks/decode.py --output artifacts/h100-decode/arms-sweep.json \
  --model unsloth/Llama-3.2-1B-Instruct \
  --sequence-length 512 1024 2048 4096 8192 --batch-size 1 4 8 --decode-steps 100
```

`artifacts/` is scratch and not committed, as everywhere else in this repo. Each
run rewrites its JSON after every configuration and carries a `complete` flag,
because a metered studio can be reclaimed mid-sweep — one of these runs was,
before the flag existed.

Correctness is not in the benchmark; it is in
`tests/test_generation_integration.py`, which requires the runner's tokens to
equal `model.generate`'s for both an unpadded and a left-padded batch, under
`LM7_RUN_HF_TESTS=1`.

## Limits

- **One sequence at a time, at one fixed batch size.** The cache and the mask are
  runner state, so a runner serves one batch from prefill to the last token.
  `max_batch_size` is the allocated batch, not a ceiling: a smaller batch is
  refused rather than padded, because it would also be a new graph shape and
  would recompile the decode step. There is no continuous batching, no request
  interleaving, no prefix reuse, and no paged attention.
- **The cache cannot grow.** `max_sequence_length` covers the prompt *and*
  everything decoded from it; running past it raises rather than evicting.
- **`generate()` is greedy.** `prefill` and `decode` return the logits, so
  sampling is the caller's to do; nothing above the two graphs is provided.
- **Prefill is compiled per prompt length.** That is the point of the split — the
  decode graph is insulated from it — but a workload with many distinct prompt
  lengths pays a compile for each. `compile_prefill=False` leaves the prompt pass
  eager, which is the boundary Transformers' own compiled generation draws.
- **`backend` accepts `eager` and `inductor` only.** Not an oversight: a stateful
  decode loop is a different problem from a forward pass, and
  [the TensorRT measurement](huggingface-generation.md#why-tensorrt-is-not-offered-here)
  is the reason to demand evidence per backend — it compiled the decode loop
  successfully, ran 3.17x faster, and generated different text without raising.
- **JIT only.** Nothing here outlives the process; the exported artifact path is
  still prefill-only. See [limitations](limitations.md).
- **Measured on `sm90` and `sm89`.** The tables above are an H100 80GB HBM3; the
  path also runs on an RTX 4070 SUPER (Ada, `sm89`) and on CPU, where the
  portable tests exercise it. Nothing here has been run on AMD, Apple, or
  anything reached through XLA.
