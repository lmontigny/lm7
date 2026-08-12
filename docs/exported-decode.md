# Exported KV-cache decode

Every artifact LM7 writes is a pure function of its inputs, except this one.

```bash
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct build/decode.lm7 \
  --decode --max-cache-len 2048 --target cpu --dtype float32
```

That captures a **decode step** rather than a prefill forward pass: one token in,
one position of logits out, against a static KV cache the artifact carries as
buffers and writes into. The artifact reloads in a fresh process, on a machine
that never had the checkpoint, and decodes.

```python
import torch, lm7

artifact = lm7.load_artifact("build/decode.lm7")
logits = artifact(input_ids=torch.tensor([[42]]), cache_position=torch.tensor([0]))
```

`cache_position` is where in the cache this token belongs. It is an input rather
than internal state, which is the whole reason this works — see
[below](#why-the-cache-can-be-a-buffer-but-not-an-input).

> [!NOTE]
> This is the AOT counterpart to [`lm7.compile_generation`](kv-cache-decode.md),
> not a replacement. That path is faster, takes a whole prompt in one call, and
> dies with the process. This one survives the process and is the only way to get
> a decode loop onto a machine with no PyTorch compile stack. Both exist because
> neither is the other's superset.

## What was actually blocking this

The repo used to record the blocker as the pytree problem `_LogitsOnly` solves:
`CausalLMOutputWithPast` cannot be deserialized by `torch.export.load`. That is
true, and it is a fact about the **output**. It was never what stopped the cache
from being captured.

The real question was where to put the state. A `Cache` is not a tensor, so it
cannot be a graph input, and a graph that takes its cache as an argument would
need the cache functionalized into inputs and outputs — the "cache as graph
input" design this was assumed to require. The answer turned out to be neither:
hold the cache as **buffers on the exported module**. `torch.export` lifts
buffers, and the writes survive as `index_copy_` on them.

Transformers already ships that wrapper —
`TorchExportableModuleWithStaticCache`, which is what ExecuTorch's LLM export
uses. LM7 wraps it in `_DecodeStep` for a two-tensor named signature and exports
that. As everywhere else in this repo, LM7 writes no cache and no kernels.

### Why the cache can be a buffer but not an input

One line inside that wrapper is what makes a stateful graph safe to call:

```python
layer.cumulative_length.copy_(cache_position[0])
```

The write position is **re-derived from an input on every call** rather than
advanced by one per execution. That inverts the hazard the JIT path spends real
effort on. There, `StaticLayer.update` advances `cumulative_length` itself, so an
extra execution silently spends a cache slot and desynchronizes the positions
from the cache — which is why `lm7.compile_generation` needs the Inductor
backend's `warmup: False` option, and why a 512-token prompt against a 533-slot
cache dies in a device-side assert. See [that page](kv-cache-decode.md#a-backend-that-compiles-by-executing-consumes-a-cache-slot).

Here, calling the same graph twice at the same `cache_position` writes the same
slot twice and stays correct. `tests/test_export_decode_integration.py::test_the_cache_actually_accumulates`
pins both halves: idempotent at one slot, different one slot on.

## Strict export, and what non-strict costs

Decode artifacts are captured with `strict=True`, where the prefill path is not.
This is not a preference:

| capture | `export` backend | `aot_inductor` backend |
| --- | --- | --- |
| `strict=False` | works | **fails to package** |
| `strict=True` | works | works |

Under non-strict export the cache tensors arrive as *lifted constants* rather
than buffers, and lowering one dies inside functionalization:

```text
RuntimeError: false INTERNAL ASSERT FAILED ... mutating a non-functional tensor
with a functional tensor is not allowed
```

pointing at `cumulative_length.copy_`. Dynamo's tracing keeps them as buffers.
Capturing strictly for both backends keeps the artifact independent of which one
was asked for, and it is what Transformers' own wrapper defaults to.

## Which backends are allowed, and why only those

`--decode` accepts `export` and `aot_inductor`. Both were checked the only way
this can be checked: export, save, **reload in a fresh process**, decode, and
compare every token against eager.

That procedure is the point rather than diligence. A backend that functionalizes
the cache writes away raises nothing. It returns the correct first token — the
cache is empty then, so there is nothing to have lost — and diverges from the
second onward into fluent, wrong text. There is no exception to catch and no
warning to read.

Every other export backend is **unmeasured, not known-bad**. ExecuTorch is the
likeliest next one to work, since this wrapper exists for it. TensorRT, OpenVINO,
Core ML, QNN, LiteRT, IREE Vulkan, TVM and StableHLO each lower through their own
toolchain and each would need the same check. Until someone runs it, `--decode`
refuses them rather than producing an artifact whose wrongness is silent.

## Measured

One machine: **Apple M-series, `cpu:arm64`, float32**, torch 2.13.0 /
transformers 5.14.1, `SmolLM2-135M-Instruct`, `--max-cache-len 64`, 12 greedy
tokens after a 5-token prompt.

| backend | export | artifact | tokens vs eager |
| --- | --- | --- | --- |
| `export` | 11.2 s | 524.2 MiB | 12/12 exact |
| `aot_inductor` | 47.7 s | 1045.6 MiB | 12/12 exact |

Both produced `' Paris. Paris is the largest city in France and the capital'`,
matching an eager run of the same weights through the same one-token-at-a-time
loop. Reproduce with:

```bash
python examples/exported_decode.py --backend aot_inductor
```

The artifact is roughly the weights plus the cache: 134.5M float32 parameters is
513 MiB, and the 64-slot cache is 2.81 MiB of it — `lm7 artifact inspect` reports
that number. `aot_inductor` doubles the total because the package holds the
compiled wrapper *and* the source program.

**No speed claim is made here.** These numbers are export cost and size, not
throughput, and nothing on this page has been compared against
`lm7.compile_generation` or run on a GPU.

## Limits

- **One token per call.** The prompt goes through the same graph a token at a
  time. That is fine at 5 tokens and the wrong shape at 512: a prompt costs a
  forward pass per token instead of one batched call. A separate exported prefill
  graph is the obvious next step — the same wrapper captures a multi-token
  example — and it is not built here.
- **Batch 1, cache length fixed at export.** `--max-cache-len` bounds prompt plus
  completion together and cannot grow afterwards, because the cache is buffers
  inside the artifact.
- **The artifact is a single session.** State lives in the payload, so two
  concurrent callers share one cache. Re-anchoring with `cache_position=0`
  restarts the write pointer but does not zero the stale slots behind it;
  whether that leaks depends on the masking, and it has **not** been tested.
  Load a second artifact for a second session.
- **No attention mask, so no padded batch.** The decode mask is built against the
  whole cache from `cache_position`, so there is nothing for a caller to pass.
- **No quantization.** The quantizing export backends are `executorch` and
  `openvino`, neither of which is validated for a stateful graph.
- **CPU and float32 only, so far.** Nothing here has run on CUDA, MPS, or in a
  narrower dtype. The export path itself is target-generic and the gate is not,
  deliberately: an untested target would fail silently in exactly the way this
  whole page is about.
- **Greedy is the caller's problem.** The artifact returns logits; sampling, stop
  sequences and the loop itself are not in it.
