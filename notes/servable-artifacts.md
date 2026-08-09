# Serving a `.lm7` artifact: what it would take

Status: Design note, nothing implemented. Written 2026-08-09.

`lm7 model serve ./model.lm7` is the obvious next thing to ask for, and the
payoff is real — compile on a GPU box, copy one file to a laptop or an NPU
machine, serve it with no Hub, no token and no network. This is why it does not
work today and what it would cost, so the next person picking it up starts from
the constraints rather than from the CLI.

## Why routing the CLI is not the missing piece

`serve` is HF-only at the front door, and it would be easy to assume the fix is
a branch in `resolve_model_source` that calls
[`load_artifact`](../src/lm7/exporting.py) or `load_bundle` instead. It is not.
Two things an artifact does not contain are things the server cannot run
without.

**No tokenizer.** `ArtifactManifest` carries the target, the graph hash, the
input signature, checksums and the payload files. Nothing else. The server needs
a tokenizer for three separate jobs — encoding the prompt, `apply_chat_template`
(the turn delimiters are a property of the checkpoint, not a convention), and
decoding tokens back to text, which is also what stop-sequence matching runs
against. An artifact plus a Hub id for the tokenizer would work, but that is no
longer an offline story.

**No decode loop.** `export_hf_model` captures a *single forward pass*: it
tokenizes one prompt and exports the model over those `input_ids`. There is no
`past_key_values` and no `cache_position` in the captured signature, so there is
no KV cache to advance. Generating from that artifact means re-running the whole
prompt for every token — and it would not even get that far, because
`_validate_shape_profile` rejects a call whose shape is not the captured one.
A `dynamic_sequence` export widens that to a bounded range, which makes the
refusal go away but leaves the quadratic re-forward.

So the artifact is the wrong *shape*, not merely missing a loader.

## What a servable artifact would need

1. **A second export mode that captures the generation structure** rather than
   one forward — the prefill graph, the decode graph, and the static cache's
   dimensions — mirroring what `compile_generation` builds at runtime. This is
   the bulk of the work and it lands in `exporting.py`, not in `serve/`.
2. **Tokenizer files inside the artifact**, with the manifest recording what is
   there. This is what makes the offline claim true; without it the story is
   "no weights download", which is a much smaller promise.
3. **A `format_version` bump**, since both of the above change what a `.lm7`
   contains. Old artifacts must keep loading, and a new one must refuse to load
   on an LM7 that predates the format — the existing rejection path already does
   this and its message is the model to follow.
4. **A generation config** — EOS ids and the default sampling settings — or the
   served model silently behaves differently from the same checkpoint served
   from the Hub, which is the kind of divergence nobody notices until the output
   is subtly wrong.
5. **Bundles for free, afterwards.** `load_bundle(...).load(target="auto")`
   already picks the right per-target artifact, so once a single artifact is
   servable the bundle case is mostly routing. The architecture guard matters
   more here, not less: an `sm89` payload on `sm120` must refuse before it loads,
   and it already does.

## What to be careful about

- **The offline claim is the whole point, so it has to be true end to end.**
  An artifact that still resolves a tokenizer from the Hub on first request has
  not delivered it. Test it with the network off, not with a warm HF cache.
- **`lm7 model export` today produces something that cannot be served.** Once a
  servable mode exists, exporting the wrong one and finding out at serve time is
  the obvious trap — the refusal should name the flag that would have produced a
  servable artifact.
- **This does not widen what can be served.** The same causal-LM contract still
  applies; a VLM needs a processor and a different model class, and none of the
  above brings that closer.

## Related

- [docs/serving.md](../docs/serving.md) — what the server does today.
- [docs/kv-cache-decode.md](../docs/kv-cache-decode.md) — the prefill/decode
  split an export mode would have to capture.
- [docs/aot-artifact-compatibility.md](../docs/aot-artifact-compatibility.md) —
  what artifacts already refuse, measured across three GPU architectures.
- [docs/jit-vs-aot.md](../docs/jit-vs-aot.md) — the two export levels and
  bundles.
