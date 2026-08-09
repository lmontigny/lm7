# Notes

Working material, not user-facing documentation — nothing here is linked from
`docs/README.md`. Kept in-repo for continuity across sessions, not for a reader
evaluating LM7.

- [`idea_on_the_fly.txt`](idea_on_the_fly.txt) — running backlog of ideas and
  next steps, added to between PRs.
- [`lm7_codex_brief.txt`](lm7_codex_brief.txt) — the original development brief
  used to bootstrap the project.
- [`ZML_details.md`](ZML_details.md) — technical notes on a third-party
  project ([ZML](https://zml.ai/)), kept for comparison while designing LM7's
  own PJRT/StableHLO path. Referenced from
  [docs/stablehlo-pjrt-evaluation.md](../docs/stablehlo-pjrt-evaluation.md).
- [`servable-artifacts.md`](servable-artifacts.md) — design note on what it
  would take to `lm7 model serve ./model.lm7`, and why the artifact format
  cannot do it today. Nothing implemented.
- [`project-summary-draft.md`](project-summary-draft.md) — unpublished one-page
  external summary of LM7: the compressed pitch, the use cases it is the
  shortest path for, where it is the wrong tool, and every claim traced back to
  the doc it came from.
- [`competition.md`](competition.md) — competitive landscape survey (Modular,
  ZML, Roofline.ai, TVM/OctoML, IREE, tinygrad, Thunder, ONNX Runtime,
  PyTorch/XLA) against LM7's positioning.
