# CLAUDE.md

Project-level context for Claude Code sessions working in this repo. Read the
[README](README.md) first for what LM7 is and how it's used; this file is
about how the *project itself* is built and how to work in it without
rediscovering the same conventions every session.

## What this is

LM7 is a PyTorch-first compiler/hardware orchestration layer: `lm7.compile()`
and `lm7.export()` take a target string (`cpu`, `nvidia`, `apple`, `tpu`, …)
and dispatch to whichever vendor compiler backend handles it, falling back to
plain PyTorch eager when nothing else can. It writes no kernels and no
compiler of its own — every backend wraps an existing vendor toolchain.

- `src/lm7/targets.py` — parses a target string into a `TargetSpec`.
- `src/lm7/planner.py` + `src/lm7/backends/registry.py` — picks the
  highest-priority backend that supports the resolved target.
- `src/lm7/backends/*.py` — one file per backend, each implementing the
  `Backend` protocol in `backends/base.py` (`probe`, `supports`, `compile`,
  `load`).
- `src/lm7/module.py` — the lazy `CompiledModule` returned by `lm7.compile()`.
- `src/lm7/exporting.py` / `bundles.py` — the `.lm7` artifact format and
  multi-target bundles.
- `src/lm7/cli.py` — the `lm7` CLI (`doctor`, `targets`, `backends`, `explain`,
  `model compatibility/run/generate/export`, `bundle`, `artifact`, `hexagon`).

Full architecture: [docs/architecture.md](docs/architecture.md). Current gaps
and unproven claims: [docs/limitations.md](docs/limitations.md) — read that
one before asserting anything is validated.

## Adding or changing a backend

Nearly every backend PR in this repo's history follows the same shape — do
the same when adding one:

1. Implement `src/lm7/backends/<name>.py` against the `Backend` protocol and
   register it in `backends/registry.py`.
2. Add a pytest marker in `pyproject.toml`'s `[tool.pytest.ini_options]`, and
   split tests into `tests/test_<name>_backend.py` (mocked, fast, always
   runs) and `tests/test_<name>_integration.py` (real toolchain, marked, skips
   itself when the package/hardware is absent).
3. Write `docs/<name>.md` (or `docs/<name>-evaluation.md` if it's a measured
   investigation rather than an adopted backend) and add it to
   [docs/README.md](docs/README.md)'s index — that file is the index of
   everything; an undiscoverable doc is as good as no doc.
4. If it installs on an ordinary hosted CPU runner, add it to the `backend`
   matrix in `.github/workflows/ci.yml` so its tests actually run instead of
   silently skipping. If it needs real hardware GitHub can't provide (GPU,
   TPU, Tenstorrent, a specific vendor NPU/SDK behind a login), say so in a
   comment next to the job list rather than adding a job that always skips.
5. Add one line to [docs/changelog.md](docs/changelog.md) under the right
   section once the PR merges, title copied verbatim. **This file has drifted
   badly out of date** (only ever touched by the commit that created it) —
   reviving it is worth doing whenever you're already in the area, not just
   for your own PR.

## Writing docs in this repo's voice

The existing docs (`docs/*.md`) are measured and specific, not
marketing copy — match that:

- **State exactly what hardware a claim was measured on** (e.g. "RTX 4070
  SUPER, Ada `sm89`, 12 GiB", not "an NVIDIA GPU"). See
  [docs/tested-hardware.md](docs/tested-hardware.md) for the actual machines
  available to this project.
- **Don't claim something is validated because it should work.** LM7's own
  running theme is that assumptions like this get corrected in later PRs
  (see `docs/limitations.md`'s MoE section, or the AMX flags: "reported, not
  consulted"). If it wasn't run, say it wasn't.
- **Link out instead of restating.** The README stays lean by pointing to
  `docs/*.md` for depth rather than inlining every number; new docs should be
  linked from both `docs/README.md` and whatever else references them, and
  cross-links should be checked (anchors are generated from headings, so a
  reworded heading breaks any `#anchor` pointing at it).
- **Long, unindexed personal notes go in `notes/`, not `docs/`.** `docs/` is
  the public, indexed documentation set (what an outside reader — including a
  Show HN reader — clicks through); `notes/` is backlog/planning/competitor
  research that isn't meant to be discovered that way. See
  [notes/README.md](notes/README.md).

## Git workflow

- **Branch off `main` per task**, `agent/<short-description>`, then PR back
  into `main`. Don't commit directly to `main`.
- **This repo gets worked on by several parallel sessions at once.** Expect
  the working tree to change under you — a `git pull` mid-task pulling in
  unrelated merges, another session's branch checked out, files you didn't
  touch showing as modified. Run `git status` before anything destructive,
  and don't fold unrelated changes you find sitting in the working tree into
  your own commit.
- Pushing a change to `.github/workflows/*.yml` needs the `workflow` OAuth
  scope on top of `repo` — a plain `gh auth login`/`setup-git` often won't
  have it, and the push fails with "refusing to allow an OAuth App to
  create/update workflow". `gh auth refresh -s workflow` fixes it (needs the
  user's browser approval).
- `main` has no branch protection and no required status checks, so a red CI
  job blocks nothing automatically — it's signal, not a gate, until someone
  configures it otherwise.

## Testing and CI

- `python -m pytest` from a plain `[dev]` install reports most
  backend-specific tests as skipped — that's expected, not a failure; it only
  proves the portable path. Install the extra and use `-m <marker>` (matching
  the marker list in `pyproject.toml`) to actually exercise one.
- CI (`.github/workflows/ci.yml`) mirrors this: a `quality` job for the
  portable suite/lint/mypy, real-hardware jobs where GitHub-hosted runners
  happen to provide the hardware (Apple Silicon MPS/Core ML on `macos-26`,
  ARM64 on `ubuntu-24.04-arm`, Windows on `windows-2025`), and a CPU-only
  `backend` matrix for extras that don't need special hardware. CUDA, ROCm,
  TPU, Tenstorrent, TensorRT, and GPU-accelerated jobs in general aren't
  there: GitHub's GPU-hosted runners exist but are gated to Organizations on
  the Team/Enterprise Cloud plan, which this personal-account repo isn't on.
- `docs/development.md` has the non-portable validation commands (real CUDA,
  TensorRT, ONNX Runtime GPU, LiteRT, etc.) that don't run in CI at all.
