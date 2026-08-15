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

## What we run, and on what

Both lists are conventions rather than rules — LM7 allowlists no models and
gates no hardware — but a measurement that reuses them is comparable to
everything already in `docs/`, and one that invents its own is not.

### Models

The canonical set lives in the `HF_MODELS` dicts of `benchmarks/*.py`. Reuse
those keys rather than adding a near-duplicate checkpoint; a new entry should
answer a question the ladder does not.

**Dense**, in increasing order of what they can tell you:

| | why it's in the set |
| --- | --- |
| hand-built MLP | isolates the compiler from everything a real model adds — and the case where compiling *loses* |
| ResNet-18, MobileNetV2, ViT | vision, and `torch.compile` parity in CI |
| BERT | an encoder, so not everything is a causal LM |
| SmolLM2-135M | the fast causal-LM smoke test; 30 layers makes it launch-bound |
| LFM2.5-230M, Qwen3.5-0.8B, DeepSeek-Coder-1.3B | architecture variety at low cost |
| Llama-3.2-1B | the reference model for quantization; most modes are validated here first |
| Llama-3.1-8B | the smallest model large enough that GEMM time dominates |
| LFM2.5-350M, Qwen3-1.7B | **named but unmeasured** — added to the ladder so they can be reached by name; nothing here has run them, so say "not measured", not "supported". Mistral-7B-Instruct-v0.3 moved out of this bucket on the MI300X. See [limitations](docs/limitations.md#model-coverage) |

**Sparse MoE** is tracked separately because it behaves differently, and
because this repo's most-corrected claims are about it — see
[limitations](docs/limitations.md#compilation-and-artifacts):

| | notes |
| --- | --- |
| `mixtral-tiny`, `olmoe-tiny` | hand-built 2-layer configs, no download; exercise the routing |
| OLMoE-1B-7B (6.92B/1B active) | the largest MoE that fits an 80 GB card |
| Qwen3-30B-A3B (~61 GB), Mixtral-8x7B (~93 GB) | need a 96 GB card |

Four MoE traps, each of which has already cost someone a wrong conclusion:

- **Behaviour is a property of (model, transformers version), not of "MoE".**
  Mixtral pre-5.x fails `torch.export`; OLMoE never did; 5.x fixed both. Check
  the pair, don't generalize from one architecture.
- **Tiny config dimensions must be multiples of 16.** The 5.x `grouped_mm` path
  needs strides that are multiples of 16 bytes and raises in *eager* otherwise.
- **On transformers 5.x the experts are not `nn.Linear`.** They are the
  parameter tensors `grouped_mm` consumes, so a 2-layer MoE has nine linears —
  attention plus `lm_head` — and any selector keyed on `.mlp.` (the `fp8` modes)
  matches nothing and refuses.
- **Llama checkpoints use the `unsloth/` mirrors**, because the Meta repos are
  gated and a rented box has no HF token.

### Hardware

[docs/tested-hardware.md](docs/tested-hardware.md) is the authority on what has
actually run; this is what to reach for and why.

| | use it for |
| --- | --- |
| RTX 4070 SUPER (Ada `sm89`, 12 GiB) | the local dev GPU — most NVIDIA numbers in this repo start here |
| H100 80 GB HBM3 (Hopper `sm90`) | the datacenter part, rented; where a claim aimed at deployers gets checked |
| RTX PRO 6000 Blackwell (`sm120`, 96 GiB) | rented; FP4, and the only card that fits Mixtral-8x7B |
| AMD EPYC 7B13 (Zen 3) | CPU baselines — **AVX2 only**, so INT8 latency does not transfer |
| Intel Xeon 8470 (Sapphire Rapids) | the H100 box's host, and the only AMX part available |
| Apple M3 Pro / M4 / M4 Pro | MPS and Core ML; the only accelerator with real-hardware CI |
| TPU v6e, single chip | `openxla`; one chip says nothing about sharding |
| Snapdragon 8 Elite | `qualcomm:sm8750`, cloud-rented physical device |

- **The GPUs above `sm89` are rented Lightning Studios and metered.** Collect
  JSON into `artifacts/` on the box and author docs locally afterwards; don't
  write prose on a rented GPU. Per-box setup traps (the `python3.12-dev` header
  Inductor needs, the pinned TensorRT venv, no `rsync`) are worth re-reading
  before starting rather than rediscovering.
- **Card capacity decides the model list.** 80 GB rules out Mixtral-8x7B at
  BF16; check before planning a sweep around it.
- **Never mix harnesses in one comparison.** `benchmarks/moe.py` and
  `benchmarks/nvidia_matrix.py` build inputs differently and disagree by 2.3x on
  the same card and model — enough to invert a conclusion. State which harness
  produced a number.
- **Intel XPU, Tenstorrent, the Intel NPU, and AWS Trainium have never run on
  real hardware.** Those adapters are unit-tested against mocks, so say
  "implemented" and not "validated".
- **AMD ROCm has run, once, on a rented MI300X** (`gfx942`, CDNA 3, 191.7 GiB) —
  detection, the core matrix, a quantization sweep and an AOTInductor artifact.
  See [docs/amd-mi300x.md](docs/amd-mi300x.md). It has no CI and no bare-metal
  part, so it is as validated as the TPU row and no more. Two things from that
  session generalize: torchao skips its cpp extensions on torch < 2.11, which
  confounds every quantized latency it produces, and the ROCm PyTorch on the AMD
  Developer Cloud is a pre-pulled Docker image rather than a system install.

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
  portable suite/lint/mypy, a `quality-arm64` job running that same suite on
  `ubuntu-24.04-arm` (the only Linux Arm coverage — lint/mypy aren't repeated
  there because they can't vary by architecture), real-hardware jobs where
  GitHub-hosted runners happen to provide the hardware (Apple Silicon MPS/Core
  ML on `macos-26`, ExecuTorch on `ubuntu-24.04-arm`, Windows on
  `windows-2025`), and a CPU-only `backend` matrix for extras that don't need
  special hardware. CUDA, ROCm,
  TPU, Tenstorrent, TensorRT, and GPU-accelerated jobs in general aren't
  there: GitHub's GPU-hosted runners exist but are gated to Organizations on
  the Team/Enterprise Cloud plan, which this personal-account repo isn't on.
- `docs/development.md` has the non-portable validation commands (real CUDA,
  TensorRT, ONNX Runtime GPU, LiteRT, etc.) that don't run in CI at all.
