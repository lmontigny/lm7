# Project summary draft (Show HN)

Status: Draft, not published. Written 2026-08-09.

A one-page external summary of LM7 for a general technical audience, in the
shape of a Show HN post. Kept here rather than in `docs/` because it is
positioning copy, not documentation — see [README.md](README.md).

Every number and claim below is sourced from a doc in this repo. Re-check them
against the source before posting; the whole credibility of a post like this
rests on nothing in it being overstated.

## The pitch, compressed

LM7 is a **portability layer, not an inference engine**. One PyTorch model, one
target string, and it drives whichever vendor compiler already exists for that
hardware. It owns no kernels and no compiler — of the 16 registered backends,
15 wrap an existing vendor toolchain and the 16th is the eager PyTorch fallback.

What a user gets is not having to write the per-vendor dispatch ladder, and not
having to relearn it for the next device. Speed is *evidence* for that, not the
pitch itself. Leading with performance invites a comparison against dedicated
single-target engines, which is not the axis LM7 wins on.

## Title options

- `Show HN: LM7 – Run the same PyTorch model on any accelerator by changing one string`
- `Show HN: LM7 – One call to compile a PyTorch model for CPU, NVIDIA, Apple, or TPU`

## Body

> LM7 is a PyTorch-first compiler orchestration layer for local inference. You
> hand it a model you already have and a target string, and you get back a
> normal `nn.Module`.
>
> ```python
> model = lm7.compile(model, target="auto")         # detect what's in this box
> model = lm7.compile(model, target="nvidia:sm89")  # or pin it exactly
> ```
>
> The problem it solves is unglamorous. There is no single "which accelerator do
> I have" call in PyTorch, and every vendor compiler has a different entry point
> and its own semantics. `torch.cuda.is_available()` is True for AMD ROCm builds
> too — `torch.version.hip` is what actually tells them apart. Apple's device
> string takes no ordinal while every other vendor's does. An importable
> `torch_xla` doesn't prove you have a TPU; the PJRT runtime may be pointed at
> CPU, so you have to ask it. TPU needs `torch.no_grad()` rather than the
> faster-sounding `torch.inference_mode()`, which silently breaks XLA tracing.
> And `torch.compile` is lazy, so a compilation failure surfaces from your first
> *call*, not from `torch.compile` — which means a fallback has to wrap the
> call, not the compile.
>
> I wrote that ladder more than once and got a different part of it wrong each
> time. LM7 is that ladder in one call, with the per-vendor branches in one
> place.
>
> **LM7 writes no kernels and no compiler of its own.** Every vendor already
> ships one; LM7 drives whichever handles your target — TorchInductor,
> AOTInductor, TensorRT, OpenVINO, ONNX Runtime, ExecuTorch/XNNPACK, Core ML,
> LiteRT, IREE/Vulkan, TVM, PyTorch/XLA, QNN — and falls back to eager PyTorch
> with a warning when none can. Target (where it runs) and backend (which
> compiler gets it there) are separate strings: pin either, or let LM7 choose.
> That split is what actually makes the hardware swappable.
>
> `lm7.export()` writes a portable `.lm7` artifact instead of running in-process,
> and `lm7 model serve` puts an OpenAI-compatible endpoint in front of a compiled
> decode loop, so an existing client can talk to a model on whatever silicon is
> in front of you.
>
> What compiling is worth, measured rather than asserted — H100 80GB,
> Llama-3.2-1B in bf16, 512-token prompt at batch 1: eager decode 13.72 ms/token,
> Inductor 4.45, Inductor + CUDA Graphs 1.77 (7.75x). Two caveats I'd rather
> state than have found: more than half of that isn't better kernels, it's not
> launching them one at a time from Python — and the advantage shrinks as real
> work grows, down to 1.86x at 8192 tokens and batch 8. Full 60-config grid and
> the repro command are in the docs.
>
> Where it honestly stands: early, inference-only, and model coverage is not
> stable. Hardware it has actually run on is RTX 4070 SUPER, H100, RTX PRO 6000
> Blackwell, AMD EPYC 7B13, Apple M3 Pro/M4/M4 Pro, TPU v6e (one chip), and a
> Snapdragon 8 Elite. AMD ROCm GPUs, Intel XPU, the Intel NPU, Tenstorrent and
> AWS Trainium have adapters unit-tested against mocks that have never touched
> real hardware — those say "implemented", not "validated". `serve` is
> single-user and takes one request at a time by design; it is not a serving
> engine, and there is no serving benchmark in the repo, so please don't read a
> latency claim into it. `docs/limitations.md` lists everything unproven and is
> maintained as carefully as the README.
>
> I'd most like feedback on whether the target/backend split holds up against
> hardware you actually own, and which vendor toolchain is worth wiring next.

## Where each claim comes from

| Claim in the body | Source |
| --- | --- |
| the detection ladder and its traps | [docs/what-this-replaces.md](../docs/what-this-replaces.md) |
| target vs backend, eager fallback | [docs/architecture.md](../docs/architecture.md) |
| 13.72 → 4.45 → 1.77 ms/token, 7.75x, 1.86x at 8192/batch 8 | [docs/kv-cache-decode.md](../docs/kv-cache-decode.md#at-one-shape) |
| hardware actually run on | [docs/tested-hardware.md](../docs/tested-hardware.md) |
| ROCm/XPU/NPU/Tenstorrent/Trainium are mock-tested only | [docs/limitations.md](../docs/limitations.md) |
| `serve` is single-user, one request at a time, unbenchmarked | [docs/serving.md](../docs/serving.md), [limitations](../docs/limitations.md#serving) |

## Judgment calls made in this draft

- **The 4070 SUPER leads the hardware list, not the H100.** "Measured on a card
  you might own" persuades this audience more than a rented Hopper — but the
  H100 is what the quoted numbers come from, because it is the configuration
  with the full 60-config grid behind it.
- **Performance appears once, mid-post, with its own caveats attached.** Stating
  the shrinking speedup before a commenter finds it is worth more than the
  larger number would have been.
- **`serve` is framed as reach, not throughput** — "talk to a model on whatever
  silicon is in front of you", not "serve traffic". The repo cannot support a
  throughput claim and should not imply one.

## Before posting

- Re-read [notes/competition.md](competition.md) — it is marked "refresh before
  quoting externally" and any named comparison in the comments should be checked
  against a current search first.
- Expect "why not just use Hugging Face Optimum?" as the most likely top
  comment; [competition.md](competition.md) has the answer worked out (same bet,
  but scoped to HF's own model libraries rather than an arbitrary `nn.Module`).
- Have `docs/limitations.md` open. Half of a Show HN's value is answering
  "what doesn't work yet" faster and more specifically than the asker expected.
