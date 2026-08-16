# Project summary draft

Status: Draft, not published. Written 2026-08-09.

A one-page summary of LM7 for an external technical audience — a launch post, a
project README rewrite, or the answer to "so what is this?". Kept here rather
than in `docs/` because it is positioning copy, not documentation — see
[README.md](README.md).

Every number and claim below is sourced from a doc in this repo. Re-check them
against the source before publishing; the whole credibility of a summary like
this rests on nothing in it being overstated.

## The pitch, compressed

LM7 is a **portability layer, not an inference engine**. One PyTorch model, one
target string, and it drives whichever vendor compiler already exists for that
hardware. It owns no kernels and no compiler — of the 16 registered backends,
15 wrap an existing vendor toolchain and the 16th is the eager PyTorch fallback.

What a user gets is not having to write the per-vendor dispatch ladder, and not
having to relearn it for the next device. Speed is *evidence* for that, not the
pitch itself. Leading with performance invites a comparison against dedicated
single-target engines, which is not the axis LM7 wins on.

## Who this is for, concretely

The abstract pitch persuades nobody. These are the situations where LM7 is the
shortest path, with what has actually been run against each.

| Use case | What LM7 does | Status |
| --- | --- | --- |
| **Develop on one machine, deploy on another** — write on a Mac, ship to a Linux GPU box | The target string is the only line that changes; detection, device placement, input movement and compile caching are LM7's problem | Run on Apple Silicon, NVIDIA `sm89`/`sm90`/`sm120`, AMD EPYC and Intel Xeon CPUs |
| **Decide whether a model will run at all, before downloading 8 GB of weights** | `lm7 model compatibility` reads the Hugging Face config only — no weights, no tokenizer, no accelerator memory, no compile — and reports `run`/`generate`/`export`/quantization separately | [model-compatibility.md](../docs/model-compatibility.md); `--json` for automation |
| **Evaluate candidate hardware without rewriting the harness per vendor** | Same model, same call site, different target; `lm7 explain` says which backend was chosen and why, `lm7 doctor` says what is missing | Benchmarks in `benchmarks/`; never mix two harnesses in one comparison — they build inputs differently |
| **Ship a model to a phone or an edge device** | `lm7.export()` writes a `.lm7` artifact through ExecuTorch/XNNPACK, Core ML, QNN or LiteRT. The device never sees Python, PyTorch, or LM7 — just the artifact and a small C++ runner | Validated on a physical Snapdragon 8 Elite over adb, and on Apple Silicon for Core ML |
| **Build once, deploy to several different machines** | A multi-target bundle carries per-target artifacts; `load_bundle(...).load(target="auto")` picks at load time on the deployment host | [jit-vs-aot.md](../docs/jit-vs-aot.md) |
| **Put an API in front of a model on hardware a serving engine won't install on** | `lm7 model serve` gives an OpenAI-compatible endpoint over the compiled decode loop, so existing clients work unchanged | Single-user, one request at a time, by design — see the limits below |
| **Try quantization without learning four vendor quantization APIs** | `--quantize int8/fp8/nvfp4` behind one flag, with hardware gates that refuse loudly rather than silently producing an unquantized model | Llama-3.2-1B is the reference model; FP8 measured on H100, NVFP4 on Blackwell |

## Where LM7 is the wrong tool

Worth stating plainly, in the summary itself. It costs nothing and it is the
fastest way to be believed about everything else.

- **Production serving at throughput.** LM7's server holds one model and one KV
  cache and serves one request at a time. Continuous batching, paged attention
  and prefix caching are what make a serving engine fast, and LM7 has none of
  them. Use a real serving engine; LM7 can hand the port to one.
- **You will only ever ship to one vendor.** Then that vendor's toolchain,
  driven directly, is one fewer layer between you and the thing that compiles.
  LM7 pays off across the second and third target, not the first.
- **Training.** Inference only.
- **A model the vendor toolchain doesn't support.** LM7's reach is bounded by
  what those toolchains already handle. It cannot compile something TensorRT or
  OpenVINO cannot, it can only tell you sooner and fall back cleanly.

## Headline options

- `LM7 – Run the same PyTorch model on any accelerator by changing one string`
- `LM7 – One call to compile a PyTorch model for CPU, NVIDIA, Apple, or TPU`

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
> Concretely, that buys you a few things. Develop on a Mac and deploy to a GPU
> box by changing one string. Ask `lm7 model compatibility` whether a checkpoint
> will run on a target before downloading its weights — it reads the config
> only. Compare candidate hardware without rewriting the harness per vendor.
> Export a `.lm7` artifact for a phone, where the device never sees Python or
> PyTorch, only the artifact and a small C++ runner. Or put an OpenAI-compatible
> endpoint in front of a compiled model on hardware a serving engine will not
> even install on.
>
> What compiling is worth, measured rather than asserted — H100 80GB,
> Llama-3.2-1B in bf16, 512-token prompt at batch 1: eager decode 13.72 ms/token,
> Inductor 4.45, Inductor + CUDA Graphs 1.77 (7.75x). Two caveats I'd rather
> state than have found: more than half of that isn't better kernels, it's not
> launching them one at a time from Python — and the advantage shrinks as real
> work grows, down to 1.86x at 8192 tokens and batch 8. Full 60-config grid and
> the repro command are in the docs.
>
> Where LM7 is the wrong tool, so you don't have to find out later: production
> serving at throughput (its server takes one request at a time by design),
> anything single-vendor forever (drive that vendor's toolchain directly — LM7
> pays off across the second and third target), training (inference only), and
> any model the underlying vendor toolchain doesn't support, since LM7 cannot
> compile what TensorRT or OpenVINO cannot.
>
> Where it honestly stands: early, inference-only, and model coverage is not
> stable. Hardware it has actually run on is RTX 4070 SUPER, H100, RTX PRO 6000
> Blackwell, AMD EPYC 7B13, AMD Instinct MI300X, Apple M3 Pro/M4/M4 Pro, TPU v6e
> (one chip), and a Snapdragon 8 Elite. AMD ROCm has one rented MI300X validation
> point, not broad AMD coverage. Intel XPU, the Intel NPU, Tenstorrent and AWS
> Trainium have adapters unit-tested against mocks that have never touched real
> hardware — those say "implemented", not "validated". There is no serving
> benchmark in the repo, so please don't read a latency claim into the server.
> `docs/limitations.md` lists everything unproven and is maintained as carefully
> as the README.
>
> I'd most like feedback on whether the target/backend split holds up against
> hardware you actually own, and which vendor toolchain is worth wiring next.

## Where each claim comes from

| Claim in the body | Source |
| --- | --- |
| the detection ladder and its traps | [docs/what-this-replaces.md](../docs/what-this-replaces.md) |
| target vs backend, eager fallback | [docs/architecture.md](../docs/architecture.md) |
| config-only compatibility preflight | [docs/model-compatibility.md](../docs/model-compatibility.md) |
| phone deployment, no Python on the device | [docs/android-device-testing.md](../docs/android-device-testing.md), [docs/executorch.md](../docs/executorch.md) |
| multi-target bundles | [docs/jit-vs-aot.md](../docs/jit-vs-aot.md) |
| 13.72 → 4.45 → 1.77 ms/token, 7.75x, 1.86x at 8192/batch 8 | [docs/kv-cache-decode.md](../docs/kv-cache-decode.md#at-one-shape) |
| hardware actually run on | [docs/tested-hardware.md](../docs/tested-hardware.md), [docs/amd-mi300x.md](../docs/amd-mi300x.md) |
| XPU/NPU/Tenstorrent/Trainium are mock-tested only; ROCm has one MI300X validation point | [docs/limitations.md](../docs/limitations.md), [docs/amd-mi300x.md](../docs/amd-mi300x.md) |
| the server is single-user and unbenchmarked | [docs/serving.md](../docs/serving.md), [limitations](../docs/limitations.md#serving) |
| quantization modes and their hardware gates | [docs/quantization.md](../docs/quantization.md) |

## Judgment calls made in this draft

- **The 4070 SUPER leads the hardware list, not the H100.** "Measured on a card
  you might own" persuades this audience more than a rented Hopper — but the
  H100 is what the quoted numbers come from, because it is the configuration
  with the full 60-config grid behind it.
- **Performance appears once, mid-post, with its own caveats attached.** Stating
  the shrinking speedup before a reader finds it is worth more than the larger
  number would have been.
- **"Where LM7 is the wrong tool" is in the body, not just this file.** A
  summary that only lists strengths gets read as marketing, and the specific
  weaknesses here are ones a reader would hit within an hour anyway.
- **The server is framed as reach, not throughput** — "an endpoint on hardware a
  serving engine won't install on", never "serve traffic". The repo cannot
  support a throughput claim and should not imply one.

## Before publishing

- Re-read [notes/competition.md](competition.md) — it is marked "refresh before
  quoting externally", and any named comparison should be checked against a
  current search first.
- Expect "why not just use Hugging Face Optimum?" as the most likely first
  question; [competition.md](competition.md) has the answer worked out (same
  bet, but scoped to HF's own model libraries rather than an arbitrary
  `nn.Module`).
- Have `docs/limitations.md` open. Half the value of a launch is answering
  "what doesn't work yet" faster and more specifically than the asker expected.
