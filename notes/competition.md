# Competitive landscape

Status: Personal reference, not exhaustive — refresh before quoting externally.
Last reviewed: 2026-08-05.

LM7's bet: keep the PyTorch `nn.Module` programming model unchanged, own no
kernels or compiler, and orchestrate whichever vendor toolchain already exists
for a target. That combination — zero model porting, zero owned compiler,
broad target list — is the axis every entry below should be checked against.

## Comparison table

| Project | Frontend a user writes | Owns compiler/kernels? | Target breadth | Closest to LM7 on |
| --- | --- | --- | --- | --- |
| [Modular MAX/Mojo](https://www.modular.com/) | Python graph API + Mojo kernels (PyTorch/ONNX import) | Yes — own graph compiler, kernel language, runtime | NVIDIA, AMD, Apple GPU, x86/ARM CPU | Breadth of hardware ambition |
| [ZML](https://zml.ai/) | Zig model `struct` (`zml.Tensor`) | No — reuses OpenXLA/PJRT | CPU, NVIDIA, AMD, Intel, TPU, AWS Neuron | "Reuse existing compilers" philosophy |
| [Roofline.ai](https://www.roofline.ai/) | Import from major frameworks (edge-focused) | Yes — own retargetable compiler/IR | CPUs, MPUs, MCUs, mobile GPUs, NPUs (edge) | "One call, many targets" pitch |
| [Apache TVM](https://tvm.apache.org/) / OctoML legacy | TVM Relax / Relay import | Yes — own compiler (Unity/Relax) | CPU, GPU, many embedded/NPU backends | Original "compile once, run anywhere" precedent |
| [IREE](https://iree.dev/) (OpenXLA) | Imports PyTorch/JAX/ONNX via MLIR | Yes — own compiler + HAL/VM runtime | CUDA, ROCm, Vulkan, Metal, CPU | Compiler substrate other frontends target — also one of LM7's own backends (`iree_vulkan`) |
| [tinygrad](https://github.com/tinygrad/tinygrad) | Own minimal tensor library, not PyTorch | Yes — writes its own backends/drivers per vendor | NVIDIA, AMD (incl. bare-metal driver stack), Apple, Qualcomm | "No vendor toolchain needed" — opposite philosophy to LM7 |
| [Lightning Thunder](https://github.com/Lightning-AI/lightning-thunder) | Existing PyTorch model, unchanged | No — dispatches to nvFuser/Inductor/cuDNN/TransformerEngine | Primarily NVIDIA; training-and-inference | Closest to LM7 in "don't touch the model," but single-vendor-centric and executor-level, not device-portability-level |
| [Hugging Face Optimum](https://github.com/huggingface/optimum) (+ `optimum-intel`, `optimum-nvidia`, `optimum-neuron`) | `transformers`/`diffusers`/`timm`/`sentence-transformers` model classes | No — dispatches to OpenVINO/IPEX/Neural Compressor, TensorRT-LLM, ONNX Runtime, or AWS Neuron SDK per sub-package | Intel CPU/GPU, NVIDIA GPU, AWS Trainium/Inferentia, broad via ONNX Runtime | The real "why not just use X" comparison — same "swap the backend, not the model" bet, at production scale and HF's backing, but scoped to HF's own model libraries rather than arbitrary `nn.Module` |
| [ONNX Runtime](https://onnxruntime.ai/) | Requires ONNX export first | No — dispatches to execution providers (TensorRT, OpenVINO, CUDA, DirectML, …) | Very broad via EPs | Multi-vendor execution-provider dispatch — also one of LM7's own backends |
| [PyTorch/XLA](https://github.com/pytorch/xla) | Existing PyTorch model | No — hands off to OpenXLA | TPU, and CUDA/CPU via PJRT | Same "PyTorch in, vendor compiler out" idea, but single-family (XLA) rather than best-compiler-per-target |
| NVIDIA TensorRT / AMD stack / Intel OpenVINO (native) | Vendor-specific export/APIs | Yes, per vendor | Single vendor each | The status quo LM7 replaces — not a portability layer at all |

## Notes per competitor

### Modular (MAX + Mojo)
Best-funded, most vertically integrated competitor. Owns the graph compiler
(MLIR-based), the kernel-programming language (Mojo), and the serving
runtime — the opposite bet from LM7's "own nothing." Ingests PyTorch/ONNX
models but the performance story depends on Mojo kernels, so heavy adoption
implicitly migrates a team off pure PyTorch. Strongest where a team is willing
to invest in a new stack for the performance ceiling; weakest where the ask is
"change one string, keep my existing PyTorch code and toolchain exactly as
is." Track their multi-GPU/tensor-parallel work (26.x releases) — that's the
direction most likely to erode LM7's differentiation if it lands well and
stays PyTorch-import-compatible.

### ZML
Already tracked in detail in [`ZML_details.md`](ZML_details.md). Reuses
OpenXLA/PJRT like LM7 reuses vendor compilers, but requires porting the model
into Zig — no PyTorch compatibility. Good comparison for architecture
decisions (buffer/tensor separation, overlapping compile with weight
loading), not a head-to-head product competitor given the porting cost.

### Roofline.ai
Founded 2024, RWTH Aachen spin-off. Same "single call, many targets" pitch as
LM7 but squarely aimed at edge/embedded (MCUs, mobile NPUs) rather than
datacenter/workstation accelerators (NVIDIA, TPU, Tenstorrent) that LM7
targets today. Watch if they move upmarket into GPU/datacenter — their
compiler is theirs (not vendor-toolchain orchestration), so the philosophy
differs even where the target list might start to overlap.

### Apache TVM / OctoML
Cautionary precedent, not an active threat: OctoML raised to a ~$900M
valuation on the "compile any model for any hardware" pitch, then NVIDIA
acquired and shut down the commercial service within five weeks (Sept–Oct
2024). TVM itself lives on as an Apache project (Tianqi Chen's MLC-LLM
carries the "run anywhere" torch forward, mostly for consumer devices and
browsers). Worth remembering when framing LM7's own commercial prospects:
the "own the compiler" version of this pitch has a rough monetization
history; LM7's "own nothing" bet is a different answer to the same problem.

### IREE / OpenXLA
Less a competitor than an upstream option — IREE is `iree_vulkan` in LM7's
own backend table. Its ambitions as a general MLIR-based substrate (any
frontend, any HAL target) overlap with what LM7 does at the orchestration
layer, but IREE competes with the *compilers* LM7 dispatches to (Inductor,
TensorRT, OpenVINO), not directly with LM7. If IREE's PyTorch import path
matured enough to be a strong standalone target, it could reduce LM7's need to
special-case per-vendor backends — worth periodic re-check.

### tinygrad
Philosophically the antithesis of LM7: writes its own kernels and, per Tiny
Corp's "sovereign stack" work, even its own AMD driver/assembler rather than
depending on vendor toolchains at all. Small (~20K LOC), NVIDIA/AMD/Apple/
Qualcomm backends, actively developed by George Hotz. Not PyTorch-compatible
(own tensor library), so not a drop-in substitute, but relevant if LM7 ever
needs a fallback path independent of vendor compiler quality/availability.

### Lightning Thunder
Closest in spirit to LM7's "don't touch the model" rule — takes an unmodified
PyTorch model and traces it to a mix of executors (nvFuser, torch.compile,
cuDNN, TransformerEngine). But it optimizes *within* a target (mostly NVIDIA,
training-heavy) rather than choosing *between* vendor targets — no CPU/Apple/
TPU/Tenstorrent story. More adjacent tooling than a device-portability
competitor.

### Hugging Face Optimum
The most directly comparable prior art, and the most likely "why not just use
X" question LM7 gets. `optimum` (plus `optimum-intel`, `optimum-nvidia`,
`optimum-neuron`) does "keep your model code, swap the execution backend" at
real production scale, HF-backed, owning no compiler of its own — the same
bet as LM7. The scope difference is the whole story: `optimum-intel` wires up
OpenVINO, Intel Extension for PyTorch, and Neural Compressor; `optimum-nvidia`
wires up TensorRT-LLM; `optimum-neuron` wires up AWS Trainium/Inferentia via
the Neuron SDK — but all of it is scoped to loading a model through
`transformers`, `diffusers`, `timm`, or `sentence-transformers`, not an
arbitrary `nn.Module`. That scoping buys optimum per-model-family tuning and
years of production usage LM7 doesn't have; LM7's bet is that "any PyTorch
module, one target string" is worth more once a model falls outside those
four libraries — a custom architecture, a research model, a non-HF causal LM.
Worth being ready for this comparison specifically, since it's the one an HN
commenter is most likely to already know and reach for.

### ONNX Runtime
The other backend LM7 already wraps (`onnxruntime`). As a standalone
competitor it requires an ONNX export step (LM7 skips that for eager/Inductor
paths) and its execution-provider model is the same "one core dispatching to
many vendor backends" idea LM7 generalizes across compilers, not just EPs.
Mature and broadly deployed — the bar LM7's ONNX-backed targets are measured
against.

### PyTorch/XLA
Google/Meta's own answer to "run PyTorch elsewhere" — but scoped to the XLA
family (TPU, and CUDA/CPU via PJRT) rather than picking the best compiler per
target. LM7 uses the same building blocks (`openxla`, `stablehlo` backends)
for TPU and Tenstorrent, so PyTorch/XLA is partly upstream dependency, partly
narrower-scope alternative.

## Open questions to revisit

- Is there a well-funded 2026-era startup doing "PyTorch model in, vendor
  compiler chosen automatically" with no owned compiler, closer to LM7's
  exact bet than anything above? Optimum is the closest found so far, but
  it's still per-vendor subclasses a user picks explicitly
  (`ORTModelForCausalLM`, `OVModelForCausalLM`, …), not one call that
  auto-detects the target the way `lm7.compile(model, target="auto")` does —
  worth re-checking whether that gap closes. Otherwise, closest analogues
  either own a compiler (Modular, Roofline, TVM/IREE) or stay single-vendor
  (Thunder, PyTorch/XLA). Re-run this search periodically; this space moves
  fast.
- Track Modular's PyTorch-import fidelity and multi-GPU progress — the
  biggest risk to LM7's differentiation is Modular's import path getting good
  enough that "one string, no Mojo required" becomes true in practice.
- Track whether NVIDIA does to any of these (or to LM7-shaped tooling) what
  it did to OctoAI — an acqui-shutdown is a real outcome in this category.
