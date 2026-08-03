# RISC-V evaluation

> [!NOTE]
> **Verdict: nothing to build in LM7 yet, and the reason is upstream.**
> `cpu:riscv64` already parses, round-trips, and would be detected on a RISC-V
> host — LM7's target model is architecture-agnostic by construction. What does
> not exist is a PyTorch that installs there. Until that lands, a RISC-V
> "backend" or "target" in LM7 would be a string with nothing behind it.

## Why this is not a new target

Every other entry in [device_list.md](device_list.md) is a *vendor* with its own
toolchain. RISC-V is an *instruction set*, and LM7 already treats CPU
architecture as a qualifier rather than a vendor:

```python
>>> from lm7.targets import parse_target
>>> parse_target("cpu:riscv64")
TargetSpec(vendor='cpu', kind='cpu', architecture='riscv64')
```

`detection.py` fills that qualifier from `platform.machine()`, which returns
`riscv64` on a RISC-V host, so `lm7 targets` would report `cpu:riscv64` on real
hardware today without a line of new code. That is the same mechanism that makes
`cpu:x86_64` and `cpu:arm64` work, and it is why Arm needed no target of its own.

The interesting question is therefore not "what target do we add" but "does the
stack underneath run at all".

## What the ecosystem actually supports

### PyTorch itself — the blocker

There is no official riscv64 wheel. The latest PyTorch release publishes no
RISC-V file at all, and PyPI's `manylinux` images and `auditwheel` do not
support riscv64, which is the mechanical reason wheels cannot ship through
normal channels.

In January 2026 the XuanTie team at Alibaba DAMO Academy filed
[an RFC for a RISC-V support roadmap](https://github.com/pytorch/pytorch/issues/171659)
with five phases: CI infrastructure and donated boards, a micro-kernel library
(explicitly "KleidiAI for Arm" as the model), a `torch.compile` backend, Triton
and TileLang support, then vLLM and SGLang. The issue is open and triaged, with
no target dates.

A third-party `pytorch-riscv64` exists on PyPI at **2.3.0a1**, against an
official 2.13.0. LM7's floor is `torch>=2.0`, so that would *install* — and then
fail on everything LM7 does with `torch.export`, which is a worse outcome than
refusing.

### Compilers and kernels — further along than PyTorch

- **LLVM and GCC** support RVV 1.0, including auto-vectorisation and the vector
  intrinsics. This is the part that is genuinely ready.
- **XNNPACK** is gaining RVV microkernels, with SiFive publishing a
  [walkthrough of contributing an RVV F32-GEMM](https://www.sifive.com/blog/sifive-accelerates-risc-v-vector-integration-in-xnnpack-for-optimized-ai-inference)
  and asking for contributors. ExecuTorch's XNNPACK backend documentation still
  describes the delegate as Arm and x86.
- **IREE** registers `riscv32` and `riscv64` as `llvm-cpu` targets and documents
  [cross-compilation](https://iree.dev/building-from-source/riscv/); recent work
  adds RISC-V microkernels and `linalg.mmt4d` lowering
  ([arXiv:2508.14899](https://arxiv.org/abs/2508.14899)).
- **TVM** has a `riscv_cpu` target through LLVM, with known rough edges — it
  [read VLEN as 128 bits instead of 256](https://github.com/apache/tvm/issues/17625)
  on a Banana Pi K1.
- **llama.cpp and ncnn** already show real RVV 1.0 gains, which is the clearest
  evidence the hardware path works when someone writes the kernels.

### Hardware — no longer the constraint

RVA23-profile silicon with RVV 1.0 is shipping. SpacemiT's K3 is among the first
RVA23 parts and quotes 60 TOPS INT4; SiFive, Ventana and Andes all ship
server-and-edge cores, and Andes reports RVV plus high-bandwidth vector memory
taking DGEMM to 92.8% of theoretical peak. Boards are buyable, which is more than
could be said for the [Hexagon plan](qualcomm-hexagon.md) when it was written.

## What LM7's backends would do on RISC-V

Assuming a working PyTorch, in the order LM7 would try them:

| Backend | On riscv64 | Why |
| --- | --- | --- |
| `eager` | works | whatever the PyTorch build supports |
| `inductor` | likely works | emits C++ and hands it to the host compiler, which supports RVV |
| `aot_inductor` | likely works | same code generation, written to an artifact |
| `tvm` | plausible | has a `riscv_cpu` LLVM target, with the VLEN caveat above |
| `iree_vulkan` | plausible for CPU, not Vulkan | IREE targets riscv64 through `llvm-cpu`; LM7 wires only the Vulkan path |
| `executorch`, `litert` | export works, kernels do not yet | both lower through XNNPACK, whose RVV microkernels are in progress |
| `onnxruntime` | unverified | no RISC-V build documented in what LM7 depends on |
| `openvino`, `zentorch`, `tensorrt`, `openxla`, `qnn`, `tenstorrent` | no | x86/Arm, AMD x86, NVIDIA, TPU, Qualcomm and Tenstorrent respectively |

Note what this table says: the two backends LM7 would *pick automatically* —
`inductor` and `aot_inductor` — are also the two most likely to work, because
they delegate to the host C++ compiler rather than shipping their own kernels.
RISC-V support is largely a question of whether PyTorch builds, not whether LM7
plans correctly.

## Testing without a board

There is no hosted RISC-V CI runner, so this cannot follow the
[ARM64 job](../.github/workflows/ci.yml) pattern.

The equivalent of Arm's Corstone FVP is **QEMU**. `qemu-riscv64` in user mode
runs a riscv64 binary on an x86-64 host and implements RVV 1.0, so the export
and numerics half of a validation is reachable without hardware — the same trade
made in [android-device-testing.md](android-device-testing.md), where the host
half runs everywhere and only the device half needs silicon. QEMU says nothing
about performance, and an RVV kernel's whole point is performance, so a board
remains necessary for any latency claim.

## What would have to be true first

In order, and none of it is LM7 work:

1. **PyTorch installs on riscv64** — an official wheel, or a documented source
   build LM7 can point at. Everything else is blocked on this.
2. **The version is current enough** for `torch.export`, which LM7 uses
   everywhere. 2.3.0a1 is not.
3. **XNNPACK's RVV microkernels land** in a released ExecuTorch, which is what
   would make `executorch` and `litert` more than an export format on RISC-V.

## Recommendation

Do not add a target, a backend, or a preflight. `cpu:riscv64` already parses,
and a preflight that refuses it would have to be removed again the moment
PyTorch ships.

Revisit when [pytorch#171659](https://github.com/pytorch/pytorch/issues/171659)
reaches its first phase. At that point the cheap first step is a QEMU
user-mode run of the existing CPU tests, which measures whether LM7 needs any
change at all — the honest expectation being that it does not, because Arm
needed none.
