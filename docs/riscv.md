# RISC-V evaluation

> [!NOTE]
> **Measured: LM7 runs on RISC-V today, unchanged.** On a RISE RISC-V runner,
> LM7 imports, detects the machine as `cpu:riscv64`, passes its target and
> planner tests, and executes a model — falling back to eager when Inductor
> cannot compile, with output identical to eager. What does *not* work is any
> claim about speed: the only rentable silicon exposes no vector extension at
> all. See [what actually happened](#measured-on-a-rise-risc-v-runner).

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

## Measured on a RISE RISC-V runner

[RISE](https://riseproject.dev/) gives open source projects free native RISC-V
GitHub Actions runners (`runs-on: ubuntu-24.04-riscv`), with no approval step.
`.github/workflows/riscv-experiment.yml` is the job; it is non-blocking, because
its purpose is to record what breaks rather than to gate pull requests.

The machine, as it describes itself:

```
uname -m         riscv64
isa              rv64imafdcsu
cores / memory   4 / 15 GiB
gcc              13.3.0
python           3.12.3 only
```

**The ISA string carries no `v`.** This is stronger than "the wrong RVV
version": the hardware advertises no vector extension to the kernel at all, and
LM7's own detection agrees, reporting `isa_extensions: ()`. Whatever
`XTHeadVector` the C910 implements is not reachable here, so this runner cannot
produce a vector measurement even with T-Head's own toolchain.

What LM7 did on it, with no changes to LM7:

```
parsed:            cpu:riscv64 | architecture: riscv64
platform.machine() riscv64
detected targets   DeviceInfo(target=cpu:riscv64, name='riscv64',
                   total_memory_bytes=16486711296, logical_cores=4)
tests              23 passed  (test_targets.py, test_planner.py)
compile            Backend inductor compilation failed for cpu:riscv64;
                   falling back to PyTorch eager
                   max abs diff vs eager: 0.0
```

Three things follow. Detection works by construction, as argued above, and now
by measurement. Inductor does not compile on this PyTorch — expected, since the
only available build is 2.3.0a1 — and the **eager fallback caught it and
returned a correct answer**, on an architecture nobody designed that fallback
for. And the planner is architecture-agnostic in practice, not just on paper.

### Getting a PyTorch onto it

Neither obvious route worked, and both failures are worth recording:

- `python3-torch` is **not packaged for riscv64** on Ubuntu 24.04; `apt-cache
  policy` returns nothing.
- The runner ships **CPython 3.12 only**, and the one riscv64 wheel on PyPI is
  tagged `cp311`. `uv python install 3.11` fetches a riscv64 build in about nine
  seconds, which solves it.
- `uv venv` does not install `pip`, so the wheel has to go in with `uv pip`.
- The wheel then imports only after `libopenblas0` and `libgomp1` are installed
  from the distro. It was never `auditwheel`-repaired — the same reason it
  carries a `none-any` tag on an archive full of riscv64 binaries — so its
  shared libraries are not bundled.

That last point sharpens the warning above. `pytorch-riscv64` is not merely old;
it is an unrepaired build that raises `ImportError: libopenblas.so.0` on any
machine without OpenBLAS already present.

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

Still add nothing. The measurement supports the original reasoning rather than
overturning it: `cpu:riscv64` resolves, the planner plans, and the eager
fallback covers the case where no backend can compile. A preflight refusing the
architecture would now be refusing something demonstrably working.

Keep the experimental job. It costs nothing, it is non-blocking, and it converts
every future claim in this document into something checkable — the eager
fallback result above is exactly the kind of thing that would otherwise be
assumed.

Revisit when [pytorch#171659](https://github.com/pytorch/pytorch/issues/171659)
reaches its first phase. Two things would change the picture: a current PyTorch,
which is what stands between this and a working `inductor`, and RVV 1.0 silicon
to rent, which is what stands between this and any statement about speed.
