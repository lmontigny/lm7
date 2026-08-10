# IREE Vulkan artifacts

LM7 can lower a fixed-shape PyTorch export through IREE to a persistent VMFB
whose kernels target Vulkan/SPIR-V. This is an experimental, explicit AOT path:
it is available through `lm7.export`, not `lm7.compile`, and automatic backend
selection never chooses it.

## Install and export

Install the compiler, Turbine bridge, and runtime together:

```bash
uv pip install -e ".[iree-vulkan]"
```

Then export a tensor-only module:

```python
import torch
import lm7

model = torch.nn.Sequential(
    torch.nn.Linear(16, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 4),
).eval()
example = torch.randn(8, 16)

artifact = lm7.export(
    model,
    args=(example,),
    target="nvidia:sm89",
    backend="iree_vulkan",
    output="model-vulkan.lm7",
)

result = lm7.load_artifact("model-vulkan.lm7")(example)
```

The artifact contains `compiled_model.vmfb`, its checksum, and the source
`ExportedProgram`. IREE is loaded only on the first invocation. Consequently a
build host can produce the artifact without seeing a Vulkan device, while the
deployment host must have `iree-base-runtime` and a driver meeting IREE's
baseline — Vulkan 1.3 with a compute queue and the `timelineSemaphore`,
`scalarBlockLayout`, and `synchronization2` features. See [Arm Mali](#arm-mali)
for why that is checked on the device rather than assumed.

The LM7 target continues to describe hardware (`nvidia`, `amd`, `intel`, or
`arm`); `iree_vulkan` describes the compiler/runtime path. The backend does not
accept CPU, Apple Metal, or TPU targets.

## Portability and tuning

By default LM7 asks IREE for a portable Vulkan artifact and does not guess a GPU
microarchitecture from the LM7 target qualifier. This avoids making an artifact
unloadable when IREE's accepted Vulkan target names differ between releases.

Advanced users can pass an IREE target explicitly:

```python
artifact = lm7.export(
    model,
    args=(example,),
    target="nvidia:sm89",
    backend="iree_vulkan",
    output="model-ampere.lm7",
    options={"vulkan_target": "ampere", "opt_level": "O3"},
)
```

Only target names supported by the installed IREE compiler are valid. A
particular device can be selected at runtime with `options={"device_uri":
"..."}`; this is recorded in the manifest and is not passed to the compiler.

### What IREE 3.11 actually accepts

`--iree-vulkan-target` takes architecture code names and a *sparse* table of
product names, and the two disagree about which parts exist. Compiling a
`linalg.matmul` module with `iree-compile 3.11.0` (macOS arm64) gives:

| | accepted | rejected |
| --- | --- | --- |
| NVIDIA | `ampere` | `ada`, `sm_89` |
| AMD | `rdna3` | |
| Qualcomm | `adreno` | |
| Arm | `valhall`, `valhall1`–`valhall4`, `mali-g77`, `mali-g78`, `mali-g715` | `valhall5`, `bifrost`, `midgard`, `mali`, `arm`, `mali-g52`, `mali-g610`, `mali-g720`, `mali-g925` |

Two things follow. The product table misses common parts — `mali-g610` ships in
a great many MediaTek phones and SBCs and is not in it — so a product name is a
worse bet than the generation. And `mali-g715` produces a **byte-identical VMFB
to `valhall4`**, so the product names are aliases onto the generations rather
than finer tuning.

The portable default therefore stays the reliable choice everywhere, which is
why LM7 does not derive this flag from the LM7 target. The flag is not inert —
`valhall1`, `valhall4`, `ampere`, and the portable default all produced
different VMFBs for the same input — it is just not something to guess at.

## Windows and WSL

Compilation was validated under Ubuntu on WSL and execution was validated in a
native Windows Python process on an NVIDIA GeForce RTX 4070 SUPER. The same VMFB
crossed that boundary and produced the expected FP32 result.

On this machine, NVIDIA CUDA is exposed inside WSL but the IREE Vulkan runtime
enumerates no WSL Vulkan device. That is a driver/runtime boundary, not an
offline compilation failure. Compile under WSL and execute from native Windows,
or install a WSL Vulkan ICD that makes `vulkaninfo` and IREE enumerate the GPU.

Useful diagnostics are:

```powershell
vulkaninfo.exe --summary
python -c "import iree.runtime as rt; print(rt.get_driver('vulkan').query_available_devices())"
```

```bash
python -c "from lm7.backends.iree_vulkan import query_vulkan_devices; print(query_vulkan_devices())"
```

## Arm Mali

`target="arm"`, `arm:valhall4`, and `arm:mali-g715` parse and export. **Nothing
has executed on an Arm GPU.** Only the compile half is plumbed, and the
paragraph above is a statement about `iree-compile`'s flag parser, not about
Mali silicon.

Mali is nonetheless the most interesting destination on this backend, because
it is one of the two architectures IREE's Vulkan path is aimed at. IREE's own
[support matrix](https://iree.dev/guides/deployment-configurations/gpu-vulkan/)
rates it:

| GPU vendor | category | performance | focus architecture |
| --- | --- | --- | --- |
| Arm Mali | mobile | good | Valhall+ |
| AMD | desktop/server | good | RDNA+ |
| Qualcomm Adreno | mobile | reasonable | 640+ |
| NVIDIA | desktop/server | reasonable | Turing+ |

That is IREE's claim, not a measurement — this project has run none of it. But
it is upstream's stated priority, and it lines up exactly with what the compiler
accepts: `valhall1`–`valhall4` compile and `bifrost`/`midgard` do not, because
"Valhall+" is where the focus starts. It also means the one card LM7 *has*
validated this backend on, an RTX 4070 SUPER, sits in the weaker half of the
matrix.

The missing half is the runtime. `load_artifact` goes through `iree.runtime` in
Python, which a phone does not have, so reaching a Mali needs an NDK
cross-compile of the IREE runtime with the Vulkan HAL driver — either
`iree-run-module` or an IREE counterpart to `tools/android_runner`, the small
C++ binary the ExecuTorch path pushes to the device. IREE publishes no prebuilt
Android runtime, and this project owns no Mali hardware: its only handset is a
Snapdragon 8 Elite whose GPU is an Adreno, reached today through LiteRT's
OpenCL delegate. See [Android device testing](android-device-testing.md).

`iree-run-module --dump_devices` is worth cross-compiling first regardless: it
is upstream's own device-side compatibility check, and it answers the question
below before any artifact is involved.

An Arm target is `remote=True` for the same reason `qualcomm:sm8750` is: it
describes deployment hardware the compiler host does not own. `torch_device()`
maps an unrecognized vendor to the CPU, so the eager backend explicitly declines
`arm` rather than reporting a host run under a Mali's name.

Two cautions before anyone reads a first number off this path.

**The device baseline is narrower than "Vulkan 1.3".** IREE requires 1.3 *plus*
a compute queue and the `timelineSemaphore`, `scalarBlockLayout`, and
`synchronization2` device features, and upstream warns that the Android version
alone does not settle it — the driver does. So a phone is qualified by running
`vulkaninfo` or `iree-run-module --dump_devices` on it, not by reading its spec
sheet. Older Mali generations are out, which is a second reason `arm:bifrost`
parses as hardware but is not a compiler target.

**A small graph will lose to the CPU.** The only mobile-GPU measurement this
project has — LiteRT on the Adreno — was ~660x *slower* than the same phone's
CPU delegate on a 3-layer MLP, because the graph was too small to amortise
dispatch and shader compilation. That is a property of the workload rather than
of IREE's Mali codegen, and it survives a "good" rating in the matrix above: the
models big enough to repay a mobile GPU are the ones this backend cannot export
yet. A first Mali number taken on an MLP would measure dispatch overhead and
nothing else.

## Current scope

- Fixed input shapes only. Dynamic shape profiles are rejected.
- Tensor inputs and tensor/tuple/list outputs only; Python scalars, dictionaries,
  caches, and model-specific output dataclasses are outside the initial ABI.
- FP32 MLP compilation and native Windows execution are validated on an RTX
  4070 SUPER. FP16 is a goal of the path but does not yet have the same checked
  hardware result, and no Arm GPU has executed an artifact at all.
- Operator coverage is determined by `torch.export`, IREE Turbine, and IREE's
  Vulkan lowering. Unsupported graphs fail explicitly; LM7 does not fall back to
  PyTorch for an exported VMFB.
- Full Hugging Face causal LMs are not claimed yet. Attention operators, dynamic
  sequence lengths, KV caches, and structured outputs need model-specific
  coverage before this can be presented as an arbitrary-HF-model path.
- VMFB compatibility depends on the IREE runtime/compiler generation and the
  target driver. It is not a stable cross-version ABI.

## Vulkan, SPIR-V, and WebGPU

This backend uses IREE's Vulkan HAL and emits SPIR-V inside a VMFB. It does not
run through a browser or WebGPU. WebGPU uses WGSL/SPIR-V-adjacent compilation
and a different runtime contract; IREE's current Python deployment path is
Vulkan-native. A browser/WebGPU backend therefore remains separate future work,
not a switch on this artifact.
