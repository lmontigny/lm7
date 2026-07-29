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
deployment host must have a Vulkan 1.3-capable driver and `iree-base-runtime`.

The LM7 target continues to describe hardware (`nvidia`, `amd`, or `intel`);
`iree_vulkan` describes the compiler/runtime path. The initial backend does not
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

Only target names supported by the installed IREE compiler are valid. IREE
3.11 accepts `ampere`; it does not accept `ada` or `sm_89`, so the portable
default is the reliable choice for an RTX 4070-class deployment. A particular
device can be selected at runtime with `options={"device_uri": "..."}`; this is
recorded in the manifest and is not passed to the compiler.

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

## Current scope

- Fixed input shapes only. Dynamic shape profiles are rejected.
- Tensor inputs and tensor/tuple/list outputs only; Python scalars, dictionaries,
  caches, and model-specific output dataclasses are outside the initial ABI.
- FP32 MLP compilation and native Windows execution are validated. FP16 is a
  goal of the path but does not yet have the same checked hardware result.
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
