# Inspecting compiler IR and generated code

LM7 can keep the intermediate files a backend emits while it turns a PyTorch
module into an artifact. This is useful for answering questions such as:

- what graph did `torch.export` capture?
- which values are parameters versus user inputs?
- what did Inductor schedule before and after fusion?
- what source code or device code did the backend package?

This is **inspection**, not cross-compilation. A Mac can inspect a NVIDIA
artifact built elsewhere, but it cannot generate CUDA, PTX, CUBIN, or TensorRT
engines without a NVIDIA host and the NVIDIA stack.

## Local Mac baseline

On this Mac, the reproducible baseline is a CPU/ARM64 AOTInductor artifact. It
does not show CUDA device code, but it does show every LM7 debug layer that is
present before a target-specific GPU backend starts emitting device binaries.

```bash
python - <<'PY'
from pathlib import Path
import shutil

import torch
import lm7

out = Path("artifacts/ir-inspection/mac-cpu-mlp.lm7")
shutil.rmtree(out, ignore_errors=True)

model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.GELU(),
    torch.nn.Linear(8, 2),
).eval()
example = torch.randn(3, 4)

artifact = lm7.export(
    model,
    args=(example,),
    target="cpu",
    backend="aot_inductor",
    output=out,
    debug=True,
)

for path in artifact.debug_files():
    print(path.relative_to(artifact.path))
PY
```

The same run here produced:

```text
debug/exported_graph.py
debug/exported_program.txt
debug/graph_signature.txt
debug/package/compiled_model__data__aotinductor__model__...kernel.cpp
debug/package/compiled_model__data__aotinductor__model__...wrapper.cpp
debug/torchinductor/model___0.0/fx_graph_readable.py
debug/torchinductor/model___0.0/fx_graph_runnable.py
debug/torchinductor/model___0.0/fx_graph_transformed.py
debug/torchinductor/model___0.0/ir_post_fusion.txt
debug/torchinductor/model___0.0/ir_pre_fusion.txt
debug/torchinductor/model___0.0/output_code.cpp
```

`lm7 artifact inspect` checks the manifest and payloads without loading the
compiled runtime:

```bash
lm7 artifact inspect artifacts/ir-inspection/mac-cpu-mlp.lm7
```

```text
Backend:          aot_inductor
Target:           cpu:arm64
Payload:          compiled_model.pt2
Device-bound:     no
Host executable:  yes
Deployment:       requires the aot_inductor runtime
Checksums:        valid
```

## Reading the layers

Start at the top and move downward only when the previous layer does not explain
the behavior.

| File | Layer | What it answers |
| --- | --- | --- |
| `debug/exported_graph.py` | PyTorch export graph | What ATen ops did the model become? |
| `debug/graph_signature.txt` | Export signature | Which tensors are parameters, buffers, and user inputs? |
| `debug/torchinductor/.../fx_graph_*.py` | Inductor FX graph | What graph did Inductor receive and transform? |
| `debug/torchinductor/.../ir_pre_fusion.txt` | Inductor scheduler IR | What buffers, extern kernels, and loop bodies existed before fusion? |
| `debug/torchinductor/.../ir_post_fusion.txt` | Inductor scheduler IR | What survived fusion? |
| `debug/torchinductor/.../output_code.cpp` | Generated host code | What code will be compiled for this host target? |
| `debug/package/...kernel.cpp` | Packaged kernel source | What source was embedded in the `.pt2` package? |
| `debug/package/...wrapper.cpp` | AOTInductor runtime wrapper | How the compiled model is loaded and called through the runtime ABI. |

For the tiny MLP above, `exported_graph.py` stays close to the original module:

```python
def forward(self, p_0_weight, p_0_bias, p_2_weight, p_2_bias, input):
    linear = torch.ops.aten.linear.default(input, p_0_weight, p_0_bias)
    gelu = torch.ops.aten.gelu.default(linear)
    linear_1 = torch.ops.aten.linear.default(gelu, p_2_weight, p_2_bias)
    return (linear_1,)
```

The signature explains why the weights appear as function arguments:

```text
# inputs
p_0_weight: PARAMETER target='0.weight'
p_0_bias: PARAMETER target='0.bias'
p_2_weight: PARAMETER target='2.weight'
p_2_bias: PARAMETER target='2.bias'
input: USER_INPUT

# outputs
linear_1: USER_OUTPUT
```

The Inductor IR is where the graph stops looking like the Python module and
starts looking like scheduled work. In this run, the two linear layers lowered to
external `addmm` kernels, while GELU became an elementwise loop:

```text
op1.node.kernel = extern_kernels.addmm

class op2_loop_body:
    var_ranges = {p0: 24}
    def body(self, ops):
        load = ops.load('buf1', get_index)
        mul = ops.mul(load, 0.5)
        erf = ops.erf(ops.mul(load, 0.7071067811865476))
        store = ops.store('buf2', get_index, ops.mul(mul, ops.add(erf, 1.0)), None)
```

The generated C++ shows the AOTInductor runtime boundary. On CPU it includes the
CPU AOTI headers and creates a model container for `"cpu"`:

```cpp
#include <torch/csrc/inductor/aoti_include/cpu.h>

auto* model = new torch::aot_inductor::AOTInductorModel(
    constant_map,
    constant_array,
    "cpu",
    "");
```

That is the useful baseline for a host-only run: export graph, signature, FX,
Inductor IR, generated C++, and packaged C++ are all visible without a GPU.

## The same run on NVIDIA

The rest of this page is the same model exported for `nvidia` on an RTX 4070
SUPER (Ada `sm89`, 12 GiB) under WSL2, PyTorch 2.13.0+cu130, Inductor's default
Triton backend. Only the target, the device the model lives on, and the output
path change:

```python
model = model.cuda()
example = example.cuda()

artifact = lm7.export(
    model,
    args=(example,),
    target="nvidia",
    backend="aot_inductor",
    output="artifacts/ir-inspection/nvidia-mlp.lm7",
    debug=True,
)
```

Comparing that against the Mac baseline above would confound the target with a
different host, OS, and PyTorch build, so every diff below is against a
**second CPU export from this same box and same PyTorch build**
(`target="cpu"`, x86-64 with AVX2). What differs is then the target and nothing
else. The NVIDIA run produced (content hashes truncated):

```text
debug/exported_graph.py
debug/exported_program.txt
debug/graph_signature.txt
debug/package/compiled_model__data__aotinductor__model__c75rwsi...kernel.cpp
debug/package/compiled_model__data__aotinductor__model__cgpfa7m...wrapper.cpp
debug/package/compiled_model__data__aotinductor__model__cia234z....cubin
debug/package/compiled_model__data__aotinductor__model__clfhtal....cubin
debug/torchinductor/model___0.0/fx_graph_readable.py
debug/torchinductor/model___0.0/fx_graph_runnable.py
debug/torchinductor/model___0.0/fx_graph_transformed.py
debug/torchinductor/model___0.0/ir_post_fusion.txt
debug/torchinductor/model___0.0/ir_pre_fusion.txt
debug/torchinductor/model___0.0/output_code.cpp
```

### The top layers are byte-identical

`exported_graph.py`, `exported_program.txt`, `graph_signature.txt`, and
`fx_graph_readable.py` are identical between the CPU and NVIDIA exports. The
capture is target-independent: `torch.export` produced the same ATen graph and
the same parameter/user-input split, and Inductor received the same FX graph.

### The first divergence is in `fx_graph_transformed.py`

Inductor's own pre-lowering passes are where the two stacks part. On CPU the
first linear stays a single `addmm`; on CUDA it is split into `mm` plus a
separate bias add:

```diff
-        addmm: "f32[3, 8]" = torch.ops.aten.addmm.default(_0_bias, arg4_1, permute)
+        mm_default: "f32[3, 8]" = torch.ops.aten.mm.default(arg4_1, permute)
+        add_tensor: "f32[3, 8]" = torch.ops.aten.add.Tensor(_0_bias, mm_default)
```

That is Inductor's `unfuse_bias_add_to_pointwise` post-grad pattern, whose
`should_prefer_unfused_addmm` check returns `False` outright for a non-GPU input
device. It looks like a pessimization read alone — it is the setup for a fusion
that only pays off on GPU, which the scheduler IR shows.

### Scheduler IR: what fuses, and what stays extern

| | CPU (x86-64) | NVIDIA (`sm89`) |
| --- | --- | --- |
| `op0` | bias constants for linear 0 | `extern_kernels.mm` |
| `op1` | `extern_kernels.addmm` | bias constants + bias add |
| `op2` | GELU | GELU |
| `op3` | bias constants for linear 2 | bias constants for linear 2 |
| `op4` | `extern_kernels.addmm` | `extern_kernels.addmm` |
| after fusion | unchanged — nothing fused | `op1` and `op2` fuse into `op1_op2` |

Both targets keep both GEMMs in vendor libraries rather than generating them —
at 3x4x8 there is nothing for a generated GEMM to win, and `max_autotune_gemm`
declined anyway (`Not enough SMs to use max_autotune_gemm mode` on a 56-SM
card). What differs is the epilogue: unfusing the first bias let CUDA merge the
bias add and GELU into a single elementwise node, so the bias never round-trips
through global memory as its own kernel. The second linear's bias stays inside
`addmm` even on CUDA, because the same check also requires every user of the
`addmm` output to be a pointwise op — and that output is the graph output.

### Generated code is Triton, not C++ loops

`output_code.cpp` and the packaged `wrapper.cpp` switch to the CUDA AOTInductor
header and the driver-level launch path:

```cpp
#include <torch/csrc/inductor/aoti_include/cuda.h>

static inline CUfunction loadKernel(...);
static inline void launchKernel(...);
    CUfunction triton_poi_fused_addmm_gelu_0{nullptr};
    CUfunction triton_poi_fused_1{nullptr};
```

The kernels themselves are Triton, embedded in the wrapper as comments next to
the `loadKernel` call that will fetch the matching `.cubin` at runtime. The
fused node above compiled to one kernel that materializes the bias, adds it, and
applies GELU in a single pass:

```python
@triton.jit
def triton_poi_fused_addmm_gelu_0(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x2 = xindex
    tmp31 = tl.load(in_out_ptr0 + (x2), xmask)   # the mm result, updated in place
    tmp0 = (x0).to(tl.int64)
    # ... 28 elided lines selecting this row's bias element by index
    tmp30 = tl.where(tmp3, tmp16, tmp29)
    tmp32 = tmp30 + tmp31
    tmp33 = tl.full([1], 0.5, tl.float32)
    tmp34 = tmp32 * tmp33
    tmp35 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp36 = tmp32 * tmp35
    tmp37 = libdevice.erf(tmp36)
    tmp38 = tl.full([1], 1.0, tl.float32)
    tmp39 = tmp37 + tmp38
    tmp40 = tmp34 * tmp39
    tl.store(in_out_ptr0 + (x2), tmp40, xmask)
```

The bias is not loaded from memory at all: with the weights frozen into the
package, Inductor constant-folded the eight bias values into a `tl.where` chain
that the kernel evaluates per element. That is a small-model artifact, not a
general property — it is what "constant materialization" means when it shows up
as `op1`/`op3` in the scheduler IR above.

The extern GEMMs appear as runtime-ABI calls rather than generated code:

```cpp
aoti_torch_cuda_mm_out(buf0, arg4_1, ...);
aoti_torch_cuda_addmm_out(buf4, buf3, buf2, ..., 1L, 1L);
```

One consequence worth knowing before you go looking: on CUDA
`debug/package/...kernel.cpp` is a **6-line stub** holding only the compile and
link commands, because every real kernel is Triton and lives in the wrapper. The
CPU export's `kernel.cpp` is 127 lines of `at::vec` loop code. Read the wrapper
on CUDA, the kernel file on CPU.

### Which lower-level files actually appear

LM7 indexes debug files by extension:

| Indexed level | File kind | Present in this NVIDIA run |
| --- | --- | --- |
| `generated_code` | `.cpp`, `.c`, `.cu`, `.py` | yes — wrapper, kernel stub, `output_code.cpp` |
| `device_code` | `.ptx` | no |
| `machine_code` | `.s`, `.asm` | no |
| `device_binary` | `.cubin` | yes — one per Triton kernel |

The `.ptx` row is worth stating plainly rather than assuming: **no PTX reaches
the artifact on this stack.** LM7 lifts debug files out of the `.pt2` package,
and AOTInductor packages only the wrapper, the kernel file, and one `.cubin` per
Triton kernel. PTX is generated — it is left behind in Inductor's Triton cache
(`/tmp/torchinductor_<user>/triton/…/triton_poi_fused_addmm_gelu_0.ptx`) and
never copied into the package. Anything expecting `device_code` to be populated
for AOTInductor/CUDA will find it empty.

The device code is still recoverable from the artifact, using the `cuobjdump`
and `nvdisasm` that ship inside the Triton wheel:

```bash
.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump --dump-sass \
  artifacts/ir-inspection/nvidia-mlp.lm7/debug/package/*clfhtal*.cubin
```

```text
code for sm_89
nvdisasm warning : Disassembling Std Elf to Old format or Old Elf to Std format not supported yet, switching to Old format
		Function : triton_poi_fused_addmm_gelu_0
	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM89 ..."
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/                   S2R R0, SR_TID.X ;
```

The `nvdisasm` warning appears on every cubin here; it falls back to the old
format and still disassembles, so it is noise rather than a failure.
`--dump-elf` reports `.target sm_89`; `--dump-ptx` prints nothing, since the
cubins are compiled binaries with no embedded PTX.

### What `lm7 artifact inspect` reports

```bash
lm7 artifact inspect artifacts/ir-inspection/nvidia-mlp.lm7
```

```text
Backend:          aot_inductor
Target:           nvidia:sm89
Payload:          compiled_model.pt2
Device-bound:     yes
Host executable:  yes
Deployment:       requires a matching GPU architecture (sm89) and a CUDA 13.0 PyTorch runtime
Checksums:        valid
```

Against the CPU artifact's `Device-bound: no`, that line is the whole point: the
architecture was resolved at export time and baked into the payload. Inspection
does not need that architecture — `lm7 artifact inspect` checks the manifest and
hashes without loading the runtime, and every debug file above is plain text or
a file `cuobjdump` will read. Execution does: reloading this artifact through
`lm7.load_artifact` and running it against eager on the same 4070 SUPER gives a
max absolute difference of `0.0`, but a different GPU architecture or CUDA
runtime is what the `Deployment` line is warning about, and that was not tried
here.

## What this comparison establishes

Same PyTorch program, same PyTorch build, one target change:

- the exported ATen graph and graph signature did not change;
- the FX graph Inductor received did not change;
- Inductor's transformed FX graph did — CUDA unfused a bias that CPU kept;
- fusion differed as a result — one fused elementwise node on CUDA, none on CPU;
- both targets still delegated both GEMMs to vendor libraries;
- generated code changed form entirely — Triton kernels plus `.cubin`, against
  vectorized C++ loops;
- `.ptx` and `.s` never appeared, on either side.

Everything above the Inductor lowering is a property of the model. Everything
below it is a property of the compiler stack the target selected — which is
exactly the boundary LM7 is meant to make visible.
