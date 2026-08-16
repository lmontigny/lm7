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
present before a target-specific GPU backend starts emitting PTX or binaries.

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

## Extending the same run on NVIDIA

Run the same shape on the NVIDIA machine, changing only the target and output:

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

The expected structure is the same at the top:

- `exported_graph.py`
- `graph_signature.txt`
- `fx_graph_*.py`
- `ir_pre_fusion.txt`
- `ir_post_fusion.txt`

The lower layers are where NVIDIA becomes different. A CUDA build can add CUDA
source, PTX, assembly, and CUBIN files under `debug/` or inside
`debug/package/`, depending on what the active PyTorch/Inductor build emits. LM7
indexes those files as:

| Indexed level | File kind |
| --- | --- |
| `generated_code` | `.cpp`, `.c`, `.cu`, `.py` |
| `device_code` | `.ptx` |
| `machine_code` | `.s`, `.asm` |
| `device_binary` | `.cubin` |

After copying the artifact back to the Mac, this remains safe:

```bash
lm7 artifact inspect artifacts/ir-inspection/nvidia-mlp.lm7
find artifacts/ir-inspection/nvidia-mlp.lm7/debug -type f | sort
```

Loading or executing the NVIDIA artifact is different: AOTInductor and TensorRT
payloads are tied to the runtime and GPU architecture that built them. Inspect
on the Mac; execute on a compatible NVIDIA host.

## What to compare tomorrow

When adding the NVIDIA run, compare these points against the Mac CPU baseline:

- whether the exported ATen graph is identical;
- whether the graph signature is identical;
- where Inductor keeps GEMMs as extern kernels versus generated kernels;
- whether fusion changes between CPU and NVIDIA;
- which CUDA source, PTX, assembly, or CUBIN files appear;
- which architecture and runtime fields `lm7 artifact inspect` reports.

If the exported graph differs, the model capture changed. If only the lower
files differ, the same PyTorch program reached different compiler stacks, which
is exactly the boundary LM7 is meant to make visible.
