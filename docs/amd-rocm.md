# AMD GPU support

LM7 has initial support for local AMD GPUs through a ROCm-enabled PyTorch build
and TorchInductor. It does not install ROCm, GPU drivers, or PyTorch itself.

Follow the
[official AMD installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
for a PyTorch build compatible with the host ROCm release and GPU architecture.
Then install LM7 without replacing that build:

```bash
uv pip install -e ".[dev]"
```

PyTorch intentionally exposes ROCm devices through the `torch.cuda` API. LM7
uses `torch.version.hip` to distinguish AMD from NVIDIA and records the
normalized `gfx` architecture reported by the runtime.

## Verify the runtime

```bash
rocminfo | grep gfx
python - <<'PY'
import torch

print("available:", torch.cuda.is_available())
print("ROCm:", torch.version.hip)
print("GPU:", torch.cuda.get_device_name(0))
print("architecture:", torch.cuda.get_device_properties(0).gcnArchName)
PY
```

## What `lm7 targets` says about the card, and how much to trust it

`lm7 targets` names the ISA family and the formats the silicon computes:

```console
$ lm7 targets
Detected targets (2):
  amd:gfx942: AMD Instinct MI300X (CDNA 3), 192.0 GiB
    precision: native fp32, fp16, bf16, int8, fp8
```

The `gfx942` row has now been confirmed on a rented AMD Instinct MI300X VF:
generation `CDNA 3`, native `fp8`, absent `fp4`, and `fp8_format: fnuz` all
matched LM7's table. Other `gfx` entries are still documentation-derived until a
machine with that architecture runs LM7. See [AMD MI300X](amd-mi300x.md) for the
hardware run and [limitations](limitations.md#hardware-validation) for the
remaining gaps.

Two AMD-specific things the report gets right that a `gfx`-to-`sm` analogy would
get wrong:

- **`gfx` numbers do not order by capability.** `gfx1100` is larger than
  `gfx942` and is a consumer RDNA 3 part with no FP8 at all, while `gfx942` is a
  datacenter CDNA 3 part that has it. The two are separate product lines sharing
  a numbering space, so LM7 matches the string exactly rather than comparing it
  the way it compares `smXX`. An unrecognized `gfx` costs the label and nothing
  else.
- **`fp8` names two incompatible encodings.** CDNA 3 implements the `fnuz`
  variants — no infinities, one NaN, an exponent bias one greater than OCP's —
  so `torch.float8_e4m3fnuz` is the dtype that exists on `gfx942`, not the
  `torch.float8_e4m3fn` that `sm89`+ implements. CDNA 4 and RDNA 4 moved to the
  OCP encoding. `lm7 targets --json` reports which, under `fp8_format`:

  ```console
  $ lm7 targets --json | jq '.[0].capabilities.fp8_format'
  "fnuz"
  ```

  An FP8 number from a MI300X and one from an H100 were therefore not produced
  in the same format, and are not directly comparable.

`--json` also carries `cuda_build`, which answers whether the installed wheel
ships kernels for this chip or will fall back. The key keeps its CUDA-era name
because ROCm reports through the same `torch.cuda` API, but the answer matters
more here: a missing `sm_` target still runs by JIT-ing PTX, while a missing
`gfx` target fails at load with "no kernel image is available".

## Compile and test

```bash
python examples/rocm_mlp.py
python -m pytest -m rocm -q
```

`-m rocm` covers both AMD files: `tests/test_amd_integration.py` (Inductor
against eager) and `tests/test_amd_aot_integration.py` (packaging, manifest
provenance, cross-process reload, and the architecture guard). Both skip
themselves without a ROCm GPU, and the AOT file skips again without a ROCm
installation.

The equivalent API is:

```python
compiled = lm7.compile(
    model.eval(),
    target="amd",
    backend="inductor",
    transfers="automatic",
    fallback="error",
)
output = compiled(cpu_input)
```

Use an explicit architecture when required:

```python
compiled = lm7.compile(model, target="amd:gfx942")
```

## Benchmark

```bash
python benchmarks/gpu.py \
  --target amd \
  --model mlp \
  --backend eager inductor \
  --dtype float16
```

## Packaging an artifact

`aot_inductor` accepts an AMD target and writes a `.lm7` the way it does for
NVIDIA:

```bash
lm7 model export hf://HuggingFaceTB/SmolLM2-135M-Instruct artifacts/mi300x.lm7 \
  --target amd --backend aot_inductor
```

The wrapper build needs a ROCm installation on the host — the PyTorch ROCm wheel
links against `/opt/rocm` rather than bundling it — and LM7 checks for one
before packaging starts, so a missing install is a named refusal rather than an
error from inside `g++`. Set `ROCM_HOME` if it lives somewhere else. This is the
counterpart of the `[cuda-aot]` extra on NVIDIA and not the same problem: the
CUDA case is a *partial* toolkit where the wheel omits the compiler front end,
while ROCm either is or is not installed.

The manifest records `hip` and `gcn_architecture` where a CUDA artifact records
`cuda` and `compute_capability`. That distinction is load-bearing rather than
cosmetic: `torch.version.cuda` is `None` on ROCm and
`torch.cuda.get_device_capability()` returns `(9, 4)` on a `gfx942`, so writing
the CUDA fields would have produced a manifest claiming no runtime and an
architecture — `sm94` — that no NVIDIA part has ever had.

This path has run on the MI300X. The wrapper links against ROCm, the artifact
records `hip` and `gcn_architecture`, reloads through `lm7.load_artifact` in a
fresh process, and rejects architecture/runtime mismatches with AMD-specific
messages. See [AMD MI300X](amd-mi300x.md#an-aotinductor-artifact-and-the-bug-it-found)
and [artifact compatibility](aot-artifact-compatibility.md#scope).

## Scope

The AMD integration covers local single-GPU inference through TorchInductor,
AOTInductor packaging, and the `lm7 model serve` path. It has one hardware
validation point: a single-card MI300X VF in an SPX partition, not bare metal and
not multi-GPU.

Quantization is available only for the FP8 modes admitted by the MI300X run:
`fp8`, `fp8-dynamic`, and `fp8-dynamic-rowwise`. INT8 remains refused on AMD
because it measured about 10x slower than BF16, and NVFP4 remains refused
because CDNA 3 has no FP4 silicon and the weight-only result missed the accuracy
bar. See [quantization](quantization.md) and [AMD MI300X](amd-mi300x.md#quantization-and-why-none-of-it-changes-the-gate).

Multi-GPU execution, CPX partitioning, CI on physical AMD hardware, quantized
serving, larger-model serving, and long-context decode remain unvalidated. The
vLLM ROCm handoff starts and answers on the MI300X, but its throughput is still
unmeasured.

MIGraphX was evaluated as an AMD-specific compiler path and was not adopted:
where it ran it was slower than eager and Inductor, and it failed the causal-LM
shapes. See [AMD MIGraphX evaluation](amd-migraphx.md).
