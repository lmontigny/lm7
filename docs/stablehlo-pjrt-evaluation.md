# StableHLO and PJRT evaluation

LM7 reaches OpenXLA through PyTorch/XLA, and PyTorch/XLA has retreated to TPU:
its PyPI wheels are Linux x86-64 with no CUDA variants, XLA:CUDA is deprecated
and warns on initialization since 2.8, and no ROCm wheels exist. That makes
`openxla` a TPU-only backend in LM7 — see [Google TPU support](google-tpu.md).

OpenXLA itself is not TPU-only. [ZML](../notes/ZML_details.md) compiles Zig to MLIR,
lowers through OpenXLA, and dispatches through **PJRT**, which is how one model
definition reaches CPU, NVIDIA, AMD, TPU, and AWS Neuron. PJRT — not
PyTorch/XLA — is the portable boundary, and a vendor can supply a plugin without
the frontend changing.

This evaluation asked whether LM7 can reach that same boundary while keeping its
PyTorch frontend, by lowering a captured `ExportedProgram` to StableHLO and
handing the result to a PJRT client. It can, and `stablehlo` is now a registered
export backend on the strength of the measurements below.

## The question that decides it

LM7 already has two AOT artifact formats, and each gives up something:

| Artifact | Runs without PyTorch | Vendor coverage |
| --- | --- | --- |
| AOTInductor `.pt2` | No — `aoti_load_package` is a PyTorch API | CPU, Apple, NVIDIA |
| OpenVINO IR | Yes | Intel CPU only |
| **StableHLO + PJRT** | **Yes** | **Whatever PJRT plugins exist** |

If the third row holds, it is the first LM7 artifact that is simultaneously
PyTorch-free and vendor-neutral. That is what the slice below measures, and the
reason the harness refuses to run its execute stage anywhere PyTorch is
installed.

## Candidate integration paths

1. **`torch.export` → `torch_xla.stablehlo` → PJRT plugin.** Reuses the
   `ExportedProgram` LM7 already produces for every artifact. Evaluated here.
2. **`torch.export` → torch-mlir → StableHLO → PJRT plugin.** Removes the
   PyTorch/XLA build-time dependency. Not evaluated.
3. **PJRT client embedded in LM7.** Would replace `torch_xla` at run time with a
   direct plugin loader, as ZML does. Out of scope until path 1 or 2 is proven.

## What the slice measured

Host: Intel Core i7-8086K, WSL2, PJRT CPU plugin. Capture environment: torch
2.9.1+cpu with torch_xla 2.9.0. Execution environment: jax 0.11.0 as the PJRT
client, **with PyTorch not installed**.

### Conversion (torch + torch_xla)

| Model | `torch.export` | Lower to StableHLO | Save | StableHLO | Artifact |
| --- | --- | --- | --- | --- | --- |
| MLP, fp32, batch 8 | 56 ms | 302 ms | 432 ms | 1,434 chars | 8 files, 7.8 KiB |
| SmolLM2-135M, fp32, 5 tokens | 6.0 s | 15.4 s | 14.8 s | 304,460 chars | 281 files, 622.5 MiB |

Executed through torch_xla's own runtime as a control, both matched eager: max
absolute difference 0.0 for the MLP and 4.1e-05 for SmolLM2.

### Execution (PJRT client, no PyTorch installed)

| Model | PJRT compile | Weight load | First call | Median | Max abs diff | Top-1 |
| --- | --- | --- | --- | --- | --- | --- |
| MLP | 124 ms | 1.1 ms | 3.4 ms | 0.03 ms | 6.0e-08 | match |
| SmolLM2-135M | 1.14 s | 581 ms | 646 ms | 77.9 ms | 9.2e-05 | match |

SmolLM2 predicted token 7042 (`' Paris'`) from `"The capital of France is"`,
the same token eager PyTorch produces, in a process where `importlib.util.find_spec("torch")`
returns `None`.

**The path works.** A model captured from PyTorch ran on a PJRT client with no
PyTorch present and stayed numerically faithful to eager.

## The same bytes on a second vendor

The vendor-neutrality claim was then tested rather than assumed. The **exact
same artifact**, unmodified, was handed to the CUDA PJRT plugin on an RTX 4070
SUPER (sm89), still with PyTorch absent:

| Plugin | Compile | Weight load | First call | Median | Top-1 |
| --- | --- | --- | --- | --- | --- |
| CPU | 1.14 s | 581 ms | 646 ms | 77.9 ms | match |
| CUDA (sm89) | 8.79 s | 3.20 s | 310 ms | 6.0 ms | match |
| **TPU v6e** | 2.02 s | 188 ms | 11.6 ms | **1.32 ms** | match |

One file, three vendors, 59x faster on the TPU than on the CPU and 4.5x faster
than on an RTX 4070 SUPER, with no re-export. That is the property no other LM7
artifact has: an AOTInductor package is built for one device, and OpenVINO IR is
Intel-only.

The TPU row closes what was previously this document's largest open question —
it was written as "ROCm, TPU, and Neuron are not testable on this host, so their
support is inferred from PJRT's design rather than observed." TPU is now
observed. The artifact was not rebuilt or adapted; the same
`functions/forward.bytecode` exported on an x86 host went to a third vendor's
plugin in a process where `importlib.util.find_spec("torch")` returns `None`,
and still predicted `' Paris'`.

Numerics differ by device, and the spread is wider than two vendors suggested.
Against the same CPU eager reference, on fp32 logits:

| Plugin | Max abs difference | Top-1 |
| --- | --- | --- |
| CPU | 9.2e-05 | match |
| CUDA (sm89) | 2.3e-02 | match |
| TPU v6e | **2.4e-01** | match |

Forcing `NVIDIA_TF32_OVERRIDE=0` did not close the CUDA gap (2.5e-02), so that
one is ordinary cross-device fp32 reassociation rather than TF32 — the same
order as the 0.059 fp16 difference recorded for the NVIDIA AOTInductor path.

The TPU's 0.24 is a different thing and much larger: XLA lowers fp32 matmuls to
bf16 passes on TPU by default, and 30 transformer layers compound it. It is not
a property of the artifact — the identical figure appears whether the payload is
executed through raw PJRT with no PyTorch present or through torch_xla's own
runtime as a control. The predicted token survives it here, but 0.24 on logits
is close enough to mattering that it should not be assumed for another model.
See [Google TPU](google-tpu.md) for the precision control and what it costs.

**Validate per target before trusting an artifact.** Three vendors now, three
different answers to "how close is close".

## Artifact layout

`save_as_stablehlo` writes a format that is legible without the producing
framework, which is most of why this is interesting:

```text
functions/forward.mlir       StableHLO text
functions/forward.bytecode   what the PJRT client compiles
functions/forward.meta       JSON: input/output signatures, dtypes, dynamic_dims,
                             pytree spec, and input_locations
data/<parameter name>        one .npy per weight
constants/<n>                baked constants
```

`input_locations` is the load-bearing piece. Each position is labelled
`parameter`, `constant`, or `input_arg`, so a loader can rebuild the flat
argument list with no model definition — that is what
`benchmarks/stablehlo_pjrt.py` does in 15 lines. Weights being separate `.npy`
files also matches the separation LM7's own [ZML notes](../notes/ZML_details.md) argue
for: the symbolic graph does not have to carry materialized weights.

## The registered backend

```python
lm7.export(model, args=(example,), target="cpu", backend="stablehlo", output="model.lm7")
```

Three things distinguish it from the other export backends:

- **No vendor gate.** `aot_inductor` and `openvino` reject targets they were not
  validated for. This payload is target-independent — the PJRT plugin is chosen
  by whoever loads it — so the backend does not gate, and the manifest records
  `"device_bound": false`.
- **Export-only.** `supports()` returns False for compile requests with a
  message pointing at `lm7.export`. Compiling in-process through PyTorch/XLA is
  what `openxla` already does.
- **The payload is a zip.** `save_as_stablehlo` writes a directory of roughly
  280 files for a 135M model, and an LM7 manifest records one payload name and
  one checksum, so the tree is stored (uncompressed) as
  `compiled_model.stablehlo.zip`.

Loading through `lm7.load_artifact` needs PyTorch/XLA, because turning StableHLO
back into a torch callable is what PyTorch/XLA does. The PyTorch-free route is
to unpack the zip and hand `functions/forward.bytecode` to a PJRT client, which
is what `benchmarks/stablehlo_pjrt.py` and the integration test do.

## Constraints and open questions

None of these are fatal, but all of them are real:

- **Version coupling.** `torch_xla` is ABI-tied to a matching PyTorch; 2.9.0
  needs torch 2.9, while LM7 development currently runs torch 2.13. The two
  cannot share an environment, which is why the harness has separate `export`
  and `execute` commands, and why the backend's own lowering tests are gated on
  `torch_xla` being importable. This is the same drift that affects the
  `tensorrt` extra.
- **Keyword capture is not lowerable.** `torch_xla` raises "Export to stablehlo
  doesnt support kwargs yet." for a program captured with keyword inputs, so the
  backend rejects those up front and `export_hf_model` feeds this backend its
  tensors positionally.
- **The payload duplicates the source program.** An `.lm7` holds both
  `exported_program.pt2` and the StableHLO zip, so a 135M model lands at about
  1.1 GiB. That is inherent to the current artifact design rather than to this
  backend.
- **The conversion still needs PyTorch/XLA.** Path 1 routes *around*
  PyTorch/XLA at run time but still depends on it at build time, so it does not
  escape the dependency that motivated the evaluation. Path 2 (torch-mlir) is
  the way out and is unevaluated.
- **Three plugins were exercised, not five.** CPU, CUDA and TPU are measured
  above. ROCm and Neuron remain inferred from PJRT's design rather than
  observed. The TPU measurement also cost nothing in portability work — the
  artifact was exported on x86 and handed over unmodified — which is the
  strongest evidence so far that the remaining two are a matter of access
  rather than of engineering.
- **The TPU round trip cannot run under pytest.** A TPU chip is claimed by one
  process, and merely probing the runtime claims it, so
  `test_payload_runs_without_pytorch` cannot spawn a PJRT child on a TPU host
  and skips there with that reason. The coverage comes from
  `benchmarks/stablehlo_pjrt.py`, whose export and execute stages are separate
  processes by design.
- **Dynamic shapes were not exercised.** The metadata carries a `dynamic_dims`
  field per input and `torch_xla.stablehlo` exposes
  `exported_program_has_symbolic_input_shape`, so the bounded sequence dimension
  LM7 now captures for causal LMs is plausibly expressible — but unproven.
- **Runtime dependency question.** Executing needs *a* PJRT client. Using JAX as
  that client, as this harness does, trades a PyTorch dependency for a JAX one.
  A real deployment story wants the plugin loaded directly, as ZML does.

## Validation commands

Two environments, because one cannot hold both PyTorch 2.13 and torch_xla 2.9:

```bash
python3 -m venv .venv-xla
.venv-xla/bin/python -m pip install torch==2.9.* --index-url https://download.pytorch.org/whl/cpu
.venv-xla/bin/python -m pip install torch_xla==2.9.0 numpy transformers safetensors

python3 -m venv .venv-pjrt          # deliberately no torch
.venv-pjrt/bin/python -m pip install jax numpy
```

```bash
PJRT_DEVICE=CPU .venv-xla/bin/python benchmarks/stablehlo_pjrt.py export \
  --model smollm2 --output artifacts/stablehlo/smollm2 \
  --report artifacts/benchmarks/stablehlo-export.json

PJRT_DEVICE=CPU .venv-pjrt/bin/python benchmarks/stablehlo_pjrt.py execute \
  artifacts/stablehlo/smollm2 \
  --report artifacts/benchmarks/stablehlo-execute.json
```

The `execute` stage fails loudly if PyTorch is importable, so a passing run is
evidence about the artifact rather than about a convenient environment.

On a TPU VM the same two-environment split applies, with `jax[tpu]` as the
torch-free client. Nothing about the artifact or the commands changes — only
which plugin `get_backend()` finds:

```bash
uv venv --python 3.12 .venv-pjrt-tpu       # deliberately no torch
uv pip install --python .venv-pjrt-tpu/bin/python "jax[tpu]" numpy

.venv-tpu/bin/python benchmarks/stablehlo_pjrt.py export \
  --model smollm2 --output artifacts/stablehlo/smollm2

.venv-pjrt-tpu/bin/python benchmarks/stablehlo_pjrt.py execute \
  artifacts/stablehlo/smollm2
```

The export and execute stages must not overlap: both want the TPU, and only one
process can have it.

## Status

Evaluated, working, and **now a registered export backend** — see
[`lm7.export(backend="stablehlo")`](../README.md#5-export-an-artifact). What
remains open:

1. Re-run the capture with a **bounded dynamic sequence dimension** to see
   whether it survives the lowering.
2. Evaluate **torch-mlir** as the lowering path, which is what would let the
   backend lower in the normal LM7 environment rather than an ABI-matched one.
3. Exercise the **ROCm, TPU, and Neuron plugins**, none of which are testable on
   this host.

## References

- [ZML technical notes](../notes/ZML_details.md) — PJRT as the portable boundary
- [Google TPU support](google-tpu.md) — why `openxla` is TPU-only in LM7
- [JIT vs. AOT](jit-vs-aot.md) — the artifact levels this would extend
- [PJRT examples, OpenXLA](https://openxla.org/xla/pjrt/examples)
- [PyTorch/XLA releases](https://github.com/pytorch/xla/releases)
