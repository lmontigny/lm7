# StableHLO and PJRT evaluation

LM7 reaches OpenXLA through PyTorch/XLA, and PyTorch/XLA has retreated to TPU:
its PyPI wheels are Linux x86-64 with no CUDA variants, XLA:CUDA is deprecated
and warns on initialization since 2.8, and no ROCm wheels exist. That makes
`openxla` a TPU-only backend in LM7 — see [Google TPU support](google-tpu.md).

OpenXLA itself is not TPU-only. [ZML](ZML_details.md) compiles Zig to MLIR,
lowers through OpenXLA, and dispatches through **PJRT**, which is how one model
definition reaches CPU, NVIDIA, AMD, TPU, and AWS Neuron. PJRT — not
PyTorch/XLA — is the portable boundary, and a vendor can supply a plugin without
the frontend changing.

This evaluation asks whether LM7 can reach that same boundary while keeping its
PyTorch frontend, by lowering a captured `ExportedProgram` to StableHLO and
handing the result to a PJRT client.

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
files also matches the separation LM7's own [ZML notes](ZML_details.md) argue
for: the symbolic graph does not have to carry materialized weights.

## What blocks a registered backend

None of these are fatal, but all of them are real:

- **Version coupling.** `torch_xla` is ABI-tied to a matching PyTorch; 2.9.0
  needs torch 2.9, while LM7 development currently runs torch 2.13. The two
  cannot share an environment, which is why the harness has separate `export`
  and `execute` commands. This is the same drift that affects the `tensorrt`
  extra, and it means an `lm7.export(backend="stablehlo")` could not be
  exercised by the default dev environment today.
- **The conversion still needs PyTorch/XLA.** Path 1 routes *around*
  PyTorch/XLA at run time but still depends on it at build time, so it does not
  escape the dependency that motivated the evaluation. Path 2 (torch-mlir) is
  the way out and is unevaluated.
- **Only the CPU plugin was exercised.** The claim "vendor-neutral" rests on
  PJRT plugins this evaluation did not run. A CUDA plugin is installable and
  testable on the local RTX 4070; ROCm, TPU, and Neuron are not testable here.
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

## Status

Evaluated and working on CPU; **not** a registered LM7 backend. The next
decision-relevant experiments, in order:

1. Run the same artifact through the **CUDA PJRT plugin** on the local sm89 GPU.
   That is the first real test of the vendor-neutrality claim, and it is
   testable on existing hardware.
2. Re-run the capture with a **bounded dynamic sequence dimension** to see
   whether it survives the lowering.
3. Evaluate **torch-mlir** as the lowering path, which is what would let a
   `stablehlo` export backend live in the normal LM7 environment.

## References

- [ZML technical notes](ZML_details.md) — PJRT as the portable boundary
- [Google TPU support](google-tpu.md) — why `openxla` is TPU-only in LM7
- [JIT vs. AOT](jit-vs-aot.md) — the artifact levels this would extend
- [PJRT examples, OpenXLA](https://openxla.org/xla/pjrt/examples)
- [PyTorch/XLA releases](https://github.com/pytorch/xla/releases)
