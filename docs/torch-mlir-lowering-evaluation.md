# torch-mlir as the StableHLO lowering path

The [`stablehlo` backend](stablehlo-pjrt-evaluation.md) produces the only LM7
artifact that is both PyTorch-free and vendor-neutral, but it lowers through
`torch_xla`, which is ABI-tied to a matching PyTorch. That single dependency
causes every remaining limitation of the backend:

- it cannot share an environment with the PyTorch LM7 develops against, so it
  installs into its own virtualenv;
- its lowering tests are therefore gated out of CI, leaving only packaging and
  validation covered;
- and routing *around* PyTorch/XLA at run time still depends on PyTorch/XLA at
  build time, which is the dependency the whole exercise set out to escape.

[torch-mlir](https://github.com/llvm/torch-mlir) is the other route from
`torch.export` to StableHLO. This evaluation asks whether swapping to it removes
those limitations, and what it costs.

## The decisive result

**torch-mlir is not ABI-pinned to PyTorch.** It installs and imports against
torch 2.13.0 — the exact version LM7 develops on — where `torch_xla` 2.9 requires
torch 2.9:

```bash
python3 -m venv .venv-tm
.venv-tm/bin/python -m pip install torch==2.13.* --index-url https://download.pytorch.org/whl/cpu
.venv-tm/bin/python -m pip install --pre torch-mlir \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

That alone answers the question the evaluation was posed to answer: a
torch-mlir-based lowering could live in the normal LM7 environment and be
covered by CI. `torch_mlir.fx.export_and_import` also takes `dynamic_shapes`
as a first-class parameter, which is the other item PR #11 left open.

## One obstacle, and the fix

torch-mlir emits every multi-element weight as a `DenseResourceElementsAttr`,
and XLA refuses to compile those:

```
UNIMPLEMENTED: Only dense elements attr are supported
note: see current operation: %0 = "stablehlo.constant"()
  <{value = dense_resource<torch_tensor_4_torch.float32> : tensor<4xf32>}>
```

The choice is hardcoded — there is no option, and the documented
`FxImporterHooks.resolve_literal` override requires constructing IR ops by hand
from outside the importer.

The workable fix needs no private API. MLIR prints resource blobs as hex in a
trailing `{-# dialect_resources #-}` section, and `dense<"0x...">` is valid
assembly, so the blobs can be inlined textually before the module is handed to
PJRT. Two details matter: resource names contain dots (`torch_tensor_4_torch.float32`),
so a `\w+` pattern silently matches nothing; and each blob carries a four-byte
alignment prefix that the `dense<>` form omits.

## Measurements

Host: Intel Core i7-8086K, RTX 4070 SUPER, WSL2. Lowering in torch 2.13.0 +
torch-mlir; execution in a PyTorch-free environment through a PJRT client.

| | torch_xla (shipped) | torch-mlir (this evaluation) |
| --- | --- | --- |
| Works with LM7's torch | no — needs torch 2.9 | **yes — torch 2.13** |
| `dynamic_shapes` support | via `ShapeProfile` plumbing | first-class parameter |
| MLP lower time | 302 ms | 330 ms |
| SmolLM2 lower time | 15.4 s | 27.8 s + 10.8 s inlining |
| SmolLM2 artifact | 622 MiB | **1,026 MiB** |
| SmolLM2 PJRT parse + compile | 1.14 s | **29.4 s** |
| SmolLM2 execute (CPU plugin) | 77.9 ms median | 50.0 ms median |
| Weight storage | binary `.npy` per parameter | hex text inside the module |
| Payload metadata | `forward.meta` with `input_locations` | none |

Both models round-trip correctly with PyTorch absent from the executing
process. The MLP ran on the **CUDA** plugin, matching eager to 2.1e-04 — the
cross-device fp32 band established in the
[StableHLO evaluation](stablehlo-pjrt-evaluation.md). SmolLM2-135M ran on the
CPU plugin and predicted token 7042 (`' Paris'`), the same token as eager, at
6.2e-05. **Operator coverage is not the problem**: a real causal LM lowers and
executes end to end.

## The two costs

**Artifact size, and the load time that follows from it.** torch-mlir bakes
weights into the module rather than writing them alongside. Inlining them as hex
doubles every weight byte, so SmolLM2-135M lands at 1,026 MiB against
`torch_xla`'s 622 MiB — 1.65x. The consequence at load time is worse than the
size suggests: PJRT spends **29.4 s** parsing and compiling a gigabyte of MLIR
text where the `torch_xla` payload takes 1.14 s, because the weights arrive as
assembly to be parsed rather than as `.npy` buffers to be mapped. Writing MLIR
bytecode instead of text would avoid both, but bytecode preserves resources,
which is the form XLA rejects. Resolving that is what would make this a straight
win.

**The calling convention is model-dependent, and undescribed by torch-mlir.**
For the MLP,
torch-mlir produced `@main(%arg0)` — one runtime input, every weight baked —
which is far cleaner than `torch_xla`'s 277-argument signature. That advantage
does not survive a real model. SmolLM2 lowered to:

```mlir
func.func @main(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>,
                %arg2: tensor<1x5xi64>, %arg3: tensor<1x5xi64>)
```

The two leading `tensor<32xf32>` arguments are lifted buffers —
`model.model.rotary_emb.inv_freq` and `original_inv_freq`, the RoPE inverse
frequencies — which `torch.export` lifts as inputs rather than constants.
A loader must supply them ahead of the real inputs, and **torch-mlir writes no
metadata saying so**. `torch_xla`'s `forward.meta` labels every position as
`parameter`, `constant`, or `input_arg` precisely so a non-PyTorch loader can
rebuild the call. Adopting torch-mlir means LM7 would have to generate that
manifest itself.

`benchmarks/torch_mlir_lowering.py` shows this is tractable — roughly forty
lines reading `graph_signature.input_specs`, saving lifted buffers beside the
module, and recording the order. The rule it encodes is that torch-mlir bakes
parameters and tensor constants but leaves buffers lifted, which is torch-mlir's
choice rather than anything the ExportedProgram states, so the harness checks
its own metadata against the real `@main` arity and refuses to emit a payload no
loader could call. With that manifest in place the existing PyTorch-free runner
in `benchmarks/stablehlo_pjrt.py` executes both lowering paths unchanged.

## Recommendation

**Do not swap the backend over yet, and do not add a second lowering path on
these terms.** The version-pin fix is real and valuable, but taken as-is the
trade is a 1.65x larger artifact, ~3x slower export, and a calling convention
LM7 would have to document itself — in exchange for CI coverage and one fewer
pinned dependency.

The sequence that makes it a clear win, in order:

1. **Emit bytecode with inlined constants.** This removes the size penalty and
   most of the inlining time. It needs the constants materialized as dense
   *before* serialization rather than rewritten as text afterwards, which is
   what `resolve_literal` is for.
2. **Generate the payload manifest from `graph_signature`.** LM7 already reads
   `input_specs` elsewhere; emitting a `forward.meta` equivalent would make the
   two lowering paths produce interchangeable artifacts and let the existing
   PJRT loader and harness work unchanged.
3. **Then** make the lowering path selectable, defaulting to torch-mlir where
   available, and move the lowering tests into CI.

Step 1 is the one that decides it. If constants can be materialized dense before
serialization, torch-mlir wins outright; if not, the size penalty is structural
and `torch_xla` remains the better payload producer despite its pin.

## Reproduction

```bash
python3 -m venv .venv-tm
.venv-tm/bin/python -m pip install torch==2.13.* --index-url https://download.pytorch.org/whl/cpu
.venv-tm/bin/python -m pip install --pre torch-mlir \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
.venv-tm/bin/python -m pip install transformers safetensors

PJRT_DEVICE=CPU .venv-tm/bin/python benchmarks/torch_mlir_lowering.py lower \
  --model smollm2 --output artifacts/torch-mlir/smollm2

.venv-pjrt/bin/python benchmarks/stablehlo_pjrt.py execute \
  artifacts/torch-mlir/smollm2
```

## References

- [StableHLO and PJRT evaluation](stablehlo-pjrt-evaluation.md) — the backend this would replace the lowering of
- [torch-mlir](https://github.com/llvm/torch-mlir) · [release wheels](https://github.com/llvm/torch-mlir-release)
- [Torch Export to StableHLO](https://docs.pytorch.org/xla/master/features/stablehlo.html) — the `torch_xla` path
