# DeepSeek coverage

`deepseek-ai/deepseek-coder-1.3b-instruct` is the fifth causal LM in LM7's
[tested model coverage](../README.md#tested-model-coverage), and the one taken
through the widest spread of backends rather than through Inductor alone. It
needed no source changes: DeepSeek ships this checkpoint as a
`LlamaForCausalLM`, so `AutoModelForCausalLM.from_pretrained` loads it with no
`trust_remote_code` and no custom modelling file.

Why this checkpoint: it is DeepSeek's smallest ungated release (1.3B, ~2.6 GB of
safetensors), it needs no license acceptance or Hugging Face token, and its
config declares the older `rope_scaling` spelling (`{"type": "linear"}`), which
Transformers still accepts. The larger DeepSeek releases that carry the
architecture DeepSeek is actually known for — multi-head latent attention and
fine-grained MoE, from V2 onward — start at 16B parameters, which does not fit
the 12 GB GPU this matrix was measured on.

```bash
lm7 model run hf://deepseek-ai/deepseek-coder-1.3b-instruct \
  --prompt "The capital of France is" --target auto --backend auto
```

## Measured matrix

One host: RTX 4070 SUPER (Ada, `sm89`, 12 GB) plus an AVX2 x86-64 CPU, torch
2.13 for the CPU/CUDA backends and the per-backend environments described in
[development.md](development.md) for the rest. Every row tokenizes
`"The capital of France is"`, compares against the same model run eagerly, and
checks the predicted next token. **Every row below predicted `' Paris'`, matching
eager.**

| Backend | Target | dtype | Compile / export | Max abs logit diff | Cosine |
| --- | --- | --- | --- | --- | --- |
| `inductor` (JIT) | `nvidia:sm89` | float16 | 37.3 s cold, 8.1 s warm cache | 0.44 | 0.999997 |
| `inductor` (JIT) | `cpu:x86_64` | float32 | 36.1 s | 0.000048 | 1.000000 |
| `export` | `cpu:x86_64` | float32 | 46.9 s | 0.000000 | 1.000000 |
| `aot_inductor` | `nvidia:sm89` | float16 | 120.1 s | 0.44 | 0.999997 |
| `aot_inductor` | `cpu:x86_64` | float32 | 409.0 s | 0.000048 | 1.000000 |
| `aot_inductor`, dynamic `(1, 256)` | `nvidia:sm89` | float16 | 176.9 s | 0.44 | 0.999998 |
| `openvino` | `cpu:x86_64` | float32 | 147.8 s | 0.000101 | 1.000000 |
| `stablehlo` | `cpu:x86_64` | float32 | 251.6 s | 0.000053 | 1.000000 |
| `executorch` | `cpu:x86_64` | float32 | 315.0 s | 0.000187 | 1.000000 |
| `tensorrt` (JIT) | `nvidia:sm89` | float16 | 56.8 s engine build | 0.66 | 0.999993 |
| `tvm` (JIT) | `cpu:x86_64` | float32 | 131.8 s | 0.000206 | 1.000000 |

Artifact sizes, and the cost of *not* recompiling in the loading process:

| Backend | Target | Artifact | Reload + first call |
| --- | --- | --- | --- |
| `export` | `cpu` | 5.0 GiB | 1292 ms |
| `aot_inductor` | `nvidia:sm89` | 5.0 GiB | 343 ms |
| `aot_inductor`, dynamic `(1, 256)` | `nvidia:sm89` | 5.0 GiB | 56 ms |
| `aot_inductor` | `cpu` | 10.0 GiB | 655 ms |
| `openvino` | `cpu` | 10.0 GiB | 800 ms |
| `stablehlo` | `cpu` | 10.0 GiB | 8450 ms |
| `executorch` | `cpu` | 10.0 GiB | 1228 ms |

Two things worth reading off that table. The float32 CPU artifacts are twice the
float16 GPU ones for the same weights, and every artifact carries the weights
*twice* — once in `exported_program.pt2` and once in the backend's own payload —
which is why a 2.6 GB checkpoint becomes a 5 GiB artifact. And the dynamic
capture is not the slow one to load: it reloaded fastest of them all while also
being the only artifact that serves prompt lengths other than the captured five
tokens.

## What float16 divergence looks like on this model

The 0.44 maximum logit difference on CUDA float16 looks alarming next to
SmolLM2-135M, which was re-measured on this same host at 0.059. Part of that gap
is simply scale: DeepSeek's peak absolute logit on this prompt is 68.4 against
SmolLM2's 25.5, so the same relative error shows up as a larger absolute one.

That is why the parity assertion in `tests/test_hf_integration.py` —
`rtol=0.02, atol=0.075` — needed no change. Because it scales with magnitude,
DeepSeek consumes 0.79 of the allowed tolerance where SmolLM2 consumes 0.61: a
little wider, comfortably inside, and no per-model special case. Scale does not
account for the whole difference (7.5x the absolute error against 2.7x the logit
magnitude), so this model does diverge somewhat more than SmolLM2 under float16 —
just not enough to matter for the predicted token, which matched on every backend.

On CPU float32 the difference is five orders of magnitude smaller, and plain
`torch.export` is bit-exact.

## TensorRT: runs correctly, just outside the existing assertion

TensorRT builds an engine for this model in 57 s and predicts the same token,
with a last-token p99 absolute error of 0.117 — inside the 0.15 bar that
`tests/test_tensorrt_integration.py` applies. It misses the *other* half of that
assertion: last-token cosine similarity came out at 0.999895 against a required
0.9999, a miss in the fifth decimal place.

That bar was set from SmolLM2's measured behaviour, and this model's FP16
divergence is genuinely slightly wider, so DeepSeek is **not** added to the
TensorRT regression test and the bar is **not** loosened to accommodate it. The
honest summary is that TensorRT works for this model and the existing test's
thresholds are SmolLM2's, not universal constants.

## Two backends need positional inputs

A Hugging Face causal LM is naturally called with keywords, and two backends
refuse that — at different points. `tvm` refuses at compile time:

```
The tvm backend captures positional inputs only; got keyword inputs
attention_mask, input_ids, use_cache.
```

`executorch` exports happily but refuses at *call* time, on the reloaded
artifact:

```
An ExecuTorch method takes positional tensors only; got keyword inputs
attention_mask, input_ids.
```

Neither is a defect and both errors name their fix. `lm7 model export` already
wraps the model in `lm7.huggingface._LogitsOnly`, so the export side is handled;
what a caller has to remember is that a reloaded ExecuTorch artifact takes
`artifact(input_ids, attention_mask)` positionally, in `_LogitsOnly.forward`'s
order, and that reaching `tvm` at all means wrapping the model yourself. That was
the only rough edge in the whole matrix.

## Quantization

The TorchAO weight-only path was measured against an unquantized baseline on four
prompts, the same bar [quantization.md](quantization.md) applies to every other
model:

| Mode | Target | Top-1 agreement | Max logit diff | Storage |
| --- | --- | --- | --- | --- |
| `int8` | `nvidia:sm89`, bfloat16 | 4/4 | 1.66 | 2693 → 1480 MB (1.82x) |
| `fp8` | `nvidia:sm89`, bfloat16 | 4/4 | 2.12 | 2693 → 1883 MB (1.43x) |
| `int8` | `cpu:x86_64`, float32 | 4/4 | 0.67 | 5386 → 1746 MB (3.09x) |
| `nvfp4` | `nvidia:sm89`, bfloat16 | 4/4 | **9.25** | 2693 → 948 MB (2.84x) |

The OpenVINO export path — NNCF weight compression on the IR, an entirely
separate mechanism — was measured the same way and also passes, with the smallest
logit movement of the three models tried through it:

| Path | Top-1 agreement | Max logit diff | IR weights |
| --- | --- | --- | --- |
| `--backend openvino --quantize int8` | 4/4 | 0.79 | 1348 MB, against 5386 MB of FP32 weights (3.99x) |

Because that export is static-shape, its four prompts are four five-token
prompts, not the mixed-length set used for the TorchAO rows. The model is
admitted into `VALIDATED_OPENVINO_INT8`.

`int8` and `fp8` are admitted into `VALIDATED_WEIGHT_ONLY`. **`nvfp4` is not**,
despite its 4/4: a 9.25 logit difference is more than twice the admitted
Llama-3.2-1B figure and wider than the differences that came with the SmolLM2 and
LFM2.5 rejections, so passing the top-1 check reads as four prompts being too
coarse an instrument rather than as the mode being safe here. See
[quantization.md](quantization.md#nvfp4-costs-much-more-accuracy-than-8-bit).

## Not verified here

Nothing below failed — none of it was reachable on this host, and none of it is
claimed:

| Backend / target | Why not |
| --- | --- |
| `onnxruntime` | Not installed in any local environment. Its `numpy<2.5` pin needs an environment of its own, and a 1.3B model would also test the ONNX 2 GB protobuf limit, which is untried. |
| `iree_vulkan`, `litert` | Not installed locally. |
| `openxla` / `tpu`, `tenstorrent` | Need hardware that is not present. |
| `apple` (MPS) | Needs a Mac. The other four validated models were checked there; this one was not. |
| `intel:npu` | No NPU present, as [intel-npu.md](intel-npu.md) already records. |
| Compiled generation (`lm7 model generate`) | The static KV-cache decode path was not exercised for this model; only the prefill forward pass above was. |
