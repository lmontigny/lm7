# Quantization

Quantization stores or computes a tensor in fewer bits than the model was trained
in. There are two halves to it, and **LM7 currently implements only the first**.

## Weight versus activation quantization

**Weight quantization** shrinks the stored parameters. Weights are converted
once, up front, and dequantized on the fly inside the matmul, so the arithmetic
still happens in a higher precision. What you gain is memory footprint and memory
bandwidth; what you avoid is calibration, because a weight tensor's range is
known statically.

**Activation quantization** also narrows the tensors flowing between layers, so
the matmul itself executes in low precision. That is where the larger speedups
come from, because it cuts arithmetic work rather than just bytes moved. The cost
is that activation ranges depend on the input, so you need calibration data or a
quantization-aware recipe to choose per-tensor scales, and accuracy degrades more
readily. **LM7 does not do this today.**

Everything below is therefore *weight-only*: activations, and the accumulation
inside each matmul, stay in BF16.

## Supported data types

| `--quantization` | Weight storage | Compute dtype | Requires |
| --- | --- | --- | --- |
| `none` (default) | as loaded | FP32 / FP16 / BF16 | nothing |
| `int8-weight-only` | INT8 | BF16 | NVIDIA GPU |
| `fp8-weight-only` | FP8 | BF16 | NVIDIA Ada (`sm89`), Hopper (`sm90`), or newer |

Both modes force BF16 compute. `--dtype` must be `auto` or `bfloat16`, and `auto`
resolves to BF16 whenever quantization is on. `--backend` must be `auto` or
`inductor`. Anything else raises `UnsupportedModelError` rather than silently
degrading — including a non-NVIDIA target, an FP8 request on pre-Ada hardware,
and any model id outside the validated list.

## Which layers are converted

This differs between the two modes, and it changes the memory saving you should
expect:

- `int8-weight-only` converts **every `nn.Linear` except `lm_head`** — attention
  projections included.
- `fp8-weight-only` converts **only the MLP linears** (`.mlp.` in the module
  path), leaving attention and `lm_head` in BF16.

`lm_head` is left alone in both cases because it is both large and
accuracy-sensitive: it maps hidden states to the full vocabulary, so error there
lands directly on the token distribution.

## TorchAO

The conversion itself is [TorchAO](https://github.com/pytorch/ao)'s, not LM7's.
LM7 pins `torchao==0.17.0` and calls `torchao.quantization.quantize_()` with
`Int8WeightOnlyConfig` or `Float8WeightOnlyConfig`, passing a module filter that
selects the layers above.

The quantized model is then compiled and executed through LM7's normal
`inductor` path, so quantization composes with the rest of the stack instead of
being a separate execution route. That is also why `--backend` is restricted:
TensorRT and OpenXLA have their own quantization stories that LM7 has not
integrated.

```bash
uv pip install -e ".[hf,torchao]"
lm7 model run hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target nvidia --backend inductor --dtype bfloat16 \
  --quantization int8-weight-only
```

Use `--quantization fp8-weight-only` on Ada or newer. The run reports model
storage bytes, quantization time, first-call and steady-state latency, and peak
GPU memory, so the footprint-versus-latency trade is measured rather than
assumed. Add `--json` for structured output.

## Scope and caveats

> [!NOTE]
> This path is validated for exactly one model,
> `HuggingFaceTB/SmolLM2-135M-Instruct`, and rejects every other model id.

- **NVIDIA only.** No CPU, AMD, Apple, or TPU quantization path exists.
- **It can be slower.** Weight-only quantization trades arithmetic for
  bandwidth. At small batch sizes, where a decode step is already
  memory-bound per token but the dequantization overhead is paid on every
  matmul, the net can go the wrong way. This is why it is opt-in rather than a
  default, and why the run reports latency alongside footprint.
- **Storage saving is not proportional to bit width.** Only the selected linears
  are converted, so a 4x narrower weight dtype does not mean a 4x smaller model —
  embeddings, norms, `lm_head`, and (for FP8) attention all stay in BF16.
- **No activation quantization, no INT4 or lower, no quantization-aware
  training.** Those are unexplored rather than rejected.

## Related

- [Supported hardware](../README.md#supported-hardware) — where quantization can run
- [`src/lm7/huggingface.py`](../src/lm7/huggingface.py) — validation gates and
  module filters
