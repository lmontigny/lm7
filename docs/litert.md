# LiteRT export backend

LM7 can convert a static PyTorch tensor model to a LiteRT flatbuffer and package
it in a reloadable `.lm7` artifact. The path is:

```text
nn.Module → LiteRT Torch converter → compiled_model.tflite → LiteRT/XNNPACK
```

This is the generic LiteRT tensor-model integration. It is not LiteRT-LM, which
adds tokenization, prefill/decode orchestration, KV-cache management, sampling,
and a stateful conversation API around one or more LiteRT models.

## Installation

LiteRT Torch 0.9.2 supports Python 3.10 or newer but requires PyTorch
`>=2.4,<2.13`. Use a separate conversion environment when the main LM7
environment is on PyTorch 2.13:

```bash
uv venv --python 3.12 .venv-litert
uv pip install --python .venv-litert/bin/python -e ".[dev,litert]"
```

The extra installs the `litert-torch` converter and `ai-edge-litert` runtime.
The converter stack is much larger than the runtime because it also includes
the StableHLO/JAX lowering and quantization dependencies.

## Export and reload

```python
import lm7

artifact = lm7.export(
    model.eval(),
    args=(example_input,),
    target="cpu",
    backend="litert",
    output="model-litert.lm7",
)

reloaded = lm7.load_artifact("model-litert.lm7")
result = reloaded(example_input)
```

The artifact contains:

- `exported_program.pt2`, LM7's source graph.
- `compiled_model.tflite`, the LiteRT flatbuffer.
- `manifest.json`, including checksums, converter/runtime versions, and options.

Loading verifies both payloads before opening the flatbuffer. Inputs are copied
to contiguous CPU NumPy arrays and outputs are copied back into CPU Torch
tensors. Tuples, lists, and dictionaries of tensor outputs are preserved.

Keyword tensor inputs are supported:

```python
artifact = lm7.export(
    model.eval(),
    args=(),
    kwargs={"input_ids": input_ids, "attention_mask": attention_mask},
    target="cpu",
    backend="litert",
    output="model-litert.lm7",
)
```

## Options

```python
artifact = lm7.export(
    model.eval(),
    args=(example_input,),
    target="cpu",
    backend="litert",
    output="model-litert.lm7",
    options={
        "strict_export": "auto",
        "lightweight_conversion": False,
        "enable_x64": True,
        "runtime_constant_folding": None,
    },
)
```

- `strict_export` accepts `"auto"`, `True`, or `False`. The default lets LiteRT
  Torch try its supported `torch.export` capture modes.
- `lightweight_conversion` reduces conversion memory and time for large models,
  potentially bypassing some constant-folding optimizations.
- `enable_x64=False` lets the converter downcast 64-bit values to 32-bit.
- `runtime_constant_folding` explicitly enables or disables LiteRT runtime
  constant folding. `None` lets the converter choose.

Quantization objects are not exposed through LM7 options in this first path;
those objects are not portable manifest values and need a separate typed API.

## Why export-only

LiteRT Torch exposes `litert_torch.convert(nn.Module, sample_args, ...)` and
performs its own `torch.export`/Dynamo capture. It does not expose a registered
`torch.compile` backend. Wrapping conversion as a Dynamo backend would convert
each graph-break fragment separately and would not produce the coordinated
multi-signature prefill/decode structure required by language models.

LM7 therefore keeps this backend AOT-only:

```python
lm7.export(..., backend="litert")  # supported
lm7.compile(..., backend="litert")  # rejected with guidance
```

The public converter also requires the original `nn.Module` and representative
inputs. An already-captured `ExportedProgram` cannot be used with this backend.

## Validated scope

The initial integration is deliberately narrow:

- CPU execution through LiteRT's default XNNPACK interpreter.
- Static tensor inputs.
- Tensor outputs nested in tuples, lists, or dictionaries.
- FP32 MLP conversion/reload with maximum error below `6e-8`.
- Keyword-input and tuple-output reload with exact agreement.
- LiteRT Torch 0.9.2, LiteRT 2.1.6, and PyTorch 2.12.1 on Linux x86-64.

Dynamic shapes are rejected. A bounded dynamic batch prototype reached
`torch.export` but failed in LiteRT Torch's JAX symbolic-shape lowering. GPU and
NPU execution are also outside this first backend because `litert_torch.load`
uses the default LiteRT interpreter/XNNPACK path; using LiteRT's newer Compiled
Model API needs a separate runtime adapter and hardware validation.

Full Hugging Face causal LMs are not claimed. Efficient LiteRT-LM packages use
model-specific reauthoring/checkpoint mapping, multiple prefill/decode
signatures, KV-cache configuration, tokenizer metadata, and quantization. A
future LiteRT-LM integration belongs in LM7's generation API rather than this
tensor-callable backend.

Official references:

- [LiteRT Torch](https://github.com/google-ai-edge/litert-torch)
- [PyTorch converter API](https://github.com/google-ai-edge/litert-torch/blob/main/docs/pytorch_converter/README.md)
- [LiteRT-LM Python API](https://developers.google.com/edge/litert-lm/python)
- [LiteRT Torch generative architecture](https://github.com/google-ai-edge/litert-torch/blob/main/litert_torch/generative/doc/system_overview.md)
