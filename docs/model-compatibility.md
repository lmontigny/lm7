# Hugging Face model compatibility

Use the compatibility preflight before downloading a checkpoint or starting a
compile:

```bash
lm7 model compatibility hf://HuggingFaceTB/SmolLM2-135M-Instruct \
  --target auto --backend auto
```

The command downloads the Hugging Face configuration only. It does not load a
tokenizer or model weights, instantiate the model, allocate accelerator memory,
or compile a graph. Add `--json` for structured output suitable for automation.

The report resolves the target and runtime backend, identifies the Transformers
configuration and task, and evaluates three separate areas:

- `run`: whether `lm7 model run` recognizes the config and the selected runtime
  backend supports the target.
- `generate`: whether the causal-LM generation path is available and whether
  Transformers will compile or eagerly execute static-cache decode on that
  device.
- `export`: whether the config can enter LM7's export path. This remains
  conditional until `torch.export` and the chosen export backend see real
  representative inputs.
- `int8`, `fp8`, and `nvfp4`: the existing LM7 model, target, backend, dtype, and
  hardware validation gates for runtime weight-only quantization.

## Reading the result

The top-level status has three possible values:

| Status | Meaning |
| --- | --- |
| `compatible` | The installed Transformers version registers a decoder-only config for `AutoModelForCausalLM`, and a runtime backend supports the target. Also reported for a recognized diffusion pipeline — see below. |
| `unknown` | The architecture name looks like a causal LM, but the installed Transformers version does not register its config. |
| `incompatible` | The model is encoder-decoder, multimodal, requires remote code, is not a causal LM, or the requested runtime backend cannot serve the target. |

Individual checks use `compatible`, `conditional`, `unknown`, or `unsupported`.
A `conditional` result is deliberate: static KV-cache support and compiler
operator coverage cannot be proven from configuration metadata.

For example, T5 is rejected before its weights are downloaded:

```text
$ lm7 model compatibility hf://google-t5/t5-small --target cpu
Status: incompatible
Architecture: T5ForConditionalGeneration (t5)
Task: seq2seq
```

`--backend` describes the runtime backend used by `model run` and, where
applicable, `model generate`. Export compatibility is reported separately
because `lm7 model export` has its own export-backend choices.

## Diffusion pipelines

A diffusion repository has no top-level `config.json` — it declares its
components in `model_index.json` instead — so `AutoConfig` cannot read one and
LM7 used to report it as "not registered for `AutoModelForCausalLM`". That was
technically true and told you nothing. Such a repository is now identified:

```text
$ lm7 model compatibility hf://stabilityai/sd-turbo --target cpu
Status: compatible
Architecture: StableDiffusionPipeline
Task: diffusion
```

The status is `compatible` because the question this command answers is whether
LM7 recognizes the checkpoint. Every *workflow* check is `unsupported`: `model
run`, `model generate`, and `model export` tokenize text, and a diffusion
pipeline has no tokenizable input. The quantization checks are `unsupported` for
a concrete reason rather than a general one — LM7's selectors match `.mlp.`
linears and `lm_head`, which a UNet has neither of.

Detection needs the diffusion extra (`pip install "lm7[diffusion]"`). Without it
there is no way to tell "not a diffusion pipeline" from "cannot read one", so the
original config failure is raised with a pointer to the extra attached.

## What this does not guarantee

A configuration match is a fast preflight, not a model allowlist or a compiler
proof. The repository can contain custom modeling code, a nominally supported
model can use operators a compiler does not lower, and generation can lack the
static cache implementation. The first real `model run`, `model generate`, or
`model export` remains the definitive check.

LM7 keeps `trust_remote_code=False`; a repository that requires executing custom
configuration or modeling code is reported as incompatible. Gated or private
configuration downloads still require the normal Hugging Face authentication.
See [limitations](limitations.md) for the broader support boundary and
[quantization](quantization.md) for measured precision coverage.
