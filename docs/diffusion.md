# Diffusion image generation

`lm7 image generate` runs a `diffusers` text-to-image pipeline through LM7's
backend dispatch. This page is what the path does, what it costs, and — the
short section that matters most — [what has not been
measured](#what-has-not-been-measured).

## Why it is not `lm7.compile()`

`lm7.compile()` compiles one forward pass. A diffusion pipeline is three
workloads that share no weights and do not run the same number of times:

| | input | output | how often |
| --- | --- | --- | --- |
| text encode | token ids | embeddings | once |
| **denoise** | **latents + timestep** | **predicted noise** | **N times, identical shape** |
| VAE decode | latents | pixels | once |

The middle row is the entire reason to compile, and it is the reason this needed
its own entry point rather than a flag. `lm7.compile_diffusion()` is the same
move [`compile_generation`](kv-cache-decode.md) made for autoregressive decode: a
loop is not a forward pass, so it gets its own API, its own per-phase graphs and
its own counters.

It is also the *easier* of the two loops. A decode graph writes into a KV cache,
which is why that path needs `warmup: False` and an allowlist of backends known
not to execute an artifact during compilation. A denoise step writes nothing —
its output is a pure function of its inputs — so this path compiles with LM7's
ordinary warmup and imposes **no backend restriction**.

```
lm7 image generate hf://stabilityai/sd-turbo \
    --prompt "a red bicycle" --steps 4 --seed 0 --target apple --output out.png
```

## LM7 owns the loop; diffusers owns the scheduler

The denoise loop is Python, and deliberately outside the compiled region. A
scheduler holds `sigmas`, `timesteps` and a step index, and calls `.item()` on
them; handing `StableDiffusionPipeline.__call__` to `torch.compile` compiles a
graph-break minefield instead of the three dense graphs that matter. LM7 drives
`scheduler.step()` itself and compiles only the three modules — the same division
of labour as the causal-LM path, where LM7 owns the compile boundary and
Transformers owns the cache.

Each of the three is a separate wrapper class with no shared base. That is not
style: Dynamo caches compiled code per *code object*, so one shared wrapper would
put three phases in one cache entry and report two of the three compiles as
recompilations of the first.

## Four ways to get a wrong number

Each of these turns the per-step win into a loss, or into a number that is not
measuring what it says.

1. **A Python-float timestep.** Dynamo guards on a float's *value*, so passing
   the scalar a scheduler hands out recompiles the denoise graph on every step.
   LM7 converts it to a 0-d tensor.
2. **Initial latents built outside the inference context.** Tensors created under
   `inference_mode` carry a different dispatch key, and Dynamo guards on it. Scale
   the incoming noise outside the context and step 1 enters the graph as an
   ordinary tensor while every later step arrives as an inference tensor from
   `scheduler.step` — so the guard fails once and the graph compiles twice for one
   shape. Measured on the tiny pipeline: compiles per step `[1, 1, 0, 0, 0]`
   scaling outside, `[1, 0, 0, 0, 0]` scaling inside. LM7 scales inside.
3. **Guidance changes the batch size.** Classifier-free guidance evaluates the
   step conditioned and unconditioned as one batch of 2, which keeps the shape
   constant across the loop — but makes a guided run a *different input
   signature* from an unguided one, and so a second compilation. A benchmark that
   changes `--guidance-scale` between arms is measuring a recompile. SD-Turbo
   runs unguided, at batch 1.
4. **fp16 VAE decode returns black images.** SD 1.x decoders overflow float16 at
   512×512 and produce NaNs with no exception. `diffusers` ships
   `vae.config.force_upcast` for exactly this and LM7 honours it.

`runner.counters["steady"]` exists to make the first two checkable: anything
nonzero there means a step compiled. `lm7 image generate` prints a warning when
it happens, and `benchmarks/diffusion.py` records it as `recompiled_during_loop`.

## Export

`lm7 image export --component {unet,vae_decoder,text_encoder}` captures **one
component per artifact**. The manifest records which component of which pipeline,
at what resolution and batch size, because a lowered graph has weights and no
name and running a VAE decoder where a UNet was meant fails in shape rather than
in a message.

There is no single-file bundle of a whole pipeline. `bundles.py` matches
artifacts on target and graph hash rather than on the role they play, so it has
no way to express "these three, in this order" — writing them into one bundle
would produce a file that loads and cannot be run. That is left undone rather
than half-built.

## Measuring

```
python benchmarks/diffusion.py --model sd-turbo --target nvidia \
    --output artifacts/diffusion.json
```

The headline is `ms_per_step`; encode and decode are reported separately because
they are paid once per image whatever the step count. `break_even_steps` is the
counterpart of [`break_even_tokens`](exported-decode.md): compiling costs a first
call and repays it per step, so "is compiling worth it" has a step count for an
answer, and `null` when the compiled arm is not faster at all.

Correctness is checked from **identical initial latents** — different noise gives
different images and says nothing — and reported twice: on the final image, and
on the first step's latents. Divergence in a denoise loop compounds, so a small
step-1 difference beside a large final one is this path's version of the failure
[exported decode](exported-decode.md) records: right at first, quietly wrong by
the end.

## What has been measured

Only the plumbing, and only on a toy model. Both runs are
`hf-internal-testing/tiny-stable-diffusion-pipe` (1.4M-parameter UNet) at 64×64
on an **Apple M3 Pro**, float32, DDIM, 4 steps, unguided:

| target | eager | inductor | agreement (max abs) |
| --- | --- | --- | --- |
| `cpu:arm64` | 9.6 ms/step | 12.3 ms/step | 1.0e-06 |
| `apple:metal` | 12.9 ms/step | 16.0 ms/step | 4.8e-07 |

**Compiling loses on both**, which is the expected and uninteresting result for a
1.4M-parameter model: there is not enough arithmetic per step for kernel fusion
to repay its overhead, exactly as the hand-built MLP in LM7's dense ladder shows
for causal LMs. These numbers say the path runs and agrees with eager to ~1e-6.
They say nothing about whether compiling a real diffusion model is worth it.

`steady` counters were zero in every run, at 3, 6 and 12 steps: the denoise graph
compiles once and stays compiled.

## What has not been measured

- **No full-size model has been run at all.** SD-Turbo, SD 1.5 and SDXL are
  reachable by name in `benchmarks/diffusion.py` and none of them has been
  executed. The per-step speedup on a real UNet is unknown, and the toy-model
  result above is not evidence either way.
- **Nothing has run on NVIDIA.** The RTX 4070 SUPER measurement this path was
  designed for has not happened, so there are no CUDA numbers, no `float16`
  numbers and no peak-VRAM numbers.
- **Capacity is calculated, not observed.** SD-Turbo and SD 1.5 should fit 12 GiB
  comfortably at float16 and SDXL tightly; Flux (12B) should not fit. None of
  this has been checked on hardware.
- **Export is unvalidated beyond the tiny pipeline.** UNet and VAE-decoder
  artifacts have been written and reloaded for
  `hf-internal-testing/tiny-stable-diffusion-pipe` on `cpu`. No full-size
  component has been exported, and no export backend other than `export` has been
  tried.
- **Core ML and QNN are untried.** Both refuse dynamic shapes, so a UNet artifact
  for them needs a single static resolution and CFG batch. That is expressible
  with `--size` and `--batch-size` and nobody has run it.
- **Quantization does not apply.** LM7's selectors match `.mlp.` linears and
  `lm_head`; a UNet has neither, so the match-count guard refuses rather than
  reporting a 1.00× reduction. There is no `--quantize` on `lm7 image`.
- **Only text-to-image.** No image-to-image, no inpainting, no ControlNet, no
  video. A video model is a denoise loop inside a temporal loop with a rolling
  latent cache, which this API does not express.

See [limitations](limitations.md#modality-coverage) for how this sits beside the
rest of LM7's coverage.
