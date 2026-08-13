"""Three compiled graphs for text-to-image diffusion: encode, denoise, decode.

``lm7.compile()`` compiles one forward pass. That is enough to measure a model and
not enough to generate an image, because a diffusion pipeline is three different
workloads that do not share weights and do not run the same number of times:

    text encode   prompt              -> embeddings          once
    denoise       latents + timestep  -> predicted noise     N times, same shape
    vae decode    latents             -> pixels              once

The middle one is the entire reason to compile. Its shape is identical on every
step -- unlike a causal LM's decode, where the cache grows -- so one compilation
serves the whole loop and the per-step win is paid for once.

    runner = lm7.compile_diffusion(pipeline, target="nvidia")
    result = runner.generate("a red bicycle", steps=4)

LM7 owns the compile boundary, the denoise loop and the counting. It does not own
the scheduler, the UNet or the VAE: those are ``diffusers``', exactly as the KV
cache in :mod:`lm7.generation` is Transformers'. The loop is deliberately Python
and deliberately outside the graph -- a scheduler holds ``sigmas``, ``timesteps``
and a step index, and calls ``.item()`` on them, so handing a pipeline's
``__call__`` to ``torch.compile`` compiles a graph-break minefield instead of the
three dense graphs that matter.

One difference from :mod:`lm7.generation` is worth stating because its absence
would look like an oversight: **these graphs are stateless.** A decode graph
writes into a KV cache, so a backend that compiles by executing the artifact it
just built spends cache slots nobody asked for -- which is what ``warmup: False``
and ``_PLANNABLE_BACKENDS`` exist to prevent there. A denoise step writes
nothing, its output is a pure function of its inputs, and an extra execution
costs time and changes no result. So this path compiles with LM7's ordinary
warmup and imposes no backend allowlist.

See docs/diffusion.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .api import compile as compile_module
from .detection import inference_context, resolve_target, synchronize, torch_device
from .errors import UnsupportedModelError
from .generation import ZERO_COUNTERS, GraphCounters, graph_counters
from .targets import TargetSpec

# The components LM7 compiles, in the order they run. `tokenizer` and `scheduler`
# are not here on purpose: neither is a torch module and neither has a forward
# pass to capture.
COMPONENTS = ("text_encoder", "unet", "vae_decoder")

# SD-Turbo's regime, and the one this path was measured in: four steps and no
# guidance. A model trained for guidance needs a scale above 1.0, which doubles
# the denoise batch and so compiles a second variant of the graph.
DEFAULT_STEPS = 4
DEFAULT_GUIDANCE = 0.0
DEFAULT_SIZE = 512


@dataclass(frozen=True)
class DiffusionResult:
    """One generated batch, and where its time went.

    ``images`` is a float tensor in [0, 1] with shape (batch, 3, height, width) --
    pixels, not a PIL image, because this layer measures and the caller decides
    what to write.
    """

    images: torch.Tensor
    prompt: str
    steps: int
    guidance_scale: float
    seed: int | None
    encode_ms: float
    denoise_ms: float
    decode_ms: float
    counters: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return self.encode_ms + self.denoise_ms + self.decode_ms

    @property
    def ms_per_step(self) -> float:
        """Denoise milliseconds per step, which is the number that scales.

        Mirrors ``GenerationResult.ms_per_decoded_token``: encode and decode are
        paid once per image whatever the step count, so a comparison between step
        counts is only meaningful against this.
        """
        return self.denoise_ms / self.steps if self.steps else 0.0


class _TextEncodeGraph(torch.nn.Module):
    """Token ids in, one embeddings tensor out.

    A separate class from the two below with no shared base, for the reason
    ``generation._PrefillGraph`` spells out: Dynamo caches compiled code per
    *code object*, so a shared wrapper would put three phases in one cache entry
    and report two of the three compiles as recompilations of the first.

    Returns ``[0]`` rather than the output dataclass because Transformers'
    ``BaseModelOutputWithPooling`` cannot be deserialized by ``torch.export.load``
    -- the same pytree problem ``huggingface._LogitsOnly`` exists for.
    """

    def __init__(self, text_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids)[0]


class _DenoiseGraph(torch.nn.Module):
    """One denoise step: noisy latents and a timestep in, predicted noise out.

    The graph that runs N times, and the only one whose cost scales with the
    request. See ``_TextEncodeGraph`` for why this is a separate class.

    ``timestep`` is a tensor rather than the Python float a scheduler hands out.
    Dynamo specializes on a float's *value*, so passing one would guard on it and
    recompile this graph on every step -- turning the one win this module exists
    for into a loss that looks like "compiling does not help diffusion".
    """

    def __init__(self, unet: torch.nn.Module) -> None:
        super().__init__()
        self.unet = unet

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(latents, timestep, encoder_hidden_states=encoder_hidden_states).sample


class _VaeDecodeGraph(torch.nn.Module):
    """Latents in, pixels out. See ``_TextEncodeGraph`` for why it is its own class."""

    # `vae` is typed loosely because `decode` is not part of `nn.Module`: it is
    # an `AutoencoderKL` method, and `Module.__getattr__` types every attribute
    # lookup as a tensor-or-module union that mypy will not let anyone call.
    def __init__(self, vae: Any) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents).sample


class DiffusionRunner:
    """A compiled text encoder, a compiled denoise step, a compiled VAE decoder.

    Built by :func:`compile_diffusion`, which documents the arguments.
    """

    def __init__(
        self,
        pipeline: Any,
        target: TargetSpec,
        *,
        backend: str,
        compile_mode: str | None,
        compile_text_encoder: bool,
        compile_vae: bool,
        height: int,
        width: int,
    ) -> None:
        self.pipeline = pipeline
        self.target = target
        self.backend = backend
        self.compile_mode = compile_mode
        self.height = height
        self.width = width
        self.device = torch_device(target)

        self.unet = _require(pipeline, "unet")
        self.vae = _require(pipeline, "vae")
        self.text_encoder = _require(pipeline, "text_encoder")
        self.tokenizer = _require(pipeline, "tokenizer")
        self.scheduler = _require(pipeline, "scheduler")

        self.dtype = next(self.unet.parameters()).dtype

        # How much smaller a latent is than the image it decodes to: one halving
        # per VAE downsampling block.
        block_out_channels = getattr(self.vae.config, "block_out_channels", (1,))
        self.vae_scale_factor = 2 ** (len(block_out_channels) - 1)
        if height % self.vae_scale_factor or width % self.vae_scale_factor:
            raise ValueError(
                f"height and width must be multiples of {self.vae_scale_factor} for this VAE; "
                f"got {height}x{width}."
            )
        self.latent_channels = int(getattr(self.unet.config, "in_channels", 4))
        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0))

        # The weights move before compiling, not during, for the reason
        # `GenerationRunner` records: LM7's backends move the model as part of
        # compiling, compiling happens inside a call made under `inference_mode`,
        # and `Module.to` cannot swap a parameter that is a tensor subclass there.
        for module in (self.unet, self.vae, self.text_encoder):
            module.to(self.device)

        options: dict[str, Any] = {}
        if compile_mode:
            options["compile_mode"] = compile_mode
        # No `warmup: False` here, unlike the decode path. These graphs are
        # stateless, so a warmup execution costs time and changes nothing -- see
        # the module docstring.
        self._encode_graph = compile_module(
            _TextEncodeGraph(self.text_encoder).eval(),
            target=target,
            backend=backend if compile_text_encoder else "eager",
            transfers="explicit",
            fallback="error",
            cache=False,
            options=options if compile_text_encoder else None,
        )
        self._denoise_graph = compile_module(
            _DenoiseGraph(self.unet).eval(),
            target=target,
            backend=backend,
            transfers="explicit",
            fallback="error",
            cache=False,
            options=options or None,
        )
        self._decode_graph = compile_module(
            _VaeDecodeGraph(self.vae).eval(),
            target=target,
            backend=backend if compile_vae else "eager",
            transfers="explicit",
            fallback="error",
            cache=False,
            options=options if compile_vae else None,
        )

        self.encode_compile = ZERO_COUNTERS
        self.denoise_compile = ZERO_COUNTERS
        self.decode_compile = ZERO_COUNTERS
        self.steady = ZERO_COUNTERS
        # Set by the phases that measure them, so a caller driving the phases by
        # hand still gets a coherent result out of `generate`.
        self._encode_ms = 0.0
        self._decode_ms = 0.0

    # -- introspection ----------------------------------------------------

    @property
    def selected_backend(self) -> str:
        """What ``backend="auto"`` actually resolved to, once something has compiled.

        ``self.backend`` is what was *asked for*, and reporting that as the result
        would print "auto" as though it were a backend. Planning is lazy, so the
        answer does not exist until the denoise graph has run at least once;
        before that this falls back to the request.
        """
        artifact = getattr(self._denoise_graph, "artifact", None)
        return getattr(artifact, "backend", None) or self.backend

    @property
    def counters(self) -> dict[str, dict[str, int]]:
        """Compilation counters per phase, as deltas.

        ``denoise`` covers the first step, which is where its compilation
        happens. ``steady`` accumulates every step after that, and is the number
        this path exists to make checkable: anything nonzero in
        ``steady["frames"]`` means a step triggered a compile, and the usual cause
        is a timestep or a batch size that changed shape mid-loop.
        """
        return {
            "encode": self.encode_compile.to_dict(),
            "denoise": self.denoise_compile.to_dict(),
            "decode": self.decode_compile.to_dict(),
            "steady": self.steady.to_dict(),
        }

    # -- phases -----------------------------------------------------------

    def encode_prompt(self, prompt: str, *, guidance_scale: float = 0.0) -> torch.Tensor:
        """Embed the prompt, and the empty prompt too when guidance needs it.

        Classifier-free guidance evaluates the denoise step twice per iteration,
        conditioned and unconditioned. Doing that as one batch of 2 rather than
        two batches of 1 keeps the step's shape constant, which is what lets one
        compilation serve the loop -- but it does mean a guided run and an
        unguided one are *different input signatures* and compile separately.
        """
        prompts = [""] * (1 if _uses_guidance(guidance_scale) else 0) + [prompt]
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        started = _now(self.target)
        with inference_context(self.target):
            embeddings = self._encode_graph(input_ids)
        self.encode_compile = _delta(started.counters)
        self._encode_ms = _elapsed_ms(self.target, started)
        return embeddings.to(self.dtype)

    def initial_latents(self, *, seed: int | None = None, batch_size: int = 1) -> torch.Tensor:
        """Seeded noise to start from, generated on the CPU and then moved.

        The generator is a CPU one regardless of target: MPS has no equivalent,
        and seeding on the accelerator would make a fixed seed mean a different
        image per device -- which is precisely what a benchmark comparing two
        targets needs it not to mean.
        """
        generator = torch.Generator("cpu")
        if seed is not None:
            generator.manual_seed(seed)
        shape = (
            batch_size,
            self.latent_channels,
            self.height // self.vae_scale_factor,
            self.width // self.vae_scale_factor,
        )
        latents = torch.randn(shape, generator=generator, dtype=torch.float32)
        return latents.to(self.device, dtype=self.dtype)

    def denoise(
        self,
        latents: torch.Tensor,
        embeddings: torch.Tensor,
        *,
        steps: int,
        guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        """Run the scheduler's loop, calling the compiled step once per timestep.

        The scheduler is driven from here rather than compiled: it is Python and
        NumPy state that changes every iteration, and capturing it would break the
        graph in the one place a graph break is expensive.
        """
        guided = _uses_guidance(guidance_scale)
        self.scheduler.set_timesteps(steps, device=self.device)
        timesteps = self.scheduler.timesteps

        first = True
        with inference_context(self.target):
            # Scaling the incoming noise happens *inside* the inference context,
            # and that placement is load-bearing rather than tidy. Tensors created
            # under `inference_mode` carry a different dispatch key from ordinary
            # ones, and Dynamo guards on it. Scale outside and step 1 enters the
            # graph as an ordinary tensor while every later step arrives as an
            # inference tensor from `scheduler.step`, so the guard fails once and
            # the denoise graph compiles twice for one shape. Measured on the
            # tiny pipeline: compiles per step [1, 1, 0, 0, 0] scaling outside,
            # [1, 0, 0, 0, 0] scaling here.
            latents = latents * self.scheduler.init_noise_sigma
            for timestep in timesteps:
                model_input = torch.cat([latents] * 2) if guided else latents
                model_input = self.scheduler.scale_model_input(model_input, timestep)
                # A 0-d tensor, never the Python float a scheduler may hand back:
                # Dynamo guards on a float's value and would recompile per step.
                if torch.is_tensor(timestep):
                    step_t = timestep.to(self.device)
                else:
                    step_t = torch.tensor(timestep, device=self.device)
                started = _now(self.target)
                noise = self._denoise_graph(model_input, step_t, embeddings)
                if first:
                    self.denoise_compile = _delta(started.counters)
                    first = False
                else:
                    self.steady = self.steady + _delta(started.counters)
                if guided:
                    uncond, cond = noise.chunk(2)
                    noise = uncond + guidance_scale * (cond - uncond)
                latents = self.scheduler.step(noise, timestep, latents).prev_sample
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents to pixels in [0, 1].

        The VAE runs in float32 when its config says to. SD 1.x decoders overflow
        float16 at 512x512 and return NaNs -- a black image, with no exception --
        which is why ``force_upcast`` exists in the config and is honoured here
        rather than treated as a diffusers detail.
        """
        scaled = latents / self.scaling_factor if self.scaling_factor else latents
        if getattr(self.vae.config, "force_upcast", False):
            self.vae.to(torch.float32)
            scaled = scaled.to(torch.float32)
        started = _now(self.target)
        with inference_context(self.target):
            image = self._decode_graph(scaled)
        self.decode_compile = _delta(started.counters)
        self._decode_ms = _elapsed_ms(self.target, started)
        return (image.float() / 2 + 0.5).clamp(0, 1)

    def generate(
        self,
        prompt: str,
        *,
        steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int | None = None,
        latents: torch.Tensor | None = None,
    ) -> DiffusionResult:
        """Encode, denoise, decode, and report where the time went.

        ``latents`` overrides the seeded noise, so two arms of a benchmark can be
        given byte-identical starting points and their outputs compared directly.
        """
        if steps < 1:
            raise ValueError("steps must be at least 1.")
        embeddings = self.encode_prompt(prompt, guidance_scale=guidance_scale)
        if latents is None:
            latents = self.initial_latents(seed=seed)
        started = _now(self.target)
        denoised = self.denoise(latents, embeddings, steps=steps, guidance_scale=guidance_scale)
        denoise_ms = _elapsed_ms(self.target, started)
        images = self.decode_latents(denoised)
        return DiffusionResult(
            images=images,
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            encode_ms=self._encode_ms,
            denoise_ms=denoise_ms,
            decode_ms=self._decode_ms,
            counters=self.counters,
        )


def component_graph(runner: DiffusionRunner, component: str) -> torch.nn.Module:
    """The uncompiled wrapper module for one component, ready to export.

    Export captures the same wrapper the runner compiles -- the one that returns
    a plain tensor rather than a ``UNet2DConditionOutput`` -- so an artifact and a
    JIT run are the same graph. Built fresh here rather than pulled out of the
    compiled module, because what a ``CompiledModule`` holds is an implementation
    detail and this is a supported way to ask.
    """
    if component not in COMPONENTS:
        choices = ", ".join(COMPONENTS)
        raise ValueError(f"component must be one of: {choices}; got {component!r}.")
    if component == "unet":
        return _DenoiseGraph(runner.unet).eval()
    if component == "vae_decoder":
        return _VaeDecodeGraph(runner.vae).eval()
    return _TextEncodeGraph(runner.text_encoder).eval()


def compile_diffusion(
    pipeline: Any,
    target: str | TargetSpec | None = None,
    *,
    backend: str = "auto",
    compile_mode: str | None = None,
    compile_text_encoder: bool = True,
    compile_vae: bool = True,
    height: int = 512,
    width: int = 512,
) -> DiffusionRunner:
    """Compile a diffusers text-to-image pipeline into three separate graphs.

    ``pipeline`` is used exactly as given -- LM7 constructs nothing and replaces
    no component, so a pipeline whose scheduler was swapped before this call
    denoises with that scheduler.

    ``backend`` is unrestricted, unlike :func:`lm7.compile_generation`: these
    graphs hold no state, so there is no backend that can quietly break them by
    executing one during compilation.

    ``compile_text_encoder=False`` and ``compile_vae=False`` leave those two in
    eager. They run once per image whatever the step count, so compiling them is
    worth its compile time for a served workload and usually not for a single
    image -- and leaving them out isolates the denoise step when measuring.

    ``height`` and ``width`` size the initial latents and must be multiples of the
    VAE's scale factor. They are fixed at construction because changing them
    changes the denoise graph's input shape, which is a recompilation.
    """
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive.")
    resolved = resolve_target(target if target is not None else "auto")
    return DiffusionRunner(
        pipeline,
        resolved,
        backend=backend,
        compile_mode=compile_mode,
        compile_text_encoder=compile_text_encoder,
        compile_vae=compile_vae,
        height=height,
        width=width,
    )


def _uses_guidance(guidance_scale: float) -> bool:
    """Whether guidance is on, which decides the denoise batch size.

    The threshold is 1.0 rather than 0.0 because a scale of exactly 1 weights the
    conditioned prediction at one and the unconditioned at zero: the arithmetic is
    the identity, so evaluating the second half of the batch is pure waste.
    SD-Turbo is trained to run here.
    """
    return guidance_scale > 1.0


@dataclass(frozen=True)
class _Mark:
    perf: float
    counters: GraphCounters


def _now(target: TargetSpec) -> _Mark:
    synchronize(target)
    return _Mark(time.perf_counter(), graph_counters())


def _elapsed_ms(target: TargetSpec, started: _Mark) -> float:
    synchronize(target)
    return (time.perf_counter() - started.perf) * 1000.0


def _delta(before: GraphCounters) -> GraphCounters:
    return graph_counters() - before


def _require(pipeline: Any, name: str) -> Any:
    component = getattr(pipeline, name, None)
    if component is None:
        raise UnsupportedModelError(
            f"This pipeline has no {name!r}. LM7's diffusion path is written against a "
            "text-to-image pipeline with a text encoder, a UNet, a VAE and a scheduler."
        )
    return component
