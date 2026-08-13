"""The diffusion loop, tested against a stand-in pipeline rather than diffusers.

Everything here exercises LM7's code -- the loop, the phase counters, the
classifier-free-guidance batching, the seeding -- so the pipeline is built from
plain torch modules that answer the same duck-typed interface a diffusers
pipeline does. That keeps this file in the portable suite: it runs on a `[dev]`
install with no diffusion extra and no download, exactly as the tiny MoE configs
do for the sparse path. Real diffusers coverage lives in
`test_image_integration.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import lm7
from lm7.errors import UnsupportedModelError
from lm7.image_generation import DiffusionRunner, compile_diffusion

LATENT_CHANNELS = 4
CROSS_ATTENTION_DIM = 8
VAE_BLOCKS = 2  # -> vae_scale_factor 2


class FakeUNet(torch.nn.Module):
    """Predicts noise from latents, and records the shapes it was called with."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(in_channels=LATENT_CHANNELS)
        self.conv = torch.nn.Conv2d(LATENT_CHANNELS, LATENT_CHANNELS, 3, padding=1)
        self.project = torch.nn.Linear(CROSS_ATTENTION_DIM, 1)
        self.calls: list[tuple[tuple[int, ...], Any]] = []

    def forward(self, sample, timestep, encoder_hidden_states=None):
        self.calls.append((tuple(sample.shape), timestep))
        out = self.conv(sample)
        if encoder_hidden_states is not None:
            # Make the prompt actually reach the output, so a test that changes
            # the prompt sees a different image.
            bias = self.project(encoder_hidden_states).mean()
            out = out + bias
        return SimpleNamespace(sample=out)


class FakeVAE(torch.nn.Module):
    def __init__(self, *, force_upcast: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            block_out_channels=(4,) * VAE_BLOCKS,
            scaling_factor=2.0,
            force_upcast=force_upcast,
        )
        self.up = torch.nn.ConvTranspose2d(LATENT_CHANNELS, 3, 2, stride=2)

    def decode(self, latents):
        return SimpleNamespace(sample=self.up(latents))


class FakeTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(32, CROSS_ATTENTION_DIM)

    def forward(self, input_ids):
        return (self.embed(input_ids),)


class FakeTokenizer:
    model_max_length = 6

    def __call__(self, prompts, **kwargs):
        # One row per prompt, deterministic ids derived from the text so a
        # different prompt is a different tensor.
        ids = [
            [(len(prompt) + index) % 32 for index in range(self.model_max_length)]
            for prompt in prompts
        ]
        return SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.long))


class FakeScheduler:
    """A linear scheduler with the four members the runner touches."""

    init_noise_sigma = 1.0

    def __init__(self) -> None:
        self.timesteps = torch.tensor([])
        self.scaled = 0
        self.steps_taken = 0

    def set_timesteps(self, steps, device=None):
        self.timesteps = torch.linspace(900, 1, steps, dtype=torch.float32)
        self.steps_taken = 0

    def scale_model_input(self, sample, timestep):
        self.scaled += 1
        return sample

    def step(self, noise, timestep, latents):
        self.steps_taken += 1
        return SimpleNamespace(prev_sample=latents - 0.1 * noise)


def fake_pipeline(**kwargs):
    return SimpleNamespace(
        unet=FakeUNet(),
        vae=FakeVAE(**kwargs),
        text_encoder=FakeTextEncoder(),
        tokenizer=FakeTokenizer(),
        scheduler=FakeScheduler(),
    )


def runner(**kwargs) -> DiffusionRunner:
    options = {"height": 16, "width": 16, "backend": "eager"}
    options.update(kwargs)
    return compile_diffusion(fake_pipeline(), "cpu", **options)


def test_generate_produces_pixels_in_unit_range():
    result = runner().generate("a red bicycle", steps=3, seed=0)

    assert result.images.shape == (1, 3, 16, 16)
    assert float(result.images.min()) >= 0.0
    assert float(result.images.max()) <= 1.0
    assert result.steps == 3


def test_the_denoise_graph_runs_exactly_once_per_step():
    """The loop is LM7's, so its step count is LM7's to get right."""
    run = runner()

    run.generate("x", steps=5, seed=0)

    assert len(run.unet.calls) == 5
    assert run.scheduler.steps_taken == 5
    assert run.scheduler.scaled == 5


def test_every_step_sees_the_same_shape():
    """The property that lets one compilation serve the whole loop."""
    run = runner()

    run.generate("x", steps=6, seed=0)

    assert len({shape for shape, _ in run.unet.calls}) == 1


def test_guidance_doubles_the_denoise_batch():
    run = runner()

    run.generate("x", steps=2, seed=0, guidance_scale=7.5)

    assert all(shape[0] == 2 for shape, _ in run.unet.calls)


def test_no_guidance_leaves_the_batch_at_one():
    """SD-Turbo runs here, and it is a different input signature from the above."""
    run = runner()

    run.generate("x", steps=2, seed=0, guidance_scale=0.0)

    assert all(shape[0] == 1 for shape, _ in run.unet.calls)


def test_a_guidance_scale_of_exactly_one_is_not_guided():
    """At 1.0 the guidance arithmetic is the identity, so the second half is waste."""
    run = runner()

    run.generate("x", steps=1, seed=0, guidance_scale=1.0)

    assert run.unet.calls[0][0][0] == 1


def test_the_timestep_reaches_the_graph_as_a_tensor():
    """A Python float would make Dynamo guard on its value and recompile per step."""
    run = runner()

    run.generate("x", steps=3, seed=0)

    assert all(torch.is_tensor(timestep) for _, timestep in run.unet.calls)


def test_the_same_seed_reproduces_the_same_image():
    # One runner for both calls: a second `runner()` would build a second
    # pipeline with freshly randomized weights, and this would be testing those.
    run = runner()

    first = run.generate("x", steps=3, seed=11).images
    second = run.generate("x", steps=3, seed=11).images

    assert torch.equal(first, second)


def test_a_different_seed_gives_a_different_image():
    run = runner()

    first = run.generate("x", steps=3, seed=11).images
    other = run.generate("x", steps=3, seed=12).images

    assert not torch.equal(first, other)


def test_supplied_latents_override_the_seed():
    """How a benchmark gives two arms byte-identical starting points."""
    run = runner()
    latents = run.initial_latents(seed=3)

    first = run.generate("x", steps=2, seed=999, latents=latents).images
    second = run.generate("x", steps=2, seed=1, latents=latents).images

    assert torch.equal(first, second)


def test_ms_per_step_divides_denoise_time_only():
    result = runner().generate("x", steps=4, seed=0)

    assert result.ms_per_step == pytest.approx(result.denoise_ms / 4)
    assert result.total_ms == pytest.approx(result.encode_ms + result.denoise_ms + result.decode_ms)


def test_counters_are_reported_per_phase():
    result = runner().generate("x", steps=3, seed=0)

    assert set(result.counters) == {"encode", "denoise", "decode", "steady"}


def test_latent_size_follows_the_vae_scale_factor():
    run = runner(height=32, width=64)

    latents = run.initial_latents(seed=0)

    assert run.vae_scale_factor == 2
    assert latents.shape == (1, LATENT_CHANNELS, 16, 32)


def test_a_size_the_vae_cannot_halve_is_refused():
    with pytest.raises(ValueError, match="multiples of 2"):
        runner(height=15, width=16)


def test_a_pipeline_missing_a_component_is_refused_by_name():
    pipeline = fake_pipeline()
    pipeline.vae = None

    with pytest.raises(UnsupportedModelError, match="'vae'"):
        compile_diffusion(pipeline, "cpu", backend="eager", height=16, width=16)


def test_zero_steps_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        runner().generate("x", steps=0)


def test_force_upcast_decodes_in_float32():
    """SD 1.x decoders overflow float16 and return a black image with no error."""
    run = compile_diffusion(
        fake_pipeline(force_upcast=True), "cpu", backend="eager", height=16, width=16
    )

    run.generate("x", steps=1, seed=0)

    assert next(run.vae.parameters()).dtype is torch.float32


def test_compile_diffusion_is_exported_from_the_package():
    assert lm7.compile_diffusion is compile_diffusion
