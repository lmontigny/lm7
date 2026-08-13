"""The diffusion path against real diffusers, on a tiny downloaded pipeline.

`test_image_generation.py` covers LM7's loop with a stand-in pipeline and runs
everywhere. This file answers the question that stand-in cannot: whether the
duck-typed interface LM7 is written against is the one diffusers actually has,
and whether a real UNet compiles once and stays compiled.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

import lm7

pytestmark = pytest.mark.diffusion

# A ~1 MB pipeline with real diffusers component classes and a real scheduler.
TINY_PIPELINE = "hf-internal-testing/tiny-stable-diffusion-pipe"

requires_diffusers = pytest.mark.skipif(
    importlib.util.find_spec("diffusers") is None,
    reason="diffusers is unavailable; install the diffusion extra",
)


@pytest.fixture(scope="module")
def pipeline():
    diffusers = pytest.importorskip("diffusers")
    return diffusers.DiffusionPipeline.from_pretrained(TINY_PIPELINE, safety_checker=None)


@requires_diffusers
def test_a_real_pipeline_generates_pixels(pipeline):
    runner = lm7.compile_diffusion(pipeline, "cpu", backend="eager", height=64, width=64)

    result = runner.generate("a red bicycle", steps=2, seed=0)

    assert result.images.shape == (1, 3, 64, 64)
    assert torch.isfinite(result.images).all()
    assert 0.0 <= float(result.images.min()) and float(result.images.max()) <= 1.0


@requires_diffusers
def test_the_component_geometry_is_read_from_the_real_configs(pipeline):
    runner = lm7.compile_diffusion(pipeline, "cpu", backend="eager", height=64, width=64)

    assert runner.vae_scale_factor == 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    assert runner.latent_channels == pipeline.unet.config.in_channels
    assert runner.initial_latents(seed=0).shape == (
        1,
        runner.latent_channels,
        64 // runner.vae_scale_factor,
        64 // runner.vae_scale_factor,
    )


@requires_diffusers
def test_the_denoise_graph_does_not_recompile_per_step(pipeline):
    """The claim the whole module rests on, against a real UNet and scheduler.

    A nonzero `steady` here means a step changed something Dynamo guards on, and
    the per-step win is being paid for once per step instead of once. The two
    causes this has actually had are a Python-float timestep and initial latents
    built outside the inference context -- see `image_generation.denoise`.
    """
    runner = lm7.compile_diffusion(pipeline, "cpu", backend="inductor", height=64, width=64)

    result = runner.generate("a red bicycle", steps=4, seed=0)

    assert result.counters["steady"]["frames"] == 0
    assert result.counters["steady"]["recompiles"] == 0


@requires_diffusers
def test_compiled_and_eager_agree_from_identical_latents(pipeline):
    """Same starting noise through both arms, compared as images.

    Divergence in a diffusion loop compounds, so this is the check that a
    compiled denoise step is the same function as an eager one rather than
    merely a plausible one.
    """
    eager = lm7.compile_diffusion(pipeline, "cpu", backend="eager", height=64, width=64)
    latents = eager.initial_latents(seed=0)
    reference = eager.generate("a red bicycle", steps=3, latents=latents).images

    compiled = lm7.compile_diffusion(pipeline, "cpu", backend="inductor", height=64, width=64)
    candidate = compiled.generate("a red bicycle", steps=3, latents=latents).images

    assert torch.allclose(reference, candidate, atol=1e-4)
