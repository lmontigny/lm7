"""`hf://` entry points for diffusion, the way `huggingface.py` is for causal LMs.

Loading, dtype selection and artifact provenance are shared with the text path
through :mod:`lm7.hub`; what is different is everything downstream of the load,
because a diffusion pipeline is several modules rather than one and only one of
them is worth compiling per step.

Export here is deliberately **one component per artifact**. The three graphs have
different inputs, different shapes and different call counts, and LM7's bundle
format matches artifacts on target and graph hash rather than on the role they
play in a pipeline -- so there is no way today to say "these three go together,
in this order". Writing them into one bundle would produce a file that loads and
cannot be run. See docs/diffusion.md.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .detection import resolve_target
from .errors import UnsupportedModelError
from .exporting import export as export_artifact
from .hub import (
    load_diffusers,
    parse_model_uri,
    peak_memory,
    reset_peak_memory,
    resolve_dtype,
    source_metadata,
)
from .image_generation import (
    COMPONENTS,
    DEFAULT_GUIDANCE,
    DEFAULT_SIZE,
    DEFAULT_STEPS,
    DiffusionResult,
    compile_diffusion,
    component_graph,
)
from .targets import TargetSpec


@dataclass(frozen=True)
class ImageGenerateResult:
    model_uri: str
    model_id: str
    pipeline_class: str
    target: str
    backend: str
    dtype: str
    prompt: str
    steps: int
    guidance_scale: float
    seed: int | None
    height: int
    width: int
    output: str
    load_ms: float
    encode_ms: float
    denoise_ms: float
    decode_ms: float
    ms_per_step: float
    total_ms: float
    peak_memory_bytes: int | None
    counters: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageExportResult:
    model_uri: str
    model_id: str
    component: str
    target: str
    backend: str
    dtype: str
    output: str
    height: int
    width: int
    batch_size: int
    parameter_count: int
    export_ms: float
    artifact_bytes: int
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_pipeline(model_uri: str, *, target: TargetSpec, dtype: str = "auto") -> tuple[Any, str]:
    """Load a diffusers pipeline, returning it and the resolved dtype name."""
    diffusers = load_diffusers()
    model_id = parse_model_uri(model_uri)
    torch_dtype = resolve_dtype(dtype, target)
    try:
        pipeline = diffusers.DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype, safety_checker=None
        )
    except Exception as exc:
        raise UnsupportedModelError(
            f"Could not load {model_uri} as a diffusers pipeline: {exc}."
        ) from exc
    return pipeline, str(torch_dtype).removeprefix("torch.")


def generate_image(
    model_uri: str,
    *,
    prompt: str,
    output: str | None = None,
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
    seed: int | None = None,
    height: int = DEFAULT_SIZE,
    width: int = DEFAULT_SIZE,
    target: str | TargetSpec = "auto",
    backend: str = "auto",
    dtype: str = "auto",
    compile_mode: str | None = None,
) -> ImageGenerateResult:
    """Generate one image, and report where the time went.

    ``steps`` and ``guidance_scale`` default to SD-Turbo's regime -- four steps
    and no guidance -- because that is the configuration this path was measured
    in. A model trained for guidance needs a scale above 1.0, which doubles the
    denoise batch and compiles a second variant of the graph.
    """
    resolved_target = resolve_target(target)
    started = time.perf_counter()
    pipeline, dtype_name = load_pipeline(model_uri, target=resolved_target, dtype=dtype)
    load_ms = (time.perf_counter() - started) * 1000.0

    reset_peak_memory(resolved_target)
    runner = compile_diffusion(
        pipeline,
        resolved_target,
        backend=backend,
        compile_mode=compile_mode,
        height=height,
        width=width,
    )
    result = runner.generate(prompt, steps=steps, guidance_scale=guidance_scale, seed=seed)
    written = _write_image(result, output) if output else ""
    return ImageGenerateResult(
        model_uri=model_uri,
        model_id=parse_model_uri(model_uri),
        pipeline_class=type(pipeline).__name__,
        target=str(resolved_target),
        backend=runner.selected_backend,
        dtype=dtype_name,
        prompt=prompt,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        height=height,
        width=width,
        output=written,
        load_ms=load_ms,
        encode_ms=result.encode_ms,
        denoise_ms=result.denoise_ms,
        decode_ms=result.decode_ms,
        ms_per_step=result.ms_per_step,
        total_ms=result.total_ms,
        peak_memory_bytes=peak_memory(resolved_target),
        counters=result.counters,
    )


def export_image_component(
    model_uri: str,
    *,
    component: str,
    output: str,
    height: int = DEFAULT_SIZE,
    width: int = DEFAULT_SIZE,
    batch_size: int = 1,
    target: str | TargetSpec = "auto",
    backend: str = "export",
    dtype: str = "auto",
) -> ImageExportResult:
    """Capture one component of a diffusion pipeline into an LM7 artifact.

    ``batch_size`` is the denoise batch, which classifier-free guidance doubles.
    An artifact captured at 1 is not usable for a guided run and the reverse, so
    it is recorded in the manifest rather than left to be inferred.
    """
    if component not in COMPONENTS:
        choices = ", ".join(COMPONENTS)
        raise ValueError(f"component must be one of: {choices}; got {component!r}.")
    resolved_target = resolve_target(target)
    pipeline, dtype_name = load_pipeline(model_uri, target=resolved_target, dtype=dtype)
    runner = compile_diffusion(
        pipeline, resolved_target, backend="eager", height=height, width=width
    )
    module, args = _component_inputs(runner, component, batch_size=batch_size)

    started = time.perf_counter()
    artifact = export_artifact(
        module.eval(),
        args=args,
        target=resolved_target,
        output=output,
        backend=backend,
        source=source_metadata(
            model_uri,
            parse_model_uri(model_uri),
            next(module.parameters()).dtype,
            pipeline_id=parse_model_uri(model_uri),
            pipeline_class=type(pipeline).__name__,
            component=component,
            scheduler=type(pipeline.scheduler).__name__,
            batch_size=batch_size,
            height=height,
            width=width,
        ),
    )
    export_ms = (time.perf_counter() - started) * 1000.0
    files = tuple(sorted(str(path.relative_to(artifact.path)) for path in _files(artifact.path)))
    return ImageExportResult(
        model_uri=model_uri,
        model_id=parse_model_uri(model_uri),
        component=component,
        target=str(resolved_target),
        backend=backend,
        dtype=dtype_name,
        output=str(artifact.path),
        height=height,
        width=width,
        batch_size=batch_size,
        parameter_count=sum(p.numel() for p in module.parameters()),
        export_ms=export_ms,
        artifact_bytes=sum(path.stat().st_size for path in _files(artifact.path)),
        files=files,
    )


def _component_inputs(
    runner: Any, component: str, *, batch_size: int
) -> tuple[torch.nn.Module, tuple[Any, ...]]:
    """The module to capture and one representative call of it.

    The shapes come from the pipeline's own configs rather than from constants,
    because the whole point of capturing a component is that it is *this*
    pipeline's component.
    """
    module = component_graph(runner, component)
    latents = torch.randn(
        batch_size,
        runner.latent_channels,
        runner.height // runner.vae_scale_factor,
        runner.width // runner.vae_scale_factor,
        dtype=runner.dtype,
    )
    sequence = int(getattr(runner.tokenizer, "model_max_length", 77))
    if component == "unet":
        cross_dim = int(getattr(runner.unet.config, "cross_attention_dim", 768))
        embeddings = torch.randn(batch_size, sequence, cross_dim, dtype=runner.dtype)
        # A float32 scalar rather than the component dtype: schedulers hand out
        # float or int timesteps and the UNet's time projection accepts either,
        # so capturing at the pipeline's compute dtype would narrow the artifact
        # for no reason.
        return module, (latents, torch.tensor(1.0), embeddings)
    if component == "vae_decoder":
        return module, (latents,)
    return module, (torch.ones(batch_size, sequence, dtype=torch.long),)


def _write_image(result: DiffusionResult, output: str) -> str:
    """Write the first image as PNG, which needs Pillow and nothing else."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise UnsupportedModelError(
            'Writing an image needs Pillow. Install it with: pip install "lm7[diffusion]".'
        ) from exc
    array = (result.images[0].permute(1, 2, 0) * 255).round().clamp(0, 255)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.to(torch.uint8).cpu().numpy()).save(path)
    return str(path)


def _files(path: Path) -> list[Path]:
    return [item for item in path.rglob("*") if item.is_file()]
