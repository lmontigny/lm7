"""What kind of model a checkpoint holds, decided from configuration alone.

LM7's model commands were written when every model was a decoder-only causal LM,
so "is this loadable" and "is this a causal LM" were the same question and the
code asks it once. They are not the same question, and the place that shows is
the error: a diffusion repo has no top-level causal-LM ``config.json`` at all --
it has a ``model_index.json`` naming subfolders (``unet/``, ``vae/``,
``text_encoder/``, ``scheduler/``) -- so ``AutoConfig.from_pretrained`` fails on
it with a message about missing config files, and LM7 reported that as "not
registered for AutoModelForCausalLM". Technically true, and useless.

This module answers the narrower question first, from configuration only. No
weights are downloaded, because ``compatibility.inspect_hf_model`` promises that
in its own output and a preflight that pulls a 5 GiB UNet is not a preflight.

Detection is deliberately shallow. It reports what a repository *declares*, not
what will compile: a config that says ``StableDiffusionPipeline`` proves the repo
is a diffusion pipeline and proves nothing about whether ``torch.export`` covers
its operators. That second question is answered by running it, and every caller
here already says so.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import UnsupportedModelError
from .hub import load_diffusers

CAUSAL_LM = "causal-lm"
DIFFUSION = "diffusion"
UNKNOWN = "unknown"

# Pipeline classes LM7's diffusion path is written against. A repo declaring
# something else -- img2img, inpainting, video -- is still detected as diffusion,
# because what it *is* does not depend on what LM7 implements; the workflow checks
# are where "detected but not supported" gets said.
TEXT_TO_IMAGE_PIPELINES = frozenset(
    {
        "StableDiffusionPipeline",
        "StableDiffusionXLPipeline",
        "LatentConsistencyModelPipeline",
        "FluxPipeline",
    }
)


@dataclass(frozen=True)
class TaskDetection:
    """What a repository declares itself to be.

    ``pipeline_class`` is None for anything that is not a diffusion pipeline, and
    is the ``_class_name`` from ``model_index.json`` when it is -- that string is
    what decides which components exist and in what order they run.
    """

    task: str
    reason: str
    pipeline_class: str | None = None
    components: tuple[str, ...] = ()

    @property
    def is_text_to_image(self) -> bool:
        return self.pipeline_class in TEXT_TO_IMAGE_PIPELINES


def detect_diffusion(model_id: str) -> TaskDetection | None:
    """Detect a diffusion pipeline from its ``model_index.json``, or return None.

    Returns None rather than raising when the repository is not a diffusion
    pipeline, so a caller can fall through to its own detection. The one thing
    this does raise on is a missing ``diffusers``, because "we could not tell"
    and "it is not a diffusion model" are different answers and only one of them
    is fixed by installing an extra.
    """
    diffusers = load_diffusers()
    try:
        config = diffusers.DiffusionPipeline.load_config(model_id)
    except Exception:  # noqa: BLE001 - any failure here means "not a pipeline"
        return None
    if not isinstance(config, dict):
        return None
    pipeline_class = config.get("_class_name")
    if not isinstance(pipeline_class, str):
        return None
    # A diffusers config maps component name -> [library, class]. Keys starting
    # with an underscore are metadata (`_class_name`, `_diffusers_version`), and
    # a component whose value is [None, None] is declared-but-absent.
    components = tuple(
        sorted(
            name
            for name, value in config.items()
            if not name.startswith("_")
            and isinstance(value, (list, tuple))
            and len(value) == 2
            and value[0] is not None
        )
    )
    return TaskDetection(
        task=DIFFUSION,
        reason=f"The repository declares a {pipeline_class} in model_index.json.",
        pipeline_class=pipeline_class,
        components=components,
    )


def detect_task(model_id: str) -> TaskDetection:
    """Classify a Hugging Face repository from configuration alone.

    Only the diffusion branch is decided here. Causal-LM detection stays in
    ``compatibility.py``, which needs the loaded ``AutoConfig`` for the dozen
    other fields it reports and would otherwise fetch it twice.
    """
    try:
        detection = detect_diffusion(model_id)
    except UnsupportedModelError as exc:
        return TaskDetection(task=UNKNOWN, reason=str(exc))
    if detection is not None:
        return detection
    return TaskDetection(
        task=UNKNOWN,
        reason="The repository does not declare a diffusers pipeline.",
    )
