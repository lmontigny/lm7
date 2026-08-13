from __future__ import annotations

from types import SimpleNamespace

import pytest

from lm7 import tasks
from lm7.errors import UnsupportedModelError

# What `DiffusionPipeline.load_config` hands back for stabilityai/sd-turbo: the
# component map from model_index.json, with metadata keys prefixed by an
# underscore and each component a [library, class] pair.
SD_MODEL_INDEX = {
    "_class_name": "StableDiffusionPipeline",
    "_diffusers_version": "0.27.0",
    "scheduler": ["diffusers", "EulerAncestralDiscreteScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "unet": ["diffusers", "UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
    "safety_checker": [None, None],
}


def fake_diffusers(config=None, *, error=None):
    class DiffusionPipeline:
        @staticmethod
        def load_config(model_id):
            assert model_id == "stabilityai/sd-turbo"
            if error is not None:
                raise error
            return config

    return SimpleNamespace(DiffusionPipeline=DiffusionPipeline)


def test_a_stable_diffusion_repository_is_detected_from_its_model_index(monkeypatch):
    monkeypatch.setattr(tasks, "load_diffusers", lambda: fake_diffusers(SD_MODEL_INDEX))

    detection = tasks.detect_diffusion("stabilityai/sd-turbo")

    assert detection is not None
    assert detection.task == tasks.DIFFUSION
    assert detection.pipeline_class == "StableDiffusionPipeline"
    assert detection.is_text_to_image is True
    assert "model_index.json" in detection.reason


def test_declared_but_absent_components_are_not_reported_as_present(monkeypatch):
    """`safety_checker: [None, None]` means the repo names it and ships nothing."""
    monkeypatch.setattr(tasks, "load_diffusers", lambda: fake_diffusers(SD_MODEL_INDEX))

    detection = tasks.detect_diffusion("stabilityai/sd-turbo")

    assert detection is not None
    assert detection.components == ("scheduler", "text_encoder", "tokenizer", "unet", "vae")
    assert "safety_checker" not in detection.components


def test_a_repository_that_is_not_a_pipeline_detects_as_none(monkeypatch):
    """`load_config` raising is how diffusers says "no model_index.json here"."""
    monkeypatch.setattr(
        tasks,
        "load_diffusers",
        lambda: fake_diffusers(error=OSError("model_index.json not found")),
    )

    assert tasks.detect_diffusion("stabilityai/sd-turbo") is None


def test_a_config_without_a_class_name_detects_as_none(monkeypatch):
    monkeypatch.setattr(tasks, "load_diffusers", lambda: fake_diffusers({"unet": ["d", "U"]}))

    assert tasks.detect_diffusion("stabilityai/sd-turbo") is None


def test_a_missing_diffusers_extra_raises_rather_than_denying_the_pipeline(monkeypatch):
    """"Cannot tell" and "not a diffusion model" are different answers.

    Returning None here would let a caller report a diffusion repo as unreadable
    for a reason the user cannot act on, so the missing extra is raised instead.
    """

    def missing():
        raise UnsupportedModelError("Diffusion support is not installed.")

    monkeypatch.setattr(tasks, "load_diffusers", missing)

    with pytest.raises(UnsupportedModelError):
        tasks.detect_diffusion("stabilityai/sd-turbo")


def test_detect_task_downgrades_a_missing_extra_to_unknown(monkeypatch):
    def missing():
        raise UnsupportedModelError("Diffusion support is not installed.")

    monkeypatch.setattr(tasks, "load_diffusers", missing)

    detection = tasks.detect_task("stabilityai/sd-turbo")

    assert detection.task == tasks.UNKNOWN
    assert "not installed" in detection.reason


def test_detect_task_reports_unknown_for_a_non_pipeline(monkeypatch):
    monkeypatch.setattr(
        tasks,
        "load_diffusers",
        lambda: fake_diffusers(error=OSError("model_index.json not found")),
    )

    detection = tasks.detect_task("stabilityai/sd-turbo")

    assert detection.task == tasks.UNKNOWN
    assert detection.pipeline_class is None
