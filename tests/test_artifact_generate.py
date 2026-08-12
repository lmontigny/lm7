"""`lm7 artifact generate`, and the manifest field that makes it possible. No downloads.

The reason an artifact records what it was built from is that the failure it
prevents is silent: token ids only mean words under the tokenizer the model was
trained with, so the wrong one produces fluent text made of the wrong words and
raises nothing. Every refusal here exists because the alternative is not an error
the run could detect.
"""

from __future__ import annotations

import json

import pytest
import torch

from lm7.artifact_generation import generate_from_artifact
from lm7.cli import _build_parser
from lm7.errors import UnsupportedModelError
from lm7.exporting import ArtifactManifest, export
from lm7.inspection import inspect_artifact

SOURCE = {
    "model_uri": "hf://owner/model",
    "model_id": "owner/model",
    "tokenizer_id": "owner/model",
    "dtype": "float32",
}


class Doubler(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


def _artifact(tmp_path, name="art.lm7", **kwargs):
    return export(
        Doubler(), args=(torch.ones(2),), output=str(tmp_path / name), backend="export", **kwargs
    )


# -- the manifest field --------------------------------------------------


def test_source_reaches_the_manifest_and_inspection(tmp_path):
    artifact = _artifact(tmp_path, source=SOURCE)
    assert artifact.manifest.source == SOURCE

    written = json.loads((artifact.path / "manifest.json").read_text())
    assert written["source"]["tokenizer_id"] == "owner/model"

    inspection = inspect_artifact(artifact.path)
    assert inspection.source is not None
    assert inspection.source["model_uri"] == "hf://owner/model"
    assert inspection.to_dict()["source"]["dtype"] == "float32"


def test_an_artifact_without_a_source_still_loads(tmp_path):
    """Optional, so artifacts written before this field are not invalidated."""
    artifact = _artifact(tmp_path)
    assert artifact.manifest.source is None
    assert inspect_artifact(artifact.path).source is None
    assert inspect_artifact(artifact.path).to_dict()["source"] is None


def test_a_manifest_predating_source_deserializes():
    manifest = ArtifactManifest.from_dict(
        {
            "format_version": 1,
            "lm7_version": "0.1.0",
            "torch_version": "2.13.0",
            "created_at": "2026-01-01T00:00:00+00:00",
            "target": {"vendor": "cpu", "kind": "cpu"},
            "model_graph_hash": "abc",
            "cache_key": "def",
            "input_signature": None,
            "program_file": "exported_program.pt2",
            "program_sha256": "0" * 64,
        }
    )
    assert manifest.source is None
    assert manifest.decode is None


# -- what generate refuses -----------------------------------------------


def test_generating_from_a_plain_forward_pass_is_refused(tmp_path):
    """A prefill artifact has no cache, so there is no sequence to continue.

    It would happily return logits for one call and then repeat itself forever,
    which is a worse answer than a refusal.
    """
    artifact = _artifact(tmp_path, source=SOURCE)
    with pytest.raises(UnsupportedModelError, match="not a decode artifact"):
        generate_from_artifact(artifact.path, prompt="hello")


def test_an_artifact_with_no_recorded_tokenizer_is_refused(tmp_path):
    """The whole point of the field: without it, the ids cannot become text."""
    artifact = _artifact(tmp_path, decode={"max_cache_len": 64, "shape": "dynamic"})
    with pytest.raises(UnsupportedModelError, match="does not record which tokenizer"):
        generate_from_artifact(artifact.path, prompt="hello")


def test_a_budget_larger_than_the_cache_is_refused_before_the_tokenizer_is_fetched(tmp_path):
    """Refused before a Hub round trip, since no prompt is needed to answer it.

    The tokenizer id here is deliberately one that does not exist: if the check
    ever moves back below the tokenizer load, this test fails with a network
    error instead of passing quietly.
    """
    artifact = _artifact(
        tmp_path,
        decode={"max_cache_len": 8, "shape": "dynamic", "max_tokens_per_call": 7},
        source=SOURCE,
    )
    with pytest.raises(UnsupportedModelError, match="tokens of KV cache"):
        generate_from_artifact(artifact.path, prompt="hello", max_new_tokens=1000)


def test_a_zero_token_budget_is_refused(tmp_path):
    artifact = _artifact(tmp_path, decode={"max_cache_len": 64}, source=SOURCE)
    with pytest.raises(ValueError, match="max_new_tokens must be at least 1"):
        generate_from_artifact(artifact.path, prompt="hello", max_new_tokens=0)


# -- the CLI surface -----------------------------------------------------


def test_the_cli_exposes_artifact_generate():
    args = _build_parser().parse_args(
        ["artifact", "generate", "model.lm7", "--prompt", "hi", "--max-new-tokens", "8"]
    )
    assert args.artifact_command == "generate"
    assert args.artifact == "model.lm7"
    assert args.prompt == "hi"
    assert args.max_new_tokens == 8
    assert args.tokenizer is None


def test_the_tokenizer_override_is_available_but_not_the_default():
    args = _build_parser().parse_args(
        ["artifact", "generate", "model.lm7", "--tokenizer", "owner/other"]
    )
    assert args.tokenizer == "owner/other"
