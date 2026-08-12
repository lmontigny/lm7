"""Gates and artifact metadata for exported KV-cache decode steps. No downloads.

The thing this feature can get wrong is not loud. A decode artifact whose cache
writes were dropped somewhere in lowering still exports, still loads, still
returns a plausible first token -- the cache is empty at that point, so there is
nothing to have lost -- and only then diverges. So the gates here are deliberately
refusals rather than warnings, and the one test that can actually catch a dropped
write lives in ``test_export_decode_integration.py``, where a real model decodes
against eager.
"""

from __future__ import annotations

import json

import pytest
import torch

from lm7.cli import _build_parser
from lm7.errors import BackendUnavailableError, UnsupportedModelError
from lm7.exporting import DECODE_BACKENDS, ArtifactManifest, export
from lm7.huggingface import DEFAULT_MAX_CACHE_LEN, export_hf_model
from lm7.inspection import inspect_artifact

TINY = "hf://hf-internal-testing/tiny-random-LlamaForCausalLM"


class Doubler(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


# -- what a decode export refuses ----------------------------------------


@pytest.mark.parametrize("backend", ("executorch", "openvino", "tensorrt", "coreml", "litert"))
def test_decode_export_refuses_an_unmeasured_backend(backend):
    """Unmeasured is not the same as broken, and neither is worth guessing.

    Every one of these lowers through its own vendor toolchain, and none has been
    run against a graph that mutates buffers it carries.
    """
    with pytest.raises(UnsupportedModelError, match="validated for these backends"):
        export_hf_model(TINY, output="unused.lm7", backend=backend, decode=True)


def test_decode_export_refuses_a_dynamic_sequence():
    with pytest.raises(UnsupportedModelError, match="one token per call"):
        export_hf_model(TINY, output="unused.lm7", decode=True, dynamic_sequence=True)


def test_decode_export_refuses_quantization():
    with pytest.raises(UnsupportedModelError, match="Quantized decode export"):
        export_hf_model(TINY, output="unused.lm7", decode=True, quantization="int8")


def test_decode_export_refuses_a_cache_with_no_room_to_decode():
    with pytest.raises(UnsupportedModelError, match="max_cache_len must be at least 2"):
        export_hf_model(TINY, output="unused.lm7", decode=True, max_cache_len=1)


def test_the_export_api_refuses_decode_metadata_on_an_unmeasured_backend(tmp_path):
    """The gate is on `export` too, not only on the Hugging Face wrapper.

    `lm7.export` takes any nn.Module, so a caller who wraps their own decode step
    reaches the same hazard without passing through `export_hf_model`.
    """
    with pytest.raises(BackendUnavailableError, match="preserve those writes"):
        export(
            Doubler(),
            args=(torch.ones(2),),
            output=str(tmp_path / "art.lm7"),
            backend="openvino",
            decode={"batch_size": 1, "max_cache_len": 8},
        )


def test_every_decode_backend_is_an_export_backend():
    from lm7.exporting import EXPORT_BACKENDS

    assert DECODE_BACKENDS <= EXPORT_BACKENDS


# -- what a decode artifact records --------------------------------------


def test_decode_metadata_reaches_the_manifest_and_inspection(tmp_path):
    """A stateful artifact has to say so on disk.

    Nothing about the input signature distinguishes a decode step from any other
    two-tensor graph, so a reader who did not export it has no way to know that
    calling it twice is two different things happening.
    """
    artifact = export(
        Doubler(),
        args=(torch.ones(2),),
        output=str(tmp_path / "art.lm7"),
        backend="export",
        decode={"batch_size": 1, "max_cache_len": 64, "cache_bytes": 4096},
    )
    assert artifact.manifest.decode == {
        "batch_size": 1,
        "max_cache_len": 64,
        "cache_bytes": 4096,
    }

    written = json.loads((artifact.path / "manifest.json").read_text())
    assert written["decode"]["max_cache_len"] == 64

    inspection = inspect_artifact(artifact.path)
    assert inspection.decode is not None
    assert inspection.decode["max_cache_len"] == 64
    assert inspection.to_dict()["decode"]["cache_bytes"] == 4096


def test_an_ordinary_artifact_carries_no_decode_metadata(tmp_path):
    artifact = export(
        Doubler(), args=(torch.ones(2),), output=str(tmp_path / "art.lm7"), backend="export"
    )
    assert artifact.manifest.decode is None
    assert inspect_artifact(artifact.path).decode is None
    assert inspect_artifact(artifact.path).to_dict()["decode"] is None


def test_a_manifest_written_before_decode_existed_still_loads():
    """`decode` is optional, so an artifact from an earlier LM7 is not invalidated."""
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
    assert manifest.decode is None


# -- the artifact is built once ------------------------------------------


def test_the_unlifted_module_is_built_once(tmp_path):
    """`ExportedProgram.module()` re-runs the unlifting pass on every call.

    Invisible waste for a forward pass and per-token waste for a decode loop,
    which is the one workload whose whole cost model is per-token.
    """
    artifact = export(
        Doubler(), args=(torch.ones(2),), output=str(tmp_path / "art.lm7"), backend="export"
    )
    assert artifact.module() is artifact.module()
    assert torch.equal(artifact(torch.ones(2)), torch.full((2,), 2.0))


# -- the CLI surface -----------------------------------------------------


def test_the_cli_exposes_decode_and_a_cache_length():
    args = _build_parser().parse_args(
        ["model", "export", TINY, "out.lm7", "--decode", "--max-cache-len", "256"]
    )
    assert args.decode is True
    assert args.max_cache_len == 256


def test_the_cache_length_defaults_to_the_same_number_compile_generation_uses():
    args = _build_parser().parse_args(["model", "export", TINY, "out.lm7"])
    assert args.decode is False
    assert args.max_cache_len == DEFAULT_MAX_CACHE_LEN

    import inspect as inspect_module

    from lm7.generation import compile_generation

    default = inspect_module.signature(compile_generation).parameters["max_sequence_length"].default
    assert default == DEFAULT_MAX_CACHE_LEN
