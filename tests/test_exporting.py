import json

import pytest
import torch

import lm7
from lm7.errors import ArtifactLoadError
from lm7.exporting import artifact_cache_key
from lm7.targets import TargetSpec


def model():
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()


def test_export_and_load_round_trip(tmp_path):
    torch.manual_seed(0)
    source = model()
    example = torch.randn(2, 4)
    expected = source(example)

    artifact = lm7.export(
        source,
        args=(example,),
        target="cpu",
        output=tmp_path / "model.lm7",
    )

    assert artifact.path.name == "model.lm7"
    assert artifact.manifest.format_version == 1
    assert artifact.manifest.target["vendor"] == "cpu"
    assert (artifact.path / "manifest.json").is_file()
    assert (artifact.path / "exported_program.pt2").is_file()

    loaded = lm7.load_artifact(artifact.path)
    actual = loaded.module()(example)
    torch.testing.assert_close(actual, expected)


def test_accepts_exported_program(tmp_path):
    source = model()
    example = torch.randn(2, 4)
    exported_program = torch.export.export(source, (example,))

    artifact = lm7.export(
        exported_program,
        target="cpu",
        output=tmp_path / "program.lm7",
    )

    torch.testing.assert_close(artifact.module()(example), source(example))
    assert artifact.manifest.input_signature is None


def test_existing_output_is_not_overwritten(tmp_path):
    output = tmp_path / "existing.lm7"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        lm7.export(model(), args=(torch.randn(1, 4),), target="cpu", output=output)


def test_modified_program_fails_checksum_validation(tmp_path):
    artifact = lm7.export(
        model(),
        args=(torch.randn(1, 4),),
        target="cpu",
        output=tmp_path / "model.lm7",
    )
    with (artifact.path / "exported_program.pt2").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ArtifactLoadError, match="checksum"):
        lm7.load_artifact(artifact.path)


def test_unsupported_format_is_rejected(tmp_path):
    artifact = lm7.export(
        model(),
        args=(torch.randn(1, 4),),
        target="cpu",
        output=tmp_path / "model.lm7",
    )
    manifest_path = artifact.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactLoadError, match="Unsupported LM7 artifact format"):
        lm7.load_artifact(artifact.path)


def test_cache_key_changes_with_target_and_signature():
    graph_hash = "abc"
    cpu = TargetSpec("cpu", "cpu")
    gpu = TargetSpec("nvidia", "gpu", architecture="sm90")
    signature = (("tensor", (2, 4), "torch.float32"),)

    assert artifact_cache_key(graph_hash, signature, cpu) == artifact_cache_key(
        graph_hash, signature, cpu
    )
    assert artifact_cache_key(graph_hash, signature, cpu) != artifact_cache_key(
        graph_hash, signature, gpu
    )
    assert artifact_cache_key(graph_hash, signature, cpu) != artifact_cache_key(
        graph_hash, (("tensor", (4, 4), "torch.float32"),), cpu
    )


def test_module_export_requires_args(tmp_path):
    with pytest.raises(ValueError, match="args must be supplied"):
        lm7.export(model(), target="cpu", output=tmp_path / "model.lm7")


def test_shape_profile_is_persisted_and_validated(tmp_path):
    source = model()
    profile = lm7.ShapeProfile({"input": {0: lm7.DynamicDimension("batch", min=1, max=4)}})
    artifact = lm7.export(
        source,
        args=(torch.randn(2, 4),),
        target="cpu",
        output=tmp_path / "dynamic.lm7",
        shape_profile=profile,
    )

    assert artifact.manifest.shape_profile == {
        "argument_order": ["input"],
        "inputs": {
            "input": {
                "0": {
                    "name": "batch",
                    "min": 1,
                    "max": 4,
                }
            }
        },
    }
    assert artifact(torch.randn(3, 4)).shape == (3, 3)

    loaded = lm7.load_artifact(artifact.path)
    assert loaded(torch.randn(4, 4)).shape == (4, 3)
    with pytest.raises(ValueError, match=r"expected \[1, 4\]"):
        loaded(torch.randn(5, 4))


def test_shape_profile_rejects_unknown_inputs(tmp_path):
    profile = lm7.ShapeProfile({"missing": {0: lm7.DynamicDimension("batch", min=1, max=4)}})
    with pytest.raises(ValueError, match="unknown or unbound"):
        lm7.export(
            model(),
            args=(torch.randn(2, 4),),
            target="cpu",
            output=tmp_path / "dynamic.lm7",
            shape_profile=profile,
        )


def test_shape_profile_and_raw_dynamic_shapes_are_mutually_exclusive(tmp_path):
    profile = lm7.ShapeProfile({"input": {0: lm7.DynamicDimension("batch", min=1, max=4)}})
    with pytest.raises(ValueError, match="cannot be supplied together"):
        lm7.export(
            model(),
            args=(torch.randn(2, 4),),
            target="cpu",
            output=tmp_path / "dynamic.lm7",
            shape_profile=profile,
            dynamic_shapes={"input": None},
        )
