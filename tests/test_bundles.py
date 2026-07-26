from __future__ import annotations

import json
import shutil

import pytest
import torch

import lm7
from lm7.errors import ArtifactLoadError


def model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()


def _retarget(source, destination, *, vendor, kind, architecture):
    shutil.copytree(source, destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["target"] = {
        "vendor": vendor,
        "kind": kind,
        "architecture": architecture,
        "model": None,
        "ordinal": 0 if kind == "gpu" else None,
        "remote": False,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return destination


def test_create_and_load_multi_target_bundle(tmp_path):
    torch.manual_seed(0)
    source = model()
    example = torch.randn(2, 4)
    expected = source(example)
    cpu_artifact = lm7.export(
        source,
        args=(example,),
        target="cpu",
        output=tmp_path / "cpu.lm7",
    )
    nvidia_artifact = _retarget(
        cpu_artifact.path,
        tmp_path / "nvidia.lm7",
        vendor="nvidia",
        kind="gpu",
        architecture="sm89",
    )

    bundle = lm7.create_bundle(
        [cpu_artifact.path, nvidia_artifact],
        output=tmp_path / "model.bundle.lm7",
    )

    assert {str(target) for target in bundle.available_targets()} == {
        "cpu",
        "nvidia:sm89",
    }
    loaded_bundle = lm7.load_bundle(bundle.path)
    loaded = loaded_bundle.load(target="cpu")
    torch.testing.assert_close(loaded(example), expected)


def test_auto_uses_best_detected_target_present_in_bundle(tmp_path, monkeypatch):
    artifact = lm7.export(
        model(),
        args=(torch.randn(1, 4),),
        target="cpu",
        output=tmp_path / "cpu.lm7",
    )
    bundle = lm7.create_bundle([artifact.path], output=tmp_path / "bundle.lm7")
    monkeypatch.setattr(
        "lm7.bundles.detect_targets",
        lambda: [
            lm7.DeviceInfo(lm7.TargetSpec("nvidia", "gpu", architecture="sm89"), "GPU"),
            lm7.DeviceInfo(lm7.TargetSpec("cpu", "cpu"), "CPU"),
        ],
    )

    assert bundle.load(target="auto").manifest.target["vendor"] == "cpu"


def test_bundle_rejects_different_model_graphs(tmp_path):
    first = lm7.export(
        torch.nn.Linear(4, 3).eval(),
        args=(torch.randn(1, 4),),
        target="cpu",
        output=tmp_path / "first.lm7",
    )
    second = lm7.export(
        torch.nn.Linear(4, 5).eval(),
        args=(torch.randn(1, 4),),
        target="cpu",
        output=tmp_path / "second.lm7",
    )

    with pytest.raises(ValueError, match="same exported model graph"):
        lm7.create_bundle(
            [first.path, second.path],
            output=tmp_path / "bundle.lm7",
        )


def test_bundle_rejects_duplicate_target_backend_entries(tmp_path):
    artifact = lm7.export(
        model(),
        args=(torch.randn(1, 4),),
        target="cpu",
        output=tmp_path / "model.lm7",
    )

    with pytest.raises(ValueError, match="entries must be unique"):
        lm7.create_bundle(
            [artifact.path, artifact.path],
            output=tmp_path / "bundle.lm7",
        )


def test_bundle_manifest_checksum_is_validated(tmp_path):
    artifact = lm7.export(
        model(),
        args=(torch.randn(1, 4),),
        target="cpu",
        output=tmp_path / "model.lm7",
    )
    bundle = lm7.create_bundle([artifact.path], output=tmp_path / "bundle.lm7")
    entry = bundle.manifest.entries[0]
    nested_manifest = bundle.path / entry["path"] / "manifest.json"
    nested_manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactLoadError, match="manifest checksum"):
        lm7.load_bundle(bundle.path).load(target="cpu")


def test_bundle_does_not_overwrite_existing_output(tmp_path):
    artifact = lm7.export(
        model(),
        args=(torch.randn(1, 4),),
        target="cpu",
        output=tmp_path / "model.lm7",
    )
    output = tmp_path / "bundle.lm7"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        lm7.create_bundle([artifact.path], output=output)
