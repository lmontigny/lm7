import json
from pathlib import Path

import pytest
import torch

import lm7
from lm7.backends import registry
from lm7.backends.aot_inductor import AOTInductorBackend
from lm7.errors import ArtifactLoadError, BackendUnavailableError, CompilationError


def model():
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()


def test_backend_is_registered_and_reports_support():
    backend = registry.get("aot_inductor")
    assert isinstance(backend, AOTInductorBackend)
    assert backend.probe().available is True
    assert any(info["name"] == "aot_inductor" for info in lm7.backends())


def test_aot_export_and_load_with_package_api(tmp_path, monkeypatch):
    source = model()
    example = torch.randn(2, 4)
    expected = source(example)

    def fake_compile(exported_program, *, package_path, inductor_configs):
        Path(package_path).write_bytes(b"fake compiled package")
        return str(package_path)

    def fake_load(package_path):
        assert Path(package_path).read_bytes() == b"fake compiled package"
        return source

    monkeypatch.setattr(torch._inductor, "aoti_compile_and_package", fake_compile)
    monkeypatch.setattr(torch._inductor, "aoti_load_package", fake_load)

    artifact = lm7.export(
        source,
        args=(example,),
        target="cpu",
        backend="aot_inductor",
        output=tmp_path / "model.lm7",
        options={"max_autotune": False},
    )

    assert artifact.manifest.backend == "aot_inductor"
    assert artifact.manifest.compiled_file == "compiled_model.pt2"
    assert artifact.manifest.compiled_sha256
    torch.testing.assert_close(artifact(example), expected)

    loaded = lm7.load_artifact(artifact.path)
    torch.testing.assert_close(loaded(example), expected)


def test_compiled_checksum_is_validated(tmp_path, monkeypatch):
    source = model()
    example = torch.randn(2, 4)

    def fake_compile(exported_program, *, package_path, inductor_configs):
        Path(package_path).write_bytes(b"package")
        return str(package_path)

    monkeypatch.setattr(torch._inductor, "aoti_compile_and_package", fake_compile)
    monkeypatch.setattr(torch._inductor, "aoti_load_package", lambda path: source)
    artifact = lm7.export(
        source,
        args=(example,),
        target="cpu",
        backend="aot_inductor",
        output=tmp_path / "model.lm7",
    )
    (artifact.path / "compiled_model.pt2").write_bytes(b"tampered")

    with pytest.raises(ArtifactLoadError, match="compiled package checksum"):
        lm7.load_artifact(artifact.path)


def test_aot_export_rejects_unvalidated_target(tmp_path):
    with pytest.raises(BackendUnavailableError, match="CPU targets"):
        lm7.export(
            model(),
            args=(torch.randn(1, 4),),
            target="nvidia:sm90",
            backend="aot_inductor",
            output=tmp_path / "model.lm7",
        )


def test_aot_compile_failure_leaves_no_output(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("C++ compiler missing")

    monkeypatch.setattr(torch._inductor, "aoti_compile_and_package", fail)
    output = tmp_path / "model.lm7"
    with pytest.raises(CompilationError, match="C\\+\\+ compiler missing"):
        lm7.export(
            model(),
            args=(torch.randn(1, 4),),
            target="cpu",
            backend="aot_inductor",
            output=output,
        )
    assert not output.exists()


def test_source_only_manifest_remains_loadable(tmp_path):
    source = model()
    example = torch.randn(2, 4)
    artifact = lm7.export(
        source,
        args=(example,),
        target="cpu",
        output=tmp_path / "source.lm7",
    )
    assert artifact.manifest.backend == "export"
    torch.testing.assert_close(lm7.load_artifact(artifact.path)(example), source(example))


def test_pr2_manifest_without_backend_fields_remains_loadable(tmp_path):
    source = model()
    example = torch.randn(2, 4)
    artifact = lm7.export(
        source,
        args=(example,),
        target="cpu",
        output=tmp_path / "source.lm7",
    )
    manifest_path = artifact.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in (
        "backend",
        "backend_version",
        "compiled_file",
        "compiled_sha256",
        "runtime_requirements",
    ):
        manifest.pop(field)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = lm7.load_artifact(artifact.path)
    assert loaded.manifest.backend == "export"
    torch.testing.assert_close(loaded(example), source(example))
