import json
import zipfile
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
        debug_dir = Path(inductor_configs["trace.debug_dir"])
        trace_dir = debug_dir / "model__0"
        trace_dir.mkdir(parents=True)
        (trace_dir / "fx_graph_transformed.py").write_text("graph", encoding="utf-8")
        (trace_dir / "ir_pre_fusion.txt").write_text("pre", encoding="utf-8")
        (trace_dir / "ir_post_fusion.txt").write_text("post", encoding="utf-8")
        (trace_dir / "output_code.cpp").write_text("code", encoding="utf-8")
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
        debug=True,
    )

    assert artifact.manifest.backend == "aot_inductor"
    assert artifact.manifest.compiled_file == "compiled_model.pt2"
    assert artifact.manifest.compiled_sha256
    assert artifact.manifest.debug_requested is True
    assert {item["level"] for item in artifact.manifest.debug_artifacts} >= {
        "export",
        "fx",
        "inductor_ir_pre_fusion",
        "inductor_ir_post_fusion",
        "generated_code",
    }
    assert all(path.is_file() for path in artifact.debug_files())
    torch.testing.assert_close(artifact(example), expected)

    loaded = lm7.load_artifact(artifact.path)
    assert loaded.debug_files() == artifact.debug_files()
    torch.testing.assert_close(loaded(example), expected)


def test_source_export_debug_files(tmp_path):
    source = model()
    example = torch.randn(2, 4)
    artifact = lm7.export(
        source,
        args=(example,),
        target="cpu",
        output=tmp_path / "source.lm7",
        debug=True,
    )

    assert {path.name for path in artifact.debug_files()} == {
        "exported_graph.py",
        "exported_program.txt",
        "graph_signature.txt",
    }
    assert all(item["level"] == "export" for item in artifact.manifest.debug_artifacts)


def test_debug_is_disabled_by_default(tmp_path):
    artifact = lm7.export(
        model(),
        args=(torch.randn(1, 4),),
        target="cpu",
        output=tmp_path / "source.lm7",
    )
    assert artifact.manifest.debug_requested is False
    assert artifact.manifest.debug_artifacts == ()
    assert artifact.debug_files() == ()


def test_debug_extracts_low_level_files_from_pt2_package(tmp_path, monkeypatch):
    source = model()
    example = torch.randn(1, 4)

    def fake_compile(exported_program, *, package_path, inductor_configs):
        with zipfile.ZipFile(package_path, "w") as archive:
            archive.writestr("data/aot/model/kernel.cpp", "cpp")
            archive.writestr("data/aot/model/kernel.ptx", "ptx")
            archive.writestr("data/aot/model/metadata.json", "{}")
        return str(package_path)

    monkeypatch.setattr(torch._inductor, "aoti_compile_and_package", fake_compile)
    monkeypatch.setattr(torch._inductor, "aoti_load_package", lambda path: source)
    artifact = lm7.export(
        source,
        args=(example,),
        target="cpu",
        backend="aot_inductor",
        output=tmp_path / "model.lm7",
        debug=True,
    )

    package_items = [
        item
        for item in artifact.manifest.debug_artifacts
        if item["path"].startswith("debug/package/")
    ]
    assert {item["kind"] for item in package_items} == {"source", "ptx"}
    assert all(item["sha256"] for item in package_items)
    assert not any("metadata.json" in item["path"] for item in package_items)


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
