import json
import os
import zipfile
from pathlib import Path

import pytest
import torch

import lm7
from lm7 import exporting
from lm7.backends import aot_inductor, registry
from lm7.backends.aot_inductor import AOTInductorBackend
from lm7.backends.base import CompileRequest
from lm7.errors import ArtifactLoadError, BackendUnavailableError, CompilationError
from lm7.targets import parse_target


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
        assert isinstance(package_path, str)
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
        assert isinstance(package_path, str)
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


def build_environment(monkeypatch, *, torch_version, cuda, architecture):
    """Pin what this process looks like, so the hint is the same everywhere.

    Without this the assertions would depend on the test machine's PyTorch and
    GPU, and the CPU-only CI runner and a developer's card would disagree.
    """
    monkeypatch.setattr(torch, "__version__", torch_version)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(aot_inductor, "_current_compute_capability", lambda: architecture)


def test_nvidia_artifacts_record_what_they_were_built_against(monkeypatch):
    monkeypatch.setattr(
        exporting,
        "_cuda_device_requirements",
        lambda: {
            "cuda": "13.0",
            "compute_capability": "sm120",
            "device_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        },
    )

    # A CPU or Apple package is bound to its host too, but LM7 has not measured
    # how, so it claims nothing rather than guessing.
    assert exporting._aot_inductor_requirements(parse_target("cpu")) == {}

    requirements = exporting._aot_inductor_requirements(parse_target("nvidia:sm120"))
    assert requirements["device_bound"] is True
    assert requirements["cuda"] == "13.0"
    assert requirements["compute_capability"] == "sm120"
    assert "Blackwell" in requirements["device_name"]


def test_amd_artifacts_record_the_rocm_pair_and_not_the_cuda_one(monkeypatch):
    """An AMD payload is as device-bound as a CUDA one, and was recording nothing.

    Reusing the CUDA fields would have been worse than the gap it replaced:
    `torch.version.cuda` is None on ROCm, and `get_device_capability` returns
    (9, 4) for a gfx942, so the manifest would have claimed no runtime and an
    `sm94` architecture that no NVIDIA part has ever had.
    """
    monkeypatch.setattr(
        exporting,
        "_rocm_device_requirements",
        lambda: {
            "hip": "7.0.51831",
            "gcn_architecture": "gfx942",
            "device_name": "AMD Instinct MI300X",
        },
    )

    requirements = exporting._aot_inductor_requirements(parse_target("amd:gfx942"))
    assert requirements["device_bound"] is True
    assert requirements["hip"] == "7.0.51831"
    assert requirements["gcn_architecture"] == "gfx942"
    assert "MI300X" in requirements["device_name"]
    # The two vendors' pairs never appear together.
    assert "cuda" not in requirements
    assert "compute_capability" not in requirements


def test_load_failure_names_the_field_that_moved(monkeypatch, tmp_path):
    build_environment(monkeypatch, torch_version="2.13.0+cu130", cuda="13.0", architecture="sm89")

    def refuse(path):
        raise RuntimeError("no kernel image is available for execution on the device")

    monkeypatch.setattr(torch._inductor, "aoti_load_package", refuse)

    with pytest.raises(ArtifactLoadError) as failure:
        AOTInductorBackend().load_package(
            tmp_path / "compiled_model.pt2",
            built_with={
                "torch": "2.13.0+cu130",
                "cuda": "13.0",
                "compute_capability": "sm120",
            },
        )

    message = str(failure.value)
    assert "no kernel image is available" in message
    assert "GPU architecture sm120 -> sm89" in message
    assert "re-export the model" in message
    # PyTorch and CUDA match here, so neither belongs in the list of differences.
    assert "PyTorch 2.13.0+cu130 ->" not in message


def test_load_failure_does_not_blame_a_matching_environment(monkeypatch, tmp_path):
    build_environment(monkeypatch, torch_version="2.13.0+cu130", cuda="13.0", architecture="sm120")

    def refuse(path):
        raise RuntimeError("boom")

    monkeypatch.setattr(torch._inductor, "aoti_load_package", refuse)

    with pytest.raises(ArtifactLoadError) as failure:
        AOTInductorBackend().load_package(
            tmp_path / "compiled_model.pt2",
            built_with={
                "torch": "2.13.0+cu130",
                "cuda": "13.0",
                "compute_capability": "sm120",
            },
        )

    message = str(failure.value)
    assert "which is what this process has" in message
    assert "re-export" not in message


def test_load_failure_without_recorded_metadata_keeps_the_general_hint(monkeypatch, tmp_path):
    # Artifacts written before the manifest recorded a build environment still
    # have to fail readably.
    def refuse(path):
        raise RuntimeError("boom")

    monkeypatch.setattr(torch._inductor, "aoti_load_package", refuse)

    with pytest.raises(ArtifactLoadError, match="compatible PyTorch runtime"):
        AOTInductorBackend().load_package(tmp_path / "compiled_model.pt2")


def test_load_artifact_reports_the_manifest_build_environment(tmp_path, monkeypatch):
    """The manifest's record has to reach the error, not just the file."""
    source = model()

    def fake_compile(exported_program, *, package_path, inductor_configs):
        Path(package_path).write_bytes(b"package")
        return str(package_path)

    monkeypatch.setattr(torch._inductor, "aoti_compile_and_package", fake_compile)
    monkeypatch.setattr(torch._inductor, "aoti_load_package", lambda path: source)
    artifact = lm7.export(
        source,
        args=(torch.randn(2, 4),),
        target="cpu",
        backend="aot_inductor",
        output=tmp_path / "model.lm7",
    )
    recorded = artifact.manifest.runtime_requirements["torch"]

    def refuse(path):
        raise RuntimeError("undefined symbol")

    monkeypatch.setattr(torch._inductor, "aoti_load_package", refuse)
    with pytest.raises(ArtifactLoadError) as failure:
        lm7.load_artifact(artifact.path)

    assert f"PyTorch {recorded}" in str(failure.value)


def test_aot_export_rejects_unsupported_target(tmp_path):
    """`amd:gfx942` used to be the case here and is now supported, so the refusal
    is demonstrated on a vendor AOTInductor genuinely has no path to: PyTorch has
    no Tenstorrent device for it to lower to."""
    with pytest.raises(BackendUnavailableError, match="CPU, Apple Silicon, NVIDIA, and AMD"):
        lm7.export(
            model(),
            args=(torch.randn(1, 4),),
            target="tenstorrent:blackhole",
            backend="aot_inductor",
            output=tmp_path / "model.lm7",
        )


def cuda_request() -> CompileRequest:
    return CompileRequest(
        model=model(),
        target=parse_target("nvidia:sm89"),
        mode="lazy",
        transfers="automatic",
        fallback="error",
    )


def write_cuda_toolkit(root: Path) -> Path:
    for header in ("include/crt/host_defines.h", "include/nv/target"):
        path = root / header
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return root


def test_cuda_support_requires_a_toolkit(monkeypatch, tmp_path):
    backend = AOTInductorBackend()
    monkeypatch.setattr(aot_inductor, "_cuda_toolkit_home", lambda: None)
    support = backend.supports(cuda_request())
    assert support.supported is False
    assert "CUDA toolkit" in support.reason
    assert "cuda-aot" in support.reason

    monkeypatch.setattr(aot_inductor, "_cuda_toolkit_home", lambda: tmp_path)
    support = backend.supports(cuda_request())
    assert support.supported is True
    assert support.priority == 90


def amd_request() -> CompileRequest:
    return CompileRequest(
        model=model(),
        target=parse_target("amd:gfx942"),
        mode="lazy",
        transfers="automatic",
        fallback="error",
    )


def test_amd_support_requires_a_rocm_installation(monkeypatch, tmp_path):
    """The AMD counterpart of the CUDA toolkit gate, and not the same problem.

    The CUDA case is a *partial* toolkit: the PyTorch wheel bundles the runtime
    headers and omits the compiler front end, so LM7 has to find and splice in
    the missing half. ROCm ships as one tree that either is or is not installed.
    """
    backend = AOTInductorBackend()
    monkeypatch.setattr(aot_inductor, "_rocm_home", lambda: None)
    support = backend.supports(amd_request())
    assert support.supported is False
    assert "ROCm" in support.reason
    assert "ROCM_HOME" in support.reason

    monkeypatch.setattr(aot_inductor, "_rocm_home", lambda: tmp_path)
    support = backend.supports(amd_request())
    assert support.supported is True
    assert support.priority == 90


def test_amd_compile_fails_before_packaging_without_rocm(monkeypatch, tmp_path):
    monkeypatch.setattr(aot_inductor, "_rocm_home", lambda: None)

    def unreachable(*args, **kwargs):
        raise AssertionError("packaging must not start without a ROCm installation")

    monkeypatch.setattr(torch._inductor, "aoti_compile_and_package", unreachable)
    exported = torch.export.export(model(), (torch.randn(2, 4),))

    with pytest.raises(CompilationError, match="no ROCm installation was found"):
        AOTInductorBackend().compile_exported(
            exported, tmp_path / "model.pt2", target=parse_target("amd:gfx942")
        )


def test_rocm_home_prefers_the_environment_over_the_default_root(monkeypatch, tmp_path):
    monkeypatch.delenv("ROCM_HOME", raising=False)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.setattr(aot_inductor, "_ROCM_DEFAULT_ROOT", tmp_path / "absent")
    assert aot_inductor._rocm_home() is None

    installed = tmp_path / "opt-rocm"
    installed.mkdir()
    monkeypatch.setattr(aot_inductor, "_ROCM_DEFAULT_ROOT", installed)
    assert aot_inductor._rocm_home() == installed

    explicit = tmp_path / "explicit"
    explicit.mkdir()
    monkeypatch.setenv("ROCM_HOME", str(explicit))
    assert aot_inductor._rocm_home() == explicit

    # A path that does not exist is not an installation, so discovery continues.
    monkeypatch.setenv("ROCM_HOME", str(tmp_path / "nowhere"))
    assert aot_inductor._rocm_home() == installed


def test_compute_capability_is_not_invented_for_a_rocm_host(monkeypatch):
    """`torch.cuda.get_device_capability` answers on ROCm -- it returns (9, 4) on
    a gfx942 -- so calling it unconditionally would record `sm94`, an NVIDIA
    architecture that has never existed."""
    monkeypatch.setattr(torch.version, "hip", "7.0-test")
    assert aot_inductor._current_compute_capability() is None


def test_environment_mismatch_hint_names_rocm_for_an_amd_artifact():
    """The hint told every reader to check their CUDA runtime. For an artifact
    built on ROCm that names a runtime the host does not have and never had."""
    hint = aot_inductor._environment_mismatch_hint(
        {"torch": "2.13.0+rocm7.0", "hip": "7.0", "gcn_architecture": "gfx942"}
    )
    assert "ROCm runtime 7.0" in hint
    assert "GPU architecture gfx942" in hint
    assert "one ROCm runtime" in hint
    assert "CUDA" not in hint


def test_cuda_compile_fails_before_packaging_without_a_toolkit(monkeypatch, tmp_path):
    monkeypatch.setattr(aot_inductor, "_cuda_toolkit_home", lambda: None)

    def unreachable(*args, **kwargs):
        raise AssertionError("packaging must not start without a CUDA toolkit")

    monkeypatch.setattr(torch._inductor, "aoti_compile_and_package", unreachable)
    exported = torch.export.export(model(), (torch.randn(2, 4),))

    with pytest.raises(CompilationError, match="no CUDA toolkit was found"):
        AOTInductorBackend().compile_exported(
            exported, tmp_path / "model.pt2", target=parse_target("nvidia:sm89")
        )


def test_cuda_toolkit_discovery_needs_every_header(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    assert aot_inductor._cuda_toolkit_home() != tmp_path

    (tmp_path / "include" / "crt").mkdir(parents=True)
    (tmp_path / "include" / "crt" / "host_defines.h").write_text("", encoding="utf-8")
    assert aot_inductor._cuda_toolkit_home() != tmp_path

    write_cuda_toolkit(tmp_path)
    assert aot_inductor._cuda_toolkit_home() == tmp_path


def test_cuda_driver_library_dirs_need_a_linkable_stub(monkeypatch, tmp_path):
    # Pin the WSL candidate away from the host so the result is the same whether
    # or not the test machine has a WSL driver directory.
    monkeypatch.setattr(aot_inductor, "_WSL_DRIVER_DIR", tmp_path / "absent")
    stubs = tmp_path / "lib64" / "stubs"
    stubs.mkdir(parents=True)
    assert aot_inductor._cuda_driver_library_dirs(tmp_path) == []

    (stubs / "libcuda.so").write_text("", encoding="utf-8")
    assert aot_inductor._cuda_driver_library_dirs(tmp_path) == [stubs]


def test_cuda_build_environment_fills_gaps_and_restores(monkeypatch, tmp_path):
    toolkit = write_cuda_toolkit(tmp_path / "toolkit")
    stubs = toolkit / "lib64" / "stubs"
    stubs.mkdir(parents=True)
    (stubs / "libcuda.so").write_text("", encoding="utf-8")
    monkeypatch.setattr(aot_inductor, "_cuda_toolkit_home", lambda: toolkit)
    monkeypatch.setattr(aot_inductor, "_WSL_DRIVER_DIR", tmp_path / "absent")
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv("LIBRARY_PATH", "/existing")

    with aot_inductor._cuda_build_environment(parse_target("nvidia:sm89")):
        assert os.environ["CUDA_HOME"] == str(toolkit)
        assert os.environ["LIBRARY_PATH"].split(os.pathsep) == [str(stubs), "/existing"]

    assert "CUDA_HOME" not in os.environ
    assert os.environ["LIBRARY_PATH"] == "/existing"


def test_cuda_build_environment_keeps_an_explicit_toolkit(monkeypatch, tmp_path):
    monkeypatch.setattr(aot_inductor, "_cuda_toolkit_home", lambda: tmp_path / "discovered")
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda")

    with aot_inductor._cuda_build_environment(parse_target("nvidia:sm89")):
        assert os.environ["CUDA_HOME"] == "/opt/cuda"


def test_cpu_compile_leaves_the_cuda_environment_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(aot_inductor, "_cuda_toolkit_home", lambda: tmp_path)
    monkeypatch.delenv("CUDA_HOME", raising=False)

    with aot_inductor._cuda_build_environment(parse_target("cpu")):
        assert "CUDA_HOME" not in os.environ


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


def test_aot_compile_failure_can_preserve_debug_output(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(torch._inductor, "aoti_compile_and_package", fail)
    debug_output = tmp_path / "failure-debug"
    monkeypatch.setenv("LM7_DEBUG_FAILURE_DIR", str(debug_output))
    with pytest.raises(CompilationError, match="compiler failed"):
        lm7.export(
            model(),
            args=(torch.randn(1, 4),),
            target="cpu",
            backend="aot_inductor",
            output=tmp_path / "model.lm7",
            debug=True,
        )

    assert (debug_output / "exported_program.txt").is_file()
    assert (debug_output / "exported_graph.py").is_file()


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
