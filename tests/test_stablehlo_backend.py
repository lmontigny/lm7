from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch

import lm7
import lm7.exporting
from lm7.backends import registry
from lm7.backends.base import BackendInfo, CompileRequest
from lm7.backends.stablehlo import StableHLOBackend
from lm7.errors import ArtifactLoadError, BackendUnavailableError, CompilationError
from lm7.exporting import COMPILED_STABLEHLO_NAME
from lm7.targets import parse_target


def model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()


def request_for(target: str = "cpu") -> CompileRequest:
    return CompileRequest(
        model=model(),
        target=parse_target(target),
        mode="lazy",
        transfers="automatic",
        fallback="error",
    )


def fake_tree(root: Path, *, constants: bool = False, meta: bool = True) -> Path:
    """Build the directory shape torch_xla.stablehlo.save_as_stablehlo produces."""
    (root / "functions").mkdir(parents=True, exist_ok=True)
    (root / "functions" / "forward.bytecode").write_bytes(b"StableHLO bytecode")
    (root / "functions" / "forward.mlir").write_text("module @main {}", encoding="utf-8")
    if meta:
        (root / "functions" / "forward.meta").write_text(
            json.dumps({"input_locations": [], "input_signature": []}), encoding="utf-8"
        )
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "0.weight").write_bytes(b"\x93NUMPY")
    if constants:
        (root / "constants").mkdir(exist_ok=True)
        (root / "constants" / "0").write_bytes(b"\x93NUMPY")
    return root


def install_fake_torch_xla(monkeypatch, *, constants: bool = False, meta: bool = True) -> dict:
    """Stand in for torch_xla so the export path is testable without it.

    torch_xla is ABI-tied to a matching PyTorch and cannot be installed next to
    the PyTorch this suite runs against, so these tests exercise LM7's packaging
    and validation rather than the lowering itself. The real lowering is covered
    by tests/test_stablehlo_integration.py and benchmarks/stablehlo_pjrt.py.
    """
    import sys
    from types import SimpleNamespace

    calls: dict = {}

    def save_as_stablehlo(exported_program, path):
        calls["exported_program"] = exported_program
        calls["path"] = path
        fake_tree(Path(path), constants=constants, meta=meta)

    class StableHLOGraphModule:
        @staticmethod
        def load(path):
            calls["loaded_from"] = path
            source = model()
            return lambda *args, **kwargs: source(*args, **kwargs)

    module = SimpleNamespace(
        save_as_stablehlo=save_as_stablehlo,
        StableHLOGraphModule=StableHLOGraphModule,
    )
    monkeypatch.setitem(sys.modules, "torch_xla", SimpleNamespace(__version__="2.9.0"))
    monkeypatch.setitem(sys.modules, "torch_xla.stablehlo", module)
    monkeypatch.setattr(
        StableHLOBackend,
        "probe",
        lambda self: BackendInfo("stablehlo", "2.9.0", True, "PyTorch/XLA can lower to StableHLO."),
    )
    return calls


def test_backend_is_registered():
    backend = registry.get("stablehlo")
    assert isinstance(backend, StableHLOBackend)
    assert any(info["name"] == "stablehlo" for info in lm7.backends())


def test_probe_reports_the_missing_dependency():
    probe = StableHLOBackend().probe()
    if probe.available:  # torch_xla is genuinely installed in this environment
        pytest.skip("PyTorch/XLA is installed; the unavailable path cannot be exercised")
    assert "PyTorch/XLA is not installed" in probe.reason
    assert "stablehlo" in probe.reason


def test_backend_is_export_only():
    """There is no JIT path here; openxla already fills that role."""
    support = StableHLOBackend().supports(request_for())
    assert support.supported is False
    assert "export-only" in support.reason

    with pytest.raises(CompilationError, match="does not compile in-process"):
        StableHLOBackend().compile(request_for(), (), {})


def test_export_writes_a_zipped_payload(tmp_path, monkeypatch):
    calls = install_fake_torch_xla(monkeypatch)
    output = tmp_path / "model.lm7"

    artifact = lm7.export(
        model(), args=(torch.randn(2, 4),), target="cpu", backend="stablehlo", output=output
    )

    assert artifact.manifest.backend == "stablehlo"
    assert artifact.manifest.backend_version == "2.9.0"
    assert artifact.manifest.compiled_file == COMPILED_STABLEHLO_NAME
    assert artifact.manifest.compiled_sha256
    assert calls["path"]

    with zipfile.ZipFile(output / COMPILED_STABLEHLO_NAME) as archive:
        names = set(archive.namelist())
    assert {"functions/forward.bytecode", "functions/forward.meta"} <= names
    assert any(name.startswith("data/") for name in names)


def test_manifest_records_that_the_payload_is_not_device_bound(tmp_path, monkeypatch):
    """The PJRT plugin is chosen at load time, so one payload serves any vendor."""
    install_fake_torch_xla(monkeypatch)
    artifact = lm7.export(
        model(),
        args=(torch.randn(2, 4),),
        target="cpu",
        backend="stablehlo",
        output=tmp_path / "model.lm7",
    )
    requirements = artifact.manifest.runtime_requirements
    assert requirements["device_bound"] is False
    assert requirements["pjrt_plugin"] == "any"


@pytest.mark.parametrize("target", ["cpu", "nvidia:sm89", "amd:gfx942", "tpu"])
def test_export_does_not_gate_on_vendor(tmp_path, monkeypatch, target):
    """Unlike aot_inductor and openvino, the captured payload is target-independent.

    Only the gate is under test. Whether this host can hold a model on the named
    device is a separate question, so a device-move failure is not a gate failure.
    """
    install_fake_torch_xla(monkeypatch)
    monkeypatch.setattr(
        lm7.exporting, "_artifact_target", lambda value, backend="export": parse_target(target)
    )
    try:
        artifact = lm7.export(
            model(),
            args=(torch.randn(2, 4),),
            target="cpu",
            backend="stablehlo",
            output=tmp_path / f"{target.replace(':', '-')}.lm7",
        )
    except BackendUnavailableError as error:  # pragma: no cover - the failure we assert against
        pytest.fail(f"stablehlo rejected target {target!r}: {error}")
    except (NotImplementedError, RuntimeError, AssertionError):
        pytest.skip(f"this host cannot place a model on {target}")
    else:
        assert artifact.manifest.backend == "stablehlo"


def test_reload_validates_the_payload_checksum(tmp_path, monkeypatch):
    install_fake_torch_xla(monkeypatch)
    output = tmp_path / "model.lm7"
    lm7.export(model(), args=(torch.randn(2, 4),), target="cpu", backend="stablehlo", output=output)
    (output / COMPILED_STABLEHLO_NAME).write_bytes(b"tampered")

    with pytest.raises(ArtifactLoadError, match="checksum"):
        lm7.load_artifact(output)


def test_reload_recreates_directories_a_zip_cannot_store(tmp_path, monkeypatch):
    """A model with no baked constants leaves an empty dir the loader still stats."""
    calls = install_fake_torch_xla(monkeypatch, constants=False)
    output = tmp_path / "model.lm7"
    lm7.export(model(), args=(torch.randn(2, 4),), target="cpu", backend="stablehlo", output=output)

    lm7.load_artifact(output)
    unpacked = Path(calls["loaded_from"])
    assert (unpacked / "constants").is_dir()
    assert (unpacked / "data").is_dir()


def test_packaging_rejects_a_tree_without_a_loadable_program(tmp_path, monkeypatch):
    install_fake_torch_xla(monkeypatch, meta=False)
    with pytest.raises(CompilationError, match="forward.meta"):
        lm7.export(
            model(),
            args=(torch.randn(2, 4),),
            target="cpu",
            backend="stablehlo",
            output=tmp_path / "model.lm7",
        )
    assert not (tmp_path / "model.lm7").exists()


def test_export_requires_pytorch_xla(tmp_path):
    if StableHLOBackend().probe().available:
        pytest.skip("PyTorch/XLA is installed; the unavailable path cannot be exercised")
    with pytest.raises(BackendUnavailableError, match="PyTorch/XLA is not installed"):
        lm7.export(
            model(),
            args=(torch.randn(2, 4),),
            target="cpu",
            backend="stablehlo",
            output=tmp_path / "model.lm7",
        )


def test_program_entries_reports_an_unreadable_package(tmp_path):
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not a zip")
    with pytest.raises(ArtifactLoadError, match="unreadable"):
        StableHLOBackend().program_entries(broken)


def test_export_works_when_the_cache_directory_does_not_exist(tmp_path, monkeypatch):
    """A first run on a clean machine has no LM7 cache directory yet."""
    install_fake_torch_xla(monkeypatch)
    monkeypatch.setenv("LM7_CACHE_DIR", str(tmp_path / "absent" / "cache"))

    artifact = lm7.export(
        model(),
        args=(torch.randn(2, 4),),
        target="cpu",
        backend="stablehlo",
        output=tmp_path / "model.lm7",
    )
    assert artifact.manifest.compiled_file == COMPILED_STABLEHLO_NAME
    lm7.load_artifact(artifact.path)
