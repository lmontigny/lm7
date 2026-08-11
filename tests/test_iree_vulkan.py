from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import lm7
from lm7.backends import iree_vulkan, registry
from lm7.backends.base import BackendInfo, CompileRequest
from lm7.backends.eager import EagerBackend
from lm7.backends.iree_vulkan import IREEVulkanBackend
from lm7.errors import ArtifactLoadError, BackendUnavailableError, CompilationError
from lm7.targets import parse_target


def model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()


def request_for(module: torch.nn.Module | None = None) -> CompileRequest:
    return CompileRequest(
        model=module or model(),
        target=parse_target("nvidia:sm89"),
        mode="lazy",
        transfers="automatic",
        fallback="error",
    )


def available_probe() -> BackendInfo:
    return BackendInfo("iree_vulkan", "3.11.0", True, "available")


def test_backend_is_registered_but_jit_is_not_supported(monkeypatch):
    backend = registry.get("iree_vulkan")
    assert isinstance(backend, IREEVulkanBackend)
    monkeypatch.setattr(backend, "probe", available_probe)

    support = backend.supports(request_for())

    assert not support.supported
    assert "lm7.export" in support.reason
    with pytest.raises(CompilationError, match="export-only"):
        backend.compile(request_for(), (), {})


def test_probe_reports_missing_optional_packages(monkeypatch):
    monkeypatch.setattr(iree_vulkan, "_has_module", lambda _name: False)

    probe = IREEVulkanBackend().probe()

    assert not probe.available
    assert "lm7[iree-vulkan]" in probe.reason
    assert "iree-base-compiler" in probe.reason
    assert "iree-base-runtime" in probe.reason
    assert "iree-turbine" in probe.reason


def test_probe_distinguishes_offline_compilation_from_runtime_devices(monkeypatch):
    monkeypatch.setattr(iree_vulkan, "_has_module", lambda _name: True)
    monkeypatch.setattr(iree_vulkan, "_package_version", lambda _name: "3.11.0")
    monkeypatch.setattr(iree_vulkan, "query_vulkan_devices", lambda: ())

    probe = IREEVulkanBackend().probe()

    assert probe.available
    assert "compile" in probe.reason
    assert "no Vulkan devices" in probe.reason


def test_inspect_vulkan_runtime_reports_missing_runtime(monkeypatch):
    monkeypatch.setattr(iree_vulkan, "_has_module", lambda _name: False)

    diagnostics = iree_vulkan.inspect_vulkan_runtime()

    assert diagnostics["available"] is False
    assert diagnostics["runtime_installed"] is False
    assert diagnostics["device_count"] == 0
    assert '".[iree-vulkan]"' in diagnostics["reason"]


def test_inspect_vulkan_runtime_reports_no_devices(monkeypatch):
    monkeypatch.setattr(iree_vulkan, "_has_module", lambda _name: True)
    monkeypatch.setattr(iree_vulkan, "_package_version", lambda _name: "3.11.0")
    monkeypatch.setattr(iree_vulkan, "query_vulkan_devices", lambda: ())

    diagnostics = iree_vulkan.inspect_vulkan_runtime()

    assert diagnostics["available"] is False
    assert diagnostics["runtime_installed"] is True
    assert diagnostics["runtime_version"] == "3.11.0"
    assert diagnostics["device_count"] == 0
    assert "no Vulkan devices" in diagnostics["reason"]


def test_inspect_vulkan_runtime_reports_devices_as_jsonable(monkeypatch):
    class Opaque:
        def __str__(self) -> str:
            return "opaque-value"

    monkeypatch.setattr(iree_vulkan, "_has_module", lambda _name: True)
    monkeypatch.setattr(iree_vulkan, "_package_version", lambda _name: "3.11.0")
    monkeypatch.setattr(
        iree_vulkan,
        "query_vulkan_devices",
        lambda: ({"name": "Mali-G715", "driver": Opaque(), "queue_counts": (1, 2)},),
    )

    diagnostics = iree_vulkan.inspect_vulkan_runtime()

    assert diagnostics["available"] is True
    assert diagnostics["device_count"] == 1
    assert diagnostics["devices"] == [
        {"name": "Mali-G715", "driver": "opaque-value", "queue_counts": [1, 2]}
    ]


def test_compile_exported_writes_vmfb_with_portable_flags(monkeypatch, tmp_path):
    calls = {}

    class Session:
        def set_flags(self, *flags):
            calls["flags"] = flags

    class ExportOutput:
        session = Session()

        def compile(self, path, *, target_backends):
            calls["path"] = Path(path)
            calls["target_backends"] = target_backends
            Path(path).write_bytes(b"vmfb")

    def export_program(exported_program):
        calls["program"] = exported_program
        return ExportOutput()

    fake_aot = SimpleNamespace(export=export_program)
    original_import = iree_vulkan._import_module

    def import_module(name):
        return fake_aot if name == "iree.turbine.aot" else original_import(name)

    backend = IREEVulkanBackend()
    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(iree_vulkan, "_import_module", import_module)
    exported = torch.export.export(model(), (torch.randn(2, 4),))
    output = tmp_path / "model.vmfb"

    assert backend.compile_exported(exported, output) == output
    assert calls["program"] is exported
    assert calls["target_backends"] is None
    assert calls["flags"] == (
        "--iree-hal-target-device=vulkan",
        "--iree-opt-level=O2",
    )
    assert output.read_bytes() == b"vmfb"


def test_compile_exported_passes_an_explicit_target(monkeypatch, tmp_path):
    calls = {}

    class ExportOutput:
        session = SimpleNamespace(set_flags=lambda *flags: calls.setdefault("flags", flags))

        @staticmethod
        def compile(path, *, target_backends):
            assert target_backends is None
            Path(path).write_bytes(b"vmfb")

    fake_aot = SimpleNamespace(export=lambda _program: ExportOutput())
    backend = IREEVulkanBackend()
    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(
        iree_vulkan,
        "_import_module",
        lambda _name: fake_aot,
    )
    exported = torch.export.export(model(), (torch.randn(2, 4),))

    backend.compile_exported(
        exported,
        tmp_path / "model.vmfb",
        options={"vulkan_target": "ampere", "opt_level": "O3"},
    )

    assert calls["flags"] == (
        "--iree-hal-target-device=vulkan",
        "--iree-opt-level=O3",
        "--iree-vulkan-target=ampere",
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"opt_level": "fast"}, "optimization level"),
        ({"unknown": True}, "Unsupported"),
    ],
)
def test_compile_exported_rejects_invalid_options(monkeypatch, tmp_path, options, message):
    backend = IREEVulkanBackend()
    monkeypatch.setattr(backend, "probe", available_probe)
    exported = torch.export.export(model(), (torch.randn(2, 4),))

    with pytest.raises(CompilationError, match=message):
        backend.compile_exported(exported, tmp_path / "model.vmfb", options=options)


def test_vmfb_callable_loads_lazily_and_returns_torch(monkeypatch, tmp_path):
    numpy = pytest.importorskip("numpy")
    calls = {}

    class DeviceArray:
        def __init__(self, value):
            self.value = value

        def to_host(self):
            return self.value

    class BoundModule:
        @staticmethod
        def main(value):
            calls["input"] = value
            return DeviceArray(value + 1)

    class Runtime:
        class Config:
            def __init__(self, driver_name=None, *, device=None):
                calls["config"] = (driver_name, device)
                self.vm_instance = object()

        class VmModule:
            @staticmethod
            def mmap(instance, path):
                calls["mmap"] = (instance, path)
                return object()

        @staticmethod
        def get_driver(name):
            assert name == "vulkan"
            return SimpleNamespace(query_available_devices=lambda: ({"name": "GPU"},))

        @staticmethod
        def load_vm_module(vm_module, config):
            calls["load"] = (vm_module, config)
            return BoundModule()

    vmfb = tmp_path / "model.vmfb"
    vmfb.write_bytes(b"vmfb")
    monkeypatch.setattr(
        iree_vulkan,
        "_import_module",
        lambda name: Runtime if name == "iree.runtime" else None,
    )
    compiled = IREEVulkanBackend().load_vmfb(vmfb)
    assert "mmap" not in calls

    actual = compiled(torch.tensor([[1.0, 2.0]]))

    assert calls["config"] == ("vulkan", None)
    assert calls["mmap"][1] == str(vmfb)
    assert calls["input"].dtype == numpy.float32
    torch.testing.assert_close(actual, torch.tensor([[2.0, 3.0]]))


def test_vmfb_callable_reports_missing_vulkan_device(monkeypatch, tmp_path):
    class Runtime:
        @staticmethod
        def get_driver(_name):
            return SimpleNamespace(query_available_devices=lambda: ())

    vmfb = tmp_path / "model.vmfb"
    vmfb.write_bytes(b"vmfb")
    monkeypatch.setattr(iree_vulkan, "_import_module", lambda _name: Runtime)
    compiled = IREEVulkanBackend().load_vmfb(vmfb)

    with pytest.raises(ArtifactLoadError, match="found no devices"):
        compiled(torch.ones(1))


def test_export_packages_vmfb_and_reloads(monkeypatch, tmp_path):
    source = model()
    example = torch.randn(2, 4)
    expected = source(example)
    backend = registry.get("iree_vulkan")
    assert isinstance(backend, IREEVulkanBackend)
    compile_calls = []
    load_calls = []

    def compile_exported(exported_program, path, *, options):
        compile_calls.append((exported_program, Path(path), options))
        Path(path).write_bytes(b"vmfb")
        return Path(path)

    def load_vmfb(path, *, device_uri=None, function_name="main"):
        load_calls.append((Path(path), device_uri, function_name))
        return source

    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(backend, "compile_exported", compile_exported)
    monkeypatch.setattr(backend, "load_vmfb", load_vmfb)
    output = tmp_path / "model.lm7"

    artifact = lm7.export(
        source,
        args=(example,),
        target="nvidia:sm89",
        backend="iree_vulkan",
        output=output,
        options={"vulkan_target": "ampere", "device_uri": "vulkan://gpu-id"},
    )

    assert artifact.manifest.backend == "iree_vulkan"
    assert artifact.manifest.compiled_file == "compiled_model.vmfb"
    assert artifact.manifest.compiled_sha256
    assert artifact.manifest.runtime_requirements["vulkan_target"] == "ampere"
    assert artifact.manifest.runtime_requirements["vulkan_device_uri"] == "vulkan://gpu-id"
    assert (artifact.path / "compiled_model.vmfb").read_bytes() == b"vmfb"
    assert compile_calls[0][2] == {"vulkan_target": "ampere"}
    torch.testing.assert_close(artifact(example), expected)

    loaded = lm7.load_artifact(output)
    torch.testing.assert_close(loaded(example), expected)
    assert len(load_calls) == 2
    assert load_calls[-1][1] == "vulkan://gpu-id"


def test_corrupt_vmfb_fails_checksum_validation(monkeypatch, tmp_path):
    backend = registry.get("iree_vulkan")
    assert isinstance(backend, IREEVulkanBackend)

    def compile_exported(_program, path, *, options):
        assert not options
        Path(path).write_bytes(b"vmfb")
        return Path(path)

    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(backend, "compile_exported", compile_exported)
    monkeypatch.setattr(backend, "load_vmfb", lambda path, **_kwargs: model())
    artifact = lm7.export(
        model(),
        args=(torch.randn(2, 4),),
        target="nvidia:sm89",
        backend="iree_vulkan",
        output=tmp_path / "model.lm7",
    )
    vmfb = artifact.path / "compiled_model.vmfb"
    vmfb.write_bytes(vmfb.read_bytes() + b"corrupt")

    with pytest.raises(ArtifactLoadError, match="checksum"):
        lm7.load_artifact(artifact.path)


@pytest.mark.parametrize("target", ["cpu", "apple:metal", "tpu:v5e"])
def test_export_rejects_non_vulkan_gpu_targets(target, tmp_path):
    with pytest.raises(BackendUnavailableError, match="NVIDIA, AMD, Intel, or Arm"):
        lm7.export(
            model(),
            args=(torch.randn(2, 4),),
            target=target,
            backend="iree_vulkan",
            output=tmp_path / "model.lm7",
        )


@pytest.mark.parametrize("target", ["arm", "arm:valhall4", "arm:mali-g715"])
def test_export_accepts_arm_gpu_targets(monkeypatch, tmp_path, target):
    """The vendor gate lets Mali through, and the target skips local detection.

    This proves the plumbing only. No Arm GPU has executed an LM7 VMFB -- the
    runtime half needs an NDK cross-compile of IREE. See docs/iree-vulkan.md.
    """
    source = model()
    backend = registry.get("iree_vulkan")
    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(
        backend,
        "compile_exported",
        lambda _exported, path, *, options: (Path(path).write_bytes(b"vmfb"), Path(path))[1],
    )
    monkeypatch.setattr(backend, "load_vmfb", lambda path, **_kwargs: source)

    artifact = lm7.export(
        source,
        args=(torch.randn(2, 4),),
        target=target,
        backend="iree_vulkan",
        output=tmp_path / "model.lm7",
        options={"vulkan_target": "valhall4"},
    )

    assert artifact.manifest.target["vendor"] == "arm"
    assert artifact.manifest.target["remote"] is True
    assert artifact.manifest.runtime_requirements["vulkan_target"] == "valhall4"


def test_nothing_quietly_runs_an_arm_target_on_the_host():
    """torch_device() maps an unknown vendor to the CPU, so an eager backend
    that claimed arm would report a host run as a Mali one."""
    request = CompileRequest(
        model=model(),
        target=parse_target("arm:mali-g715"),
        mode="lazy",
        transfers="automatic",
        fallback="error",
    )

    support = EagerBackend().supports(request)

    assert not support.supported
    assert "iree_vulkan" in support.reason


def test_export_rejects_dynamic_shapes(tmp_path):
    with pytest.raises(BackendUnavailableError, match="static shapes"):
        lm7.export(
            model(),
            args=(torch.randn(2, 4),),
            target="nvidia:sm89",
            backend="iree_vulkan",
            output=tmp_path / "model.lm7",
            dynamic_shapes={"input": {0: torch.export.Dim("batch", min=1, max=4)}},
        )
