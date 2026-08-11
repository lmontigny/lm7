from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import lm7
from lm7 import exporting as exporting_module
from lm7.backends import litert as litert_backend
from lm7.backends import registry
from lm7.backends.base import BackendInfo, CompileRequest
from lm7.backends.litert import LiteRTBackend, parse_options
from lm7.errors import ArtifactLoadError, BackendUnavailableError, CompilationError
from lm7.targets import parse_target


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    ).eval()


def request_for(target: str = "cpu") -> CompileRequest:
    return CompileRequest(
        model(),
        parse_target(target),
        "lazy",
        "automatic",
        "error",
    )


def available_probe() -> BackendInfo:
    return BackendInfo("litert", "0.9.2", True, "available")


def test_backend_is_registered_but_export_only(monkeypatch):
    backend = registry.get("litert")
    assert isinstance(backend, LiteRTBackend)
    monkeypatch.setattr(backend, "probe", available_probe)

    support = backend.supports(request_for())

    assert not support.supported
    assert "lm7.export" in support.reason
    with pytest.raises(CompilationError, match="export-only"):
        backend.compile(request_for(), (torch.randn(2, 4),), {})


def test_probe_reports_missing_optional_packages(monkeypatch):
    monkeypatch.setattr(litert_backend, "_has_module", lambda _name: False)
    monkeypatch.setattr(litert_backend, "_is_linux_aarch64", lambda: False)

    probe = LiteRTBackend().probe()

    assert not probe.available
    assert '".[litert]"' in probe.reason
    assert "litert-torch" in probe.reason
    assert "ai-edge-litert" in probe.reason


def test_probe_reports_linux_aarch64_litert_torch_packaging_gap(monkeypatch):
    monkeypatch.setattr(
        litert_backend,
        "_has_module",
        lambda name: name != "litert_torch",
    )
    monkeypatch.setattr(litert_backend, "_is_linux_aarch64", lambda: True)

    probe = LiteRTBackend().probe()

    assert not probe.available
    assert "Linux aarch64" in probe.reason
    assert "litert-converter==0.3.*" in probe.reason
    assert '".[litert]"' not in probe.reason


@pytest.mark.parametrize("version", [(2, 3), (2, 13)])
def test_probe_rejects_torch_versions_outside_litert_range(monkeypatch, version):
    monkeypatch.setattr(litert_backend, "_has_module", lambda _name: True)
    monkeypatch.setattr(litert_backend, "_torch_major_minor", lambda: version)

    probe = LiteRTBackend().probe()

    assert not probe.available
    assert ">=2.4,<2.13" in probe.reason


def test_parse_options_defaults_to_auto_strict_export():
    settings = parse_options(None)

    assert settings.strict_export == "auto"
    assert settings.lightweight_conversion is False
    assert settings.enable_x64 is True
    assert settings.runtime_constant_folding is None


def test_parse_options_accepts_converter_configuration():
    settings = parse_options(
        {
            "strict_export": True,
            "lightweight_conversion": True,
            "enable_x64": False,
            "runtime_constant_folding": True,
        }
    )

    assert settings.converter_options == {
        "strict_export": True,
        "lightweight_conversion": True,
        "enable_x64": False,
        "runtime_constant_folding": True,
    }


def test_parse_options_rejects_invalid_or_unknown_values():
    with pytest.raises(CompilationError, match="strict_export"):
        parse_options({"strict_export": "sometimes"})
    with pytest.raises(CompilationError, match="Unsupported"):
        parse_options({"unknown": True})


def test_convert_module_writes_tflite_with_expected_options(monkeypatch, tmp_path):
    calls = {}

    class Converted:
        @staticmethod
        def export(path):
            calls["export"] = path
            Path(path).write_bytes(b"tflite")

    def convert(module, *, sample_args, sample_kwargs, **options):
        calls["convert"] = (module, sample_args, sample_kwargs, options)
        return Converted()

    backend = LiteRTBackend()
    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(
        litert_backend,
        "_import_module",
        lambda _name: SimpleNamespace(convert=convert),
    )
    source = model()
    example = torch.randn(2, 4)
    output = tmp_path / "model.tflite"

    result = backend.convert_module(
        source,
        (example,),
        {},
        output,
        options={"strict_export": False, "lightweight_conversion": True},
    )

    assert result == output
    assert output.read_bytes() == b"tflite"
    assert calls["export"] == str(output)
    converted_model, converted_args, converted_kwargs, options = calls["convert"]
    assert converted_model is source
    assert converted_args[0].device.type == "cpu"
    assert converted_kwargs == {}
    assert options == {
        "strict_export": False,
        "lightweight_conversion": True,
        "enable_x64": True,
        "runtime_constant_folding": None,
    }


def test_load_tflite_requires_existing_file(monkeypatch, tmp_path):
    backend = LiteRTBackend()
    monkeypatch.setattr(backend, "probe", available_probe)

    with pytest.raises(ArtifactLoadError, match="does not exist"):
        backend.load_tflite(tmp_path / "missing.tflite")


def test_callable_converts_nested_outputs_to_torch():
    pytest.importorskip("numpy")

    class RuntimeModel:
        @staticmethod
        def __call__(x):
            return {"first": x + 1, "nested": (x * 2,)}

    compiled = litert_backend._LiteRTCallable(RuntimeModel())
    value = torch.tensor([1.0, 2.0])

    output = compiled(value)

    torch.testing.assert_close(output["first"], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(output["nested"][0], torch.tensor([2.0, 4.0]))


def test_export_packages_tflite_and_reloads(monkeypatch, tmp_path):
    source = model()
    example = torch.randn(2, 4)
    expected = source(example)
    backend = registry.get("litert")
    assert isinstance(backend, LiteRTBackend)
    convert_calls = []
    load_calls = []

    def convert_module(module, args, kwargs, path, *, options):
        convert_calls.append((module, args, kwargs, Path(path), options))
        Path(path).write_bytes(b"tflite")
        return Path(path)

    def load_tflite(path):
        load_calls.append(Path(path))
        return source

    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(backend, "convert_module", convert_module)
    monkeypatch.setattr(backend, "load_tflite", load_tflite)
    monkeypatch.setattr(exporting_module, "_litert_version", lambda: "0.9.2")
    output = tmp_path / "model.lm7"

    artifact = lm7.export(
        source,
        args=(example,),
        target="cpu",
        backend="litert",
        output=output,
        options={"enable_x64": False},
    )

    assert artifact.manifest.backend == "litert"
    assert artifact.manifest.backend_version == "0.9.2"
    assert artifact.manifest.compiled_file == "compiled_model.tflite"
    assert artifact.manifest.compiled_sha256
    requirements = artifact.manifest.runtime_requirements
    assert requirements["runtime"] == "LiteRT Interpreter/XNNPACK"
    assert requirements["static_shapes"] is True
    assert requirements["enable_x64"] is False
    assert (artifact.path / "compiled_model.tflite").read_bytes() == b"tflite"
    assert convert_calls[0][4]["strict_export"] == "auto"
    assert convert_calls[0][4]["enable_x64"] is False
    torch.testing.assert_close(artifact(example), expected)

    loaded = lm7.load_artifact(output)
    torch.testing.assert_close(loaded(example), expected)
    assert len(load_calls) == 2


def test_corrupt_tflite_fails_checksum_validation(monkeypatch, tmp_path):
    backend = registry.get("litert")
    assert isinstance(backend, LiteRTBackend)
    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(
        backend,
        "convert_module",
        lambda _model, _args, _kwargs, path, **_options: Path(path).write_bytes(b"tflite"),
    )
    monkeypatch.setattr(backend, "load_tflite", lambda _path: model())
    output = tmp_path / "model.lm7"

    lm7.export(
        model(),
        args=(torch.randn(2, 4),),
        target="cpu",
        backend="litert",
        output=output,
    )
    (output / "compiled_model.tflite").write_bytes(b"corrupt")

    with pytest.raises(ArtifactLoadError, match="checksum"):
        lm7.load_artifact(output)


@pytest.mark.parametrize(
    "target",
    ["nvidia:sm89", "amd:gfx942", "intel:gpu", "apple:metal", "tpu:v5e"],
)
def test_export_rejects_non_cpu_targets(target, tmp_path):
    with pytest.raises(BackendUnavailableError, match="CPU/XNNPACK"):
        lm7.export(
            model(),
            args=(torch.randn(2, 4),),
            target=target,
            backend="litert",
            output=tmp_path / "model.lm7",
        )


def test_export_rejects_dynamic_shapes(tmp_path):
    with pytest.raises(BackendUnavailableError, match="static shapes"):
        lm7.export(
            model(),
            args=(torch.randn(2, 4),),
            target="cpu",
            backend="litert",
            output=tmp_path / "model.lm7",
            dynamic_shapes={"input": {0: torch.export.Dim("batch", min=1, max=8)}},
        )


def test_export_rejects_exported_program_without_source_module(tmp_path):
    exported = torch.export.export(model(), (torch.randn(2, 4),))

    with pytest.raises(BackendUnavailableError, match="source nn.Module"):
        lm7.export(
            exported,
            target="cpu",
            backend="litert",
            output=tmp_path / "model.lm7",
        )
