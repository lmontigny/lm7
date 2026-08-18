from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import lm7
from lm7.backends import onnxruntime as ort_backend
from lm7.backends import registry
from lm7.backends.base import BackendInfo, CompileRequest
from lm7.backends.onnxruntime import ONNXRuntimeBackend, parse_options
from lm7.errors import ArtifactLoadError, BackendUnavailableError, CompilationError
from lm7.targets import parse_target


def model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)).eval()


def request_for(target: str = "cpu") -> CompileRequest:
    return CompileRequest(
        model(),
        parse_target(target),
        "lazy",
        "automatic",
        "error",
    )


def available_probe() -> BackendInfo:
    return BackendInfo("onnxruntime", "1.28.0", True, "available")


def _fake_load(locations: tuple[str, ...]):
    """Stand in for onnx.load, reporting which sidecars the graph references."""

    def load(path, load_external_data: bool = True):
        initializer = SimpleNamespace(
            external_data=[SimpleNamespace(key="location", value=name) for name in locations]
        )
        return SimpleNamespace(graph=SimpleNamespace(initializer=[initializer]))

    return load


def test_backend_is_registered_and_cpu_support_is_below_openvino(monkeypatch):
    backend = registry.get("onnxruntime")
    assert isinstance(backend, ONNXRuntimeBackend)
    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(ort_backend, "available_providers", lambda: ("CPUExecutionProvider",))

    support = backend.supports(request_for())

    assert support.supported
    assert support.priority == 70


def test_probe_reports_missing_optional_packages(monkeypatch):
    monkeypatch.setattr(ort_backend, "_has_module", lambda _name: False)

    probe = ONNXRuntimeBackend().probe()

    assert not probe.available
    assert '".[onnxruntime]"' in probe.reason
    assert "onnx" in probe.reason
    assert "onnxscript" in probe.reason
    assert "onnxruntime" in probe.reason


def test_support_requires_target_execution_provider(monkeypatch):
    backend = ONNXRuntimeBackend()
    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(ort_backend, "available_providers", lambda: ("CPUExecutionProvider",))

    support = backend.supports(request_for("nvidia:sm89"))

    assert not support.supported
    assert "CUDAExecutionProvider" in support.reason
    assert "onnxruntime-gpu" in support.reason


@pytest.mark.parametrize("target", ["amd:gfx942", "apple:metal", "tpu:v5e"])
def test_support_rejects_unvalidated_targets(monkeypatch, target):
    backend = ONNXRuntimeBackend()
    monkeypatch.setattr(backend, "probe", available_probe)

    assert not backend.supports(request_for(target)).supported


def test_parse_options_uses_strict_cuda_provider_defaults():
    settings = parse_options(parse_target("nvidia:sm89"), None)

    assert settings.provider == "CUDAExecutionProvider"
    assert settings.disable_cpu_fallback is True
    assert settings.optimize is True
    assert settings.opset_version is None


def test_parse_options_accepts_provider_configuration():
    settings = parse_options(
        parse_target("cpu"),
        {
            "provider": "AzureExecutionProvider",
            "provider_options": {"endpoint": "local"},
            "disable_cpu_fallback": True,
            "opset_version": 20,
            "optimize": False,
        },
    )

    assert settings.provider == "AzureExecutionProvider"
    assert settings.provider_options == {"endpoint": "local"}
    assert settings.disable_cpu_fallback is True
    assert settings.compiler_options == {
        "opset_version": 20,
        "optimize": False,
        "external_data": "auto",
    }


def test_parse_options_rejects_unknown_values():
    with pytest.raises(CompilationError, match="Unsupported"):
        parse_options(parse_target("cpu"), {"unknown": True})


def test_compile_exported_writes_and_validates_onnx(monkeypatch, tmp_path):
    calls = {}

    class ONNXProgram:
        def save(self, path, *, external_data):
            calls["save"] = (Path(path), external_data)
            Path(path).write_bytes(b"onnx")

    class Checker:
        @staticmethod
        def check_model(value):
            calls["checked"] = value

    fake_onnx = SimpleNamespace(load=_fake_load(()), checker=Checker())
    backend = ONNXRuntimeBackend()
    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(
        ort_backend, "_import_module", lambda name: fake_onnx if name == "onnx" else None
    )

    def export(program, **kwargs):
        calls["program"] = program
        calls["export"] = kwargs
        return ONNXProgram()

    monkeypatch.setattr(torch.onnx, "export", export)
    exported = torch.export.export(model(), (torch.randn(2, 4),))
    output = tmp_path / "model.onnx"

    assert (
        backend.compile_exported(
            exported,
            output,
            options={"opset_version": 20, "optimize": False},
        )
        == output
    )
    assert calls["program"] is exported
    assert calls["export"]["dynamo"] is True
    assert calls["export"]["external_data"] is False
    assert calls["export"]["opset_version"] == 20
    assert calls["export"]["optimize"] is False
    assert calls["save"] == (output, False)
    assert calls["checked"] == str(output)


def test_load_onnx_uses_requested_provider_and_disables_fallback(monkeypatch, tmp_path):
    calls = {}

    class SessionOptions:
        def add_session_config_entry(self, name, value):
            calls["config"] = (name, value)

    class Session:
        def __init__(self, path, *, sess_options, providers):
            calls["session"] = (path, sess_options, providers)

        @staticmethod
        def get_inputs():
            return (SimpleNamespace(name="input"),)

        @staticmethod
        def get_outputs():
            # Read at construction now: the callable binds outputs by name.
            return (SimpleNamespace(name="logits"),)

        @staticmethod
        def get_providers():
            return ("CUDAExecutionProvider", "CPUExecutionProvider")

        @staticmethod
        def disable_fallback():
            calls["disabled"] = True

    runtime = SimpleNamespace(
        SessionOptions=SessionOptions,
        InferenceSession=Session,
        get_available_providers=lambda: ("CUDAExecutionProvider", "CPUExecutionProvider"),
    )
    backend = ONNXRuntimeBackend()
    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(ort_backend, "_import_module", lambda _name: runtime)
    path = tmp_path / "model.onnx"
    path.write_bytes(b"onnx")

    backend.load_onnx(
        path,
        provider="CUDAExecutionProvider",
        provider_options={"device_id": "1"},
    )

    assert calls["config"] == ("session.disable_cpu_ep_fallback", "1")
    assert calls["session"][2] == [("CUDAExecutionProvider", {"device_id": "1"})]
    assert calls["disabled"] is True


def test_load_onnx_rejects_missing_provider(monkeypatch, tmp_path):
    runtime = SimpleNamespace(get_available_providers=lambda: ("CPUExecutionProvider",))
    backend = ONNXRuntimeBackend()
    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(ort_backend, "_import_module", lambda _name: runtime)
    path = tmp_path / "model.onnx"
    path.write_bytes(b"onnx")

    with pytest.raises(ArtifactLoadError, match="unavailable"):
        backend.load_onnx(path, provider="CUDAExecutionProvider")


def test_export_packages_onnx_and_reloads(monkeypatch, tmp_path):
    source = model()
    example = torch.randn(2, 4)
    expected = source(example)
    backend = registry.get("onnxruntime")
    assert isinstance(backend, ONNXRuntimeBackend)
    compile_calls = []
    load_calls = []

    def compile_exported(program, path, *, options):
        compile_calls.append((program, Path(path), options))
        Path(path).write_bytes(b"onnx")
        return Path(path)

    def load_onnx(path, **options):
        load_calls.append((Path(path), options))
        return source

    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(backend, "compile_exported", compile_exported)
    monkeypatch.setattr(backend, "load_onnx", load_onnx)
    output = tmp_path / "model.lm7"

    artifact = lm7.export(
        source,
        args=(example,),
        target="cpu",
        backend="onnxruntime",
        output=output,
        options={"provider_options": {"arena_extend_strategy": "kSameAsRequested"}},
    )

    assert artifact.manifest.backend == "onnxruntime"
    assert artifact.manifest.compiled_file == "compiled_model.onnx"
    assert artifact.manifest.compiled_sha256
    requirements = artifact.manifest.runtime_requirements
    assert requirements["execution_provider"] == "CPUExecutionProvider"
    assert requirements["disable_cpu_fallback"] is False
    assert requirements["provider_options"] == {"arena_extend_strategy": "kSameAsRequested"}
    assert (artifact.path / "compiled_model.onnx").read_bytes() == b"onnx"
    assert compile_calls[0][2] == {
        "opset_version": None,
        "optimize": True,
        "external_data": "auto",
    }
    torch.testing.assert_close(artifact(example), expected)

    loaded = lm7.load_artifact(output)
    torch.testing.assert_close(loaded(example), expected)
    assert len(load_calls) == 2
    assert load_calls[-1][1]["provider"] == "CPUExecutionProvider"


def test_corrupt_onnx_fails_checksum_validation(monkeypatch, tmp_path):
    backend = registry.get("onnxruntime")
    assert isinstance(backend, ONNXRuntimeBackend)

    def compile_exported(_program, path, *, options):
        assert options == {"opset_version": None, "optimize": True, "external_data": "auto"}
        Path(path).write_bytes(b"onnx")
        return Path(path)

    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(backend, "compile_exported", compile_exported)
    monkeypatch.setattr(backend, "load_onnx", lambda path, **_kwargs: model())
    artifact = lm7.export(
        model(),
        args=(torch.randn(2, 4),),
        target="cpu",
        backend="onnxruntime",
        output=tmp_path / "model.lm7",
    )
    onnx_path = artifact.path / "compiled_model.onnx"
    onnx_path.write_bytes(onnx_path.read_bytes() + b"corrupt")

    with pytest.raises(ArtifactLoadError, match="checksum"):
        lm7.load_artifact(artifact.path)


@pytest.mark.parametrize("target", ["amd:gfx942", "apple:metal", "tpu:v5e"])
def test_export_rejects_unvalidated_targets(target, tmp_path):
    with pytest.raises(BackendUnavailableError, match="CPU and NVIDIA"):
        lm7.export(
            model(),
            args=(torch.randn(2, 4),),
            target=target,
            backend="onnxruntime",
            output=tmp_path / "model.lm7",
        )


def test_order_feeds_binds_keyword_tensors_by_name_not_call_order():
    # Capture order deliberately differs from the caller's.
    names = ("y", "x")
    x = torch.tensor([1.0])
    y = torch.tensor([2.0])

    assert ort_backend.order_feeds(names, (), {"x": x, "y": y}) == {"x": x, "y": y}
    assert ort_backend.order_feeds(names, (), {"y": y, "x": x}) == {"x": x, "y": y}


def test_order_feeds_fills_unnamed_inputs_left_to_right():
    x = torch.tensor([1.0])
    y = torch.tensor([2.0])

    # "y" is named, so the positional tensor can only be "x" -- whichever side of
    # the capture order it sits on.
    assert ort_backend.order_feeds(("y", "x"), (x,), {"y": y}) == {"x": x, "y": y}
    assert ort_backend.order_feeds(("x", "y"), (x, y), {}) == {"x": x, "y": y}


def test_order_feeds_rejects_an_argument_count_the_graph_cannot_take():
    with pytest.raises(ValueError, match="expects 2 tensor inputs, got 1"):
        ort_backend.order_feeds(("x", "y"), (torch.tensor([1.0]),), {})


def test_parse_options_defaults_external_data_to_auto():
    settings = parse_options(parse_target("cpu"), None)

    assert settings.external_data == "auto"
    assert settings.compiler_options["external_data"] == "auto"


@pytest.mark.parametrize("value", [True, False, "auto"])
def test_parse_options_accepts_every_external_data_setting(value):
    settings = parse_options(parse_target("cpu"), {"external_data": value})

    assert settings.external_data == value


def test_parse_options_rejects_a_non_boolean_external_data():
    with pytest.raises(CompilationError, match="external_data must be True, False, or 'auto'"):
        parse_options(parse_target("cpu"), {"external_data": "yes"})


def test_weight_bytes_counts_a_tied_weight_once():
    tied = torch.nn.Linear(4, 4, bias=False)

    class Tied(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = tied
            self.second = tied

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.second(self.first(x))

    exported = torch.export.export(Tied().eval(), (torch.randn(1, 4),))

    # 4x4 float32 once, not twice.
    assert ort_backend._weight_bytes(exported) == 4 * 4 * 4


def test_compile_exported_switches_to_external_data_above_the_limit(monkeypatch, tmp_path):
    calls = {}
    output = tmp_path / "model.onnx"

    class ONNXProgram:
        def save(self, path, *, external_data):
            calls["save"] = (Path(path), external_data)
            Path(path).write_bytes(b"onnx")
            if external_data:
                ort_backend.external_data_path(Path(path)).write_bytes(b"weights")

    fake_onnx = SimpleNamespace(
        load=_fake_load((output.name + ".data",)),
        checker=SimpleNamespace(check_model=lambda value: calls.__setitem__("checked", value)),
    )
    backend = ONNXRuntimeBackend()
    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(
        ort_backend, "_import_module", lambda name: fake_onnx if name == "onnx" else None
    )
    monkeypatch.setattr(torch.onnx, "export", lambda program, **kwargs: ONNXProgram())
    # Any real model is above a zero-byte budget, so "auto" has to choose the sidecar.
    monkeypatch.setattr(ort_backend, "EMBEDDED_WEIGHT_LIMIT", 0)
    exported = torch.export.export(model(), (torch.randn(2, 4),))

    backend.compile_exported(exported, output)

    assert calls["save"] == (output, True)
    assert ort_backend.external_data_path(output).read_bytes() == b"weights"


def test_compile_exported_drops_a_sidecar_the_graph_never_references(monkeypatch, tmp_path):
    # Below roughly a kilobyte the exporter keeps tensors inline whatever it was
    # asked for, and leaves an empty sidecar behind.
    output = tmp_path / "model.onnx"

    class ONNXProgram:
        def save(self, path, *, external_data):
            Path(path).write_bytes(b"onnx")
            ort_backend.external_data_path(Path(path)).write_bytes(b"")

    fake_onnx = SimpleNamespace(
        load=_fake_load(()), checker=SimpleNamespace(check_model=lambda value: None)
    )
    backend = ONNXRuntimeBackend()
    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(
        ort_backend, "_import_module", lambda name: fake_onnx if name == "onnx" else None
    )
    monkeypatch.setattr(torch.onnx, "export", lambda program, **kwargs: ONNXProgram())
    exported = torch.export.export(model(), (torch.randn(2, 4),))

    backend.compile_exported(exported, output, options={"external_data": True})

    assert not ort_backend.external_data_path(output).exists()


def test_compile_exported_refuses_a_sidecar_name_it_cannot_package(monkeypatch, tmp_path):
    output = tmp_path / "model.onnx"

    class ONNXProgram:
        def save(self, path, *, external_data):
            Path(path).write_bytes(b"onnx")

    fake_onnx = SimpleNamespace(
        load=_fake_load(("somewhere-else.bin",)),
        checker=SimpleNamespace(check_model=lambda value: None),
    )
    backend = ONNXRuntimeBackend()
    monkeypatch.setattr(backend, "probe", available_probe)
    monkeypatch.setattr(
        ort_backend, "_import_module", lambda name: fake_onnx if name == "onnx" else None
    )
    monkeypatch.setattr(torch.onnx, "export", lambda program, **kwargs: ONNXProgram())
    exported = torch.export.export(model(), (torch.randn(2, 4),))

    with pytest.raises(CompilationError, match="somewhere-else.bin"):
        backend.compile_exported(exported, output, options={"external_data": True})
