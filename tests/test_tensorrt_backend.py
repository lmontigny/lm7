from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import lm7
from lm7.backends.base import BackendInfo, CompileRequest
from lm7.backends.tensorrt import TensorRTBackend
from lm7.errors import ArtifactLoadError, BackendUnavailableError, CompilationError
from lm7.exporting import EXPORT_BACKENDS
from lm7.targets import TargetSpec

tensorrt_backend_module = importlib.import_module("lm7.backends.tensorrt")


def request(*, vendor: str = "nvidia", options=None) -> CompileRequest:
    return CompileRequest(
        torch.nn.Identity(),
        TargetSpec(vendor, "gpu" if vendor != "cpu" else "cpu"),
        "lazy",
        "automatic",
        "error",
        options or {},
    )


def test_probe_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setattr(tensorrt_backend_module.importlib.util, "find_spec", lambda name: None)

    info = TensorRTBackend().probe()

    assert not info.available
    assert ".[tensorrt]" in info.reason


def test_probe_rejects_rocm_runtime(monkeypatch):
    monkeypatch.setattr(
        tensorrt_backend_module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(),
    )
    monkeypatch.setattr(
        tensorrt_backend_module.importlib.metadata,
        "version",
        lambda name: "test-version",
    )
    monkeypatch.setattr(torch.version, "hip", "7.0-test")

    info = TensorRTBackend().probe()

    assert not info.available
    assert "ROCm" in info.reason


def test_support_is_nvidia_only(monkeypatch):
    backend = TensorRTBackend()
    monkeypatch.setattr(
        backend,
        "probe",
        lambda: SimpleNamespace(available=True, reason="available"),
    )

    assert backend.supports(request()).supported
    assert backend.supports(request(vendor="cpu")).reason == (
        "TensorRT supports NVIDIA GPU targets only."
    )
    assert backend.supports(request()).priority < 100


def test_compile_uses_registered_tensorrt_backend(monkeypatch):
    backend = TensorRTBackend()
    calls = {}

    monkeypatch.setattr(
        tensorrt_backend_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(__version__="test-version"),
    )
    monkeypatch.setattr(tensorrt_backend_module, "torch_device", lambda target: torch.device("cpu"))

    def fake_compile(model, **kwargs):
        calls.update(kwargs)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    artifact = backend.compile(
        request(options={"dynamic": True, "min_block_size": 2}),
        (torch.ones(2),),
        {},
    )

    assert calls == {
        "backend": "tensorrt",
        "dynamic": True,
        "options": {"min_block_size": 2},
    }
    assert artifact.metadata["torch_tensorrt_version"] == "test-version"
    torch.testing.assert_close(artifact.callable(torch.ones(2)), torch.ones(2))


def test_compile_wraps_lazy_backend_failure(monkeypatch):
    backend = TensorRTBackend()
    monkeypatch.setattr(
        tensorrt_backend_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(__version__="test-version"),
    )
    monkeypatch.setattr(tensorrt_backend_module, "torch_device", lambda target: torch.device("cpu"))

    def fail_on_call(model, **kwargs):
        def compiled(*args, **call_kwargs):
            raise RuntimeError("TensorRT build failed")

        return compiled

    monkeypatch.setattr(torch, "compile", fail_on_call)

    with pytest.raises(CompilationError, match="TensorRT build failed"):
        backend.compile(request(), (torch.ones(2),), {})


def _fake_torch_tensorrt(monkeypatch, calls: dict) -> None:
    """A torch_tensorrt whose engine build and save are recorded, not run."""

    def compile_exported(exported_program, arg_inputs=None, kwarg_inputs=None, **settings):
        calls["arg_inputs"] = arg_inputs
        calls["kwarg_inputs"] = kwarg_inputs
        calls["settings"] = settings
        return SimpleNamespace(name="trt-module")

    def save(module, file_path, **kwargs):
        calls["saved"] = (module.name, file_path, kwargs)
        Path(file_path).write_bytes(b"engine")

    monkeypatch.setattr(
        tensorrt_backend_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            __version__="test-version",
            dynamo=SimpleNamespace(compile=compile_exported),
            save=save,
            load=lambda file_path: SimpleNamespace(
                module=lambda: lambda *args, **kwargs: torch.ones(2)
            ),
        ),
    )
    monkeypatch.setattr(TensorRTBackend, "probe", lambda self: _AVAILABLE)


_AVAILABLE = BackendInfo("tensorrt", "test-version", True, "available")


def test_compile_exported_builds_and_saves_an_engine(monkeypatch, tmp_path):
    calls: dict = {}
    # Capture before patching importlib: torch.export imports through it too.
    exported = torch.export.export(torch.nn.Linear(4, 4).eval(), (torch.randn(2, 4),))
    example = torch.randn(2, 4)
    _fake_torch_tensorrt(monkeypatch, calls)

    path = TensorRTBackend().compile_exported(
        exported,
        tmp_path / "engine.trt.pt2",
        arg_inputs=(example,),
        kwarg_inputs={"mask": example},
        options={"min_block_size": 2},
    )

    assert path.read_bytes() == b"engine"
    # Keyword inputs stay keyword: saving re-exports against them, so flattening
    # here would change how the reloaded artifact has to be called.
    assert calls["arg_inputs"] == [example]
    assert calls["kwarg_inputs"] == {"mask": example}
    assert calls["saved"][2]["kwarg_inputs"] == {"mask": example}
    # Caller options reach Torch-TensorRT untouched.
    assert calls["settings"] == {"min_block_size": 2}
    assert calls["saved"][2]["output_format"] == "exported_program"


def test_compile_exported_explains_the_enabled_precisions_trap(monkeypatch, tmp_path):
    """Torch-TensorRT 2.12 enables explicit typing and then rejects the option,
    with an error that does not say what to do about it."""

    exported = torch.export.export(torch.nn.Linear(4, 4).eval(), (torch.randn(2, 4),))

    def explode(exported_program, arg_inputs=None, kwarg_inputs=None, **settings):
        raise AssertionError("use_explicit_typing was set to True, however found ...")

    monkeypatch.setattr(
        tensorrt_backend_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(dynamo=SimpleNamespace(compile=explode)),
    )
    monkeypatch.setattr(TensorRTBackend, "probe", lambda self: _AVAILABLE)

    with pytest.raises(CompilationError, match="Drop options="):
        TensorRTBackend().compile_exported(
            exported,
            tmp_path / "engine.trt.pt2",
            arg_inputs=(torch.randn(2, 4),),
            options={"enabled_precisions": {torch.float16}},
        )


def test_load_engine_reports_an_unloadable_artifact(monkeypatch, tmp_path):
    def explode(file_path):
        raise RuntimeError("serialized engine version mismatch")

    monkeypatch.setattr(
        tensorrt_backend_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(load=explode),
    )
    monkeypatch.setattr(TensorRTBackend, "probe", lambda self: _AVAILABLE)

    with pytest.raises(ArtifactLoadError, match="GPU architecture"):
        TensorRTBackend().load_engine(tmp_path / "engine.trt.pt2")


def test_export_rejects_tensorrt_for_non_nvidia_targets(tmp_path):
    with pytest.raises(BackendUnavailableError, match="NVIDIA GPUs only"):
        lm7.export(
            torch.nn.Linear(4, 4).eval(),
            args=(torch.randn(2, 4),),
            target="cpu",
            backend="tensorrt",
            output=tmp_path / "model.lm7",
        )


def test_export_rejects_tensorrt_with_dynamic_shapes(tmp_path):
    with pytest.raises(BackendUnavailableError, match="require static shapes"):
        lm7.export(
            torch.nn.Linear(4, 4).eval(),
            args=(torch.randn(2, 4),),
            target="nvidia",
            backend="tensorrt",
            output=tmp_path / "model.lm7",
            dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=8)},),
        )


def test_tensorrt_is_an_export_backend():
    # The JIT path was the only one until engines could be serialized.
    assert "tensorrt" in EXPORT_BACKENDS
