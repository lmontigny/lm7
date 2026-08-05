from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lm7.backends import registry
from lm7.backends.base import CompileRequest
from lm7.backends.tvm import TVMBackend
from lm7.errors import ArtifactLoadError, CompilationError
from lm7.targets import parse_target

tvm_backend_module = importlib.import_module("lm7.backends.tvm")


def model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()


def request_for(target: str = "cpu", options=None) -> CompileRequest:
    return CompileRequest(
        model=model(),
        target=parse_target(target),
        mode="lazy",
        transfers="automatic",
        fallback="error",
        options=options or {},
    )


def patch_find_spec(monkeypatch, *, present: bool) -> None:
    """Answer only for the tvm namespace.

    Replacing importlib.util.find_spec outright lies to every lazy import in the
    process, including torch's own, which breaks torch.export inside compile().
    """
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "tvm" or name.startswith("tvm."):
            # A real ModuleSpec, not a stand-in: torch inspects `.origin` on the
            # tvm spec for its own (Relay-era, non-functional) tvm backend.
            return (
                importlib.machinery.ModuleSpec(name, loader=None, origin="lm7-test")
                if present
                else None
            )
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(tvm_backend_module.importlib.util, "find_spec", fake_find_spec)


class FakeArray(list):
    """Stands in for the Relax VM's own array type, which nests its outputs."""


def install_fake_tvm(monkeypatch, *, output=None, build_error: Exception | None = None) -> dict:
    """Stand in for Apache TVM so the adapter is testable without the wheel.

    The real lowering is covered by tests/test_tvm_integration.py.
    """
    calls: dict = {}
    result = torch.zeros(8, 3) if output is None else output

    class FakeVM:
        def __getitem__(self, name):
            calls["entry"] = name

            def run(*inputs):
                calls["input_shapes"] = [tuple(i.shape) for i in inputs]
                return FakeArray([FakeArray([_FakeTensor(result)])])

            return run

    class _FakeTensor:
        """Exposes __dlpack__ like a real TVM tensor, so no NumPy is involved."""

        def __init__(self, value):
            self._value = value
            self.shape = tuple(value.shape)

        def __dlpack__(self, *args, **kwargs):
            return self._value.__dlpack__(*args, **kwargs)

        def __dlpack_device__(self):
            return self._value.__dlpack_device__()

    def from_dlpack(tensor):
        return _FakeTensor(tensor)

    class FakeExecutable:
        """Stands in for tvm.runtime.executable.Executable, AOT export's payload."""

        def export_library(self, path):
            calls["exported_library_path"] = path
            Path(path).write_bytes(b"fake-tvm-library")

    def build(mod, target=None):
        calls["built_target"] = str(target)
        if build_error is not None:
            raise build_error
        return FakeExecutable()

    def load_module(path):
        calls["loaded_library_path"] = path
        if not Path(path).is_file():
            raise RuntimeError(f"cannot find file {path}")
        return SimpleNamespace()

    class Target:
        """Mirrors real TVM 0.25's parser closely enough to catch drift.

        TVM dropped the CLI-string target form ("llvm -mcpu=x"); only a bare
        kind name or the JSON-dict form parses. LM7 hit this for real -- see
        docs/tvm.md -- so the fake validates it too, instead of accepting
        anything and giving false confidence.
        """

        def __init__(self, value):
            if isinstance(value, str) and " " in value:
                raise ValueError(
                    f'Cannot parse target string "{value}". CLI target string form '
                    '(e.g. "llvm -mcpu=xxx") is no longer supported. Please use JSON '
                    'dict form (e.g. {"kind": "llvm", "mcpu": "xxx"}) instead.'
                )
            self.value = value

        def __str__(self):
            return str(self.value)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    relax = SimpleNamespace(build=build, VirtualMachine=lambda ex, dev: FakeVM())

    def from_exported_program(exported, keep_params_as_input=None):
        calls["keep_params_as_input"] = keep_params_as_input
        return SimpleNamespace()

    frontend = SimpleNamespace(from_exported_program=from_exported_program)
    tvm_module = SimpleNamespace(
        __version__="0.25.0-test",
        target=SimpleNamespace(Target=Target),
        cpu=lambda index: SimpleNamespace(name="cpu", index=index),
        runtime=SimpleNamespace(from_dlpack=from_dlpack, load_module=load_module),
    )
    for name, module in {
        "tvm": tvm_module,
        "tvm.relax": relax,
        "tvm.relax.frontend.torch": frontend,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    patch_find_spec(monkeypatch, present=True)
    monkeypatch.setattr(
        tvm_backend_module.importlib.metadata, "version", lambda name: "0.25.0-test"
    )
    return calls


def test_probe_reports_missing_optional_dependency(monkeypatch):
    patch_find_spec(monkeypatch, present=False)

    info = TVMBackend().probe()

    assert not info.available
    assert ".[tvm]" in info.reason


def test_probe_rejects_a_relay_era_tvm(monkeypatch):
    """A pre-Relax TVM has no relax frontend; say so instead of failing later."""
    patch_find_spec(monkeypatch, present=True)
    monkeypatch.setattr(
        tvm_backend_module.importlib.metadata, "version", lambda name: "0.14.0-test"
    )

    def no_relax(name):
        raise ImportError("No module named 'tvm.relax'")

    monkeypatch.setattr(tvm_backend_module.importlib, "import_module", no_relax)

    info = TVMBackend().probe()

    assert not info.available
    assert "Relax-era TVM" in info.reason


def test_probe_available(monkeypatch):
    install_fake_tvm(monkeypatch)

    info = TVMBackend().probe()

    assert info.available
    assert info.version == "0.25.0-test"


def test_supports_cpu_at_priority_zero(monkeypatch):
    """Priority 0 ties with eager, and auto breaks ties by name -- so eager wins."""
    install_fake_tvm(monkeypatch)

    support = TVMBackend().supports(request_for())

    assert support.supported
    assert support.priority == 0


def test_supports_rejects_non_cpu_targets(monkeypatch):
    install_fake_tvm(monkeypatch)

    support = TVMBackend().supports(request_for("nvidia"))

    assert not support.supported
    assert "CPU (LLVM) targets only" in support.reason


def test_compile_uses_the_exported_program_frontend(monkeypatch):
    calls = install_fake_tvm(monkeypatch)

    artifact = TVMBackend().compile(request_for(), (torch.randn(8, 4),), {})

    # from_fx cannot lower `embedding`; the exported-program frontend can, and
    # baking the params in keeps the VM signature equal to the call signature.
    assert calls["keep_params_as_input"] is False
    assert calls["built_target"] == "llvm"
    assert calls["entry"] == "main"
    assert artifact.metadata["frontend"] == "relax.from_exported_program"
    assert artifact.metadata["tvm_version"] == "0.25.0-test"


def test_compile_honours_a_target_option(monkeypatch):
    calls = install_fake_tvm(monkeypatch)

    TVMBackend().compile(
        request_for(options={"target": {"kind": "llvm", "mcpu": "x"}}),
        (torch.randn(8, 4),),
        {},
    )

    assert calls["built_target"] == "{'kind': 'llvm', 'mcpu': 'x'}"


def test_compile_rejects_a_cli_style_target_string(monkeypatch):
    """TVM 0.25 dropped "llvm -mcpu=x" in favour of the JSON-dict form; a user
    following the old-style example should get a clear error, not a crash."""
    install_fake_tvm(monkeypatch)

    with pytest.raises(CompilationError, match="no longer supported"):
        TVMBackend().compile(
            request_for(options={"target": "llvm -mcpu=x"}), (torch.randn(8, 4),), {}
        )


def test_compile_rejects_keyword_inputs(monkeypatch):
    install_fake_tvm(monkeypatch)

    with pytest.raises(CompilationError, match="positional inputs only"):
        TVMBackend().compile(request_for(), (), {"input_ids": torch.ones(1, 4)})


def test_compile_wraps_build_failure(monkeypatch):
    install_fake_tvm(monkeypatch, build_error=RuntimeError("cannot legalize op"))

    with pytest.raises(CompilationError, match="cannot legalize op"):
        TVMBackend().compile(request_for(), (torch.randn(8, 4),), {})


def test_compiled_callable_returns_a_torch_tensor(monkeypatch):
    expected = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    calls = install_fake_tvm(monkeypatch, output=expected)

    artifact = TVMBackend().compile(request_for(), (torch.randn(8, 4),), {})
    result = TVMBackend().load(artifact)(torch.randn(8, 4))

    assert calls["input_shapes"] == [(8, 4)]
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, expected)


def test_compiled_callable_rejects_keyword_inputs(monkeypatch):
    install_fake_tvm(monkeypatch)
    artifact = TVMBackend().compile(request_for(), (torch.randn(8, 4),), {})

    with pytest.raises(CompilationError, match="positional tensors only"):
        TVMBackend().load(artifact)(x=torch.randn(8, 4))


def test_to_torch_unwraps_nested_arrays():
    tensor = torch.ones(2, 2)

    class Holder:
        def __dlpack__(self, *args, **kwargs):
            return tensor.__dlpack__(*args, **kwargs)

        def __dlpack_device__(self):
            return tensor.__dlpack_device__()

    assert isinstance(tvm_backend_module._to_torch(Holder()), torch.Tensor)
    nested = FakeArray([FakeArray([Holder()])])
    torch.testing.assert_close(tvm_backend_module._to_torch(nested), tensor)


def test_to_torch_keeps_multiple_outputs_as_a_tuple():
    class Holder:
        def __init__(self, value):
            self._value = value

        def __dlpack__(self, *args, **kwargs):
            return self._value.__dlpack__(*args, **kwargs)

        def __dlpack_device__(self):
            return self._value.__dlpack_device__()

    out = tvm_backend_module._to_torch([Holder(torch.ones(2)), Holder(torch.zeros(2))])

    assert isinstance(out, tuple)
    assert len(out) == 2


def test_backend_is_registered():
    assert isinstance(registry.get("tvm"), TVMBackend)


def exported_program_for(module: torch.nn.Module = None, args=None, kwargs=None):
    with torch.no_grad():
        return torch.export.export(module or model(), args or (torch.randn(8, 4),), kwargs or {})


def test_compile_exported_writes_and_validates_a_library(monkeypatch, tmp_path):
    calls = install_fake_tvm(monkeypatch)
    library_path = tmp_path / "compiled_model.tvm.so"

    result = TVMBackend().compile_exported(exported_program_for(), library_path)

    assert result == library_path
    # from_fx cannot lower `embedding`; baking params in keeps the reloaded
    # library's call signature equal to the exported program's real arguments.
    assert calls["keep_params_as_input"] is False
    assert calls["exported_library_path"] == str(library_path)
    # Round-trips through the saved library (not the in-memory executable) to
    # validate the save/reload path itself, not just codegen.
    assert calls["loaded_library_path"] == str(library_path)
    assert library_path.is_file()


def test_compile_exported_honours_a_target_option(monkeypatch, tmp_path):
    calls = install_fake_tvm(monkeypatch)
    library_path = tmp_path / "compiled_model.tvm.so"

    TVMBackend().compile_exported(
        exported_program_for(), library_path, options={"target": {"kind": "llvm", "mcpu": "x"}}
    )

    assert calls["built_target"] == "{'kind': 'llvm', 'mcpu': 'x'}"


def test_compile_exported_rejects_keyword_inputs(monkeypatch, tmp_path):
    install_fake_tvm(monkeypatch)
    library_path = tmp_path / "compiled_model.tvm.so"

    class TwoArgs(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    exported = exported_program_for(
        TwoArgs().eval(), args=(torch.randn(4),), kwargs={"y": torch.randn(4)}
    )

    with pytest.raises(CompilationError, match="positional inputs only"):
        TVMBackend().compile_exported(exported, library_path)

    assert not library_path.exists()


def test_compile_exported_wraps_build_failure_and_cleans_up(monkeypatch, tmp_path):
    install_fake_tvm(monkeypatch, build_error=RuntimeError("cannot legalize op"))
    library_path = tmp_path / "compiled_model.tvm.so"

    with pytest.raises(CompilationError, match="cannot legalize op"):
        TVMBackend().compile_exported(exported_program_for(), library_path)

    assert not library_path.exists()


def test_compile_exported_reports_missing_optional_dependency(monkeypatch, tmp_path):
    patch_find_spec(monkeypatch, present=False)
    library_path = tmp_path / "compiled_model.tvm.so"

    with pytest.raises(CompilationError, match=".\\[tvm\\]"):
        TVMBackend().compile_exported(exported_program_for(), library_path)


def test_load_library_returns_a_working_callable(monkeypatch, tmp_path):
    expected = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    calls = install_fake_tvm(monkeypatch, output=expected)
    library_path = tmp_path / "compiled_model.tvm.so"
    TVMBackend().compile_exported(exported_program_for(), library_path)

    result = TVMBackend().load_library(library_path)(torch.randn(8, 4))

    assert calls["loaded_library_path"] == str(library_path)
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, expected)


def test_load_library_reports_a_missing_file(monkeypatch, tmp_path):
    install_fake_tvm(monkeypatch)

    with pytest.raises(ArtifactLoadError, match="embed the exporting host's CPU architecture"):
        TVMBackend().load_library(tmp_path / "does-not-exist.so")


def test_load_library_reports_missing_optional_dependency(monkeypatch, tmp_path):
    patch_find_spec(monkeypatch, present=False)

    with pytest.raises(ArtifactLoadError, match=".\\[tvm\\]"):
        TVMBackend().load_library(tmp_path / "compiled_model.tvm.so")
