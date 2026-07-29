from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from lm7.backends import registry
from lm7.backends.base import CompileRequest
from lm7.backends.tvm import TVMBackend
from lm7.errors import CompilationError
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
        def __init__(self, value):
            self._value = value
            self.shape = tuple(value.shape)

        def numpy(self):
            return self._value.numpy()

    def fake_tensor(array, device=None):
        return _FakeTensor(torch.from_numpy(array))

    def build(mod, target=None):
        calls["built_target"] = str(target)
        if build_error is not None:
            raise build_error
        return SimpleNamespace()

    class Target:
        def __init__(self, value):
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
        runtime=SimpleNamespace(tensor=fake_tensor),
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

    TVMBackend().compile(request_for(options={"target": "llvm -mcpu=x"}), (torch.randn(8, 4),), {})

    assert calls["built_target"] == "llvm -mcpu=x"


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
        def numpy(self):
            return tensor.numpy()

    assert isinstance(tvm_backend_module._to_torch(Holder()), torch.Tensor)
    nested = FakeArray([FakeArray([Holder()])])
    torch.testing.assert_close(tvm_backend_module._to_torch(nested), tensor)


def test_to_torch_keeps_multiple_outputs_as_a_tuple():
    class Holder:
        def __init__(self, value):
            self._value = value

        def numpy(self):
            return self._value.numpy()

    out = tvm_backend_module._to_torch([Holder(torch.ones(2)), Holder(torch.zeros(2))])

    assert isinstance(out, tuple)
    assert len(out) == 2


def test_backend_is_registered():
    assert isinstance(registry.get("tvm"), TVMBackend)
