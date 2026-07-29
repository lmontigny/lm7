from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import lm7
from lm7.backends import registry
from lm7.backends.base import CompileRequest
from lm7.backends.executorch import ExecuTorchBackend
from lm7.errors import ArtifactLoadError, BackendUnavailableError, CompilationError
from lm7.exporting import COMPILED_PTE_NAME
from lm7.targets import parse_target

executorch_backend_module = importlib.import_module("lm7.backends.executorch")


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


def patch_find_spec(monkeypatch, *, present: bool) -> None:
    """Make `executorch` look present or absent without touching the import system.

    `importlib.util` is a shared module, so replacing `find_spec` outright lies to
    every lazy import in the process -- including torch's own, which breaks
    `torch.export` in any test that exports under the patch. Only answer for the
    executorch namespace and delegate the rest to the real implementation.
    """
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "executorch" or name.startswith("executorch."):
            return SimpleNamespace() if present else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(executorch_backend_module.importlib.util, "find_spec", fake_find_spec)


def install_fake_executorch(monkeypatch, tmp_path, *, delegated: int = 1, total: int = 2) -> dict:
    """Stand in for ExecuTorch so the packaging path is testable without it.

    The prebuilt runtime extension is ABI-linked to libtorch and cannot be
    installed next to the PyTorch this suite runs against, so these tests cover
    LM7's packaging, manifest, and validation rather than the lowering itself.
    The real lowering is covered by tests/test_executorch_integration.py.
    """
    calls: dict = {}

    def make_node(op: str, target_name: str) -> SimpleNamespace:
        return SimpleNamespace(op=op, target=target_name)

    nodes = [make_node("call_function", "executorch_call_delegate")] * delegated
    nodes += [make_node("call_function", "aten.add.Tensor")] * (total - delegated)
    nodes += [make_node("placeholder", "x")]

    lowered = SimpleNamespace(
        buffer=b"PTE\x00fake-program",
        exported_program=lambda: SimpleNamespace(
            graph_module=SimpleNamespace(graph=SimpleNamespace(nodes=nodes))
        ),
    )

    class FakeEdge:
        def to_executorch(self):
            return lowered

    def to_edge_transform_and_lower(exported_program, partitioner):
        calls["partitioner"] = [type(item).__name__ for item in partitioner]
        return FakeEdge()

    exir = SimpleNamespace(to_edge_transform_and_lower=to_edge_transform_and_lower)

    class XnnpackPartitioner:
        pass

    partition_module = SimpleNamespace(XnnpackPartitioner=XnnpackPartitioner)

    class FakeMethod:
        def execute(self, inputs):
            calls["executed"] = [tuple(item.shape) for item in inputs]
            return [torch.zeros(8, 3)]

    class FakeProgram:
        def load_method(self, name):
            calls["method"] = name
            return FakeMethod()

    class FakeRuntime:
        @staticmethod
        def get():
            return SimpleNamespace(load_program=lambda path: FakeProgram())

    runtime_module = SimpleNamespace(Runtime=FakeRuntime)

    modules = {
        "executorch": SimpleNamespace(__file__=str(tmp_path / "executorch" / "__init__.py")),
        "executorch.exir": exir,
        "executorch.backends.xnnpack.partition.xnnpack_partitioner": partition_module,
        "executorch.runtime": runtime_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    patch_find_spec(monkeypatch, present=True)
    monkeypatch.setattr(
        executorch_backend_module.importlib.metadata, "version", lambda name: "1.3.1-test"
    )
    flatc = tmp_path / "bin" / "flatc"
    flatc.parent.mkdir(parents=True, exist_ok=True)
    flatc.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(executorch_backend_module, "_flatc_path", lambda: flatc)
    return calls


def test_probe_reports_missing_optional_dependency(monkeypatch):
    patch_find_spec(monkeypatch, present=False)

    info = ExecuTorchBackend().probe()

    assert not info.available
    assert ".[executorch]" in info.reason


def test_probe_reports_unresolvable_flatc(monkeypatch, tmp_path):
    install_fake_executorch(monkeypatch, tmp_path)
    monkeypatch.setattr(executorch_backend_module, "_flatc_path", lambda: None)

    info = ExecuTorchBackend().probe()

    assert not info.available
    assert "flatc" in info.reason


def test_probe_available(monkeypatch, tmp_path):
    install_fake_executorch(monkeypatch, tmp_path)

    info = ExecuTorchBackend().probe()

    assert info.available
    assert info.version == "1.3.1-test"
    assert "xnnpack" in info.reason


def test_supports_is_export_only(monkeypatch, tmp_path):
    install_fake_executorch(monkeypatch, tmp_path)

    support = ExecuTorchBackend().supports(request_for())

    assert not support.supported
    assert "export-only" in support.reason


def test_compile_refuses_in_process_use(monkeypatch, tmp_path):
    install_fake_executorch(monkeypatch, tmp_path)

    with pytest.raises(CompilationError, match="does not compile in-process"):
        ExecuTorchBackend().compile(request_for(), (torch.ones(1, 4),), {})


def test_compile_exported_writes_pte_and_reports_partition(monkeypatch, tmp_path):
    calls = install_fake_executorch(monkeypatch, tmp_path, delegated=3, total=7)
    exported = torch.export.export(model(), (torch.randn(8, 4),))
    destination = tmp_path / "out" / COMPILED_PTE_NAME

    lowered = ExecuTorchBackend().compile_exported(exported, destination)

    assert calls["partitioner"] == ["XnnpackPartitioner"]
    assert lowered.path == destination
    assert destination.read_bytes() == b"PTE\x00fake-program"
    assert (lowered.delegated_calls, lowered.total_calls) == (3, 7)


def test_compile_exported_wraps_lowering_failure(monkeypatch, tmp_path):
    install_fake_executorch(monkeypatch, tmp_path)
    exir = sys.modules["executorch.exir"]
    monkeypatch.setattr(
        exir,
        "to_edge_transform_and_lower",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unsupported operator")),
    )
    exported = torch.export.export(model(), (torch.randn(8, 4),))

    with pytest.raises(CompilationError, match="unsupported operator"):
        ExecuTorchBackend().compile_exported(exported, tmp_path / COMPILED_PTE_NAME)


def test_loaded_method_rejects_keyword_inputs(monkeypatch, tmp_path):
    install_fake_executorch(monkeypatch, tmp_path)
    program = tmp_path / COMPILED_PTE_NAME
    program.write_bytes(b"PTE\x00fake-program")

    loaded = ExecuTorchBackend().load_pte(program)

    with pytest.raises(ArtifactLoadError, match="positional tensors only"):
        loaded(input_ids=torch.ones(1, 4))


def test_loaded_method_returns_a_single_tensor(monkeypatch, tmp_path):
    calls = install_fake_executorch(monkeypatch, tmp_path)
    program = tmp_path / COMPILED_PTE_NAME
    program.write_bytes(b"PTE\x00fake-program")

    loaded = ExecuTorchBackend().load_pte(program)
    output = loaded(torch.randn(8, 4))

    assert calls["method"] == "forward"
    assert calls["executed"] == [(8, 4)]
    assert isinstance(output, torch.Tensor)


def test_export_writes_an_executorch_artifact(monkeypatch, tmp_path):
    install_fake_executorch(monkeypatch, tmp_path, delegated=2, total=5)
    output = tmp_path / "model.lm7"

    artifact = lm7.export(
        model(), args=(torch.randn(8, 4),), target="cpu", backend="executorch", output=output
    )

    assert artifact.manifest.backend == "executorch"
    assert artifact.manifest.backend_version == "1.3.1-test"
    assert artifact.manifest.compiled_file == COMPILED_PTE_NAME
    requirements = artifact.manifest.runtime_requirements
    assert requirements["delegate"] == "xnnpack"
    assert (requirements["delegated_calls"], requirements["total_calls"]) == (2, 5)
    # The .pte carries its own weights and the XNNPACK delegate spans ARM64 and
    # x86-64, so unlike every other compiled payload it is not host-bound.
    assert requirements["device_bound"] is False
    assert (output / COMPILED_PTE_NAME).is_file()


def test_export_rejects_a_non_cpu_target(monkeypatch, tmp_path):
    install_fake_executorch(monkeypatch, tmp_path)

    with pytest.raises(BackendUnavailableError, match="XNNPACK delegate, which is a CPU target"):
        lm7.export(
            model(),
            args=(torch.randn(8, 4),),
            target="nvidia",
            backend="executorch",
            output=tmp_path / "model.lm7",
        )


def test_export_reports_an_unavailable_backend(monkeypatch, tmp_path):
    patch_find_spec(monkeypatch, present=False)

    with pytest.raises(BackendUnavailableError, match="ExecuTorch is not installed"):
        lm7.export(
            model(),
            args=(torch.randn(8, 4),),
            target="cpu",
            backend="executorch",
            output=tmp_path / "model.lm7",
        )


def test_load_artifact_verifies_the_pte_checksum(monkeypatch, tmp_path):
    install_fake_executorch(monkeypatch, tmp_path)
    output = tmp_path / "model.lm7"
    lm7.export(
        model(), args=(torch.randn(8, 4),), target="cpu", backend="executorch", output=output
    )
    (output / COMPILED_PTE_NAME).write_bytes(b"PTE\x00tampered")

    with pytest.raises(ArtifactLoadError, match="checksum does not match"):
        lm7.load_artifact(output)


def test_backend_is_registered():
    assert isinstance(registry.get("executorch"), ExecuTorchBackend)


def test_flatc_context_restores_path(monkeypatch, tmp_path):
    """The lowering puts the wheel's flatc on PATH; it must not leak afterwards."""
    flatc = tmp_path / "bin" / "flatc"
    flatc.parent.mkdir(parents=True, exist_ok=True)
    flatc.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(executorch_backend_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(executorch_backend_module, "_flatc_path", lambda: flatc)
    monkeypatch.setenv("PATH", "/original")

    import os

    with executorch_backend_module._flatc_on_path():
        assert str(tmp_path / "bin") in os.environ["PATH"]
        assert "/original" in os.environ["PATH"]
    assert os.environ["PATH"] == "/original"


def test_flatc_context_leaves_a_resolvable_flatc_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(executorch_backend_module.shutil, "which", lambda name: "/usr/bin/flatc")
    monkeypatch.setenv("PATH", "/original")

    import os

    with executorch_backend_module._flatc_on_path():
        assert os.environ["PATH"] == "/original"


def test_delegate_counts_tolerates_an_opaque_program():
    assert executorch_backend_module._delegate_counts(object()) == (0, 0)


def test_load_pte_requires_the_runtime(monkeypatch, tmp_path):
    patch_find_spec(monkeypatch, present=False)

    with pytest.raises(ArtifactLoadError, match="ExecuTorch is not installed"):
        ExecuTorchBackend().load_pte(Path(tmp_path / COMPILED_PTE_NAME))
