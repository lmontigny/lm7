from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

import lm7
from lm7.backends import registry
from lm7.backends.base import CompileRequest
from lm7.backends.coreml import ExecuTorchCoreMLBackend, parse_options
from lm7.errors import ArtifactLoadError, BackendUnavailableError, CompilationError
from lm7.exporting import COMPILED_PTE_NAME
from lm7.targets import parse_target

coreml_backend_module = importlib.import_module("lm7.backends.coreml")


def model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()


def request_for(target: str = "apple") -> CompileRequest:
    return CompileRequest(
        model=model(),
        target=parse_target(target),
        mode="lazy",
        transfers="automatic",
        fallback="error",
    )


def patch_find_spec(monkeypatch, *, present: bool) -> None:
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "executorch" or name.startswith("executorch."):
            return SimpleNamespace() if present else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(coreml_backend_module.importlib.util, "find_spec", fake_find_spec)


def install_fake_coreml(
    monkeypatch,
    tmp_path,
    *,
    delegated: int = 1,
    total: int = 2,
    execute_output: torch.Tensor | None = None,
    darwin: bool = True,
) -> dict:
    """Stand in for ExecuTorch's Core ML delegate and coremltools.

    Unlike QNN, a Core ML .pte actually executes on the host that built it, so
    this also fakes the runtime load path (mirroring
    tests/test_executorch_backend.py's install_fake_executorch), not just the
    lowering. The real lowering and execution are covered by
    tests/test_coreml_integration.py.
    """
    calls: dict = {}
    output = torch.zeros(8, 3) if execute_output is None else execute_output

    def make_node(op: str, target_name: str) -> SimpleNamespace:
        return SimpleNamespace(op=op, target=target_name)

    nodes = [make_node("call_function", "executorch_call_delegate")] * delegated
    nodes += [make_node("call_function", "aten.add.Tensor")] * (total - delegated)
    lowered = SimpleNamespace(
        buffer=b"PTE\x00fake-coreml-program",
        exported_program=lambda: SimpleNamespace(
            graph_module=SimpleNamespace(graph=SimpleNamespace(nodes=nodes))
        ),
    )

    class FakeEdge:
        def to_executorch(self):
            calls["to_executorch"] = True
            return lowered

    def to_edge_transform_and_lower(exported_program, partitioner):
        calls["exported_program"] = exported_program
        calls["partitioner"] = partitioner
        return FakeEdge()

    exir = SimpleNamespace(to_edge_transform_and_lower=to_edge_transform_and_lower)

    class FakePartitioner:
        def __init__(self, *, compile_specs):
            calls["compile_specs"] = compile_specs

    partitioner_module = SimpleNamespace(CoreMLPartitioner=FakePartitioner)

    class FakeCoreMLBackend:
        @staticmethod
        def generate_compile_specs(*, compute_unit, compute_precision):
            calls["compute_unit"] = compute_unit
            calls["compute_precision"] = compute_precision
            return "coreml-compile-specs"

    compiler_module = SimpleNamespace(CoreMLBackend=FakeCoreMLBackend)

    ct = SimpleNamespace(
        ComputeUnit=SimpleNamespace(
            ALL="ALL", CPU_ONLY="CPU_ONLY", CPU_AND_GPU="CPU_AND_GPU", CPU_AND_NE="CPU_AND_NE"
        )
    )
    precision_module = SimpleNamespace(
        ComputePrecision=SimpleNamespace(FLOAT16="FLOAT16", FLOAT32="FLOAT32")
    )

    class FakeMethod:
        def execute(self, inputs):
            calls["executed"] = [tuple(item.shape) for item in inputs]
            return [output]

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
        "executorch.backends.apple.coreml.partition.coreml_partitioner": partitioner_module,
        "executorch.backends.apple.coreml.compiler": compiler_module,
        "executorch.runtime": runtime_module,
        "coremltools": ct,
        "coremltools.converters.mil.mil.passes.defs.quantization": precision_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    patch_find_spec(monkeypatch, present=True)
    monkeypatch.setattr(coreml_backend_module, "_executorch_version", lambda: "1.3.1-test")
    monkeypatch.setattr(coreml_backend_module, "_flatc_path", lambda: tmp_path / "flatc")
    monkeypatch.setattr(coreml_backend_module.sys, "platform", "darwin" if darwin else "linux")
    return calls


def test_probe_reports_missing_optional_dependency(monkeypatch):
    patch_find_spec(monkeypatch, present=False)

    info = ExecuTorchCoreMLBackend().probe()

    assert not info.available
    assert "ExecuTorch is not installed" in info.reason


def test_probe_rejects_non_macos(monkeypatch, tmp_path):
    install_fake_coreml(monkeypatch, tmp_path, darwin=False)

    info = ExecuTorchCoreMLBackend().probe()

    assert not info.available
    assert "macOS-only" in info.reason


def test_probe_available(monkeypatch, tmp_path):
    install_fake_coreml(monkeypatch, tmp_path)

    info = ExecuTorchCoreMLBackend().probe()

    assert info.available
    assert info.version == "1.3.1-test"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"compute_unit": "gpu_only"}, "Unsupported Core ML compute_unit"),
        ({"compute_precision": "int8"}, "Unsupported Core ML compute_precision"),
        ({"foo": "bar"}, "Unsupported Core ML options"),
    ],
)
def test_parse_options_rejects_bad_values(options, message):
    with pytest.raises(CompilationError, match=message):
        parse_options(options)


def test_supports_and_compile_are_export_only(monkeypatch, tmp_path):
    install_fake_coreml(monkeypatch, tmp_path)
    backend = ExecuTorchCoreMLBackend()

    assert not backend.supports(request_for()).supported
    with pytest.raises(CompilationError, match="does not compile in-process"):
        backend.compile(request_for(), (torch.ones(1, 4),), {})


def test_compile_exported_writes_pte_and_records_partition(monkeypatch, tmp_path):
    calls = install_fake_coreml(monkeypatch, tmp_path, delegated=1, total=2)
    exported = torch.export.export(model(), (torch.randn(8, 4),))
    destination = tmp_path / COMPILED_PTE_NAME

    lowered = ExecuTorchCoreMLBackend().compile_exported(exported, destination)

    assert calls["compute_unit"] == "ALL"
    assert calls["compute_precision"] == "FLOAT16"
    assert calls["compile_specs"] == "coreml-compile-specs"
    assert destination.read_bytes() == b"PTE\x00fake-coreml-program"
    assert (lowered.delegated_calls, lowered.total_calls) == (1, 2)
    assert lowered.compute_unit == "all"
    assert lowered.compute_precision == "float16"


def test_compile_exported_honours_options(monkeypatch, tmp_path):
    calls = install_fake_coreml(monkeypatch, tmp_path)
    exported = torch.export.export(model(), (torch.randn(8, 4),))

    ExecuTorchCoreMLBackend().compile_exported(
        exported,
        tmp_path / COMPILED_PTE_NAME,
        options={"compute_unit": "cpu_only", "compute_precision": "float32"},
    )

    assert calls["compute_unit"] == "CPU_ONLY"
    assert calls["compute_precision"] == "FLOAT32"


def test_compile_exported_rejects_zero_delegation(monkeypatch, tmp_path):
    install_fake_coreml(monkeypatch, tmp_path, delegated=0, total=4)
    exported = torch.export.export(model(), (torch.randn(8, 4),))

    with pytest.raises(CompilationError, match="delegated zero"):
        ExecuTorchCoreMLBackend().compile_exported(exported, tmp_path / COMPILED_PTE_NAME)


def test_compile_exported_rejects_dynamic_shapes(monkeypatch, tmp_path):
    install_fake_coreml(monkeypatch, tmp_path)
    batch = torch.export.Dim("batch", min=1, max=16)
    exported = torch.export.export(model(), (torch.randn(8, 4),), dynamic_shapes=({0: batch},))

    with pytest.raises(CompilationError, match="requires static shapes"):
        ExecuTorchCoreMLBackend().compile_exported(exported, tmp_path / COMPILED_PTE_NAME)


def test_load_pte_executes_through_the_runtime(monkeypatch, tmp_path):
    expected = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    calls = install_fake_coreml(monkeypatch, tmp_path, execute_output=expected)
    exported = torch.export.export(model(), (torch.randn(8, 4),))
    destination = tmp_path / COMPILED_PTE_NAME
    ExecuTorchCoreMLBackend().compile_exported(exported, destination)

    result = ExecuTorchCoreMLBackend().load_pte(destination)(torch.randn(8, 4))

    assert calls["method"] == "forward"
    assert calls["executed"] == [(8, 4)]
    torch.testing.assert_close(result, expected)


def test_load_pte_requires_the_runtime(monkeypatch, tmp_path):
    patch_find_spec(monkeypatch, present=False)

    with pytest.raises(ArtifactLoadError, match="ExecuTorch is not installed"):
        ExecuTorchCoreMLBackend().load_pte(tmp_path / COMPILED_PTE_NAME)


def test_export_rejects_non_apple_target(monkeypatch, tmp_path):
    install_fake_coreml(monkeypatch, tmp_path)

    with pytest.raises(BackendUnavailableError, match="target='apple'"):
        lm7.export(
            model(),
            args=(torch.randn(8, 4),),
            target="cpu",
            backend="coreml",
            output=tmp_path / "model.lm7",
        )


def test_export_writes_portable_coreml_manifest(monkeypatch, tmp_path):
    install_fake_coreml(monkeypatch, tmp_path, delegated=1, total=2)

    artifact = lm7.export(
        model(),
        args=(torch.randn(8, 4),),
        target="apple",
        backend="coreml",
        output=tmp_path / "model.lm7",
    )

    assert artifact.manifest.backend == "coreml"
    assert artifact.manifest.backend_version == "1.3.1-test"
    assert artifact.manifest.compiled_file == COMPILED_PTE_NAME
    requirements = artifact.manifest.runtime_requirements
    assert requirements["delegate"] == "coreml"
    assert requirements["compute_unit"] == "all"
    assert requirements["compute_precision"] == "float16"
    assert requirements["device_bound"] is False
    assert (requirements["delegated_calls"], requirements["total_calls"]) == (1, 2)
    assert (artifact.path / COMPILED_PTE_NAME).is_file()


def test_load_artifact_verifies_coreml_pte_checksum(monkeypatch, tmp_path):
    install_fake_coreml(monkeypatch, tmp_path)
    output = tmp_path / "model.lm7"
    lm7.export(
        model(),
        args=(torch.randn(8, 4),),
        target="apple",
        backend="coreml",
        output=output,
    )
    (output / COMPILED_PTE_NAME).write_bytes(b"tampered")

    with pytest.raises(ArtifactLoadError, match="checksum does not match"):
        lm7.load_artifact(output)


def test_backend_is_registered():
    assert isinstance(registry.get("coreml"), ExecuTorchCoreMLBackend)
