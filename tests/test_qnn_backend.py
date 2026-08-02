from __future__ import annotations

import contextlib
import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

import lm7
from lm7.backends import registry
from lm7.backends.base import CompileRequest
from lm7.backends.qnn import ExecuTorchQNNBackend, parse_options
from lm7.errors import ArtifactLoadError, BackendUnavailableError, CompilationError
from lm7.exporting import COMPILED_PTE_NAME
from lm7.targets import parse_target

qnn_backend_module = importlib.import_module("lm7.backends.qnn")


def model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()


def request_for(target: str = "qualcomm:sm8750") -> CompileRequest:
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

    monkeypatch.setattr(qnn_backend_module.importlib.util, "find_spec", fake_find_spec)


def install_fake_qnn(
    monkeypatch,
    tmp_path,
    *,
    delegated: int = 2,
    total: int = 5,
) -> dict:
    calls: dict = {}
    sdk_root = tmp_path / "qairt" / "2.37.0"
    sdk_root.mkdir(parents=True)
    (sdk_root / "QNN_README.txt").write_text("QNN SDK\n", encoding="utf-8")
    monkeypatch.setenv("QNN_SDK_ROOT", str(sdk_root))

    def make_node(op: str, target_name: str) -> SimpleNamespace:
        return SimpleNamespace(op=op, target=target_name)

    nodes = [make_node("call_function", "executorch_call_delegate")] * delegated
    nodes += [make_node("call_function", "aten.add.Tensor")] * (total - delegated)
    nodes += [make_node("placeholder", "x")]
    lowered = SimpleNamespace(
        buffer=b"PTE\x00fake-qnn-program",
        exported_program=lambda: SimpleNamespace(
            graph_module=SimpleNamespace(graph=SimpleNamespace(nodes=nodes))
        ),
    )

    class FakeEdge:
        def to_executorch(self):
            calls["to_executorch"] = True
            return lowered

    def generate_htp_compiler_spec(*, use_fp16):
        calls["use_fp16"] = use_fp16
        return "htp-options"

    def generate_qnn_executorch_compiler_spec(*, soc_model, backend_options):
        calls["soc_model"] = soc_model
        calls["backend_options"] = backend_options
        return ["qnn-compile-spec"]

    def to_edge_transform_and_lower_to_qnn(module, inputs, compiler_specs):
        calls["module"] = module
        calls["inputs"] = tuple(tuple(item.shape) for item in inputs)
        calls["compiler_specs"] = compiler_specs
        return FakeEdge()

    utils = SimpleNamespace(
        generate_htp_compiler_spec=generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec=generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn=to_edge_transform_and_lower_to_qnn,
    )
    schema = SimpleNamespace(QcomChipset=SimpleNamespace(SM8750="sm8750-enum"))
    modules = {
        "executorch": SimpleNamespace(__file__=str(tmp_path / "executorch" / "__init__.py")),
        "executorch.backends.qualcomm.partition.qnn_partitioner": SimpleNamespace(),
        "executorch.backends.qualcomm.utils.utils": utils,
        "executorch.backends.qualcomm.serialization.qc_schema": schema,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    patch_find_spec(monkeypatch, present=True)
    monkeypatch.setattr(qnn_backend_module, "_executorch_version", lambda: "1.3.1-test")
    monkeypatch.setattr(qnn_backend_module, "_flatc_path", lambda: tmp_path / "flatc")
    monkeypatch.setattr(qnn_backend_module, "_flatc_on_path", contextlib.nullcontext)
    return calls


def test_probe_reports_missing_executorch(monkeypatch):
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    patch_find_spec(monkeypatch, present=False)

    info = ExecuTorchQNNBackend().probe()

    assert not info.available
    assert "ExecuTorch is not installed" in info.reason


def test_probe_requires_qnn_sdk_root(monkeypatch):
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    patch_find_spec(monkeypatch, present=True)
    monkeypatch.setattr(qnn_backend_module, "_executorch_version", lambda: "1.3.1-test")

    info = ExecuTorchQNNBackend().probe()

    assert not info.available
    assert "QNN_SDK_ROOT is not set" in info.reason


def test_probe_rejects_non_sdk_directory(monkeypatch, tmp_path):
    patch_find_spec(monkeypatch, present=True)
    monkeypatch.setenv("QNN_SDK_ROOT", str(tmp_path))
    monkeypatch.setattr(qnn_backend_module, "_executorch_version", lambda: "1.3.1-test")

    info = ExecuTorchQNNBackend().probe()

    assert not info.available
    assert "QNN_README.txt is missing" in info.reason


def test_probe_available(monkeypatch, tmp_path):
    install_fake_qnn(monkeypatch, tmp_path)

    info = ExecuTorchQNNBackend().probe()

    assert info.available
    assert info.version == "1.3.1-test"
    assert "SM8750" in info.reason
    assert "v79" in info.reason


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"precision": "int8"}, "must be 'fp16'"),
        ({"quantization": "8a8w"}, "Unsupported QNN options"),
    ],
)
def test_parse_options_rejects_unimplemented_modes(options, message):
    with pytest.raises(CompilationError, match=message):
        parse_options(options)


def test_supports_and_compile_are_export_only(monkeypatch, tmp_path):
    install_fake_qnn(monkeypatch, tmp_path)
    backend = ExecuTorchQNNBackend()

    assert not backend.supports(request_for()).supported
    with pytest.raises(CompilationError, match="does not compile in-process"):
        backend.compile(request_for(), (torch.ones(1, 4),), {})


def test_compile_exported_writes_qnn_pte_and_records_partition(monkeypatch, tmp_path):
    calls = install_fake_qnn(monkeypatch, tmp_path, delegated=3, total=7)
    exported = torch.export.export(model(), (torch.randn(8, 4),))
    destination = tmp_path / COMPILED_PTE_NAME

    lowered = ExecuTorchQNNBackend().compile_exported(
        exported,
        destination,
        target=parse_target("qualcomm:sm8750"),
    )

    assert calls["use_fp16"] is True
    assert calls["soc_model"] == "sm8750-enum"
    assert calls["backend_options"] == "htp-options"
    assert calls["compiler_specs"] == ["qnn-compile-spec"]
    assert calls["inputs"] == ((8, 4),)
    assert destination.read_bytes() == b"PTE\x00fake-qnn-program"
    assert (lowered.delegated_calls, lowered.total_calls) == (3, 7)
    assert lowered.soc_model == "SM8750"
    assert lowered.htp_arch == "v79"
    assert lowered.precision == "fp16"


def test_compile_exported_rejects_zero_delegation(monkeypatch, tmp_path):
    install_fake_qnn(monkeypatch, tmp_path, delegated=0, total=4)
    exported = torch.export.export(model(), (torch.randn(8, 4),))

    with pytest.raises(CompilationError, match="delegated zero"):
        ExecuTorchQNNBackend().compile_exported(
            exported,
            tmp_path / COMPILED_PTE_NAME,
            target=parse_target("qualcomm:sm8750"),
        )


def test_compile_exported_rejects_dynamic_shapes(monkeypatch, tmp_path):
    install_fake_qnn(monkeypatch, tmp_path)
    batch = torch.export.Dim("batch", min=1, max=16)
    exported = torch.export.export(
        model(),
        (torch.randn(8, 4),),
        dynamic_shapes=({0: batch},),
    )

    with pytest.raises(CompilationError, match="requires static shapes"):
        ExecuTorchQNNBackend().compile_exported(
            exported,
            tmp_path / COMPILED_PTE_NAME,
            target=parse_target("qualcomm:sm8750"),
        )


def test_export_rejects_wrong_target(monkeypatch, tmp_path):
    install_fake_qnn(monkeypatch, tmp_path)

    with pytest.raises(BackendUnavailableError, match="qualcomm:sm8750"):
        lm7.export(
            model(),
            args=(torch.randn(8, 4),),
            target="cpu",
            backend="qnn",
            output=tmp_path / "model.lm7",
        )


def test_export_writes_device_bound_qnn_manifest(monkeypatch, tmp_path):
    install_fake_qnn(monkeypatch, tmp_path, delegated=2, total=5)

    artifact = lm7.export(
        model(),
        args=(torch.randn(8, 4),),
        target="qualcomm:sm8750",
        backend="qnn",
        output=tmp_path / "model.lm7",
    )

    assert artifact.manifest.backend == "qnn"
    assert artifact.manifest.backend_version == "1.3.1-test"
    assert artifact.manifest.compiled_file == COMPILED_PTE_NAME
    requirements = artifact.manifest.runtime_requirements
    assert requirements["delegate"] == "qnn"
    assert requirements["backend"] == "htp"
    assert requirements["soc_model"] == "SM8750"
    assert requirements["htp_arch"] == "v79"
    assert requirements["vtcm_mb"] == 8
    assert requirements["precision"] == "fp16"
    assert requirements["qnn_sdk"] == "2.37.0"
    assert requirements["device_bound"] is True
    assert (requirements["delegated_calls"], requirements["total_calls"]) == (2, 5)
    assert "libQnnHtp.so" in requirements["runtime_libraries"]
    assert (artifact.path / COMPILED_PTE_NAME).is_file()

    with pytest.raises(ArtifactLoadError, match="bound to SM8750"):
        artifact(torch.randn(8, 4))


def test_load_artifact_verifies_qnn_pte_checksum(monkeypatch, tmp_path):
    install_fake_qnn(monkeypatch, tmp_path)
    output = tmp_path / "model.lm7"
    lm7.export(
        model(),
        args=(torch.randn(8, 4),),
        target="qualcomm:sm8750",
        backend="qnn",
        output=output,
    )
    (output / COMPILED_PTE_NAME).write_bytes(b"tampered")

    with pytest.raises(ArtifactLoadError, match="checksum does not match"):
        lm7.load_artifact(output)


def test_backend_is_registered():
    assert isinstance(registry.get("qnn"), ExecuTorchQNNBackend)
