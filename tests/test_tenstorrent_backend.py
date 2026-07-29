from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from lm7.backends.base import CompileRequest
from lm7.backends.tenstorrent import TenstorrentBackend
from lm7.errors import CompilationError
from lm7.targets import TargetSpec

tenstorrent_backend_module = importlib.import_module("lm7.backends.tenstorrent")


def request(*, vendor: str = "tenstorrent", options=None) -> CompileRequest:
    return CompileRequest(
        torch.nn.Identity(),
        TargetSpec(vendor, "accelerator" if vendor == "tenstorrent" else "cpu"),
        "lazy",
        "automatic",
        "error",
        options or {},
    )


def _install_plugin(monkeypatch, runtime) -> None:
    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    monkeypatch.setattr(
        tenstorrent_backend_module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(),
    )
    monkeypatch.setattr(
        tenstorrent_backend_module.importlib.metadata,
        "version",
        lambda name: "test-version",
    )
    monkeypatch.setattr(
        tenstorrent_backend_module.importlib,
        "import_module",
        lambda name: runtime,
    )


def test_probe_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setattr(tenstorrent_backend_module.importlib.util, "find_spec", lambda name: None)

    info = TenstorrentBackend().probe()

    assert not info.available
    assert "pjrt-plugin-tt" in info.reason


def test_probe_selects_the_tt_pjrt_device(monkeypatch):
    selected = {}
    runtime = SimpleNamespace(
        device_type=lambda: selected.get("device_type", "CPU"),
        set_device_type=lambda value: selected.update(device_type=value),
        addressable_device_count=lambda: 2,
    )
    _install_plugin(monkeypatch, runtime)

    info = TenstorrentBackend().probe()

    assert selected == {"device_type": "TT"}
    assert info.available
    assert info.version == "test-version"
    assert "2 addressable Tenstorrent device(s)" in info.reason


def test_probe_never_hijacks_a_tpu_runtime(monkeypatch):
    runtime = SimpleNamespace(
        device_type=lambda: "TPU",
        set_device_type=lambda value: pytest.fail("must not reassign a live PJRT runtime"),
        addressable_device_count=lambda: 4,
    )
    _install_plugin(monkeypatch, runtime)

    info = TenstorrentBackend().probe()

    assert not info.available
    assert "the PJRT device is TPU, not TT" in info.reason


def test_probe_honours_an_explicit_pjrt_device(monkeypatch):
    runtime = SimpleNamespace(
        device_type=lambda: "CUDA",
        set_device_type=lambda value: pytest.fail("must not override an explicit PJRT_DEVICE"),
        addressable_device_count=lambda: 1,
    )
    _install_plugin(monkeypatch, runtime)
    monkeypatch.setenv("PJRT_DEVICE", "CUDA")

    info = TenstorrentBackend().probe()

    assert not info.available
    assert "the PJRT device is CUDA, not TT" in info.reason


def test_probe_reports_a_missing_card(monkeypatch):
    runtime = SimpleNamespace(
        device_type=lambda: "TT",
        set_device_type=lambda value: None,
        addressable_device_count=lambda: 0,
    )
    _install_plugin(monkeypatch, runtime)
    monkeypatch.setattr(tenstorrent_backend_module, "tenstorrent_device_nodes", list)

    info = TenstorrentBackend().probe()

    assert not info.available
    assert "no /dev/tenstorrent node exists" in info.reason


def test_probe_separates_a_present_card_from_a_broken_runtime(monkeypatch):
    runtime = SimpleNamespace(
        device_type=lambda: "TT",
        set_device_type=lambda value: None,
        addressable_device_count=lambda: 0,
    )
    _install_plugin(monkeypatch, runtime)
    monkeypatch.setattr(tenstorrent_backend_module, "tenstorrent_device_nodes", lambda: ["0", "1"])

    info = TenstorrentBackend().probe()

    assert not info.available
    assert "tt-kmd published 0, 1" in info.reason


def test_support_is_tenstorrent_only(monkeypatch):
    backend = TenstorrentBackend()
    monkeypatch.setattr(
        backend,
        "probe",
        lambda: SimpleNamespace(available=True, reason="available"),
    )

    assert backend.supports(request()).supported
    assert backend.supports(request()).priority == 100
    assert backend.supports(request(vendor="cpu")).reason == (
        "tt-xla supports Tenstorrent targets only in LM7."
    )


def test_compile_uses_the_registered_tt_backend(monkeypatch):
    backend = TenstorrentBackend()
    calls = {}
    torch_xla = SimpleNamespace(
        __version__="test-version",
        device=lambda index: torch.device("cpu"),
        sync=lambda **kwargs: calls.update(sync=kwargs),
    )
    monkeypatch.setattr(
        tenstorrent_backend_module.importlib,
        "import_module",
        lambda name: torch_xla,
    )
    monkeypatch.setattr(
        tenstorrent_backend_module.importlib.metadata,
        "version",
        lambda name: "plugin-test-version",
    )

    def fake_compile(model, **kwargs):
        calls.update(compile=kwargs)

        def compiled(value):
            calls["grad_enabled"] = torch.is_grad_enabled()
            calls["inference_mode"] = torch.is_inference_mode_enabled()
            return model(value)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    artifact = backend.compile(
        request(options={"dynamic": False, "enable_optimizer": True}),
        (torch.ones(2),),
        {},
    )

    assert calls == {
        "compile": {
            "backend": "tt",
            "dynamic": False,
            "options": {"enable_optimizer": True},
        },
        "sync": {"wait": True},
        "grad_enabled": False,
        "inference_mode": False,
    }
    assert artifact.metadata["torch_xla_version"] == "test-version"
    assert artifact.metadata["pjrt_plugin_tt_version"] == "plugin-test-version"


def test_compile_wraps_backend_failure(monkeypatch):
    backend = TenstorrentBackend()
    torch_xla = SimpleNamespace(
        __version__="test-version",
        device=lambda index: torch.device("cpu"),
        sync=lambda **kwargs: None,
    )
    monkeypatch.setattr(
        tenstorrent_backend_module.importlib,
        "import_module",
        lambda name: torch_xla,
    )
    monkeypatch.setattr(
        torch,
        "compile",
        lambda model, **kwargs: (
            lambda *args, **call_kwargs: (_ for _ in ()).throw(RuntimeError("tt-mlir lowering"))
        ),
    )

    with pytest.raises(CompilationError, match="tt-mlir lowering"):
        backend.compile(request(), (torch.ones(2),), {})
