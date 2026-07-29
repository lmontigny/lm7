from __future__ import annotations

import importlib.metadata

import pytest
import torch

import lm7
from lm7.benchmarking import _environment, _package_version, _percentile, _synchronize
from lm7.targets import TargetSpec


def test_cpu_eager_benchmark_reports_stable_schema():
    result = lm7.benchmark(
        torch.nn.Linear(4, 3).eval(),
        args=(torch.randn(2, 4),),
        target="cpu",
        backend="eager",
        warmup=1,
        repeats=3,
    )

    assert result.target.startswith("cpu")
    assert result.backend == "eager"
    assert result.first_call_ms >= 0
    assert result.latency_median_ms > 0
    assert result.latency_p95_ms >= result.latency_median_ms
    assert result.samples_per_second > 0
    assert result.peak_memory_bytes is None
    assert result.batch_size == 2
    assert result.environment["torch"] == torch.__version__
    assert result.environment["device_name"]
    assert result.environment["logical_cpu_count"]
    assert result.environment["torch_threads"] > 0
    assert result.to_dict()["repeats"] == 3


def test_explicit_cpu_benchmark_does_not_initialize_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def unexpected_cuda_init():
        raise AssertionError("CPU benchmark initialized CUDA")

    monkeypatch.setattr(torch.cuda, "init", unexpected_cuda_init)

    result = lm7.benchmark(
        torch.nn.Identity().eval(),
        args=(torch.randn(2, 4),),
        target="cpu",
        backend="eager",
        warmup=0,
        repeats=1,
    )

    assert result.target.startswith("cpu")


def test_benchmark_rejects_invalid_iteration_counts():
    model = torch.nn.Identity().eval()
    with pytest.raises(ValueError, match="warmup"):
        lm7.benchmark(model, args=(torch.tensor(1),), warmup=-1)
    with pytest.raises(ValueError, match="repeats"):
        lm7.benchmark(model, args=(torch.tensor(1),), repeats=0)


def test_percentile_interpolates():
    assert _percentile([1.0], 0.95) == 1.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def test_batch_size_is_detected_in_keyword_inputs():
    result = lm7.benchmark(
        torch.nn.Identity().eval(),
        kwargs={"input": torch.randn(3, 4)},
        target="cpu",
        backend="eager",
        warmup=0,
        repeats=1,
    )

    assert result.batch_size == 3


def test_tpu_synchronize_uses_torch_xla(monkeypatch):
    calls = {}

    class FakeTorchXla:
        @staticmethod
        def sync(*, wait):
            calls["wait"] = wait

    monkeypatch.setattr(
        "lm7.benchmarking.importlib.import_module",
        lambda name: FakeTorchXla if name == "torch_xla" else None,
    )

    _synchronize(TargetSpec("tpu", "accelerator"))

    assert calls == {"wait": True}


def test_tpu_environment_reports_pjrt_metadata(monkeypatch):
    class FakeRuntime:
        @staticmethod
        def device_type():
            return "TPU"

        @staticmethod
        def addressable_device_count():
            return 8

    monkeypatch.setattr(
        "lm7.benchmarking.importlib.import_module",
        lambda name: FakeRuntime if name == "torch_xla.runtime" else None,
    )

    environment = _environment(TargetSpec("tpu", "accelerator"))

    assert environment["device_name"] == "Google TPU"
    assert environment["pjrt_device"] == "TPU"
    assert environment["addressable_device_count"] == 8


def test_tensorrt_environment_reports_compiler_versions(monkeypatch):
    versions = {"torch-tensorrt": "2.12.1", "tensorrt": "10.16.1.11"}
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _ordinal: "Fake NVIDIA GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _ordinal: (8, 9))
    monkeypatch.setattr(
        "lm7.benchmarking.importlib.metadata.version",
        lambda distribution: versions[distribution],
    )

    environment = _environment(TargetSpec("nvidia", "gpu", "sm89"), "tensorrt")

    assert environment["torch_tensorrt"] == "2.12.1"
    assert environment["tensorrt"] == "10.16.1.11"


def test_missing_optional_package_version_is_none(monkeypatch):
    def missing(_distribution):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr("lm7.benchmarking.importlib.metadata.version", missing)

    assert _package_version("torch-tensorrt") is None
