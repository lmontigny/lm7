from __future__ import annotations

import pytest
import torch

import lm7
from lm7.benchmarking import _percentile


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
    assert result.to_dict()["repeats"] == 3


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
