from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "gpu.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("benchmarks_gpu", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparisons_report_only_matched_execution_paths():
    harness = _load_harness()
    results = [
        {"backend": "torch-eager", "latency_median_ms": 12.0},
        {"backend": "torch-compile", "latency_median_ms": 3.0},
        {"backend": "inductor", "latency_median_ms": 2.0},
    ]

    comparisons = harness._comparisons(results)

    assert comparisons == [
        {
            "baseline": "torch-eager",
            "optimized": "torch-compile",
            "latency_speedup": pytest.approx(4.0),
        }
    ]


def test_comparisons_cover_lm7_automatic_and_placed_pairs():
    harness = _load_harness()
    results = [
        {"backend": "eager", "latency_median_ms": 10.0},
        {"backend": "inductor", "latency_median_ms": 5.0},
        {"backend": "eager-placed", "latency_median_ms": 9.0},
        {"backend": "inductor-placed", "latency_median_ms": 3.0},
    ]

    comparisons = harness._comparisons(results)

    assert comparisons == [
        {
            "baseline": "eager-placed",
            "optimized": "inductor-placed",
            "latency_speedup": pytest.approx(3.0),
        },
        {
            "baseline": "eager",
            "optimized": "inductor",
            "latency_speedup": pytest.approx(2.0),
        },
    ]


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (4.0, "4.00x speedup"),
        (0.5, "2.00x slowdown"),
    ],
)
def test_comparison_summary_names_the_direction(ratio, expected):
    harness = _load_harness()

    summary = harness._comparison_summary(
        {
            "baseline": "torch-eager",
            "optimized": "torch-compile",
            "latency_speedup": ratio,
        }
    )

    assert expected in summary
