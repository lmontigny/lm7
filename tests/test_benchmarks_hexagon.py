"""Tests for benchmarks/hexagon.py.

The Qualcomm Hexagon paths cannot run in CI or on any host without the
hexagon-mlir toolchain and an NPU, so what is verifiable here is the harness
itself: that it degrades cleanly when the toolchain is absent, that the CLI
matches what docs/qualcomm-hexagon.md tells people to type, and that the
host-only run produces the documented JSON shape.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "hexagon.py"


def _load_harness():
    # benchmarks/ is not an installed package, so load the script by path.
    spec = importlib.util.spec_from_file_location("benchmarks_hexagon", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


def test_host_paths_are_always_available(harness):
    assert harness._unavailable_reason("eager", "mlp") is None
    assert harness._unavailable_reason("inductor", "mlp") is None


def test_hexagon_paths_report_a_reason_without_the_toolchain(harness):
    if harness._module_available("torch_mlir") and harness._hexagon_backend_available():
        pytest.skip("hexagon-mlir is installed; the unavailable path cannot be exercised")
    for path in ("hexagon", "hexagon-sim"):
        reason = harness._unavailable_reason(path, "mlp")
        assert isinstance(reason, str) and reason
        assert "docs/qualcomm-hexagon.md" in reason or "hexagon-mlir" in reason


def test_backend_probe_is_false_and_does_not_raise(harness):
    assert isinstance(harness._hexagon_backend_available(), bool)


def test_parser_accepts_the_documented_invocation(harness):
    arguments = harness.build_parser().parse_args(
        [
            "--model",
            "gpt2",
            "--path",
            "eager",
            "inductor",
            "hexagon",
            "--dtype",
            "float16",
            "--iterations",
            "10",
            "--option",
            "enableVTCMTiling=True",
            "--output",
            "out.json",
        ]
    )
    assert arguments.model == "gpt2"
    assert arguments.path == ["eager", "inductor", "hexagon"]
    assert arguments.iterations == 10
    assert arguments.option == [("enableVTCMTiling", "True")]
    assert arguments.layers == 2


def test_parser_rejects_an_unknown_path(harness):
    with pytest.raises(SystemExit):
        harness.build_parser().parse_args(["--path", "migraphx"])


def test_parser_rejects_a_malformed_option(harness):
    with pytest.raises(SystemExit):
        harness.build_parser().parse_args(["--option", "enableLWP"])


def test_logits_normalizes_tensor_list_and_model_output(harness):
    tensor = torch.zeros(2, 3)

    class Output:
        logits = tensor

    assert harness._logits(tensor) is tensor
    assert harness._logits([tensor]) is tensor
    assert harness._logits((tensor, tensor)) is tensor
    assert harness._logits(Output()) is tensor
    with pytest.raises(TypeError):
        harness._logits(object())


def test_simulator_context_restores_the_environment(harness):
    previous = os.environ.get("RUN_ON_SIM")
    try:
        os.environ.pop("RUN_ON_SIM", None)
        with harness._simulator(True):
            assert os.environ["RUN_ON_SIM"] == "1"
        assert "RUN_ON_SIM" not in os.environ

        os.environ["RUN_ON_SIM"] = "0"
        with harness._simulator(True):
            assert os.environ["RUN_ON_SIM"] == "1"
        assert os.environ["RUN_ON_SIM"] == "0"

        with harness._simulator(False):
            assert os.environ["RUN_ON_SIM"] == "0"
    finally:
        if previous is None:
            os.environ.pop("RUN_ON_SIM", None)
        else:
            os.environ["RUN_ON_SIM"] = previous


@pytest.mark.cpu
def test_host_only_run_writes_the_documented_report(harness, tmp_path):
    output = tmp_path / "report.json"
    harness.main(
        [
            "--model",
            "mlp",
            "--path",
            "eager",
            "hexagon",
            "--dtype",
            "float32",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--output",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["workload"]["model"] == "mlp"
    assert report["workload"]["target"] == "qualcomm"
    assert report["workload"]["atol"] == pytest.approx(1e-4)

    by_path = {result["path"]: result for result in report["results"]}
    assert set(by_path) == {"eager", "hexagon"}

    eager = by_path["eager"]
    assert eager["available"] is True
    assert eager["max_abs_diff_vs_eager"] == pytest.approx(0.0)
    assert eager["within_tolerance"] is True
    assert eager["latency_median_ms"] > 0
    assert eager["samples_per_second"] > 0

    # Without the Qualcomm toolchain the NPU path must be recorded as skipped
    # rather than silently dropped or crashing the run.
    hexagon = by_path["hexagon"]
    if hexagon["available"]:
        pytest.skip("hexagon-mlir is installed; this host can run the NPU path")
    assert hexagon["reason"]
