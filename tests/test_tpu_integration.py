from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest
import torch

import lm7


def _tpu_available() -> bool:
    try:
        if importlib.util.find_spec("torch_xla") is None:
            return False
        import torch_xla.runtime as xr

        return xr.device_type() == "TPU" and xr.addressable_device_count() > 0
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError):
        return False


pytestmark = [
    pytest.mark.tpu,
    pytest.mark.skipif(not _tpu_available(), reason="Google TPU runtime is unavailable"),
]


def _run_in_fresh_process(body: str) -> str:
    """Run a snippet in its own interpreter and return its stdout.

    XLA's fp32 matmul precision is process-global and is read while lowering the
    first computation, so a test that picks a setting cannot share a process
    with one that already ran something. Each case gets its own.
    """
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_openxla_tpu_matches_cpu_eager():
    torch.manual_seed(0)
    source = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()
    example_input = torch.randn(8, 16)
    expected = source(example_input)

    compiled = lm7.compile(
        source,
        target="tpu",
        backend="openxla",
        transfers="automatic",
        fallback="error",
    )
    actual = compiled(example_input)

    assert compiled.selected_backend == "openxla"
    assert compiled.target is not None
    assert compiled.target.vendor == "tpu"
    assert actual.device.type == "xla"
    # Deliberately loose. XLA lowers fp32 matmuls to bf16 passes on TPU unless
    # asked otherwise, so this MLP lands ~2e-3 from CPU eager -- the earlier
    # 2e-3 bound here was a near miss that would have flaked on another seed.
    # The tight comparison lives in the mat_mul_precision test below; see
    # docs/google-tpu.md for the measured error at each setting.
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.2, atol=5e-3)


def test_mat_mul_precision_highest_matches_cpu_eager():
    output = _run_in_fresh_process(
        """
        import torch
        import lm7

        torch.manual_seed(0)
        source = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 4),
        ).eval()
        example_input = torch.randn(8, 16)
        expected = source(example_input)

        compiled = lm7.compile(
            source,
            target="tpu",
            backend="openxla",
            transfers="automatic",
            fallback="error",
            options={"mat_mul_precision": "highest"},
        )
        actual = compiled(example_input).cpu()
        print((actual - expected).abs().max().item())
        """
    )

    # Three orders of magnitude tighter than the default setting above.
    assert float(output.strip()) < 1e-5


def test_mat_mul_precision_is_refused_once_a_computation_has_run():
    output = _run_in_fresh_process(
        """
        import torch
        import torch_xla
        import lm7
        from lm7.errors import CompilationError

        # Force the setting to be locked in before LM7 is asked to change it.
        device = torch_xla.device()
        (torch.ones(2, 2).to(device) @ torch.ones(2, 2).to(device)).cpu()

        compiled = lm7.compile(
            torch.nn.Linear(4, 4).eval(),
            target="tpu",
            backend="openxla",
            transfers="automatic",
            fallback="error",
            options={"mat_mul_precision": "highest"},
        )
        try:
            # Compilation is lazy, so the refusal surfaces on the first call.
            compiled(torch.randn(2, 4))
        except CompilationError as exc:
            print("refused:", exc)
        else:
            print("accepted")
        """
    )

    assert "refused:" in output
    assert "already run an XLA computation" in output
