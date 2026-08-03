from __future__ import annotations

import importlib.util
import os

import pytest
import torch

import lm7
from lm7.errors import CompilationError
from lm7.huggingface import generate_hf_model


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


def _has_executed_xla() -> bool:
    """Whether an XLA computation has already run in this process.

    XLA reads its fp32 matmul precision while lowering the first computation and
    ignores every later change, so this is also the point after which a
    precision can no longer be chosen.
    """
    import torch_xla.debug.metrics as met

    return met.counter_value("ExecuteComputation") is not None


def _mlp() -> tuple[torch.nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()
    return model, torch.randn(8, 16)


# Deliberately the first test in this module.
#
# The other suites that need process isolation -- executorch, openvino,
# stablehlo -- get it with subprocess.run. That is not available here: a TPU chip
# is claimed by a single process, and merely probing the runtime is enough to
# claim it, so a pytest process that can see the TPU is also a pytest process
# whose children cannot. The precision is therefore one choice per pytest run,
# and only a test running before any XLA work can make it.
def test_mat_mul_precision_highest_matches_cpu_eager():
    if _has_executed_xla():
        pytest.skip(
            "an XLA computation already ran in this process, so the matmul precision is "
            "already fixed; run this module on its own to exercise the setting"
        )
    source, example_input = _mlp()
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

    # Three orders of magnitude tighter than the TPU default checked below.
    assert (actual - expected).abs().max().item() < 1e-5


def test_openxla_tpu_matches_cpu_eager():
    source, example_input = _mlp()
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
    # Deliberately loose, and loose in the direction the hardware actually
    # errs. XLA lowers fp32 matmuls to bf16 passes on TPU unless asked
    # otherwise, so this MLP lands ~2e-3 from CPU eager; the previous 2e-3
    # bound was a near miss that would have flaked on another seed. See
    # docs/google-tpu.md for the measured error at each setting.
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.2, atol=5e-3)


@pytest.mark.hf
@pytest.mark.skipif(os.environ.get("LM7_RUN_HF_TESTS") != "1", reason="set LM7_RUN_HF_TESTS=1")
def test_generation_runs_on_tpu():
    """Multi-token decode, which inference_mode() used to make impossible here.

    Generation ran under torch.inference_mode() and died partway through with
    "Cannot set version_counter for inference tensor" -- PyTorch/XLA needs the
    version counters inference mode disables.
    """
    result = generate_hf_model(
        "hf://HuggingFaceTB/SmolLM2-135M-Instruct",
        prompt="The capital of France is",
        max_new_tokens=8,
        target="tpu",
    )

    assert result.generated_tokens == 8
    assert result.generated_text.strip()
    # Transformers' compile criteria list "tpu", but a Cloud TPU is torch device
    # type "xla" and does not match, so the decode loop is eager here.
    assert result.backend == "eager"


def test_mat_mul_precision_is_refused_once_a_computation_has_run():
    import torch_xla

    # Establish the precondition rather than depending on test order.
    device = torch_xla.device()
    (torch.ones(2, 2).to(device) @ torch.ones(2, 2).to(device)).cpu()
    assert _has_executed_xla()

    compiled = lm7.compile(
        torch.nn.Linear(4, 4).eval(),
        target="tpu",
        backend="openxla",
        transfers="automatic",
        fallback="error",
        options={"mat_mul_precision": "highest"},
    )

    # torch_xla would accept the setting here and silently not apply it, so LM7
    # raises. Compilation is lazy, so the refusal surfaces on the first call.
    with pytest.raises(CompilationError, match="already run an XLA computation"):
        compiled(torch.randn(2, 4))
