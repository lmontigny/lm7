from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
import zipfile

import pytest
import torch

import lm7
from lm7.backends.stablehlo import PROGRAM_ENTRY, PROGRAM_META_ENTRY, StableHLOBackend
from lm7.errors import CompilationError
from lm7.exporting import COMPILED_STABLEHLO_NAME

pytestmark = [
    pytest.mark.stablehlo,
    pytest.mark.skipif(
        importlib.util.find_spec("torch_xla") is None,
        reason="PyTorch/XLA is not installed",
    ),
]


def model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    ).eval()


def _executing_device_type() -> str:
    """Report the device torch_xla will actually run this artifact on.

    Loading a StableHLO payload back into a torch callable goes through
    torch_xla, and torch_xla runs on whatever PJRT device the host exposes. On a
    TPU VM that is the TPU, even though these artifacts are exported for
    ``target="cpu"`` -- the payload is target-independent, so the *execution*
    device is a property of the host rather than of the artifact.
    """
    try:
        import torch_xla.runtime as xr

        return str(xr.device_type() or "CPU")
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError):
        return "CPU"


def _eager_tolerance() -> dict[str, float]:
    """Tolerance for comparing an executed payload against CPU eager.

    TPU lowers fp32 matmuls to bf16 passes by default, which moves this MLP by
    ~2.5e-3 -- four orders of magnitude more than the CPU plugin's 1e-6. That is
    the device's default precision rather than a defect in the artifact; see
    docs/google-tpu.md.
    """
    if _executing_device_type() == "TPU":
        return {"rtol": 0.2, "atol": 5e-3}
    return {}


def test_stablehlo_artifact_matches_eager(tmp_path):
    source = model()
    example = torch.randn(8, 16)
    expected = source(example)

    artifact = lm7.export(
        source, args=(example,), target="cpu", backend="stablehlo", output=tmp_path / "model.lm7"
    )

    assert artifact.manifest.backend == "stablehlo"
    assert artifact.manifest.compiled_file == COMPILED_STABLEHLO_NAME
    tolerance = _eager_tolerance()
    torch.testing.assert_close(artifact(example).cpu(), expected, **tolerance)

    reloaded = lm7.load_artifact(artifact.path)
    torch.testing.assert_close(reloaded(example).cpu(), expected, **tolerance)


def test_payload_carries_what_a_pjrt_client_needs(tmp_path):
    """The point of the backend: the payload is loadable without this framework."""
    artifact = lm7.export(
        model(),
        args=(torch.randn(8, 16),),
        target="cpu",
        backend="stablehlo",
        output=tmp_path / "model.lm7",
    )
    entries = StableHLOBackend().program_entries(artifact.path / COMPILED_STABLEHLO_NAME)

    assert PROGRAM_ENTRY in entries
    assert PROGRAM_META_ENTRY in entries
    # Weights ship as individual .npy files rather than inside the program, which
    # is what lets a non-PyTorch loader rebuild the call without a model class.
    assert any(name.startswith("data/") for name in entries)


def test_keyword_capture_is_rejected_with_an_actionable_message(tmp_path):
    """torch_xla cannot lower a kwargs program; the message should say so."""

    class Named(torch.nn.Module):
        def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            return first + second

    with pytest.raises(CompilationError, match="positional args"):
        lm7.export(
            Named().eval(),
            kwargs={"first": torch.randn(4), "second": torch.randn(4)},
            args=(),
            target="cpu",
            backend="stablehlo",
            output=tmp_path / "model.lm7",
        )


def test_payload_runs_without_pytorch(tmp_path):
    """Unpack the payload and execute it through a PJRT client in a torch-free process.

    Skipped unless a PyTorch-free interpreter with a PJRT client is configured
    via LM7_PJRT_PYTHON; see docs/stablehlo-pjrt-evaluation.md for the setup.
    """
    import os

    interpreter = os.environ.get("LM7_PJRT_PYTHON")
    if not interpreter:
        pytest.skip("set LM7_PJRT_PYTHON to a torch-free interpreter with a PJRT client")

    source = model()
    example = torch.randn(8, 16)
    expected = source(example)
    artifact = lm7.export(
        source, args=(example,), target="cpu", backend="stablehlo", output=tmp_path / "model.lm7"
    )

    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(artifact.path / COMPILED_STABLEHLO_NAME) as archive:
        archive.extractall(unpacked)
    (unpacked / "constants").mkdir(exist_ok=True)
    numpy = pytest.importorskip("numpy")
    numpy.save(tmp_path / "input.npy", example.numpy())
    numpy.save(tmp_path / "expected.npy", expected.detach().numpy())

    script = textwrap.dedent(
        """
        import json, sys
        import numpy as np
        assert "torch" not in sys.modules
        import importlib.util
        assert importlib.util.find_spec("torch") is None, "PyTorch is installed"
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.extend.backend
        import jaxlib._jax as jx
        from jax._src.lib import xla_client

        root, input_path, expected_path = sys.argv[1:4]
        meta = json.load(open(f"{root}/functions/forward.meta"))
        backend = jax.extend.backend.get_backend()
        devices = xla_client.DeviceList(tuple(backend.devices()[:1]))
        program = jx.ifrt_programs.make_hlo_program(
            open(f"{root}/functions/forward.bytecode", "rb").read()
        )
        options = jx.ifrt_programs.make_xla_compile_options(
            xla_client.CompileOptions(), devices, []
        )
        executable = backend.compile_and_load_ifrt_program(program, options)

        runtime = np.load(input_path)
        inputs = []
        for loc, sig in zip(meta["input_locations"], meta["input_signature"]):
            if loc["type_"] == "parameter":
                value = np.load(f"{root}/data/{loc['name']}")
            elif loc["type_"] == "constant":
                value = np.load(f"{root}/constants/{loc['position']}")
            else:
                value = runtime
            inputs.append(np.asarray(value, dtype=sig["dtype"]))

        result = np.asarray(executable.execute([jax.device_put(v) for v in inputs])[0])
        expected = np.load(expected_path)
        assert result.shape == expected.shape, (result.shape, expected.shape)
        # The plugin is the loader's choice, so this may run on a different device
        # than the eager reference. Cross-device fp32 reassociation moves the last
        # few digits; the tolerance reflects that rather than bitwise agreement.
        # TPU is looser on purpose: it lowers fp32 matmuls to bf16 passes by
        # default, which is a device precision choice, not a broken payload.
        limit = 5e-3 if backend.platform == "tpu" else 1e-3
        difference = float(np.abs(result - expected).max())
        assert difference < limit, (backend.platform, difference)
        print(f"OK platform={backend.platform} max_abs_diff={difference:.3e}")
        """
    )
    environment = dict(os.environ)
    # A GPU plugin otherwise grabs most of the card just to run one small graph.
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    completed = subprocess.run(
        [
            interpreter,
            "-c",
            script,
            str(unpacked),
            f"{tmp_path}/input.npy",
            f"{tmp_path}/expected.npy",
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "OK" in completed.stdout
    assert sys.executable != interpreter
