"""Side-by-side evaluation of host CPU paths and Qualcomm Hexagon NPU execution.

This is the first implementation slice of the Hexagon evaluation plan in
``docs/qualcomm-hexagon.md``. It does not add an LM7 backend: it runs one model
through eager CPU (the correctness reference), TorchInductor on CPU (a host
baseline), and Qualcomm's torch-mlir to Hexagon path, so compile cost and
accuracy against eager are directly comparable.

Unlike the other benchmarks here, the compiled path does not run on this host.
``TorchMLIRHexagonLauncher`` lowers Linalg IR to a ``.so`` and executes it on an
adb-reachable Hexagon NPU (``ANDROID_HOST``/``ANDROID_SERIAL``) or on the
Hexagon simulator (``RUN_ON_SIM=1``), which the ``hexagon-sim`` path selects.

Run the host baseline anywhere:

    python benchmarks/hexagon.py --model gpt2 --path eager inductor

Run the full comparison inside a built hexagon-mlir environment:

    python benchmarks/hexagon.py --model gpt2 --path eager inductor hexagon \
      --dtype float16 --iterations 10 \
      --output artifacts/benchmarks/hexagon-gpt2-fp16-v75.json

Paths whose runtime is missing are reported as unavailable and skipped rather
than failing the run, so the CPU baseline still works before the Qualcomm
toolchain is built.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

# torch.testing.assert_close defaults are tight for float32 but unrealistic for
# compiled low-precision inference. The float16 value matches the atol=0.03 that
# Qualcomm's own GPT-2 tutorial uses to compare Hexagon against x86.
_DEFAULT_ATOL = {"float32": 1e-4, "float16": 3e-2, "bfloat16": 5e-2}

_HOST_PATHS = ("eager", "inductor")
_HEXAGON_PATHS = ("hexagon", "hexagon-sim")
_ALL_PATHS = _HOST_PATHS + _HEXAGON_PATHS

# run_torch_mlir() lowers, links, deploys, and executes in one call, so timing a
# loop of calls measures compilation rather than inference. Reporting a median
# and a throughput here would describe the compiler, not the NPU.
_LATENCY_UNAVAILABLE = (
    "TorchMLIRHexagonLauncher.run_torch_mlir() recompiles and redeploys on every call, so "
    "wall-clock repeats measure compilation, not inference. Use --iterations for on-device "
    "repetition; true steady-state latency needs the launcher's profiler output or a "
    "compile_torch_mlir()/execute_kernel() split. See docs/qualcomm-hexagon.md."
)


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _hexagon_backend_available() -> bool:
    # This module only exists inside a hexagon-mlir build of Triton, and importing
    # it is the cheapest honest check that the Qualcomm toolchain is on PYTHONPATH.
    # It reads the build's environment at import time, so a half-configured
    # toolchain fails with more than just ImportError.
    try:
        importlib.import_module("triton.backends.qcom_hexagon_backend.compiler")
    except (ImportError, AttributeError, KeyError, OSError, RuntimeError, ValueError):
        return False
    return True


def _transformers_available() -> bool:
    return _module_available("transformers")


def _unavailable_reason(path: str, model: str) -> str | None:
    """Return why ``path`` cannot run here, or None if it can."""
    if model == "gpt2" and not _transformers_available():
        return 'transformers is not installed; install LM7 with ".[hf]"'
    if path in _HOST_PATHS:
        return None
    if not _module_available("torch_mlir"):
        return "torch-mlir is not importable; build hexagon-mlir (see docs/qualcomm-hexagon.md)"
    if not _hexagon_backend_available():
        return (
            "triton.backends.qcom_hexagon_backend is not importable; source the hexagon-mlir "
            "environment (scripts/set_local_env.sh) so the patched Triton is on PYTHONPATH"
        )
    if not os.environ.get("HEXAGON_ARCH_VERSION"):
        return "HEXAGON_ARCH_VERSION is unset; export 73, 75, 79, or 81 for your device"
    if path == "hexagon" and not (
        os.environ.get("ANDROID_HOST") and os.environ.get("ANDROID_SERIAL")
    ):
        return (
            "ANDROID_HOST and ANDROID_SERIAL are unset; attach a Hexagon device or use "
            "--path hexagon-sim for a device-free correctness check"
        )
    return None


def _mlp(
    batch_size: int, dtype: torch.dtype, layers: int
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...], str]:
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096),
        torch.nn.GELU(),
        torch.nn.Linear(4096, 1024),
    ).eval()
    # Hexagon inputs stay on the CPU; the launcher moves them to the device.
    inputs = (torch.randn(batch_size, 1024, dtype=dtype),)
    return model.to(dtype=dtype), inputs, "MLP"


def _gpt2(
    batch_size: int, dtype: torch.dtype, layers: int
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...], str]:
    """GPT-2 trimmed to ``layers`` blocks, mirroring Qualcomm's torch-mlir tutorial."""
    transformers = importlib.import_module("transformers")
    model_name = "openai-community/gpt2"
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_name)
    config = transformers.GPT2Config.from_pretrained(model_name)
    # One layer is not enough to make an accuracy comparison meaningful.
    config.n_layer = layers
    model = transformers.GPT2LMHeadModel.from_pretrained(
        model_name, config=config, torch_dtype=dtype
    ).eval()
    encoding = tokenizer("What is nature of our existence?", return_tensors="pt")
    input_ids = encoding["input_ids"].expand(batch_size, -1).contiguous()
    return model, (input_ids,), model.__class__.__name__


_MODELS = {"mlp": _mlp, "gpt2": _gpt2}


def _logits(output: Any) -> torch.Tensor:
    """Normalize eager, inductor, and launcher outputs to one comparable tensor."""
    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, (list, tuple)) and output:
        return _logits(output[0])
    raise TypeError(f"Could not find a comparable tensor in output of type {type(output)!r}")


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _measure_host(
    path: str,
    model: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    fn = model if path == "eager" else torch.compile(model)
    with torch.inference_mode():
        started = time.perf_counter()
        reference = fn(*args)
        first_call_ms = (time.perf_counter() - started) * 1000

        for _ in range(warmup):
            fn(*args)

        latencies_ms: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            fn(*args)
            latencies_ms.append((time.perf_counter() - started) * 1000)

    median_ms = statistics.median(latencies_ms)
    batch_size = args[0].shape[0] if args and args[0].ndim else 1
    return {
        "output": reference,
        "first_call_ms": first_call_ms,
        "compile_and_run_ms": None,
        "latency_median_ms": median_ms,
        "latency_p95_ms": _percentile(latencies_ms, 0.95),
        "samples_per_second": batch_size * 1000 / median_ms if median_ms else float("inf"),
    }


@contextmanager
def _simulator(enabled: bool) -> Any:
    """Set RUN_ON_SIM for the launcher, restoring whatever the caller had."""
    if not enabled:
        yield
        return
    previous = os.environ.get("RUN_ON_SIM")
    os.environ["RUN_ON_SIM"] = "1"
    try:
        yield
    finally:
        if previous is None:
            del os.environ["RUN_ON_SIM"]
        else:
            os.environ["RUN_ON_SIM"] = previous


def _measure_hexagon(
    path: str,
    model: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    func_name: str,
    iterations: int,
    overrides: dict[str, str],
    artifacts_dir: Path,
) -> dict[str, Any]:
    fx = importlib.import_module("torch_mlir.fx")
    compiler_utils = importlib.import_module("torch_mlir.compiler_utils")
    hexagon_compiler = importlib.import_module("triton.backends.qcom_hexagon_backend.compiler")
    launcher_module = importlib.import_module(
        "triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher"
    )

    started = time.perf_counter()
    module = fx.export_and_import(
        model,
        *args,
        output_type=compiler_utils.OutputType.LINALG_ON_TENSORS,
        func_name=func_name,
    )
    export_ms = (time.perf_counter() - started) * 1000

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    bytecode_path = artifacts_dir / f"{func_name}.mlirbc"
    bytecode_path.write_bytes(module.operation.get_asm(binary=True))

    options = dict(hexagon_compiler.HexagonOptions().__dict__)
    # Both are disabled in Qualcomm's GPT-2 tutorial; --option overrides them.
    options["enableVTCMTiling"] = False
    options["enableConvertToHexagonmem"] = False
    options.update(overrides)

    started = time.perf_counter()
    with _simulator(path == "hexagon-sim"):
        outputs = launcher_module.TorchMLIRHexagonLauncher().run_torch_mlir(
            str(bytecode_path),
            list(args),
            func_name,
            base_dir_for_artifacts=str(artifacts_dir),
            iterations=iterations,
            options=options,
        )
    compile_and_run_ms = (time.perf_counter() - started) * 1000

    return {
        "output": outputs,
        "first_call_ms": None,
        "export_to_linalg_ms": export_ms,
        "compile_and_run_ms": compile_and_run_ms,
        "iterations": iterations,
        "latency_median_ms": None,
        "latency_p95_ms": None,
        "samples_per_second": None,
        "latency_unavailable_reason": _LATENCY_UNAVAILABLE,
        "simulated": path == "hexagon-sim",
    }


def _parse_option(value: str) -> tuple[str, str]:
    key, separator, option_value = value.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got {value!r}")
    return key, option_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare eager CPU, TorchInductor, and the Qualcomm Hexagon NPU path."
    )
    parser.add_argument(
        "--path",
        nargs="+",
        choices=_ALL_PATHS,
        default=["eager", "inductor", "hexagon"],
        help="Execution paths to evaluate; 'eager' is always the correctness reference.",
    )
    parser.add_argument("--model", choices=tuple(_MODELS), default="mlp")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Transformer blocks to keep for --model gpt2; ignored otherwise.",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Host-path warmup calls.")
    parser.add_argument("--repeats", type=int, default=30, help="Host-path timed calls.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="On-device repetition count passed to the Hexagon launcher.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        help="Max absolute difference from eager allowed (default depends on dtype).",
    )
    parser.add_argument(
        "--option",
        action="append",
        type=_parse_option,
        default=[],
        metavar="KEY=VALUE",
        help="Override a HexagonOptions field, repeatable.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/benchmarks/hexagon"),
        help="Where to write the .mlirbc and launcher artifacts.",
    )
    parser.add_argument("--output", type=Path, help="Write machine-readable results as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if arguments.layers < 1:
        parser.error("--layers must be at least 1")
    if arguments.iterations < 1:
        parser.error("--iterations must be at least 1")

    dtype = _dtype(arguments.dtype)
    atol = arguments.atol if arguments.atol is not None else _DEFAULT_ATOL[arguments.dtype]
    overrides = dict(arguments.option)

    reason = _unavailable_reason("eager", arguments.model)
    if reason is not None:
        raise SystemExit(f"Cannot build the {arguments.model} reference: {reason}")

    # Build the model and inputs once so every path runs identical weights and
    # inputs; only then is the accuracy comparison against eager meaningful.
    torch.manual_seed(0)
    base_model, inputs, func_name = _MODELS[arguments.model](
        arguments.batch_size, dtype, arguments.layers
    )
    with torch.inference_mode():
        reference_output = _logits(base_model(*inputs))

    paths = ["eager", *[p for p in arguments.path if p != "eager"]]
    results: list[dict[str, Any]] = []
    for path in paths:
        reason = _unavailable_reason(path, arguments.model)
        if reason is not None:
            print(f"{path:>12}  unavailable: {reason}")
            results.append({"path": path, "available": False, "reason": reason})
            continue

        model = copy.deepcopy(base_model)
        if path in _HOST_PATHS:
            measured = _measure_host(path, model, inputs, arguments.warmup, arguments.repeats)
        else:
            measured = _measure_hexagon(
                path,
                model,
                inputs,
                func_name,
                arguments.iterations,
                overrides,
                arguments.artifacts_dir,
            )

        output = _logits(measured.pop("output"))
        max_abs_diff = (reference_output - output.to(reference_output.dtype)).abs().max().item()
        measured.update(
            {
                "path": path,
                "available": True,
                "max_abs_diff_vs_eager": max_abs_diff,
                "within_tolerance": max_abs_diff <= atol,
            }
        )
        results.append(measured)

        accuracy = "reference" if path == "eager" else f"maxdiff={max_abs_diff:.3e}"
        if path in _HOST_PATHS:
            print(
                f"{path:>12}  first={measured['first_call_ms']:9.2f} ms  "
                f"median={measured['latency_median_ms']:8.3f} ms  "
                f"p95={measured['latency_p95_ms']:8.3f} ms  "
                f"throughput={measured['samples_per_second']:10.2f} samples/s  {accuracy}"
            )
        else:
            print(
                f"{path:>12}  export={measured['export_to_linalg_ms']:9.2f} ms  "
                f"compile+run={measured['compile_and_run_ms']:11.2f} ms  "
                f"iterations={measured['iterations']:<4d}  latency=n/a  {accuracy}"
            )
        del model

    report = {
        "schema_version": 1,
        "workload": {
            "model": arguments.model,
            "function": func_name,
            "dtype": arguments.dtype,
            "batch_size": arguments.batch_size,
            "layers": arguments.layers if arguments.model == "gpt2" else None,
            "target": "qualcomm",
            "atol": atol,
            "iterations": arguments.iterations,
            "hexagon_arch_version": os.environ.get("HEXAGON_ARCH_VERSION"),
            "hexagon_options_overrides": overrides,
        },
        "results": results,
    }
    if arguments.output is not None:
        out = arguments.output.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON: {out}")


if __name__ == "__main__":
    main()
