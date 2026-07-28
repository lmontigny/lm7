"""Side-by-side evaluation of TorchInductor and OpenVINO on a CPU host.

This is the first implementation slice of the OpenVINO evaluation plan in
``docs/openvino-evaluation.md``. It does not add an LM7 backend: it runs one
model through eager (the correctness reference), TorchInductor, the OpenVINO
``torch.compile`` backend, and an OpenVINO IR artifact under a single
measurement harness, so conversion cost, steady-state latency, and accuracy
against eager are directly comparable.

    python benchmarks/openvino_eval.py \
      --path eager inductor openvino openvino_ir \
      --output artifacts/benchmarks/openvino-mlp-fp32-b8.json

Paths whose runtime is missing are reported as unavailable and skipped rather
than failing the run, so an eager+Inductor baseline still works before OpenVINO
is installed.

The module is named ``openvino_eval`` and not ``openvino`` on purpose: running
``python benchmarks/openvino.py`` would put ``benchmarks/`` first on
``sys.path`` and shadow the real ``openvino`` package with this script.

Four OpenVINO behaviours make a naive comparison misleading, so the harness
works around each of them explicitly:

* The CPU plugin's ``INFERENCE_PRECISION_HINT`` does not default to FP32. It is
  FP16 on ARM hosts and BF16 on x86 hosts with AMX, so OpenVINO runs an FP32
  model in reduced precision and looks both faster and less accurate than eager
  for the same nominal dtype. ``--inference-precision`` defaults to ``f32`` so
  the comparison is like-for-like, and the effective value is recorded per path.
* The ``openvino`` dynamo backend catches every compilation exception and
  silently returns ``torch._inductor.compile_fx``, and each partition
  independently falls back to eager PyTorch on a runtime exception. A path
  labelled ``openvino`` can therefore measure Inductor or eager. Every OpenVINO
  ``torch.compile`` result records ``ov_partitions``, ``ov_compiled_models``,
  and ``ov_runtime_fallbacks`` read from the backend's own caches so a silent
  fallback shows up in the JSON instead of being reported as an OpenVINO win.
* ``openvino.save_model`` compresses floating point weights to FP16 by
  default, which shows up as FP16-level error on an otherwise FP32 comparison.
  This script saves full-precision IR unless ``--compress-to-fp16`` is passed,
  and records the choice.
* Importing ``openvino.torch`` assigns ``torch._dynamo.config`` .
  ``inline_inbuilt_nn_modules = False`` process-wide. That is a deprecated
  no-op from PyTorch 2.13 on, but on older versions it also changes how
  Inductor compiles, so ``eager`` and ``inductor`` are always measured before
  the first OpenVINO import.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch

# OpenVINO's PyTorch decoder maps float32/float16 but not bfloat16, and its
# execute path round-trips tensors through NumPy, which has no bfloat16 dtype.
# Offering bfloat16 here would only ever measure a silent fallback.
_DTYPES = {"float32": torch.float32, "float16": torch.float16}

# Inductor and OpenVINO reassociate FP32 arithmetic differently; the FP16 atol
# matches the looser policy used for validated low-precision paths elsewhere in
# LM7.
_DEFAULT_ATOL = {"float32": 1e-4, "float16": 2e-2}

_ALL_PATHS = ("eager", "inductor", "openvino", "openvino_ir")


def _openvino_available() -> bool:
    # find_spec avoids importing openvino.torch as a side effect of probing:
    # the import mutates torch._dynamo config for the whole process.
    return importlib.util.find_spec("openvino") is not None


def _mlp(batch_size: int, dtype: torch.dtype) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096),
        torch.nn.GELU(),
        torch.nn.Linear(4096, 1024),
    ).eval()
    inputs = (torch.randn(batch_size, 1024, dtype=dtype),)
    return model.to(dtype=dtype), inputs


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


def _ov_config(arguments: argparse.Namespace) -> dict[str, str]:
    """Build the OpenVINO plugin config shared by the torch.compile and IR paths."""
    if arguments.inference_precision == "default":
        return {}
    return {"INFERENCE_PRECISION_HINT": arguments.inference_precision}


def _ov_execution_stats() -> dict[str, int]:
    """Read OpenVINO's dynamo caches to prove OpenVINO actually ran.

    ``compiled_cache`` holds one OpenVINO ``CompiledModel`` per executed
    partition and ``partitioned_modules`` holds the rewritten graphs, so zero
    compiled models after a call means the dynamo backend silently fell back to
    Inductor. ``perm_fallback`` is latched by ``OpenVINOGraphModule`` when a
    partition raises at runtime and permanently reverts to eager PyTorch.
    """
    from openvino.frontend.pytorch.torchdynamo import execute as ov_execute
    from openvino.frontend.pytorch.torchdynamo.execute import OpenVINOGraphModule

    fallbacks = 0
    for graph in ov_execute.partitioned_modules.values():
        for submodule in graph.modules():
            if isinstance(submodule, OpenVINOGraphModule) and submodule.perm_fallback:
                fallbacks += 1
    return {
        "ov_partitions": len(ov_execute.partitioned_modules),
        "ov_compiled_models": len(ov_execute.compiled_cache),
        "ov_runtime_fallbacks": fallbacks,
    }


def _reset_ov_caches() -> None:
    from openvino.frontend.pytorch.torchdynamo import execute as ov_execute

    ov_execute.clear_caches()


def _ov_environment(device: str) -> dict[str, Any]:
    import openvino as ov

    core = ov.Core()
    available = list(core.available_devices)
    properties: dict[str, Any] = {}
    if device in available:
        for name in ("FULL_DEVICE_NAME", "INFERENCE_PRECISION_HINT", "OPTIMIZATION_CAPABILITIES"):
            try:
                properties[name.lower()] = str(core.get_property(device, name))
            except Exception:  # noqa: BLE001 - properties are optional per plugin
                properties[name.lower()] = None
    return {
        "openvino_version": ov.get_version(),
        "available_devices": available,
        "device": device,
        # The plugin default recorded here is what an unconfigured OpenVINO run
        # would have used; --inference-precision is what this run requested.
        "device_defaults": properties,
    }


def _build_openvino_ir(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    device: str,
    ir_directory: Path,
    compress_to_fp16: bool,
    config: dict[str, str],
) -> tuple[Any, dict[str, Any]]:
    """Export to an OpenVINO IR file, then load it back through the runtime.

    The IR is saved and re-read from disk rather than compiled in memory so the
    measurement reflects the deployment path the plan cares about: an artifact
    that a fresh process can load.
    """
    import openvino as ov

    started = time.perf_counter()
    exported = torch.export.export(model, inputs)
    export_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    ov_model = ov.convert_model(exported, example_input=list(inputs))
    convert_ms = (time.perf_counter() - started) * 1000

    ir_directory.mkdir(parents=True, exist_ok=True)
    xml_path = ir_directory / "model.xml"
    started = time.perf_counter()
    ov.save_model(ov_model, xml_path, compress_to_fp16=compress_to_fp16)
    save_ms = (time.perf_counter() - started) * 1000

    core = ov.Core()
    started = time.perf_counter()
    compiled = core.compile_model(core.read_model(xml_path), device, config)
    compile_ms = (time.perf_counter() - started) * 1000
    request = compiled.create_infer_request()

    output_dtype = inputs[0].dtype

    def _call(*args: torch.Tensor) -> torch.Tensor:
        # share_outputs is left at its default: the returned buffer would
        # otherwise alias OpenVINO's output tensor and be overwritten by the
        # next inference, which silently corrupts the accuracy comparison.
        result = request.infer([a.detach().numpy() for a in args], share_inputs=True)
        return torch.from_numpy(result[compiled.outputs[0]]).to(output_dtype)

    metadata = {
        "ir_export_ms": export_ms,
        "ir_convert_ms": convert_ms,
        "ir_save_ms": save_ms,
        "ir_compile_ms": compile_ms,
        "ir_compress_to_fp16": compress_to_fp16,
        "ov_config": dict(config),
        "ir_bytes": sum(f.stat().st_size for f in (xml_path, xml_path.with_suffix(".bin"))),
        "ir_path": str(xml_path),
    }
    return _call, metadata


def _build(
    path: str,
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    arguments: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    if path == "eager":
        return model, {}
    if path == "inductor":
        return torch.compile(model, backend="inductor"), {}
    if path == "openvino":
        import openvino.torch  # noqa: F401  (registers the "openvino" dynamo backend)

        _reset_ov_caches()
        options = {"device": arguments.device, "config": _ov_config(arguments)}
        return torch.compile(model, backend="openvino", options=options), {
            "ov_config": dict(options["config"]),
        }
    if path == "openvino_ir":
        directory = arguments.ir_dir or Path(".lm7-openvino-ir")
        return _build_openvino_ir(
            model,
            inputs,
            arguments.device,
            directory,
            arguments.compress_to_fp16,
            _ov_config(arguments),
        )
    raise ValueError(f"Unknown path {path!r}")


def _measure(
    fn: Any,
    args: tuple[torch.Tensor, ...],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        started = time.perf_counter()
        # clone() so the reference survives later calls even if a runtime hands
        # back a buffer it reuses.
        first_output = fn(*args)
        first_call_ms = (time.perf_counter() - started) * 1000
        first_output = first_output.clone()

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
        "output": first_output,
        "first_call_ms": first_call_ms,
        "latency_median_ms": median_ms,
        "latency_p95_ms": _percentile(latencies_ms, 0.95),
        "samples_per_second": batch_size * 1000 / median_ms if median_ms else float("inf"),
    }


def _order_paths(requested: list[str]) -> list[str]:
    """Keep eager first and non-OpenVINO paths before any OpenVINO import.

    Importing ``openvino.torch`` disables dynamo's ``inline_inbuilt_nn_modules``
    for the rest of the process, so an Inductor measurement taken after that
    import is not the same Inductor measurement taken before it.
    """
    seen = dict.fromkeys(requested)
    native = [p for p in seen if p in ("eager", "inductor")]
    openvino_paths = [p for p in seen if p not in ("eager", "inductor")]
    return ["eager", *[p for p in native if p != "eager"], *openvino_paths]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare eager, TorchInductor, and OpenVINO on a CPU host.",
    )
    parser.add_argument(
        "--path",
        nargs="+",
        choices=_ALL_PATHS,
        default=list(_ALL_PATHS),
        help="Execution paths to evaluate; 'eager' is always the correctness reference.",
    )
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--device",
        default="CPU",
        help="OpenVINO device name, for example CPU, GPU, or NPU.",
    )
    parser.add_argument(
        "--inference-precision",
        choices=("f32", "f16", "bf16", "default"),
        default="f32",
        help=(
            "OpenVINO INFERENCE_PRECISION_HINT. The plugin default is not FP32 "
            "(FP16 on ARM, BF16 on x86 with AMX); 'default' leaves it unset."
        ),
    )
    parser.add_argument(
        "--compress-to-fp16",
        action="store_true",
        help="Save IR with OpenVINO's default FP16 weight compression instead of full precision.",
    )
    parser.add_argument("--ir-dir", type=Path, help="Directory for the generated OpenVINO IR.")
    parser.add_argument(
        "--atol",
        type=float,
        help="Max absolute difference from eager allowed (default depends on dtype).",
    )
    parser.add_argument("--output", type=Path, help="Write machine-readable results as JSON.")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if a requested path is unavailable instead of skipping it.",
    )
    arguments = parser.parse_args()

    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if arguments.repeats < 1:
        parser.error("--repeats must be at least 1")

    dtype = _DTYPES[arguments.dtype]
    atol = arguments.atol if arguments.atol is not None else _DEFAULT_ATOL[arguments.dtype]

    # Build the model and inputs once so every path runs identical weights and
    # inputs; only then is the accuracy comparison against eager meaningful.
    torch.manual_seed(0)
    base_model, inputs = _mlp(arguments.batch_size, dtype)
    with torch.inference_mode():
        reference_output = base_model(*inputs).clone()

    paths = _order_paths(arguments.path)
    openvino_ready = _openvino_available()
    results: list[dict[str, Any]] = []
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }

    for path in paths:
        needs_openvino = path in ("openvino", "openvino_ir")
        if needs_openvino and not openvino_ready:
            message = "install openvino in this environment"
            if arguments.require_all:
                raise SystemExit(f"Requested path {path!r} is unavailable: {message}")
            print(f"{path:>12}  unavailable: {message}")
            results.append({"path": path, "available": False, "reason": message})
            continue
        if needs_openvino and "openvino" not in environment:
            environment["openvino"] = _ov_environment(arguments.device)

        model = copy.deepcopy(base_model)
        try:
            callable_under_test, metadata = _build(path, model, inputs, arguments)
            measured = _measure(
                callable_under_test,
                inputs,
                arguments.warmup,
                arguments.repeats,
            )
        except Exception as error:  # one broken path must not hide the rest
            if arguments.require_all:
                raise
            reason = f"{type(error).__name__}: {error}"
            print(f"{path:>12}  failed: {reason}")
            results.append({"path": path, "available": False, "reason": reason})
            continue

        output = measured.pop("output")
        max_abs_diff = (reference_output - output.to(reference_output.dtype)).abs().max().item()
        measured.update(metadata)
        measured.update(
            {
                "path": path,
                "available": True,
                "max_abs_diff_vs_eager": max_abs_diff,
                "within_tolerance": max_abs_diff <= atol,
            }
        )
        if path == "openvino":
            measured.update(_ov_execution_stats())

        results.append(measured)

        accuracy = "reference" if path == "eager" else f"maxdiff={max_abs_diff:.3e}"
        note = ""
        if path == "openvino":
            if measured["ov_compiled_models"] == 0:
                note = "  WARNING: no OpenVINO partitions ran (silent fallback)"
            elif measured["ov_runtime_fallbacks"]:
                note = f"  WARNING: {measured['ov_runtime_fallbacks']} partition(s) fell back"
        print(
            f"{path:>12}  first={measured['first_call_ms']:9.2f} ms  "
            f"median={measured['latency_median_ms']:8.3f} ms  "
            f"p95={measured['latency_p95_ms']:8.3f} ms  "
            f"throughput={measured['samples_per_second']:10.2f} samples/s  "
            f"{accuracy}{note}"
        )
        del model

    report = {
        "schema_version": 1,
        "environment": environment,
        "workload": {
            "model": "mlp",
            "dtype": arguments.dtype,
            "batch_size": arguments.batch_size,
            "target": "cpu",
            "atol": atol,
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
            "inference_precision": arguments.inference_precision,
            "path_order": paths,
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
