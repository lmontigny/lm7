from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from typing import Any

import torch

from .api import backends as inspect_backends
from .api import version
from .backends import registry
from .backends.base import CompileRequest
from .cache import cache_dir
from .detection import detect_targets, resolve_target
from .errors import LM7Error
from .huggingface import HuggingFaceRunResult, run_hf_model
from .planner import Plan, plan
from .targets import DeviceInfo, TargetSpec


def _target_data(device: DeviceInfo) -> dict[str, Any]:
    target = device.target
    return {
        "target": str(target),
        "vendor": target.vendor,
        "kind": target.kind,
        "architecture": target.architecture,
        "model": target.model,
        "ordinal": target.ordinal,
        "remote": target.remote,
        "name": device.name,
        "total_memory_bytes": device.total_memory_bytes,
        "capabilities": dict(device.capabilities),
    }


def _target_spec_data(target: TargetSpec) -> dict[str, Any]:
    return {
        "target": str(target),
        "vendor": target.vendor,
        "kind": target.kind,
        "architecture": target.architecture,
        "model": target.model,
        "ordinal": target.ordinal,
        "remote": target.remote,
    }


def _explain_plan(target: str, backend: str) -> tuple[TargetSpec, Plan]:
    resolved = resolve_target(target)
    request = CompileRequest(torch.nn.Identity(), resolved, "lazy", "automatic", "warn", {})
    _, selected_plan = plan(request, backend, registry)
    return resolved, selected_plan


def _explain_data(target: str, backend: str) -> dict[str, Any]:
    resolved, selected_plan = _explain_plan(target, backend)
    return {
        "requested_target": target,
        "requested_backend": backend,
        "resolved_target": _target_spec_data(resolved),
        "selected_backend": selected_plan.selected,
        "candidates": [
            {
                "backend": candidate.backend,
                "supported": candidate.support.supported,
                "priority": candidate.support.priority,
                "reason": candidate.support.reason,
            }
            for candidate in selected_plan.candidates
        ],
    }


def _doctor_data() -> dict[str, Any]:
    return {
        "status": "ok",
        "lm7_version": version(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "cache_dir": str(cache_dir()),
        "targets": [_target_data(device) for device in detect_targets()],
        "backends": list(inspect_backends()),
    }


def _format_memory(total_bytes: int | None) -> str:
    if total_bytes is None:
        return ""
    gibibytes = total_bytes / 1024**3
    return f", {gibibytes:.1f} GiB"


def _print_targets(devices: Sequence[DeviceInfo]) -> None:
    print(f"Detected targets ({len(devices)}):")
    for device in devices:
        print(f"  {device.target}: {device.name}{_format_memory(device.total_memory_bytes)}")


def _print_backends(backends: Sequence[dict[str, Any]]) -> None:
    print(f"Registered backends ({len(backends)}):")
    for backend in backends:
        status = "available" if backend["available"] else "unavailable"
        version_suffix = f", version {backend['version']}" if backend["version"] else ""
        print(f"  {backend['name']}: {status}{version_suffix}")
        if backend["reason"]:
            print(f"    {backend['reason']}")


def _print_explanation(data: dict[str, Any]) -> None:
    resolved = data["resolved_target"]["target"]
    print(f"Selected {data['selected_backend']} for {resolved}")
    print()
    print("Candidates:")
    for candidate in data["candidates"]:
        status = "supported" if candidate["supported"] else "unavailable"
        print(
            f"  {candidate['backend']}: {status} "
            f"(priority {candidate['priority']}) - {candidate['reason']}"
        )


def _print_doctor(data: dict[str, Any]) -> None:
    print(f"LM7 {data['lm7_version']} diagnostics")
    print(f"Python: {data['python_version']}")
    print(f"PyTorch: {data['pytorch_version']}")
    print(f"Platform: {data['platform']}")
    print(f"Cache: {data['cache_dir']}")
    print()
    _print_targets(
        [
            DeviceInfo(
                TargetSpec(
                    target["vendor"],
                    target["kind"],
                    target["architecture"],
                    target["model"],
                    target["ordinal"],
                    target["remote"],
                ),
                target["name"],
                target["total_memory_bytes"],
                target["capabilities"],
            )
            for target in data["targets"]
        ]
    )
    print()
    _print_backends(data["backends"])


def _print_model_run(result: HuggingFaceRunResult) -> None:
    print(f"Model: {result.model_uri}")
    print(f"Target: {result.target}")
    print(f"Backend: {result.backend}")
    print(f"Dtype: {result.dtype}")
    print(f"Quantization: {result.quantization}")
    print(f"Parameters: {result.parameter_count:,}")
    baseline_mib = result.baseline_model_storage_bytes / 1024**2
    storage_mib = result.model_storage_bytes / 1024**2
    if result.quantization == "none":
        print(f"Model storage: {storage_mib:.1f} MiB")
    else:
        reduction = 1 - result.model_storage_bytes / result.baseline_model_storage_bytes
        print(
            f"Model storage: {baseline_mib:.1f} -> {storage_mib:.1f} MiB "
            f"({reduction:.1%} reduction)"
        )
    print(f"Input tokens: {result.input_tokens}")
    if result.quantization_ms:
        print(f"Quantization time: {result.quantization_ms:.2f} ms")
    print(f"First call: {result.first_call_ms:.2f} ms")
    print(f"Steady call: {result.latency_ms:.2f} ms")
    if result.peak_memory_bytes is not None:
        print(f"Peak GPU memory: {result.peak_memory_bytes / 1024**2:.1f} MiB")
    print(f"Next token: {result.next_token_id} ({result.next_token!r})")


def _emit_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def _add_json_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lm7",
        description="Run models and inspect LM7 hardware targets and compiler backends.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="show environment, target, and backend diagnostics"
    )
    _add_json_argument(doctor_parser)

    targets_parser = subparsers.add_parser("targets", help="list detected hardware targets")
    _add_json_argument(targets_parser)

    backends_parser = subparsers.add_parser("backends", help="list registered compiler backends")
    _add_json_argument(backends_parser)

    explain_parser = subparsers.add_parser("explain", help="explain backend selection for a target")
    explain_parser.add_argument("--target", default="auto", help="target selector (default: auto)")
    explain_parser.add_argument(
        "--backend", default="auto", help="backend selector (default: auto)"
    )
    _add_json_argument(explain_parser)

    model_parser = subparsers.add_parser("model", help="run models through LM7")
    model_subparsers = model_parser.add_subparsers(dest="model_command", required=True)
    run_parser = model_subparsers.add_parser("run", help="compile and run a causal-LM forward pass")
    run_parser.add_argument("model_uri", help="model URI, for example hf://owner/model")
    run_parser.add_argument("--prompt", default="The capital of France is", help="input prompt")
    run_parser.add_argument("--target", default="auto", help="target selector (default: auto)")
    run_parser.add_argument("--backend", default="auto", help="backend selector (default: auto)")
    run_parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="model dtype (default: auto)",
    )
    run_parser.add_argument(
        "--quantization",
        choices=("none", "int8-weight-only", "fp8-weight-only"),
        default="none",
        help="experimental model quantization (default: none)",
    )
    _add_json_argument(run_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            data = _doctor_data()
            _emit_json(data) if args.json else _print_doctor(data)
        elif args.command == "targets":
            devices = detect_targets()
            data = {"targets": [_target_data(device) for device in devices]}
            _emit_json(data) if args.json else _print_targets(devices)
        elif args.command == "backends":
            backends = list(inspect_backends())
            data = {"backends": backends}
            _emit_json(data) if args.json else _print_backends(backends)
        elif args.command == "explain":
            data = _explain_data(args.target, args.backend)
            _emit_json(data) if args.json else _print_explanation(data)
        elif args.command == "model" and args.model_command == "run":
            result = run_hf_model(
                args.model_uri,
                prompt=args.prompt,
                target=args.target,
                backend=args.backend,
                dtype=args.dtype,
                quantization=args.quantization,
            )
            _emit_json(result.to_dict()) if args.json else _print_model_run(result)
    except LM7Error as exc:
        if args.json:
            _emit_json({"error": {"type": type(exc).__name__, "message": str(exc)}})
        else:
            print(f"lm7: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
