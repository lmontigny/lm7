from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .api import backends as inspect_backends
from .api import version
from .backends import registry
from .backends.base import CompileRequest
from .bundles import create_bundle, load_bundle
from .cache import cache_dir
from .compatibility import ModelCompatibilityResult, inspect_hf_model
from .detection import cuda_build_targets, detect_targets, resolve_target
from .errors import LM7Error
from .exporting import EXPORT_BACKENDS
from .hexagon import HexagonToolchainDiagnostics, diagnose_hexagon_toolchain
from .huggingface import (
    HuggingFaceExportResult,
    HuggingFaceGenerateResult,
    HuggingFaceRunResult,
    export_hf_model,
    generate_hf_model,
    run_hf_model,
)
from .inspection import ArtifactInspection, inspect_artifact
from .planner import Plan, plan
from .serve.engine import ServeConfig
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
        **(
            {"cuda_build": build_targets}
            if (build_targets := cuda_build_targets(target)) is not None
            else {}
        ),
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


def _target_spec_from_data(value: dict[str, Any]) -> TargetSpec:
    return TargetSpec(
        value["vendor"],
        value["kind"],
        value.get("architecture"),
        value.get("model"),
        value.get("ordinal"),
        value.get("remote", False),
    )


def _bundle_data(path: str) -> dict[str, Any]:
    bundle = load_bundle(path)
    return {
        "path": str(bundle.path),
        "model_graph_hash": bundle.manifest.model_graph_hash,
        "entries": [
            {
                "key": entry["key"],
                "target": _target_spec_data(_target_spec_from_data(entry["target"])),
                "backend": entry["backend"],
                "path": entry["path"],
            }
            for entry in bundle.manifest.entries
        ],
    }


def _format_memory(total_bytes: int | None) -> str:
    if total_bytes is None:
        return ""
    gibibytes = total_bytes / 1024**3
    return f", {gibibytes:.1f} GiB"


def _format_precision(capabilities: Mapping[str, Any]) -> str:
    """One line naming the formats the silicon computes, and the ones it fakes.

    Emulated formats are the reason this is printed at all: a target that runs
    BF16 by emulation looks identical to one that does not until something is
    inexplicably slow.
    """
    precision = capabilities.get("precision")
    if not isinstance(precision, Mapping) or not precision:
        return ""
    native = [name for name, support in precision.items() if support == "native"]
    emulated = [name for name, support in precision.items() if support == "emulated"]
    parts = [f"native {', '.join(native)}"] if native else []
    if emulated:
        parts.append(f"emulated {', '.join(emulated)}")
    return "; ".join(parts)


def _print_targets(devices: Sequence[DeviceInfo]) -> None:
    print(f"Detected targets ({len(devices)}):")
    for device in devices:
        generation = device.capabilities.get("generation")
        suffix = f" ({generation})" if generation else ""
        print(
            f"  {device.target}: {device.name}{suffix}{_format_memory(device.total_memory_bytes)}"
        )
        precision = _format_precision(device.capabilities)
        if precision:
            print(f"    precision: {precision}")


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


def _print_hexagon_doctor(result: HexagonToolchainDiagnostics, mode: str) -> None:
    print("Hexagon-MLIR toolchain diagnostics")
    print(f"Host:        {result.host['distribution']} / {result.host['machine']}")
    print(f"Python:      {result.host['python']}")
    print(f"PyTorch:     {result.host.get('pytorch', 'unknown')}")
    print(f"Compilation: {'ready' if result.compile_ready else 'not ready'}")
    print(f"Simulator:   {'ready' if result.simulator_ready else 'not ready'}")
    print(f"Device:      {'ready' if result.device_ready else 'not ready'}")
    print(f"Requested:   {mode} ({'ready' if result.ready_for(mode) else 'not ready'})")
    print()
    print("Checks:")
    for check in result.checks:
        value = f" = {check.value}" if check.value else ""
        print(f"  [{check.status}] {check.category}/{check.name}{value}")
        if check.detail:
            print(f"    {check.detail}")
        if check.status != "ok" and check.remediation:
            print(f"    Fix: {check.remediation}")


def _print_bundle(data: dict[str, Any]) -> None:
    print(f"Bundle: {data['path']}")
    print(f"Model graph: {data['model_graph_hash']}")
    print(f"Entries ({len(data['entries'])}):")
    for entry in data["entries"]:
        print(f"  {entry['key']}: {entry['target']['target']} / {entry['backend']}")


def _print_artifact_inspection(result: ArtifactInspection) -> None:
    requirements = result.runtime_requirements
    print(f"Backend:          {result.backend}")
    print(f"Target:           {result.target}")
    print(f"Payload:          {result.payload or 'none'}")
    print(f"Device-bound:     {'yes' if result.device_bound else 'no'}")
    if precision := requirements.get("precision"):
        print(f"Precision:        {precision}")
    if result.delegated_calls is not None and result.total_calls is not None:
        print(f"Delegation:       {result.delegated_calls} / {result.total_calls} calls")
    if sdk := requirements.get("qnn_sdk"):
        print(f"QNN SDK:          {sdk}")
    if soc := requirements.get("soc_model"):
        print(f"QNN SoC:          {soc}")
    if htp := requirements.get("htp_arch"):
        print(f"HTP architecture: {htp}")
    libraries = requirements.get("runtime_libraries")
    if isinstance(libraries, list):
        print(f"Runtime libraries: {', '.join(str(item) for item in libraries)}")
    print(f"Host executable:  {'yes' if result.host_executable else 'no'}")
    print(f"Deployment:       {result.deployment}")
    print(f"Checksums:        {result.checksum_status}")
    for warning in result.warnings:
        print(f"Warning:          {warning}")
    for error in result.errors:
        print(f"Error:            {error}")


def _print_model_run(result: HuggingFaceRunResult) -> None:
    print(f"Model: {result.model_uri}")
    print(f"Target: {result.target}")
    print(f"Backend: {result.backend}")
    print(f"Dtype: {result.dtype}")
    print(f"Quantization: {result.quantization}")
    print(f"Parameters: {result.parameter_count:,}")
    baseline_mib = result.baseline_model_storage_bytes / 1024**2
    storage_mib = result.model_storage_bytes / 1024**2
    compiled_bytes = result.compiled_weight_bytes
    if result.quantization == "none":
        print(f"Model storage: {storage_mib:.1f} MiB")
    elif compiled_bytes is not None:
        # The backend compressed its own artifact, so the torch module is unchanged
        # and the saving has to be read off the compiled weights instead.
        compiled_mib = compiled_bytes / 1024**2
        reduction = 1 - compiled_bytes / result.baseline_model_storage_bytes
        print(f"Model storage: {storage_mib:.1f} MiB (unchanged; quantized by the backend)")
        print(
            f"Compiled weights: {baseline_mib:.1f} -> {compiled_mib:.1f} MiB "
            f"({reduction:.1%} reduction)"
        )
    else:
        reduction = 1 - result.model_storage_bytes / result.baseline_model_storage_bytes
        print(
            f"Model storage: {baseline_mib:.1f} -> {storage_mib:.1f} MiB "
            f"({reduction:.1%} reduction)"
        )
        print(f"Quantized layers: {result.quantized_modules}")
    print(f"Input tokens: {result.input_tokens}")
    if result.quantization_ms:
        print(f"Quantization time: {result.quantization_ms:.2f} ms")
    print(f"First call: {result.first_call_ms:.2f} ms")
    print(f"Steady call: {result.latency_ms:.2f} ms")
    if result.peak_memory_bytes is not None:
        print(f"Peak GPU memory: {result.peak_memory_bytes / 1024**2:.1f} MiB")
    print(f"Next token: {result.next_token_id} ({result.next_token!r})")


def _print_model_generate(result: HuggingFaceGenerateResult) -> None:
    print(f"Model: {result.model_uri}")
    print(f"Target: {result.target}")
    print(f"Backend: {result.backend}")
    print(f"Dtype: {result.dtype}")
    print(f"Parameters: {result.parameter_count:,}")
    print(f"Input tokens: {result.input_tokens}")
    print(f"Generated tokens: {result.generated_tokens}")
    print(f"KV cache: {result.cache_implementation}")
    # Only the compiling backends pay a compile on the first generation; saying so
    # unconditionally would explain an eager target's timings with a step it skipped.
    first_call_label = (
        "First generation (includes compile)" if result.backend != "eager" else "First generation"
    )
    print(f"{first_call_label}: {result.first_call_ms:.2f} ms")
    print(f"Steady generation: {result.latency_ms:.2f} ms")
    if result.peak_memory_bytes is not None:
        print(f"Peak GPU memory: {result.peak_memory_bytes / 1024**2:.1f} MiB")
    print()
    print(result.generated_text)


def _dynamic_sequence(value: str | None) -> bool | tuple[int, int]:
    """Turn the --dynamic-seq argument into an export_hf_model argument."""
    if value is None:
        return False
    if value == "auto":
        return True
    minimum, separator, maximum = value.partition(":")
    if not separator:
        raise LM7Error(f"Invalid --dynamic-seq value {value!r}; expected MIN:MAX, such as 1:2048.")
    try:
        bounds = (int(minimum), int(maximum))
    except ValueError:
        raise LM7Error(
            f"Invalid --dynamic-seq value {value!r}; MIN and MAX must be integers."
        ) from None
    return bounds


def _print_model_export(result: HuggingFaceExportResult) -> None:
    print(f"Model: {result.model_uri}")
    print(f"Target: {result.target}")
    print(f"Backend: {result.backend}")
    print(f"Dtype: {result.dtype}")
    print(f"Quantization: {result.quantization}")
    print(f"Parameters: {result.parameter_count:,}")
    if result.sequence_bounds is None:
        print(f"Captured shape: {result.input_tokens} tokens from {result.prompt!r}")
    else:
        minimum, maximum = result.sequence_bounds
        print(
            f"Captured shape: {result.input_tokens} tokens from {result.prompt!r}, "
            f"sequence dynamic in [{minimum}, {maximum}]"
        )
    print(f"Export time: {result.export_ms:.2f} ms")
    print(f"Artifact: {result.output} ({result.artifact_bytes / 1024**2:.1f} MiB)")
    print(f"Files: {', '.join(result.files)}")


def _print_model_compatibility(result: ModelCompatibilityResult) -> None:
    architecture = ", ".join(result.architectures) or result.config_class or "unknown"
    if result.model_type:
        architecture = f"{architecture} ({result.model_type})"
    print(f"Model: {result.model_uri}")
    print(f"Status: {result.status}")
    print(f"Architecture: {architecture}")
    print(f"Task: {result.task}")
    print(f"Target: {result.target}")
    print(f"Backend: {result.selected_backend or 'none'}")
    if result.context_length is not None:
        print(f"Context length: {result.context_length:,}")
    print()
    print("Workflows:")
    for check in result.workflows:
        print(f"  {check.name}: {check.status} - {check.reason}")
    print()
    print("Quantization:")
    for check in result.quantization:
        print(f"  {check.name}: {check.status} - {check.reason}")
    print()
    print("Notes:")
    for note in result.notes:
        print(f"  - {note}")


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
    compatibility_parser = model_subparsers.add_parser(
        "compatibility", help="inspect Hugging Face compatibility without downloading weights"
    )
    compatibility_parser.add_argument("model_uri", help="model URI, for example hf://owner/model")
    compatibility_parser.add_argument(
        "--target", default="auto", help="target selector (default: auto)"
    )
    compatibility_parser.add_argument(
        "--backend", default="auto", help="backend selector (default: auto)"
    )
    _add_json_argument(compatibility_parser)

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
        "--quantize",
        dest="quantization",
        choices=(
            "none",
            "int8",
            "fp8",
            "nvfp4",
            "fp8-dynamic",
            "fp8-dynamic-rowwise",
            "nvfp4-dynamic",
        ),
        default="none",
        help=(
            "experimental quantization (default: none). The bare names are weight-only; "
            "the '-dynamic' names also quantize activations, so the matmul runs in the "
            "narrow format. 'fp8-dynamic' scales per tensor and "
            "'fp8-dynamic-rowwise' per row"
        ),
    )
    # The pre-0.2 spelling, kept working so existing scripts do not break, plus the
    # explicit weight-only long forms.
    run_parser.add_argument(
        "--quantization",
        dest="quantization",
        choices=(
            "none",
            "int8",
            "fp8",
            "nvfp4",
            "fp8-dynamic",
            "fp8-dynamic-rowwise",
            "nvfp4-dynamic",
            "int8-weight-only",
            "fp8-weight-only",
            "nvfp4-weight-only",
        ),
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    _add_json_argument(run_parser)

    generate_parser = model_subparsers.add_parser(
        "generate", help="generate tokens with a compiled static KV-cache decode loop"
    )
    generate_parser.add_argument("model_uri", help="model URI, for example hf://owner/model")
    generate_parser.add_argument(
        "--prompt", default="The capital of France is", help="input prompt"
    )
    generate_parser.add_argument(
        "--max-new-tokens", type=int, default=32, help="tokens to generate (default: 32)"
    )
    generate_parser.add_argument("--target", default="auto", help="target selector (default: auto)")
    generate_parser.add_argument(
        "--backend", choices=("auto", "inductor"), default="auto", help="backend selector"
    )
    generate_parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="model dtype (default: auto)",
    )
    _add_json_argument(generate_parser)

    serve_parser = model_subparsers.add_parser(
        "serve", help="serve a model over an OpenAI-compatible HTTP endpoint"
    )
    serve_parser.add_argument(
        "model_uri",
        help=(
            "hf://owner/model, or the path to a directory holding a model saved by save_pretrained"
        ),
    )
    serve_parser.add_argument("--target", default="auto", help="target selector (default: auto)")
    serve_parser.add_argument(
        "--backend",
        default="auto",
        help=(
            "backend selector for the compiled decode loop: auto, eager, or inductor. "
            "'vllm' instead hands the port to vLLM, which must be installed separately"
        ),
    )
    serve_parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="model dtype (default: auto)",
    )
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)"
    )
    serve_parser.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    serve_parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help=(
            "tokens of static KV cache to allocate at startup (default: 2048). "
            "The cache never grows, so this is the hard limit on prompt plus completion"
        ),
    )
    serve_parser.add_argument(
        "--compile-mode",
        default=None,
        help="Inductor preset for both graphs, for example reduce-overhead (default: none)",
    )
    serve_parser.add_argument(
        "--no-compile-prefill",
        dest="compile_prefill",
        action="store_false",
        help="leave the prompt pass in eager, so a new prompt length does not recompile",
    )
    serve_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be served, without loading a model or binding a port",
    )
    _add_json_argument(serve_parser)

    export_parser = model_subparsers.add_parser(
        "export", help="capture a model into an LM7 artifact"
    )
    export_parser.add_argument("model_uri", help="model URI, for example hf://owner/model")
    export_parser.add_argument("output", help="output artifact directory, for example model.lm7")
    export_parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="prompt whose tokenization fixes the captured input shape",
    )
    export_parser.add_argument(
        "--dynamic-seq",
        nargs="?",
        const="auto",
        metavar="MIN:MAX",
        help=(
            "capture the sequence length as a bounded dynamic dimension so one "
            "artifact serves many prompt lengths; bounds default to the model config"
        ),
    )
    export_parser.add_argument("--target", default="auto", help="target selector (default: auto)")
    export_parser.add_argument(
        "--backend",
        choices=tuple(sorted(EXPORT_BACKENDS)),
        default="export",
        help="export backend (default: export)",
    )
    export_parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="model dtype (default: auto)",
    )
    export_parser.add_argument(
        "--quantize",
        dest="quantization",
        choices=("none", "int8"),
        default="none",
        help=(
            "INT8 export quantization: calibrated XNNPACK PTQ on the executorch "
            "backend, NNCF weight compression on openvino (default: none)"
        ),
    )
    _add_json_argument(export_parser)

    bundle_parser = subparsers.add_parser("bundle", help="create and inspect LM7 bundles")
    bundle_subparsers = bundle_parser.add_subparsers(dest="bundle_command", required=True)
    bundle_create_parser = bundle_subparsers.add_parser(
        "create", help="package target artifacts into a bundle"
    )
    bundle_create_parser.add_argument("output", help="output bundle directory")
    bundle_create_parser.add_argument(
        "artifacts", nargs="+", help="artifact directories to include"
    )
    _add_json_argument(bundle_create_parser)
    bundle_inspect_parser = bundle_subparsers.add_parser(
        "inspect", help="list bundle targets and backends"
    )
    bundle_inspect_parser.add_argument("bundle", help="bundle directory")
    _add_json_argument(bundle_inspect_parser)
    artifact_parser = subparsers.add_parser("artifact", help="inspect LM7 artifacts")
    artifact_subparsers = artifact_parser.add_subparsers(dest="artifact_command", required=True)
    artifact_inspect_parser = artifact_subparsers.add_parser(
        "inspect", help="show artifact integrity and deployment requirements"
    )
    artifact_inspect_parser.add_argument("artifact", help="artifact directory")
    _add_json_argument(artifact_inspect_parser)

    hexagon_parser = subparsers.add_parser(
        "hexagon", help="inspect the experimental Hexagon-MLIR toolchain"
    )
    hexagon_subparsers = hexagon_parser.add_subparsers(dest="hexagon_command", required=True)
    hexagon_doctor_parser = hexagon_subparsers.add_parser(
        "doctor", help="check compile, simulator, and device prerequisites"
    )
    hexagon_doctor_parser.add_argument(
        "--mode",
        choices=("compile", "simulator", "device"),
        default="compile",
        help="readiness mode that controls the exit status (default: compile)",
    )
    _add_json_argument(hexagon_doctor_parser)
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
        elif args.command == "model" and args.model_command == "compatibility":
            compatibility = inspect_hf_model(
                args.model_uri,
                target=args.target,
                backend=args.backend,
            )
            (
                _emit_json(compatibility.to_dict())
                if args.json
                else _print_model_compatibility(compatibility)
            )
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
        elif args.command == "model" and args.model_command == "generate":
            generate_result = generate_hf_model(
                args.model_uri,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                target=args.target,
                backend=args.backend,
                dtype=args.dtype,
            )
            (
                _emit_json(generate_result.to_dict())
                if args.json
                else _print_model_generate(generate_result)
            )
        elif args.command == "model" and args.model_command == "serve":
            # Imported here rather than at module scope so that building the
            # parser for any other subcommand never reaches FastAPI.
            from .serve.cli import serve_model

            return serve_model(
                ServeConfig(
                    model=args.model_uri,
                    target=args.target,
                    backend=args.backend,
                    dtype=args.dtype,
                    max_model_len=args.max_model_len,
                    compile_mode=args.compile_mode,
                    compile_prefill=args.compile_prefill,
                    host=args.host,
                    port=args.port,
                ),
                dry_run=args.dry_run,
                as_json=args.json,
            )
        elif args.command == "model" and args.model_command == "export":
            export_result = export_hf_model(
                args.model_uri,
                output=args.output,
                prompt=args.prompt,
                target=args.target,
                backend=args.backend,
                dtype=args.dtype,
                quantization=args.quantization,
                dynamic_sequence=_dynamic_sequence(args.dynamic_seq),
            )
            (
                _emit_json(export_result.to_dict())
                if args.json
                else _print_model_export(export_result)
            )
        elif args.command == "bundle" and args.bundle_command == "create":
            bundle = create_bundle(args.artifacts, output=args.output)
            data = _bundle_data(str(bundle.path))
            _emit_json(data) if args.json else _print_bundle(data)
        elif args.command == "bundle" and args.bundle_command == "inspect":
            data = _bundle_data(args.bundle)
            _emit_json(data) if args.json else _print_bundle(data)
        elif args.command == "artifact" and args.artifact_command == "inspect":
            inspection = inspect_artifact(args.artifact)
            (
                _emit_json(inspection.to_dict())
                if args.json
                else _print_artifact_inspection(inspection)
            )
            if not inspection.valid:
                return 1
        elif args.command == "hexagon" and args.hexagon_command == "doctor":
            diagnostics = diagnose_hexagon_toolchain()
            data = diagnostics.to_dict()
            data["requested_mode"] = args.mode
            data["ready"] = diagnostics.ready_for(args.mode)
            _emit_json(data) if args.json else _print_hexagon_doctor(diagnostics, args.mode)
            if not diagnostics.ready_for(args.mode):
                return 1
    except LM7Error as exc:
        if args.json:
            _emit_json({"error": {"type": type(exc).__name__, "message": str(exc)}})
        else:
            print(f"lm7: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
