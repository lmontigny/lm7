"""On-device validation of LM7 ExecuTorch artifacts on an adb-reachable Android phone.

``docs/executorch.md`` claims a ``.pte`` is portable because it is not bound to
the machine that produced it, and then concedes that LM7 has never checked this:
validation is host XNNPACK on x86-64, and "runs correctly on x86-64" is not
"runs correctly on a Pixel". This harness closes that gap. It exports a model
with LM7, runs it on a real ARM64 device through the ExecuTorch C++ runtime, and
compares the device outputs against host eager.

The device never sees LM7, PyTorch, or Python. It sees a ``.bpte`` -- the LM7
``.pte`` rewrapped as an ExecuTorch BundledProgram, which carries the example
inputs and the expected outputs alongside the program -- and a cross-compiled
``example_runner``. That is the same deployment surface a phone app would have.

Any adb-reachable device works, including a remote one. Qualcomm Device Cloud
forwards its adb server over an SSH tunnel, so a cloud phone is reached by
pointing adb at the forwarded port; see ``docs/android-device-testing.md``.

Build the runner once (see the doc), then:

    python benchmarks/android_device.py --runner /path/to/example_runner

    python benchmarks/android_device.py --runner /path/to/example_runner \
      --quantize none int8 --serial eb49fb9d \
      --output artifacts/benchmarks/android-sm8750.json

Requires the ExecuTorch environment described in ``docs/executorch.md``, because
the export half runs on the host.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

import lm7

# ExecuTorch shells out to flatc when it serializes a BundledProgram, and the
# wheel hides it under executorch/data/bin rather than putting it on PATH. LM7
# already solves this for its own lowering; reuse that rather than re-deriving
# the path here.
from lm7.backends.executorch import _flatc_on_path
from lm7.exporting import COMPILED_PTE_NAME

DEVICE_DIR = "/data/local/tmp/lm7"

# example_runner logs one "%f" per output element via ET_LOG, which prefixes
# every line with a severity, timestamp, and source location.
_LOG_FLOAT = re.compile(r"\]\s*(-?(?:\d+\.\d+|inf|nan))\s*$")

# Tolerance for device-vs-host agreement. float32 XNNPACK reorders reductions
# across architectures, so exact equality is not expected; the float value
# matches the rtol/atol that tests/test_executorch_integration.py already uses
# for host XNNPACK against eager. INT8 is a different question -- the deviation
# there is dominated by single-sample PTQ calibration, not by the architecture.
_TOLERANCE = {"none": 1e-4, "int8": 2e-2}


@dataclass
class DeviceResult:
    quantization: str
    ok: bool
    detail: str = ""
    max_abs_diff_vs_eager: float | None = None
    max_abs_diff_vs_host_runtime: float | None = None
    strict_verification: str | None = None
    pte_bytes: int | None = None
    delegated_calls: int | None = None
    total_calls: int | None = None
    quantized_ops: int | None = None
    on_device_ms: dict[str, float] = field(default_factory=dict)
    device_ops: list[str] = field(default_factory=list)
    adb_roundtrip_ms: float | None = None


def mlp() -> tuple[torch.nn.Module, torch.Tensor]:
    """The model tests/test_executorch_integration.py exports, so host and device agree on the subject."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()
    return model, torch.randn(8, 16)


def hf_causal_lm(model_id: str) -> tuple[torch.nn.Module, torch.Tensor]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    encoded = tokenizer("The capital of France is", return_tensors="pt")
    return _LogitsOnly(model), encoded["input_ids"]


class _LogitsOnly(torch.nn.Module):
    """torch.export captures a HF CausalLMOutput; the bundled runner wants plain tensors."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.inner(input_ids=input_ids).logits


def adb(serial: str | None, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["adb"]
    if serial:
        command += ["-s", serial]
    command += list(args)
    return subprocess.run(command, capture_output=True, text=True, check=check)


def adb_shell(serial: str | None, script: str) -> tuple[int, str]:
    """Run a shell script on the device and recover its exit status.

    Older adb does not propagate the remote exit code, so the status is echoed
    into the stream and parsed back out.
    """
    completed = adb(serial, "shell", f"{script}; echo __EXIT__$?", check=False)
    stream = completed.stdout + completed.stderr
    match = re.search(r"__EXIT__(\d+)", stream)
    status = int(match.group(1)) if match else completed.returncode
    return status, re.sub(r"__EXIT__\d+", "", stream)


def bundle(pte: Path, inputs: list[torch.Tensor], expected: list[torch.Tensor]) -> bytes:
    """Rewrap an LM7 .pte as a BundledProgram carrying its own inputs and expected outputs.

    This is what lets the device check itself: the reference values travel
    inside the file, so the runner needs nothing else to verify a result.
    """
    config = importlib.import_module("executorch.devtools.bundled_program.config")
    core = importlib.import_module("executorch.devtools.bundled_program.core")
    serialize = importlib.import_module("executorch.devtools.bundled_program.serialize")

    suite = config.MethodTestSuite(
        method_name="forward",
        test_cases=[config.MethodTestCase(inputs=inputs, expected_outputs=expected)],
    )
    program = core.BundledProgram(None, [suite], pte_file_path=str(pte))
    with _flatc_on_path():
        blob: bytes = serialize.serialize_from_bundled_program_to_flatbuffer(program)
    return blob


def parse_outputs(stream: str) -> list[float]:
    return [
        float(match.group(1)) for line in stream.splitlines() if (match := _LOG_FLOAT.search(line))
    ]


def inspect_etdump(etdump: Path) -> tuple[dict[str, float], list[str]]:
    """Recover real on-device timings from the runner's ETDump.

    Wall-clock around `adb shell` is useless as an inference measurement --
    process start and, for a cloud device, the network round-trip dwarf the
    model. The runner always writes an ETDump, and that records what the device
    itself measured, per operator. The operator names are worth keeping: a
    "Fully Connected (NC, QS8, QC8W) GEMM" entry is direct evidence that the
    INT8 kernels ran, rather than a float fallback.
    """
    devtools = importlib.import_module("executorch.devtools")

    with _flatc_on_path():
        inspector = devtools.Inspector(etdump_path=str(etdump))
        timings: dict[str, float] = {}
        operators: list[str] = []
        for block in inspector.event_blocks:
            for event in block.events:
                if event.perf_data is None or not event.perf_data.raw:
                    continue
                elapsed = float(event.perf_data.raw[0])
                if event.name in {"Method::execute", "DELEGATE_CALL", "Method::init"}:
                    timings[event.name] = round(elapsed, 4)
                elif block.name == "Execute":
                    operators.append(event.name)
    return timings, operators


def run_once(serial: str | None, *, verify: bool, repeats: int) -> tuple[int, str, list[float]]:
    """Execute the bundled program on the device.

    example_runner has no iteration flag, so repeats are separate invocations.
    Each one reloads the program, so the wall-clock here is a round-trip, not a
    latency; inspect_etdump() is what reports inference time.
    """
    flags = f"--bundled_program_path={DEVICE_DIR}/model.bpte --etdump_path={DEVICE_DIR}/etdump.etdp"
    if verify:
        flags += " --output_verification"
    command = f"cd {DEVICE_DIR} && ./example_runner {flags} --print_output"

    status, stream = 0, ""
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        status, stream = adb_shell(serial, command)
        durations.append((time.perf_counter() - started) * 1000.0)
        if status != 0:
            break
    return status, stream, durations


def evaluate(
    *,
    model: torch.nn.Module,
    example: torch.Tensor,
    quantization: str,
    serial: str | None,
    runner: Path,
    workdir: Path,
    repeats: int,
) -> DeviceResult:
    result = DeviceResult(quantization=quantization, ok=False)

    with torch.no_grad():
        eager = model(example)

    options: dict[str, Any] = {}
    if quantization != "none":
        options["quantization"] = quantization

    artifact_path = workdir / f"model-{quantization}.lm7"
    artifact = lm7.export(
        model,
        args=(example,),
        target="cpu",
        backend="executorch",
        output=artifact_path,
        options=options or None,
    )
    requirements = artifact.manifest.runtime_requirements or {}
    result.delegated_calls = requirements.get("delegated_calls")
    result.total_calls = requirements.get("total_calls")
    result.quantized_ops = requirements.get("quantized_ops")

    pte = artifact_path / COMPILED_PTE_NAME
    result.pte_bytes = pte.stat().st_size

    # The host runtime running the same .pte. Separating this from eager splits
    # "the export lost accuracy" from "the architecture disagrees".
    host_runtime = artifact(example)

    blob = bundle(pte, [example], [eager])
    bpte = workdir / f"model-{quantization}.bpte"
    bpte.write_bytes(blob)

    adb_shell(serial, f"mkdir -p {DEVICE_DIR}")
    adb(serial, "push", str(bpte), f"{DEVICE_DIR}/model.bpte")
    adb(serial, "push", str(runner), f"{DEVICE_DIR}/example_runner")
    adb_shell(serial, f"chmod 755 {DEVICE_DIR}/example_runner")

    # Strict verification aborts the process on mismatch, and its tolerance is
    # hardcoded at rtol=1e-3/atol=1e-5 -- realistic for float32, not for INT8.
    # Run it as a gate only where that tolerance is meaningful, and always fall
    # back to comparing the printed values ourselves.
    strict = quantization == "none"
    status, stream, durations = run_once(serial, verify=strict, repeats=repeats)

    if strict:
        result.strict_verification = "passed" if status == 0 else "failed"
    if status != 0:
        result.detail = f"example_runner exited {status}: {stream.strip()[-400:]}"
        return result

    values = parse_outputs(stream)
    if len(values) != eager.numel():
        result.detail = f"expected {eager.numel()} printed values, parsed {len(values)}"
        return result

    device = torch.tensor(values, dtype=eager.dtype).reshape(eager.shape)
    result.max_abs_diff_vs_eager = float((device - eager).abs().max())
    result.max_abs_diff_vs_host_runtime = float((device - host_runtime).abs().max())
    if durations:
        result.adb_roundtrip_ms = round(statistics.median(durations), 2)

    local_etdump = workdir / f"etdump-{quantization}.etdp"
    pulled = adb(serial, "pull", f"{DEVICE_DIR}/etdump.etdp", str(local_etdump), check=False)
    if pulled.returncode == 0 and local_etdump.is_file():
        result.on_device_ms, result.device_ops = inspect_etdump(local_etdump)

    tolerance = _TOLERANCE[quantization]
    result.ok = result.max_abs_diff_vs_eager <= tolerance
    if not result.ok:
        result.detail = f"max abs diff {result.max_abs_diff_vs_eager:.3g} exceeds {tolerance:g}"
    return result


def device_properties(serial: str | None) -> dict[str, str]:
    wanted = {
        "model": "ro.product.model",
        "soc": "ro.soc.model",
        "platform": "ro.board.platform",
        "android_release": "ro.build.version.release",
        "abi": "ro.product.cpu.abi",
    }
    properties = {}
    for key, prop in wanted.items():
        status, stream = adb_shell(serial, f"getprop {prop}")
        properties[key] = stream.strip() if status == 0 else "unknown"
    return properties


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runner", required=True, type=Path, help="arm64 example_runner binary")
    parser.add_argument(
        "--serial", default=None, help="adb serial; required when several are attached"
    )
    parser.add_argument(
        "--quantize",
        nargs="+",
        default=["none", "int8"],
        choices=["none", "int8"],
        help="export configurations to validate on the device",
    )
    parser.add_argument("--model", default="mlp", help="'mlp' or a Hugging Face causal LM id")
    parser.add_argument(
        "--repeats", type=int, default=5, help="device invocations per configuration"
    )
    parser.add_argument("--output", type=Path, default=None, help="write a JSON report here")
    args = parser.parse_args(argv)

    if shutil.which("adb") is None:
        parser.error("adb is not on PATH")
    if not args.runner.is_file():
        parser.error(f"runner not found: {args.runner}")

    attached = adb(args.serial, "devices", check=False).stdout
    if "\tdevice" not in attached:
        parser.error(f"no adb device is ready:\n{attached}")

    model, example = mlp() if args.model == "mlp" else hf_causal_lm(args.model)
    properties = device_properties(args.serial)
    print(
        f"device: {properties['model']} ({properties['soc']}, {properties['abi']}, "
        f"Android {properties['android_release']})"
    )

    results = []
    with tempfile.TemporaryDirectory() as raw:
        for quantization in args.quantize:
            print(f"\n== {quantization} ==")
            result = evaluate(
                model=model,
                example=example,
                quantization=quantization,
                serial=args.serial,
                runner=args.runner,
                workdir=Path(raw),
                repeats=args.repeats,
            )
            results.append(result)
            status = "ok" if result.ok else "FAILED"
            print(f"  {status}")
            if result.max_abs_diff_vs_eager is not None:
                print(f"  max abs diff vs host eager:    {result.max_abs_diff_vs_eager:.3g}")
                print(f"  max abs diff vs host runtime:  {result.max_abs_diff_vs_host_runtime:.3g}")
            if result.strict_verification:
                print(f"  bundled strict verification:   {result.strict_verification}")
            if result.on_device_ms:
                execute = result.on_device_ms.get("Method::execute")
                delegate = result.on_device_ms.get("DELEGATE_CALL")
                print(f"  on-device Method::execute:     {execute} ms")
                print(f"  on-device DELEGATE_CALL:       {delegate} ms")
            if result.device_ops:
                print(f"  device operators:              {', '.join(result.device_ops)}")
            if result.adb_roundtrip_ms:
                print(f"  adb round-trip (not latency):  {result.adb_roundtrip_ms} ms")
            if result.detail:
                print(f"  {result.detail}")

    report = {
        "device": properties,
        "model": args.model,
        "lm7": lm7.__version__,
        "results": [vars(result) for result in results],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.output}")

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
