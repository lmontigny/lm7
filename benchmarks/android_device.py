"""On-device validation of LM7 ExecuTorch artifacts on an adb-reachable Android phone.

``docs/executorch.md`` claims a ``.pte`` is portable because it is not bound to
the machine that produced it, and then concedes that LM7 has never checked this:
validation is host XNNPACK on x86-64, and "runs correctly on x86-64" is not
"runs correctly on a Pixel". This harness closes that gap. It exports a model
with LM7, runs it on a real ARM64 device through the ExecuTorch C++ runtime, and
compares the device outputs against host eager.

The device never sees LM7, PyTorch, or Python. It sees a ``.pte``, its inputs as
raw tensor bytes, and ``lm7_runner`` -- the small C++ binary in
``tools/android_runner`` cross-compiled for arm64. That is the same deployment
surface a phone app would have.

ExecuTorch ships ``example_runner``, which validates a BundledProgram and would
otherwise be the obvious tool. It does not scale to a real model: bundling
serializes through JSON and ``flatc``, which aborts on a 622 MB SmolLM2 ``.pte``,
and it reports outputs by logging one line per element, where a transformer's
logits run to hundreds of thousands of values. ``lm7_runner`` exists because of
those two limits; see ``tools/android_runner/lm7_runner.cpp``.

Any adb-reachable device works, including a remote one. Qualcomm Device Cloud
forwards its adb server over an SSH tunnel, so a cloud phone is reached by
pointing adb at the forwarded port; see ``docs/android-device-testing.md``.

Build the runner once (see the doc), then:

    python benchmarks/android_device.py --runner /path/to/lm7_runner

    python benchmarks/android_device.py --runner /path/to/lm7_runner \
      --model HuggingFaceTB/SmolLM2-135M-Instruct --quantize none \
      --serial eb49fb9d --output artifacts/benchmarks/android-sm8750.json

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
from lm7.exporting import COMPILED_PTE_NAME

DEVICE_DIR = "/data/local/tmp/lm7"

_ITER_MS = re.compile(r"^iter_ms\s+([0-9.]+)\s*$", re.MULTILINE)
_REPORTED = re.compile(
    r"^(output_numel|output_nbytes|output_dtype|output_outputs)\s+(\d+)\s*$", re.MULTILINE
)

# Tolerance for device-vs-host agreement. float32 XNNPACK reorders reductions
# across architectures, so exact equality is not expected; the float value
# matches the rtol/atol that tests/test_executorch_integration.py already uses
# for host XNNPACK against eager. INT8 is a different question -- the deviation
# there is dominated by single-sample PTQ calibration, not by the architecture.
_TOLERANCE = {"none": 1e-4, "int8": 2e-2}

# ExecuTorch reports its output ScalarType as PyTorch's integer code. Reading
# the buffer back at the wrong width silently yields the wrong element count, so
# the dtype travels with the result rather than being assumed.
_SCALAR_TYPES = {
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    11: torch.bool,
    15: torch.bfloat16,
}


@dataclass
class ModelCase:
    """A model, the input to capture with, and optionally the tokenizer behind it."""

    model: torch.nn.Module
    example: torch.Tensor
    tokenizer: Any = None


@dataclass
class DeviceResult:
    quantization: str
    ok: bool
    detail: str = ""
    max_abs_diff_vs_eager: float | None = None
    max_abs_diff_vs_host_runtime: float | None = None
    # For a causal LM: does the device pick the same next token as the host? A
    # logit delta is hard to interpret at INT8; a changed argmax is not.
    next_token_agrees: bool | None = None
    next_token_host: str | None = None
    next_token_device: str | None = None
    top5_overlap: int | None = None
    pte_bytes: int | None = None
    delegated_calls: int | None = None
    total_calls: int | None = None
    quantized_ops: int | None = None
    export_seconds: float | None = None
    push_seconds: float | None = None
    output_dtype: str | None = None
    on_device_ms: dict[str, float] = field(default_factory=dict)
    host_ms: dict[str, float] = field(default_factory=dict)


def mlp() -> ModelCase:
    """The model tests/test_executorch_integration.py exports, so host and device agree on the subject."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.GELU(),
        torch.nn.Linear(32, 4),
    ).eval()
    return ModelCase(model, torch.randn(8, 16))


def hf_causal_lm(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    prompt: str = "The capital of France is",
) -> ModelCase:
    """Load a Hugging Face causal LM as a plain-tensor callable.

    The dtype is passed explicitly rather than left to `from_pretrained`.
    Transformers 5 defaults to the checkpoint's dtype, which for SmolLM2 is
    bfloat16 -- so the default silently exports a different model than the
    float32 baseline recorded in docs/executorch.md.
    """
    transformers = importlib.import_module("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="eager", dtype=dtype
    ).eval()
    encoded = tokenizer(prompt, return_tensors="pt")
    return ModelCase(_NextTokenLogits(model), encoded["input_ids"], tokenizer=tokenizer)


class _NextTokenLogits(torch.nn.Module):
    """Return the next-token logits as a plain tensor.

    `torch.export` captures a Hugging Face `CausalLMOutput`, and an ExecuTorch
    method has a flat tensor output list, so the dataclass has to go.

    Taking only the last position is a separate choice. Generation consumes only
    the final row; the earlier rows are prefill positions no decoder reads. The
    `.pte` still computes all of them -- this selects what is compared and
    transferred, not what is executed.
    """

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.inner(input_ids=input_ids).logits[:, -1, :]


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


def run_on_device(
    serial: str | None, *, input_count: int, repeats: int, warmup: int
) -> tuple[int, str]:
    inputs = ",".join(f"{DEVICE_DIR}/input{i}.bin" for i in range(input_count))
    command = (
        f"cd {DEVICE_DIR} && ./lm7_runner "
        f"--model_path={DEVICE_DIR}/model.pte "
        f"--inputs={inputs} "
        f"--output_path={DEVICE_DIR}/output.bin "
        f"--repeats={repeats} --warmup={warmup}"
    )
    return adb_shell(serial, command)


def evaluate(
    *,
    case: ModelCase,
    quantization: str,
    serial: str | None,
    runner: Path,
    workdir: Path,
    repeats: int,
    warmup: int,
) -> DeviceResult:
    result = DeviceResult(quantization=quantization, ok=False)
    model, example = case.model, case.example

    with torch.no_grad():
        eager = model(example)

    options: dict[str, Any] = {}
    if quantization != "none":
        options["quantization"] = quantization

    artifact_path = workdir / f"model-{quantization}.lm7"
    started = time.perf_counter()
    artifact = lm7.export(
        model,
        args=(example,),
        target="cpu",
        backend="executorch",
        output=artifact_path,
        options=options or None,
    )
    result.export_seconds = round(time.perf_counter() - started, 1)

    requirements = artifact.manifest.runtime_requirements or {}
    result.delegated_calls = requirements.get("delegated_calls")
    result.total_calls = requirements.get("total_calls")
    result.quantized_ops = requirements.get("quantized_ops")

    pte = artifact_path / COMPILED_PTE_NAME
    result.pte_bytes = pte.stat().st_size

    # The host runtime running the same .pte. Separating this from eager splits
    # "the export lost accuracy" from "the architecture disagrees".
    host_runtime = artifact(example)

    # Time the host on the same .pte, so ARM and x86 are compared on identical
    # bytes and an identical output shape. The number in docs/executorch.md is
    # not a substitute: it was measured on a different machine and on the full
    # logits rather than the last row.
    host_durations = []
    for _ in range(warmup):
        artifact(example)
    for _ in range(repeats):
        started = time.perf_counter()
        artifact(example)
        host_durations.append((time.perf_counter() - started) * 1000.0)
    if host_durations:
        result.host_ms = {
            "median": round(statistics.median(host_durations), 4),
            "min": round(min(host_durations), 4),
        }

    input_path = workdir / "input0.bin"
    input_path.write_bytes(example.contiguous().numpy().tobytes())

    adb_shell(serial, f"mkdir -p {DEVICE_DIR}")
    started = time.perf_counter()
    adb(serial, "push", str(pte), f"{DEVICE_DIR}/model.pte")
    result.push_seconds = round(time.perf_counter() - started, 1)
    adb(serial, "push", str(input_path), f"{DEVICE_DIR}/input0.bin")
    adb(serial, "push", str(runner), f"{DEVICE_DIR}/lm7_runner")
    adb_shell(serial, f"chmod 755 {DEVICE_DIR}/lm7_runner")

    status, stream = run_on_device(serial, input_count=1, repeats=repeats, warmup=warmup)
    if status != 0:
        result.detail = f"lm7_runner exited {status}: {stream.strip()[-400:]}"
        return result

    reported = {key: int(value) for key, value in _REPORTED.findall(stream)}
    if reported.get("output_numel") != eager.numel():
        result.detail = (
            f"device reported {reported.get('output_numel')} output elements, "
            f"host expected {eager.numel()}"
        )
        return result

    durations = [float(value) for value in _ITER_MS.findall(stream)]
    if durations:
        result.on_device_ms = {
            "median": round(statistics.median(durations), 4),
            "min": round(min(durations), 4),
            "max": round(max(durations), 4),
            "iterations": len(durations),
        }

    local_output = workdir / "output.bin"
    pulled = adb(serial, "pull", f"{DEVICE_DIR}/output.bin", str(local_output), check=False)
    if pulled.returncode != 0 or not local_output.is_file():
        result.detail = "could not pull the device output"
        return result

    reported_dtype = _SCALAR_TYPES.get(reported.get("output_dtype", -1))
    if reported_dtype is None:
        result.detail = f"unhandled device output ScalarType {reported.get('output_dtype')}"
        return result
    result.output_dtype = str(reported_dtype).removeprefix("torch.")

    raw = local_output.read_bytes()
    if len(raw) != reported.get("output_nbytes"):
        result.detail = f"pulled {len(raw)} bytes, device wrote {reported.get('output_nbytes')}"
        return result

    device = torch.frombuffer(bytearray(raw), dtype=reported_dtype).reshape(eager.shape)

    # Compare in float32 whatever the artifact's dtype is, so a bfloat16 export
    # is not measured with bfloat16 subtraction.
    device = device.float()
    eager = eager.float()
    host_runtime = host_runtime.float()

    result.max_abs_diff_vs_eager = float((device - eager).abs().max())
    result.max_abs_diff_vs_host_runtime = float((device - host_runtime).abs().max())

    if case.tokenizer is not None:
        host_token = int(eager[0].argmax())
        device_token = int(device[0].argmax())
        result.next_token_agrees = host_token == device_token
        result.next_token_host = repr(case.tokenizer.decode([host_token]))
        result.next_token_device = repr(case.tokenizer.decode([device_token]))
        host_top5 = set(torch.topk(eager[0], 5).indices.tolist())
        result.top5_overlap = len(host_top5 & set(torch.topk(device[0], 5).indices.tolist()))

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
    parser.add_argument("--runner", required=True, type=Path, help="arm64 lm7_runner binary")
    parser.add_argument(
        "--serial", default=None, help="adb serial; required when several are attached"
    )
    parser.add_argument(
        "--quantize",
        nargs="+",
        default=["none"],
        choices=["none", "int8"],
        help="export configurations to validate on the device",
    )
    parser.add_argument("--model", default="mlp", help="'mlp' or a Hugging Face causal LM id")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="dtype to load a Hugging Face model at; ignored for 'mlp'",
    )
    parser.add_argument("--repeats", type=int, default=20, help="timed forward passes on device")
    parser.add_argument("--warmup", type=int, default=3, help="untimed forward passes first")
    parser.add_argument("--output", type=Path, default=None, help="write a JSON report here")
    args = parser.parse_args(argv)

    if shutil.which("adb") is None:
        parser.error("adb is not on PATH")
    if not args.runner.is_file():
        parser.error(f"runner not found: {args.runner}")

    attached = adb(args.serial, "devices", check=False).stdout
    if "\tdevice" not in attached:
        parser.error(f"no adb device is ready:\n{attached}")

    dtype = getattr(torch, args.dtype)
    case = mlp() if args.model == "mlp" else hf_causal_lm(args.model, dtype=dtype)
    properties = device_properties(args.serial)
    print(
        f"device: {properties['model']} ({properties['soc']}, {properties['abi']}, "
        f"Android {properties['android_release']})"
    )
    label = args.model if args.model == "mlp" else f"{args.model} ({args.dtype})"
    print(f"model:  {label}")

    results = []
    with tempfile.TemporaryDirectory() as raw:
        for quantization in args.quantize:
            print(f"\n== {quantization} ==")
            result = evaluate(
                case=case,
                quantization=quantization,
                serial=args.serial,
                runner=args.runner,
                workdir=Path(raw),
                repeats=args.repeats,
                warmup=args.warmup,
            )
            results.append(result)
            print(f"  {'ok' if result.ok else 'FAILED'}")
            if result.pte_bytes:
                print(f"  .pte bytes:                    {result.pte_bytes:,}")
            if result.delegated_calls is not None:
                print(
                    f"  delegate coverage:             {result.delegated_calls}/{result.total_calls}"
                )
            if result.export_seconds:
                print(f"  host export:                   {result.export_seconds} s")
            if result.push_seconds:
                print(f"  push to device:                {result.push_seconds} s")
            if result.max_abs_diff_vs_eager is not None:
                print(f"  max abs diff vs host eager:    {result.max_abs_diff_vs_eager:.3g}")
                print(f"  max abs diff vs host runtime:  {result.max_abs_diff_vs_host_runtime:.3g}")
            if result.next_token_agrees is not None:
                verdict = "same" if result.next_token_agrees else "DIFFERENT"
                print(
                    f"  next token:                    {verdict} "
                    f"(host {result.next_token_host}, device {result.next_token_device})"
                )
                print(f"  top-5 overlap with host:       {result.top5_overlap}/5")
            if result.on_device_ms:
                print(
                    f"  on-device forward:             {result.on_device_ms['median']} ms median "
                    f"({result.on_device_ms['min']}-{result.on_device_ms['max']}, "
                    f"n={result.on_device_ms['iterations']})"
                )
            if result.host_ms:
                print(f"  host forward, same .pte:       {result.host_ms['median']} ms median")
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
