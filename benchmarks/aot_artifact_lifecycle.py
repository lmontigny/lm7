"""Validate an AOTInductor artifact across a real process boundary.

Every other measurement of `lm7.export(backend="aot_inductor")` in this repo
exports and reloads inside one interpreter, which cannot see the failures that
matter for a deployed artifact: a package that only loads because the compiling
process still holds Inductor's caches, a manifest that describes a machine
rather than the payload, an artifact that silently needs the model's source
library to come back. So each stage here is a separate process:

    export   -> writes the .lm7 directory, exits
    load     -> fresh interpreter, validates numerics against a saved reference
    mismatch -> fresh interpreter, mutated metadata, records how it fails

The driver subcommand runs the whole sequence and writes one JSON summary:

    python benchmarks/aot_artifact_lifecycle.py run --model llama32-1b \
        --results-dir artifacts/aoti-sm120

Timing vocabulary, because "reload time" is ambiguous by itself:

    cold    the artifact's pages are dropped from the page cache first, so the
            read is a real disk read (POSIX_FADV_DONTNEED, no root needed)
    warm    a second fresh process, artifact still resident in the page cache

Both are fresh interpreters. The difference between them is the filesystem, not
Python or CUDA, which is why "reload" needs two numbers rather than one.

`load` also splits the two APIs, because they do different amounts of work:

    --api lm7     lm7.load_artifact(): manifest, both checksums, the
                  ExportedProgram *and* the compiled package
    --api torch   torch._inductor.aoti_load_package(): the payload alone

Nothing in this file imports torch at module scope. Interpreter startup is part
of what is being measured, so the import lands inside the phase that pays for it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

_PROCESS_START = time.perf_counter()

HF_MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "llama32-1b": "unsloth/Llama-3.2-1B-Instruct",
}
MODELS = ("mlp", "resnet18", *HF_MODELS)
PROMPT = "The capital of France is"

REFERENCE_NAME = "reference.pt"
MANIFEST_NAME = "manifest.json"
PROGRAM_NAME = "exported_program.pt2"
PAYLOAD_NAME = "compiled_model.pt2"

# Each case mutates one thing about an artifact that is otherwise valid on this
# machine. `torch-version` is the only one that leaves the bytes alone: it runs
# an unmodified artifact under a different PyTorch, which is the mismatch a user
# hits by upgrading rather than by editing anything.
MISMATCH_CASES = (
    "architecture",
    "format-version",
    "payload-checksum",
    "program-checksum",
    "missing-payload",
    "payload-consistent-corruption",
    "torch-version",
)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _since_start_ms() -> float:
    return _elapsed_ms(_PROCESS_START)


def _artifact_files(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*") if item.is_file())


def _evict_page_cache(path: Path) -> bool:
    """Drop an artifact's pages so the next read reaches the device.

    POSIX_FADV_DONTNEED only evicts clean pages, so the sync has to come first
    or a just-written artifact stays resident and "cold" would silently measure
    the page cache. Returns False where the call does not exist (non-Linux), so
    a result can say it was not evicted rather than imply it was.
    """
    if not hasattr(os, "posix_fadvise"):
        return False
    os.sync()
    for item in _artifact_files(path):
        descriptor = os.open(item, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)
    return True


def build(name: str, dtype_name: str) -> tuple[Any, tuple[Any, ...]]:
    """Return an eval-mode model and representative inputs, on the CPU.

    Deliberately a copy of `nvidia_matrix.build` rather than an import of it:
    that module imports torch at module scope, and importing it here would
    charge the import to whichever phase ran first instead of to the one being
    timed.
    """
    import torch

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype_name
    ]

    class TensorOut(torch.nn.Module):
        """Tensor in, tensor out -- torch.export cannot round-trip an HF output."""

        def __init__(self, model: torch.nn.Module, causal: bool) -> None:
            super().__init__()
            self.model = model
            self.causal = causal

        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            if self.causal:
                return self.model(input_ids=args[0], attention_mask=args[1], use_cache=False).logits
            return self.model(*args)

    if name == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 1024),
        ).eval()
        return TensorOut(model, False).to(dtype=dtype), (torch.randn(8, 1024, dtype=dtype),)
    if name == "resnet18":
        from torchvision.models import resnet18

        return (
            TensorOut(resnet18().eval(), False).to(dtype=dtype),
            (torch.randn(8, 3, 224, 224, dtype=dtype),),
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = HF_MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
    encoded = tokenizer(PROMPT, return_tensors="pt")
    return TensorOut(model, True).eval(), (encoded["input_ids"], encoded["attention_mask"])


def _environment() -> dict[str, Any]:
    """Everything needed to decide whether an artifact belongs on this machine."""
    import torch

    record: dict[str, Any] = {
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        record.update(
            {
                "architecture": f"sm{major}{minor}",
                "device_name": torch.cuda.get_device_name(),
                "driver": _driver_version(),
                "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    return record


def _driver_version() -> str | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = output.stdout.strip().splitlines()
    return value[0].strip() if value else None


def _parity(actual: Any, reference: Any) -> dict[str, Any]:
    if actual.shape != reference.shape:
        return {"parity": "shape-mismatch", "max_abs_diff": None, "argmax_agrees": None}
    return {
        "parity": "ok",
        "max_abs_diff": (actual - reference).abs().max().item(),
        "argmax_agrees": bool(
            actual.reshape(-1, actual.shape[-1])[-1].argmax()
            == reference.reshape(-1, reference.shape[-1])[-1].argmax()
        ),
    }


def run_export(arguments: argparse.Namespace) -> dict[str, Any]:
    """Process A: build, capture, compile, persist, and record the reference."""
    import torch

    import lm7
    from lm7.detection import resolve_target, synchronize

    target = resolve_target(arguments.target)
    output = Path(arguments.artifact)
    if output.exists():
        shutil.rmtree(output)

    model, inputs = build(arguments.model, arguments.dtype)
    parameters = sum(parameter.numel() for parameter in model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_model = model.to(device)
    device_inputs = tuple(tensor.to(device) for tensor in inputs)
    with torch.no_grad():
        reference = device_model(*device_inputs).detach().float().cpu()
    synchronize(target)

    # Timed on its own so the artifact's build time splits into capture and
    # compile. lm7.export captures again internally; this is a diagnostic, and
    # the artifact's own build time is `export_seconds`.
    started = time.perf_counter()
    with torch.no_grad():
        torch.export.export(device_model, device_inputs, strict=False)
    capture_seconds = time.perf_counter() - started

    started = time.perf_counter()
    lm7.export(
        device_model,
        args=device_inputs,
        target=target,
        output=output,
        backend="aot_inductor",
    )
    export_seconds = time.perf_counter() - started

    # Process B needs the inputs and the expected answer, and must not need the
    # library that built the model to get them.
    torch.save(
        {"inputs": tuple(tensor.cpu() for tensor in inputs), "reference": reference},
        output.parent / f"{output.name}.{REFERENCE_NAME}",
    )

    sizes = {item.name: item.stat().st_size for item in _artifact_files(output)}
    return {
        "stage": "export",
        "model": arguments.model,
        "dtype": arguments.dtype,
        "parameter_count": parameters,
        "target": str(target),
        "artifact": str(output),
        "capture_seconds": capture_seconds,
        "export_seconds": export_seconds,
        "compile_seconds": export_seconds - capture_seconds,
        "artifact_bytes": sum(sizes.values()),
        "artifact_file_bytes": sizes,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else None,
        "manifest": json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8")),
        "environment": _environment(),
        "works": True,
    }


def run_load(arguments: argparse.Namespace) -> dict[str, Any]:
    """Process B: reload the artifact and prove it still computes the answer."""
    artifact_path = Path(arguments.artifact)
    evicted = _evict_page_cache(artifact_path) if arguments.evict else False

    started = time.perf_counter()
    import torch

    import_torch_ms = _elapsed_ms(started)

    started = time.perf_counter()
    import lm7

    import_lm7_ms = _elapsed_ms(started)

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    cuda_init_ms = _elapsed_ms(started)

    started = time.perf_counter()
    if arguments.api == "lm7":
        loaded: Any = lm7.load_artifact(artifact_path)
    else:
        loaded = torch._inductor.aoti_load_package(str(artifact_path / PAYLOAD_NAME))
    load_ms = _elapsed_ms(started)

    saved = torch.load(
        artifact_path.parent / f"{artifact_path.name}.{REFERENCE_NAME}", weights_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tuple(tensor.to(device) for tensor in saved["inputs"])

    started = time.perf_counter()
    with torch.no_grad():
        output = loaded(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    first_call_ms = _elapsed_ms(started)
    to_first_inference_ms = _since_start_ms()
    actual = output.detach().float().cpu()

    for _ in range(arguments.warmup):
        with torch.no_grad():
            loaded(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    samples = []
    for _ in range(arguments.repeats):
        started = time.perf_counter()
        with torch.no_grad():
            loaded(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append(_elapsed_ms(started))

    # A second load in the same interpreter: what a server pays to hold two
    # copies, and the part of the cold number that is neither disk nor CUDA.
    started = time.perf_counter()
    if arguments.api == "lm7":
        lm7.load_artifact(artifact_path)
    else:
        torch._inductor.aoti_load_package(str(artifact_path / PAYLOAD_NAME))
    second_load_ms = _elapsed_ms(started)

    record = {
        "stage": "load",
        "api": arguments.api,
        "cache": "cold" if arguments.evict else "warm",
        "evicted": evicted,
        "artifact": str(artifact_path),
        "import_torch_ms": import_torch_ms,
        "import_lm7_ms": import_lm7_ms,
        "cuda_init_ms": cuda_init_ms,
        "load_ms": load_ms,
        "second_load_ms": second_load_ms,
        "first_call_ms": first_call_ms,
        "to_first_inference_ms": to_first_inference_ms,
        "latency_median_ms": statistics.median(samples),
        "latency_min_ms": min(samples),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else None,
        # An artifact that needs the modelling library back is not a portable
        # artifact, and the only way to know is to look at what got imported.
        "imported_transformers": "transformers" in sys.modules,
        "imported_torchvision": "torchvision" in sys.modules,
        "environment": _environment(),
        "works": True,
    }
    record.update(_parity(actual, saved["reference"]))
    return record


def run_jit(arguments: argparse.Namespace) -> dict[str, Any]:
    """The baseline the artifact is supposed to beat: compile in this process.

    Not a like-for-like comparison and should not be read as one -- this path
    also builds the model and downloads/loads its weights, which the artifact
    path does not. The phases are reported separately so the comparable parts
    can be compared.
    """
    if arguments.clear_inductor_cache:
        # Clear only the cache this process would use. The default lives in a
        # shared /tmp, and a glob there deletes whatever else on the machine is
        # mid-compile -- which on a shared GPU box means someone else's run, and
        # is how this harness raced a concurrent benchmark into
        # "Directory not empty". Set TORCHINDUCTOR_CACHE_DIR to stay private.
        configured = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        directories = (
            [Path(configured)]
            if configured
            else list(Path(tempfile.gettempdir()).glob("torchinductor_*"))
        )
        for directory in directories:
            shutil.rmtree(directory, ignore_errors=True)

    started = time.perf_counter()
    import torch

    import lm7

    import_ms = _elapsed_ms(started)

    started = time.perf_counter()
    model, inputs = build(arguments.model, arguments.dtype)
    build_ms = _elapsed_ms(started)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tuple(tensor.to(device) for tensor in inputs)
    compiled = lm7.compile(
        model,
        target=arguments.target,
        backend="inductor",
        transfers="automatic",
        fallback="error",
        cache=False,
    )
    started = time.perf_counter()
    with torch.no_grad():
        compiled(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    first_call_ms = _elapsed_ms(started)

    samples = []
    for _ in range(arguments.repeats):
        started = time.perf_counter()
        with torch.no_grad():
            compiled(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append(_elapsed_ms(started))

    return {
        "stage": "jit",
        "model": arguments.model,
        "cache": "cold" if arguments.clear_inductor_cache else "warm",
        "import_ms": import_ms,
        "build_model_ms": build_ms,
        "first_call_ms": first_call_ms,
        "to_first_inference_ms": _since_start_ms(),
        "latency_median_ms": statistics.median(samples),
        "environment": _environment(),
        "works": True,
    }


def _mutate(case: str, artifact_path: Path) -> Path:
    """Copy the artifact and break exactly one thing about it."""
    working = Path(tempfile.mkdtemp(prefix="lm7-mismatch-")) / artifact_path.name
    shutil.copytree(artifact_path, working)
    manifest_path = working / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if case == "architecture":
        # The artifact now claims a GPU generation it was not built on. Nothing
        # about the payload changed, which is the point: the manifest is the
        # only thing standing between a user and "no kernel image is available
        # for execution on the device". Picked relative to what the artifact
        # already records, so the case stays a mismatch on every machine --
        # hardcoding one architecture makes this silently pass on that machine.
        recorded = manifest["target"].get("architecture")
        manifest["target"]["architecture"] = "sm120" if recorded == "sm89" else "sm89"
    elif case == "format-version":
        manifest["format_version"] = manifest["format_version"] + 1
    elif case in {"payload-checksum", "program-checksum"}:
        name = PAYLOAD_NAME if case == "payload-checksum" else PROGRAM_NAME
        with (working / name).open("r+b") as handle:
            handle.seek(64)
            flipped = bytes([handle.read(1)[0] ^ 0xFF])
            handle.seek(64)
            handle.write(flipped)
    elif case == "missing-payload":
        (working / PAYLOAD_NAME).unlink()
    elif case == "payload-consistent-corruption":
        # Break the payload and re-record its checksum, so LM7's integrity check
        # passes and PyTorch is the one that has to refuse. Everything else in
        # this list is caught by metadata; this is what the failure looks like
        # when nothing in the manifest is wrong.
        payload = working / PAYLOAD_NAME
        payload.write_bytes(payload.read_bytes()[:1024] + b"\x00" * 4096)
        manifest["compiled_sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    else:
        raise ValueError(f"Unknown mismatch case {case!r}.")

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return working


def run_mismatch(arguments: argparse.Namespace) -> dict[str, Any]:
    """Load a deliberately wrong artifact and record how badly it goes.

    A clear failure is an `ArtifactLoadError` naming the artifact and the fix. A
    poor one is anything else -- a driver error at first call, a segfault, or a
    wrong answer -- and the difference is the whole point of the case list.
    """
    case = arguments.case
    artifact_path = Path(arguments.artifact)
    working = artifact_path if case == "torch-version" else _mutate(case, artifact_path)

    import torch

    import lm7
    from lm7.errors import ArtifactLoadError

    record: dict[str, Any] = {
        "stage": "mismatch",
        "case": case,
        "artifact": str(artifact_path),
        "manifest_torch": json.loads((working / MANIFEST_NAME).read_text(encoding="utf-8")).get(
            "torch_version"
        ),
        "runtime_torch": torch.__version__,
        "environment": _environment(),
    }
    try:
        loaded = lm7.load_artifact(working)
    except ArtifactLoadError as error:
        record.update(
            {
                "outcome": "rejected",
                "clear": True,
                "error_type": type(error).__name__,
                "error": str(error)[:800],
            }
        )
        return record
    except BaseException as error:  # noqa: BLE001 - an unclear rejection is the finding
        record.update(
            {
                "outcome": "rejected",
                "clear": False,
                "error_type": type(error).__name__,
                "error": str(error)[:800],
            }
        )
        return record

    # It loaded. Whether that is correct depends on the case, so run it: a
    # mutation that survives load and then produces a right answer is a
    # non-event, and one that survives load and crashes the driver is not.
    saved = torch.load(
        artifact_path.parent / f"{artifact_path.name}.{REFERENCE_NAME}", weights_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tuple(tensor.to(device) for tensor in saved["inputs"])
    try:
        with torch.no_grad():
            output = loaded(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except BaseException as error:  # noqa: BLE001 - a late failure is the finding
        record.update(
            {
                "outcome": "loaded-then-failed",
                "clear": False,
                "error_type": type(error).__name__,
                "error": str(error)[:800],
            }
        )
        return record

    record.update({"outcome": "loaded-and-ran", "clear": None})
    record.update(_parity(output.detach().float().cpu(), saved["reference"]))
    return record


def _invoke(stage: list[str], *, python: str | None = None) -> dict[str, Any]:
    """Run one stage in its own interpreter and return its JSON record.

    Wall time is measured here rather than in the child because interpreter
    startup is part of what a reload costs and the child cannot see it.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        destination = Path(handle.name)
    # `--json` belongs to the top-level parser, so it has to precede the
    # subcommand name -- argparse hands everything after that to the subparser.
    command = [
        python or sys.executable,
        str(Path(__file__).resolve()),
        "--json",
        str(destination),
        *stage,
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    wall_seconds = time.perf_counter() - started
    try:
        record = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = {
            "works": False,
            "error_type": "NoResult",
            "error": (completed.stderr or completed.stdout)[-800:],
        }
    finally:
        destination.unlink(missing_ok=True)
    record["process_wall_seconds"] = wall_seconds
    record["returncode"] = completed.returncode
    if completed.returncode != 0 and "error" not in record:
        record["error"] = (completed.stderr or completed.stdout)[-800:]
    return record


def run_driver(arguments: argparse.Namespace) -> dict[str, Any]:
    """Export once, then reload it from scratch in as many ways as we can."""
    results_dir = Path(arguments.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifact = results_dir / f"{arguments.model}.aot.lm7"
    shared = [
        "--model",
        arguments.model,
        "--dtype",
        arguments.dtype,
        "--target",
        arguments.target,
        "--artifact",
        str(artifact),
    ]

    summary: dict[str, Any] = {"model": arguments.model, "dtype": arguments.dtype, "stages": {}}
    export = _invoke(["export", *shared])
    summary["stages"]["export"] = export
    _report("export", export)
    if not export.get("works"):
        return summary

    for api in ("lm7", "torch"):
        for cache in ("cold", "warm"):
            stage = [
                "load",
                *shared,
                "--api",
                api,
                "--repeats",
                str(arguments.repeats),
                *(["--evict"] if cache == "cold" else []),
            ]
            record = _invoke(stage)
            summary["stages"][f"load-{api}-{cache}"] = record
            _report(f"load {api} {cache}", record)

    for cache in ("cold", "warm"):
        stage = [
            "jit",
            *shared,
            "--repeats",
            str(arguments.repeats),
            *(["--clear-inductor-cache"] if cache == "cold" else []),
        ]
        record = _invoke(stage)
        summary["stages"][f"jit-{cache}"] = record
        _report(f"jit {cache}", record)

    for case in MISMATCH_CASES:
        python = arguments.other_python if case == "torch-version" else None
        if case == "torch-version" and not python:
            continue
        record = _invoke(["mismatch", *shared, "--case", case], python=python)
        summary["stages"][f"mismatch-{case}"] = record
        _report(f"mismatch {case}", record)

    destination = results_dir / f"{arguments.model}__lifecycle.json"
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {destination}")
    return summary


def _report(label: str, record: dict[str, Any]) -> None:
    if record.get("stage") == "mismatch":
        print(f"{label:>26}  {record.get('outcome')}  clear={record.get('clear')}  ")
        print(f"{'':>26}  {str(record.get('error', ''))[:160]}")
        return
    if not record.get("works"):
        print(f"{label:>26}  FAILED {record.get('error_type')}: {str(record.get('error'))[:200]}")
        return
    parts = []
    if record.get("process_wall_seconds") is not None:
        parts.append(f"wall={record['process_wall_seconds']:6.2f} s")
    if record.get("load_ms") is not None:
        parts.append(f"load={record['load_ms'] / 1000:6.3f} s")
    if record.get("to_first_inference_ms") is not None:
        parts.append(f"first_inference={record['to_first_inference_ms'] / 1000:6.2f} s")
    if record.get("latency_median_ms") is not None:
        parts.append(f"median={record['latency_median_ms']:7.3f} ms")
    if record.get("export_seconds") is not None:
        parts.append(f"build={record['export_seconds']:6.2f} s")
    if record.get("artifact_bytes") is not None:
        parts.append(f"size={record['artifact_bytes'] / 1e9:5.2f} GB")
    if record.get("max_abs_diff") is not None:
        parts.append(f"diff={record['max_abs_diff']:.3e}")
    print(f"{label:>26}  " + "  ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write this stage's record here")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(target_parser: argparse.ArgumentParser) -> None:
        target_parser.add_argument("--model", choices=MODELS, default="llama32-1b")
        target_parser.add_argument(
            "--dtype", choices=("float32", "float16", "bfloat16"), default="float16"
        )
        target_parser.add_argument("--target", default="nvidia")
        target_parser.add_argument("--artifact", default="artifacts/aoti/model.aot.lm7")

    export_parser = subparsers.add_parser("export", help="process A: write the artifact")
    add_common(export_parser)

    load_parser = subparsers.add_parser("load", help="process B: reload and validate")
    add_common(load_parser)
    load_parser.add_argument("--api", choices=("lm7", "torch"), default="lm7")
    load_parser.add_argument("--evict", action="store_true", help="drop the page cache first")
    load_parser.add_argument("--warmup", type=int, default=5)
    load_parser.add_argument("--repeats", type=int, default=20)

    jit_parser = subparsers.add_parser("jit", help="baseline: compile in this process")
    add_common(jit_parser)
    jit_parser.add_argument("--clear-inductor-cache", action="store_true")
    jit_parser.add_argument("--repeats", type=int, default=20)

    mismatch_parser = subparsers.add_parser("mismatch", help="load a wrong artifact on purpose")
    add_common(mismatch_parser)
    mismatch_parser.add_argument("--case", choices=MISMATCH_CASES, required=True)

    run_parser = subparsers.add_parser("run", help="run every stage, each in its own process")
    add_common(run_parser)
    run_parser.add_argument("--results-dir", type=Path, default=Path("artifacts/aoti"))
    run_parser.add_argument("--repeats", type=int, default=20)
    run_parser.add_argument(
        "--other-python",
        default=None,
        help="interpreter with a different PyTorch, for the torch-version case",
    )

    arguments = parser.parse_args()
    handlers = {
        "export": run_export,
        "load": run_load,
        "jit": run_jit,
        "mismatch": run_mismatch,
        "run": run_driver,
    }
    try:
        record = handlers[arguments.command](arguments)
    except BaseException as error:  # noqa: BLE001 - a failed stage is a result
        record = {
            "stage": arguments.command,
            "works": False,
            "error_type": type(error).__name__,
            "error": str(error)[:800],
            "traceback": traceback.format_exc()[-1500:],
        }
    if arguments.json:
        arguments.json.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if arguments.command != "run":
        _report(arguments.command, record)
    if not record.get("works", True) and arguments.command != "mismatch":
        sys.exit(1)


if __name__ == "__main__":
    main()
