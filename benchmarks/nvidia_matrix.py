"""Run one cell of the NVIDIA backend compatibility matrix and write it as JSON.

One cell per process, deliberately. Compiler backends fail in ways that do not
stay inside a `try`: TensorRT and Inductor can abort the interpreter, a large
model can exhaust VRAM in a way that poisons the CUDA context for everything
after it, and Dynamo state leaks between compilations in the same process. A
crashed cell should cost that cell, so the driver loop invokes this script once
per (model, path) and each result lands on disk before the next one starts.

    python benchmarks/nvidia_matrix.py --model smollm2 --path inductor \
        --results-dir artifacts/matrix

The paths do not all live in one environment. `tensorrt` pins PyTorch 2.12 and
`onnxruntime` wants the GPU build that must not be installed beside the CPU one,
so the matrix is assembled from three venvs and every result records the torch
and backend versions it ran under. Rows from different environments are
comparable for "does it work" and only roughly comparable for latency.

Two rows in the requested matrix name a preset rather than a backend, and are
split so that each changes one variable:

    inductor + max-autotune   -> compile_mode="max-autotune-no-cudagraphs"
    inductor + CUDA Graphs    -> compile_mode="reduce-overhead"

`max-autotune` on its own turns on both, which would confound them.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import torch

import lm7
from lm7.detection import resolve_target, synchronize

HF_MODELS = {
    "bert": "bert-base-uncased",
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "llama32-1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama31-8b": "unsloth/Llama-3.1-8B-Instruct",
}
MODELS = ("mlp", "resnet18", *HF_MODELS)

# compile_mode per path; None means the backend takes no Inductor preset.
PATHS: dict[str, dict[str, Any]] = {
    "eager": {"backend": "eager"},
    "inductor": {"backend": "inductor"},
    "inductor-max-autotune": {
        "backend": "inductor",
        "compile_mode": "max-autotune-no-cudagraphs",
    },
    "inductor-cudagraphs": {"backend": "inductor", "compile_mode": "reduce-overhead"},
    "aot-inductor": {"backend": "aot_inductor", "export": True},
    "tensorrt": {"backend": "tensorrt"},
    "tensorrt-export": {"backend": "tensorrt", "export": True},
    "onnxruntime": {"backend": "onnxruntime"},
}

PROMPT = "The capital of France is"


class _TensorOut(torch.nn.Module):
    """Tensor in, tensor out.

    Every export backend goes through `torch.export`, which cannot round-trip a
    `BaseModelOutput` dataclass, and several runtimes accept positional tensors
    only. Wrapping here keeps the compiled and exported paths measuring the same
    computation instead of the compiled path measuring more of it.
    """

    def __init__(self, model: torch.nn.Module, kind: str) -> None:
        super().__init__()
        self.model = model
        self.kind = kind

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        if self.kind == "causal-lm":
            output = self.model(input_ids=args[0], attention_mask=args[1], use_cache=False)
            return output.logits
        if self.kind == "bert":
            output = self.model(input_ids=args[0], attention_mask=args[1])
            return output.last_hidden_state
        return self.model(*args)


def build(name: str, dtype: torch.dtype) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    if name == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 1024),
        ).eval()
        return _TensorOut(model, "plain").to(dtype=dtype), (torch.randn(8, 1024, dtype=dtype),)
    if name == "resnet18":
        from torchvision.models import resnet18

        model = resnet18().eval()
        return (
            _TensorOut(model, "plain").to(dtype=dtype),
            (torch.randn(8, 3, 224, 224, dtype=dtype),),
        )

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    model_id = HF_MODELS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if name == "bert":
        model = AutoModel.from_pretrained(model_id, dtype=dtype).eval()
        kind = "bert"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
        kind = "causal-lm"
    encoded = tokenizer(PROMPT, return_tensors="pt")
    return (
        _TensorOut(model, kind).eval(),
        (encoded["input_ids"], encoded["attention_mask"]),
    )


def _versions(backend: str) -> dict[str, str | None]:
    versions: dict[str, str | None] = {"torch": torch.__version__}
    for module, key in (("torch_tensorrt", "tensorrt"), ("onnxruntime", "onnxruntime")):
        if backend in {"tensorrt", "onnxruntime"}:
            try:
                versions[key] = __import__(module).__version__
            except Exception:  # noqa: BLE001 - a missing version is not a failure
                versions[key] = None
    return versions


def _measure(
    callable_model: Any,
    inputs: tuple[torch.Tensor, ...],
    target: Any,
    *,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    torch.cuda.reset_peak_memory_stats()
    synchronize(target)
    started = time.perf_counter()
    with torch.no_grad():
        output = callable_model(*inputs)
    synchronize(target)
    first_call_ms = (time.perf_counter() - started) * 1000.0

    # Copy the output off the device before another call can run. CUDA Graphs
    # replay into the *same* output buffer every time, so holding the tensor and
    # reading it after the timing loop raises "accessing tensor output of
    # CUDAGraphs that has been overwritten by a subsequent run". An earlier
    # revision did exactly that and recorded `reduce-overhead` as a hard failure
    # when the backend was working correctly -- the harness was the bug.
    captured = output.detach().float().cpu()

    for _ in range(warmup):
        with torch.no_grad():
            callable_model(*inputs)
    synchronize(target)

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        with torch.no_grad():
            callable_model(*inputs)
        synchronize(target)
        samples.append((time.perf_counter() - started) * 1000.0)

    return {
        "first_call_ms": first_call_ms,
        "latency_median_ms": statistics.median(samples),
        "latency_min_ms": min(samples),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }, captured


def _parity(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    if actual.shape != reference.shape:
        return {"parity": "shape-mismatch", "max_abs_diff": None, "argmax_agrees": None}
    difference = (actual - reference).abs().max().item()
    return {
        "parity": "ok",
        "max_abs_diff": difference,
        # For a causal LM the last row is the next-token distribution; for other
        # models this is still the cheapest single check that the output did not
        # merely stay close on average while moving where it counts.
        "argmax_agrees": bool(
            actual.reshape(-1, actual.shape[-1])[-1].argmax()
            == reference.reshape(-1, reference.shape[-1])[-1].argmax()
        ),
    }


def run_cell(arguments: argparse.Namespace) -> dict[str, Any]:
    specification = PATHS[arguments.path]
    backend = specification["backend"]
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        arguments.dtype
    ]
    target = resolve_target(arguments.target)
    record: dict[str, Any] = {
        "model": arguments.model,
        "path": arguments.path,
        "backend": backend,
        "compile_mode": specification.get("compile_mode"),
        "exports": bool(specification.get("export")),
        "dtype": arguments.dtype,
        "target": str(target),
        "host": platform.node(),
        "versions": _versions(backend),
    }

    model, inputs = build(arguments.model, dtype)
    record["parameter_count"] = sum(p.numel() for p in model.parameters())

    # The reference is eager on the GPU, so parity measures the backend and not
    # the device transfer or the dtype.
    reference_model = lm7.compile(
        model, target=target, backend="eager", transfers="automatic", fallback="error", cache=False
    )
    with torch.no_grad():
        reference = reference_model(*inputs).detach().float().cpu()

    options = (
        {"compile_mode": specification["compile_mode"]}
        if specification.get("compile_mode")
        else None
    )

    if specification.get("export"):
        output_path = Path(arguments.results_dir) / "artifacts" / f"{arguments.model}.{backend}.lm7"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        lm7.export(
            model,
            args=inputs,
            target=target,
            output=output_path,
            backend=backend,
            options=options,
        )
        record["export_seconds"] = time.perf_counter() - started
        loaded = lm7.load_artifact(output_path)
        record["artifact_reload"] = "ok"
        measured, actual = _measure(
            loaded, inputs, target, warmup=arguments.warmup, repeats=arguments.repeats
        )
        record["artifact_bytes"] = sum(
            item.stat().st_size for item in output_path.rglob("*") if item.is_file()
        )
    else:
        compiled = lm7.compile(
            model,
            target=target,
            backend=backend,
            transfers="automatic",
            fallback="error",
            cache=False,
            options=options,
        )
        record["selected_backend"] = compiled.selected_backend
        record["artifact_reload"] = None
        measured, actual = _measure(
            compiled, inputs, target, warmup=arguments.warmup, repeats=arguments.repeats
        )

    record.update(measured)
    record.update(_parity(actual, reference))
    record["works"] = True
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one NVIDIA compatibility matrix cell.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--path", choices=sorted(PATHS), required=True)
    parser.add_argument("--target", default="nvidia")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts/matrix"))
    arguments = parser.parse_args()

    arguments.results_dir.mkdir(parents=True, exist_ok=True)
    destination = arguments.results_dir / f"{arguments.model}__{arguments.path}.json"

    try:
        record = run_cell(arguments)
    except BaseException as error:  # noqa: BLE001 - a failed cell is a result
        record = {
            "model": arguments.model,
            "path": arguments.path,
            "backend": PATHS[arguments.path]["backend"],
            "dtype": arguments.dtype,
            "host": platform.node(),
            "works": False,
            "error_type": type(error).__name__,
            "error": str(error)[:600],
            "traceback": traceback.format_exc()[-1200:],
        }

    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = "ok" if record.get("works") else f"FAIL {record.get('error_type')}"
    latency = record.get("latency_median_ms")
    vram = record.get("peak_vram_bytes")
    print(
        f"{arguments.model:>12} {arguments.path:>22}  {status:<28}"
        + (f"  median={latency:9.3f} ms" if latency is not None else "")
        + (
            f"  first={record['first_call_ms'] / 1000:7.2f} s"
            if record.get("first_call_ms")
            else ""
        )
        + (f"  vram={vram / 1e9:6.2f} GB" if vram else "")
        + (f"  diff={record['max_abs_diff']:.3e}" if record.get("max_abs_diff") is not None else "")
    )


if __name__ == "__main__":
    main()
