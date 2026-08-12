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
from lm7.backends.inductor import cudagraph_skips, cudagraphs_requested
from lm7.detection import (
    compute_capability,
    cuda_build_targets,
    precision_support,
    resolve_target,
    synchronize,
)

HF_MODELS = {
    "bert": "bert-base-uncased",
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "llama32-1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama31-8b": "unsloth/Llama-3.1-8B-Instruct",
    # The dense validation ladder, reachable by name and not yet measured here.
    # See docs/limitations.md#model-coverage. Mistral-7B is the first dense 7B
    # in these dicts -- the Mixtral below it is sparse, and the two are not
    # interchangeable as "a 7B".
    "lfm25-350m": "LiquidAI/LFM2.5-350M",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    # Sparse MoE. OLMoE is 6.92B total / 1B active and fits anywhere; the other
    # two are here so a large card can reach them, and they are deliberately not
    # in the `moe` plan -- Mixtral-8x7B peaks at 93.4 GB in BF16, which is more
    # than an 80 GB H100 has. See docs/limitations.md for the measurements.
    "olmoe-1b-7b": "allenai/OLMoE-1B-7B-0924-Instruct",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

# Hand-built two-layer MoE configs, mirroring benchmarks/moe.py and
# examples/sparse_moe.py. They cost no download and exercise the routing, which
# is the part that behaves differently from a dense model. Dimensions are
# multiples of 16 because the transformers 5.x `grouped_mm` path requires
# strides that are multiples of 16 bytes and raises in eager otherwise.
_TINY_MOE = {
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 64,
}
TINY_MOE_MODELS = ("mixtral-tiny", "olmoe-tiny")

MODELS = ("mlp", "resnet18", *HF_MODELS, *TINY_MOE_MODELS)
MOE_MODELS = (*TINY_MOE_MODELS, "olmoe-1b-7b", "qwen3-30b-a3b", "mixtral-8x7b")

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
    # FP8 rides the default Inductor path -- what changes is the model handed to
    # it, not the compiler. Both are causal-LM only: the layer selector matches
    # on `.mlp.` module paths, so `mlp`, `resnet18` and `bert` have nothing to
    # convert and are refused rather than silently measured unquantized.
    "inductor-fp8": {"backend": "inductor", "quantize": "fp8"},
    "inductor-fp8-dynamic": {"backend": "inductor", "quantize": "fp8-dynamic"},
}

CAUSAL_LM_MODELS = ("smollm2", "llama32-1b", "llama31-8b", *MOE_MODELS)

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
        # The wrapped model is already in eval mode, but a freshly constructed
        # Module is not, so LM7 warned "compiling a model in training mode" on
        # every cell. The child's mode is what decides BatchNorm and dropout, so
        # this was cosmetic -- and a warning nobody can act on is worse than no
        # warning, because the next real one gets ignored too.
        self.eval()

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

    if name in TINY_MOE_MODELS:
        return _build_tiny_moe(name, dtype)

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


def _build_tiny_moe(
    name: str, dtype: torch.dtype
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    """A two-layer sparse MoE with random weights and no tokenizer.

    Random weights are fine because every metric here is either mechanical
    (latency, VRAM, CUDA Graph capture) or a parity check against eager on the
    same weights. They are *not* fine for accuracy, which is why no accuracy
    number is reported for these two.
    """
    if name == "olmoe-tiny":
        from transformers import OlmoeConfig, OlmoeForCausalLM

        built = OlmoeForCausalLM(OlmoeConfig(num_experts=8, num_experts_per_tok=2, **_TINY_MOE))
    else:
        from transformers import MixtralConfig, MixtralForCausalLM

        built = MixtralForCausalLM(
            MixtralConfig(num_local_experts=4, num_experts_per_tok=2, **_TINY_MOE)
        )
    built = built.eval().to(dtype=dtype)
    built.config.use_cache = False
    input_ids = torch.randint(0, _TINY_MOE["vocab_size"], (1, 16))
    return _TensorOut(built, "causal-lm"), (input_ids, torch.ones_like(input_ids))


def _versions(backend: str) -> dict[str, str | None]:
    versions: dict[str, str | None] = {"torch": torch.__version__}
    for module, key in (("torch_tensorrt", "tensorrt"), ("onnxruntime", "onnxruntime")):
        if backend in {"tensorrt", "onnxruntime"}:
            try:
                versions[key] = __import__(module).__version__
            except Exception:  # noqa: BLE001 - a missing version is not a failure
                versions[key] = None
    return versions


def environment(target_name: str = "nvidia") -> dict[str, Any]:
    """What the machine and the install are, in one block every report carries.

    A latency number is only comparable against another one taken on the same
    silicon *and* the same wheel, and this repo has already been bitten by the
    second half: `torch 2.12` and `torch 2.13` live in different venvs here, and
    rows from them are comparable for "does it work" and only roughly for speed.
    Recording both makes that checkable after the fact instead of remembered.

    `supported_precisions` is the hardware's answer and `cuda_build` is the
    wheel's -- see `lm7.detection.cuda_build_targets` for why those differ.
    """

    def module_version(name: str) -> str | None:
        try:
            return str(__import__(name).__version__)
        except Exception:  # noqa: BLE001 - an absent optional package is not a failure
            return None

    target = resolve_target(target_name)
    capability = compute_capability(target)
    precisions = precision_support(target)
    device_name = None
    driver = None
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(target.ordinal or 0)
        # `driver_version` exists only on newer torch; the CUDA runtime version
        # is always there and is the more portable of the two.
        driver = getattr(torch.version, "cuda", None)
    return {
        "gpu": device_name,
        "compute_capability": f"sm{capability}" if capability is not None else None,
        "driver": driver,
        "cuda": getattr(torch.version, "cuda", None),
        "pytorch": torch.__version__,
        "triton": module_version("triton"),
        "torchao": module_version("torchao"),
        # `native` and `emulated` both mean "this runs"; only `absent` does not.
        "supported_precisions": {
            name: state != "absent" for name, state in sorted(precisions.items())
        },
        "cuda_build": cuda_build_targets(target),
        "host": platform.node(),
        "platform": platform.platform(),
    }


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

    quantize = specification.get("quantize")
    record["quantize"] = quantize
    if quantize is not None:
        if arguments.model not in CAUSAL_LM_MODELS:
            raise SystemExit(
                f"{quantize} converts linears matched by module path (`.mlp.`), which "
                f"{arguments.model} does not have. Causal LMs only: "
                f"{', '.join(CAUSAL_LM_MODELS)}."
            )
        from lm7.huggingface import _apply_quantization

        # `_TensorOut` wraps the real module; quantize the model itself so the
        # wrapper's forward still sees the same signature.
        started = time.perf_counter()
        _, record["quantized_modules"] = _apply_quantization(model.model, target, quantize)
        record["quantize_seconds"] = time.perf_counter() - started

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
        # Requesting CUDA Graphs is not getting them: Inductor declines capture
        # for mutated inputs, dynamic shapes and more, bumping a skip counter
        # instead of raising. Recording both halves is the only way a row says
        # whether graphs were actually used -- see benchmarks/cudagraphs.py.
        mode = specification.get("compile_mode")
        record["cudagraphs_requested"] = cudagraphs_requested(mode, {}) if mode else False
        skips_before = cudagraph_skips()
        measured, actual = _measure(
            compiled, inputs, target, warmup=arguments.warmup, repeats=arguments.repeats
        )
        record["cudagraph_skips"] = cudagraph_skips() - skips_before
        record["cudagraphs_active"] = (
            record["cudagraphs_requested"] and record["cudagraph_skips"] == 0
        )

    record.update(measured)
    record.update(_parity(actual, reference))
    record["works"] = True
    return record


def plan(name: str) -> list[list[str]]:
    """Cell lists, as argument vectors a driver can hand straight to this script.

    Split by environment, because they cannot share one. `core` and `quant` run
    against the default CUDA venv; `tensorrt` needs the pinned torch 2.12 pair
    and `onnxruntime` the GPU build that must not sit beside the CPU one. Rows
    from different venvs compare for "does it work" and only roughly for speed,
    which is why every record carries its own versions.
    """
    portable = ("eager", "inductor", "inductor-cudagraphs", "inductor-max-autotune")
    if name == "core":
        return [
            ["--model", model, "--path", path]
            for model in ("mlp", "resnet18", "bert", "smollm2", "llama32-1b")
            for path in portable
        ]
    if name == "artifacts":
        return [
            ["--model", model, "--path", "aot-inductor"]
            for model in ("mlp", "resnet18", "bert", "smollm2", "llama32-1b")
        ]
    if name == "quant":
        return [
            ["--model", model, "--path", path, "--dtype", "bfloat16"]
            for model in CAUSAL_LM_MODELS
            for path in ("inductor", "inductor-fp8", "inductor-fp8-dynamic")
        ]
    if name == "large":
        return [["--model", "llama31-8b", "--path", path] for path in portable]
    if name == "moe":
        # OLMoE-1B-7B is the largest MoE that fits an 80 GB card; Mixtral-8x7B
        # (93.4 GB in BF16) and Qwen3-30B-A3B are reachable by name on a bigger
        # one.
        #
        # One FP8 cell per architecture, on the tiny configs only, because the
        # answer is a refusal and costs nothing to record: transformers 5.x
        # replaced per-expert `nn.Linear` modules with the parameter tensors
        # `grouped_mm` consumes, so a two-layer MoE has nine linears -- attention
        # plus `lm_head` -- and none under `.mlp.`. LM7 raises rather than
        # converting nothing, and pinning that refusal is worth one cheap cell.
        # Repeating it on a 6.92B download would buy the same answer.
        return [
            ["--model", model, "--path", path, "--dtype", "bfloat16"]
            for model in (*TINY_MOE_MODELS, "olmoe-1b-7b")
            for path in ("eager", "inductor", "inductor-cudagraphs")
        ] + [
            ["--model", model, "--path", "inductor-fp8", "--dtype", "bfloat16"]
            for model in TINY_MOE_MODELS
        ]
    if name == "tensorrt":
        return [
            ["--model", model, "--path", path]
            for model in ("mlp", "resnet18", "bert", "smollm2")
            for path in ("tensorrt", "tensorrt-export")
        ]
    if name == "onnxruntime":
        return [
            ["--model", model, "--path", "onnxruntime"]
            for model in ("mlp", "resnet18", "bert", "smollm2")
        ]
    raise SystemExit(
        f"Unknown plan {name!r}; choose core, artifacts, quant, large, tensorrt or onnxruntime."
    )


def summarize(directory: Path) -> str:
    records = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))
    ]
    cells = [record for record in records if "model" in record]
    if not cells:
        return f"No cells in {directory}."
    lines = [f"{len(cells)} cells, {sum(1 for c in cells if c.get('works'))} ok"]
    header = f"{'model':>12} {'path':>22} {'status':<10} {'median ms':>10} {'vram GB':>8}  parity"
    lines += [header, "-" * len(header)]
    for cell in sorted(cells, key=lambda c: (c["model"], c["path"])):
        latency = cell.get("latency_median_ms")
        vram = cell.get("peak_vram_bytes")
        diff = cell.get("max_abs_diff")
        # Every column is rendered to a string before it is padded: a width spec
        # applied straight to None raises, which turned a summary of eight good
        # cells into a traceback.
        latency_text = "-" if latency is None else f"{latency:.3f}"
        vram_text = "-" if not vram else f"{vram / 1e9:.2f}"
        diff_text = "-" if diff is None else f"{diff:.2e}"
        status = "ok" if cell.get("works") else f"FAIL {cell.get('error_type', '')}".strip()
        lines.append(
            f"{cell['model']:>12} {cell['path']:>22} "
            f"{status:<10} {latency_text:>10} {vram_text:>8}  {diff_text}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one NVIDIA compatibility matrix cell.")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--path", choices=sorted(PATHS))
    parser.add_argument(
        "--environment",
        action="store_true",
        help="print the hardware/install block every report carries, and exit",
    )
    parser.add_argument("--plan", help="print a cell list, one per line, and exit")
    parser.add_argument("--summarize", type=Path, help="summarize a results directory and exit")
    parser.add_argument("--target", default="nvidia")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts/matrix"))
    arguments = parser.parse_args()

    if arguments.environment:
        print(json.dumps(environment(arguments.target), indent=2, sort_keys=True))
        return
    if arguments.plan:
        for cell in plan(arguments.plan):
            print(" ".join(cell))
        return
    if arguments.summarize:
        print(summarize(arguments.summarize))
        return
    if not arguments.model or not arguments.path:
        raise SystemExit("--model and --path are required unless --environment/--plan/--summarize.")

    arguments.results_dir.mkdir(parents=True, exist_ok=True)
    destination = arguments.results_dir / f"{arguments.model}__{arguments.path}.json"

    # Written once per directory so a set of results is self-describing months
    # later. `summarize` skips it: it has no "model" key.
    manifest = arguments.results_dir / "environment.json"
    if not manifest.exists():
        try:
            manifest.write_text(
                json.dumps(environment(arguments.target), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception as error:  # noqa: BLE001 - a missing manifest must not cost the cell
            print(f"environment.json not written: {type(error).__name__}: {error}")

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
