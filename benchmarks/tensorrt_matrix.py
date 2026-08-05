"""Validate LM7's TensorRT path across models, precisions, shapes and reload.

`benchmarks/nvidia_matrix.py` answers "which NVIDIA backend works" for one model
at one shape. This script answers the narrower TensorRT question along the axes
that decide whether the backend is usable at all: four model families, four
precisions, several batch sizes, dynamic sequence length, and whether a
serialized engine reloads in a process that never built it.

One cell per process, for the same reason `nvidia_matrix.py` is: a TensorRT
engine build can abort the interpreter or poison the CUDA context, and a crashed
cell should cost that cell rather than the sweep.

    python benchmarks/tensorrt_matrix.py --model bert --path tensorrt-export \
        --dtype float16 --batch-size 8 --seq-len 128 --results-dir artifacts/trt

    python benchmarks/tensorrt_matrix.py --plan core        # cell list, one per line
    python benchmarks/tensorrt_matrix.py --summarize artifacts/trt

Every cell measures the same backend two ways, because they are different
products: `tensorrt` compiles in-process through `torch.compile`, and
`tensorrt-export` builds the engine, serializes it into a `.lm7` artifact and
runs the reloaded module. The second is the one worth shipping and the first is
mostly a way to find out whether a model converts.
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
from lm7.detection import resolve_target, synchronize, torch_device

HF_MODELS = {
    "bert": "bert-base-uncased",
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
}
MODELS = ("mlp", "resnet18", "fp8-matmul", *HF_MODELS)
SEQUENCE_MODELS = frozenset(HF_MODELS)

PATHS: dict[str, dict[str, Any]] = {
    "eager": {"backend": "eager"},
    "inductor": {"backend": "inductor"},
    "tensorrt": {"backend": "tensorrt"},
    "tensorrt-export": {"backend": "tensorrt", "export": True},
    "aot-inductor": {"backend": "aot_inductor", "export": True},
}

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

PROMPT = "The capital of France is"

# Sequence lengths a dynamic cell is called at, in the order it calls them. The
# first is the shape the engine was built for; the rest are what a dynamic
# engine is supposed to absorb without a rebuild.
DYNAMIC_SEQUENCE_LENGTHS = (32, 16, 64, 128)


class _TensorOut(torch.nn.Module):
    """Tensor in, tensor out.

    `torch.export` cannot round-trip a transformers output dataclass and several
    runtimes take positional tensors only, so both the compiled and the exported
    path measure the same computation instead of one of them measuring more.
    """

    def __init__(self, model: torch.nn.Module, kind: str) -> None:
        super().__init__()
        self.model = model
        self.kind = kind

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        if self.kind == "causal-lm":
            return self.model(input_ids=args[0], attention_mask=args[1], use_cache=False).logits
        if self.kind == "bert":
            return self.model(input_ids=args[0], attention_mask=args[1]).last_hidden_state
        return self.model(*args)


class _ScaledMatmulStack(torch.nn.Module):
    """A stack of FP8 matmuls, written the only way the hardware offers them.

    `torch._scaled_mm` is the FP8 tensor-core entry point: it takes two
    `float8_e4m3fn` operands plus their scales and accumulates in a wider type.
    Nothing about it is TensorRT-specific, which is the point -- it isolates
    "does this stack convert FP8 arithmetic on this card" from every question
    about a real model's other 300 operators.

    Four layers rather than one, because a single matmul is a two-node graph and
    TensorRT's partitioner declines anything under `min_block_size`. A one-layer
    probe answers "the graph was too small" while looking like it answered
    "FP8 is unsupported", which is the confusion this whole file exists to avoid.
    """

    def __init__(self, size: int, out_dtype: torch.dtype, layers: int = 4) -> None:
        super().__init__()
        self.out_dtype = out_dtype
        self.layers = layers
        for index in range(layers):
            weight = torch.randn(size, size) / size**0.5
            self.register_buffer(f"weight{index}", weight.to(torch.float8_e4m3fn).t())
        self.register_buffer("scale_a", torch.tensor(1.0))
        self.register_buffer("scale_b", torch.tensor(1.0))

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        for index in range(self.layers):
            activation = torch._scaled_mm(
                activation.to(torch.float8_e4m3fn),
                getattr(self, f"weight{index}"),
                scale_a=self.scale_a,
                scale_b=self.scale_b,
                out_dtype=self.out_dtype,
            )
        return activation


def _token_inputs(model_id: str, batch_size: int, sequence_length: int) -> tuple[torch.Tensor, ...]:
    """A (batch, sequence) prompt, tiled from a real one rather than random ids.

    Random token ids would exercise the same shapes, but they put the model far
    off distribution, and a next-token check on garbage tells a reader nothing.
    Tiling a real prompt keeps the argmax meaningful at every length.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"][0]
    repeated = ids.repeat((sequence_length // len(ids)) + 1)[:sequence_length]
    input_ids = repeated.unsqueeze(0).expand(batch_size, sequence_length).contiguous()
    return input_ids, torch.ones_like(input_ids)


def build(
    name: str,
    dtype: torch.dtype,
    *,
    batch_size: int,
    sequence_length: int,
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    if name == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(1024, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 1024),
        ).eval()
        return (
            _TensorOut(model, "plain").to(dtype=dtype),
            (torch.randn(batch_size, 1024, dtype=dtype),),
        )
    if name == "resnet18":
        from torchvision.models import resnet18

        model = resnet18().eval()
        return (
            _TensorOut(model, "plain").to(dtype=dtype),
            (torch.randn(batch_size, 3, 224, 224, dtype=dtype),),
        )
    if name == "fp8-matmul":
        model = _ScaledMatmulStack(1024, dtype).eval()
        return model, (torch.randn(batch_size, 1024, dtype=dtype),)

    from transformers import AutoModel, AutoModelForCausalLM

    model_id = HF_MODELS[name]
    if name == "bert":
        model = AutoModel.from_pretrained(model_id, dtype=dtype).eval()
        kind = "bert"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
        kind = "causal-lm"
    return _TensorOut(model, kind).eval(), _token_inputs(model_id, batch_size, sequence_length)


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"torch": torch.__version__}
    for attribute, module in (("torch_tensorrt", "torch_tensorrt"), ("tensorrt", "tensorrt")):
        try:
            versions[attribute] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - a missing version is not a failure
            versions[attribute] = None
    try:
        versions["torchao"] = __import__("torchao").__version__
    except Exception:  # noqa: BLE001
        versions["torchao"] = None
    return versions


def _device_facts() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"sm{major}{minor}",
    }


_ENGINE_BUILDS = [0]


def _count_engine_builds() -> bool:
    """Make every TensorRT engine build in this process countable.

    A cell that reports "ok" has proved that a callable ran and agreed with
    eager, and that is not the same as proving TensorRT did any of it: the
    partitioner declines graphs below `min_block_size` and hands the untouched
    graph back without raising, so a model can compile, run, match eager exactly
    and contain no engine. Counting builds is what separates the two, and
    `TRTInterpreter.run` is the single funnel both the JIT and the export path go
    through.
    """
    try:
        from torch_tensorrt.dynamo.conversion._TRTInterpreter import TRTInterpreter
    except Exception:  # noqa: BLE001 - no Torch-TensorRT means nothing to count
        return False
    original = TRTInterpreter.run

    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        _ENGINE_BUILDS[0] += 1
        return original(self, *args, **kwargs)

    TRTInterpreter.run = counted  # type: ignore[method-assign]
    return True


def _graph_coverage(module: Any) -> dict[str, Any] | None:
    """How much of a module's graph TensorRT owns, or None if it has no graph.

    Complements the build counter, and answers a different question: the counter
    says an engine was built in this process, this says what is present in the
    thing about to be called. Only the second survives serialization, so it is
    what a reloaded artifact can be judged on.

    Engine *count* alone turned out to be misleading. A BERT artifact built at
    the default `min_block_size=5` and one built at 1 both report a single
    engine, and the second is 5x faster -- because the first left a pile of
    operators running in PyTorch around it. `fallback_ops` is the number that
    distinguishes them.
    """
    graph = getattr(module, "graph", None)
    if graph is None:
        return None
    engines = 0
    fallback = 0
    for node in graph.nodes:
        target = str(node.target)
        if (
            node.op == "call_module"
            and "_run_on_acc" in target
            or node.op == "call_function"
            and "execute_engine" in target
        ):
            engines += 1
        elif node.op in ("call_function", "call_module") and "_guards_fn" not in target:
            fallback += 1
    return {"engines": engines, "fallback_ops": fallback}


def _compilation_count() -> int | None:
    """How many graphs Dynamo has compiled, process-wide.

    A dynamic cell wants to know whether a new sequence length triggered another
    compile, and the wall clock alone cannot separate "rebuilt the engine" from
    "the GPU was busy". Reading a private counter is acceptable here because a
    missing counter degrades the row to timings rather than failing it.
    """
    try:
        from torch._dynamo.utils import counters

        # `unique_graphs` is the one that means "Dynamo produced another graph".
        # The rest of `stats` counts captured calls and cache lookups, which move
        # on every call and would report a recompile that never happened.
        return int(counters["stats"]["unique_graphs"])
    except Exception:  # noqa: BLE001
        return None


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

    # Copy off the device before another call can run: CUDA Graphs replay into
    # the same output buffer, so holding the tensor across the timing loop reads
    # whatever the last call wrote. See the note in benchmarks/nvidia_matrix.py.
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

    samples.sort()
    return {
        "first_call_ms": first_call_ms,
        "latency_median_ms": statistics.median(samples),
        "latency_min_ms": samples[0],
        "latency_p95_ms": samples[min(int(len(samples) * 0.95), len(samples) - 1)],
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }, captured


def _parity(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    if actual.shape != reference.shape:
        return {"parity": "shape-mismatch", "max_abs_diff": None, "argmax_agrees": None}
    rows = actual.reshape(-1, actual.shape[-1])
    reference_rows = reference.reshape(-1, reference.shape[-1])
    return {
        "parity": "ok",
        "max_abs_diff": (actual - reference).abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(rows[-1], reference_rows[-1], dim=0).item(),
        "argmax_agrees": bool(rows[-1].argmax() == reference_rows[-1].argmax()),
    }


def _quantize(model: torch.nn.Module, target: Any, mode: str) -> dict[str, Any]:
    from lm7.huggingface import _apply_quantization, normalize_quantization

    normalized = normalize_quantization(mode)
    # The wrapper is what LM7 compiles, but the quantization filters match on
    # module paths like "model.layers.0.mlp.gate_proj", so quantize the inner
    # module and let the wrapper keep its own name out of the filter's way.
    inner = model.model if isinstance(model, _TensorOut) else model
    elapsed_ms, converted = _apply_quantization(inner, target, normalized)
    return {"quantize_ms": elapsed_ms, "quantized_layers": converted}


def _artifact_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _parse_options(raw: list[str] | None) -> dict[str, Any]:
    """`--options min_block_size=1 truncate_double=true` into backend options."""
    parsed: dict[str, Any] = {}
    for item in raw or []:
        key, _, value = item.partition("=")
        if value.lower() in {"true", "false"}:
            parsed[key] = value.lower() == "true"
        else:
            try:
                parsed[key] = int(value)
            except ValueError:
                parsed[key] = value
    return parsed


def run_cell(arguments: argparse.Namespace) -> dict[str, Any]:
    specification = PATHS[arguments.path]
    backend = specification["backend"]
    dtype = DTYPES[arguments.dtype]
    target = resolve_target(arguments.target)
    record: dict[str, Any] = {
        "cell": cell_name(arguments),
        "model": arguments.model,
        "path": arguments.path,
        "backend": backend,
        "exports": bool(specification.get("export")),
        "dtype": arguments.dtype,
        "quantize": arguments.quantize,
        "batch_size": arguments.batch_size,
        "seq_len": arguments.seq_len if arguments.model in SEQUENCE_MODELS else None,
        "dynamic": bool(arguments.dynamic),
        "target": str(target),
        "host": platform.node(),
        "versions": _versions(),
        **_device_facts(),
    }

    model, inputs = build(
        arguments.model,
        dtype,
        batch_size=arguments.batch_size,
        sequence_length=arguments.seq_len,
    )
    record["parameter_count"] = sum(p.numel() for p in model.parameters())
    if arguments.quantize != "none":
        record.update(_quantize(model, target, arguments.quantize))

    # Inputs live on the GPU for every path, for two reasons. A reloaded engine
    # is device-bound and has no LM7 transfer wrapper in front of it, so it
    # rejects host tensors outright; and leaving the JIT paths to copy per call
    # would time a host-to-device transfer for them that the artifact paths
    # never pay. Both would make the comparison mean something else.
    inputs = tuple(tensor.to(torch_device(target)) for tensor in inputs)

    # Eager on the GPU, in the same dtype and quantization, so parity measures
    # the backend rather than the device transfer or the weight format.
    reference_model = lm7.compile(
        model, target=target, backend="eager", transfers="automatic", fallback="error", cache=False
    )
    with torch.no_grad():
        reference = reference_model(*inputs).detach().float().cpu()

    options: dict[str, Any] = dict(_parse_options(arguments.options))
    if arguments.dynamic:
        options["dynamic"] = True
    record["options"] = options or None
    record["engine_counter"] = _count_engine_builds()
    _ENGINE_BUILDS[0] = 0

    if specification.get("export"):
        output_path = Path(arguments.results_dir) / "artifacts" / f"{record['cell']}.lm7"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(f"{output_path} exists; remove it or use a fresh results dir.")
        started = time.perf_counter()
        lm7.export(
            model,
            args=inputs,
            target=target,
            output=output_path,
            backend=backend,
            options=options or None,
        )
        record["build_seconds"] = time.perf_counter() - started
        loaded = lm7.load_artifact(output_path)
        record["artifact_reload"] = "ok"
        record["artifact_bytes"] = _artifact_bytes(output_path)
        record["artifact_path"] = str(output_path)
        coverage = _graph_coverage(loaded.module()) or {}
        record["engines_in_artifact"] = coverage.get("engines")
        record["fallback_ops_in_artifact"] = coverage.get("fallback_ops")
        callable_model: Any = loaded
    else:
        compiled = lm7.compile(
            model,
            target=target,
            backend=backend,
            transfers="automatic",
            fallback="error",
            cache=False,
            options=options or None,
        )
        record["selected_backend"] = compiled.selected_backend
        callable_model = compiled

    measured, actual = _measure(
        callable_model, inputs, target, warmup=arguments.warmup, repeats=arguments.repeats
    )
    record.update(measured)
    if not specification.get("export"):
        # For a JIT path the first call *is* the build, and reporting it under
        # both names would invite reading one column as if it excluded the other.
        record["build_seconds"] = measured["first_call_ms"] / 1000.0
    record.update(_parity(actual, reference))

    # Counted here rather than before the measurement: `lm7.compile` is lazy, so
    # a JIT path builds its engine inside the first call above, and reading the
    # counter any earlier would report zero for every JIT row.
    record["engines_built"] = _ENGINE_BUILDS[0] if record["engine_counter"] else None
    # The claim a reader actually wants from a "tensorrt" row. False here with
    # works=True is the interesting outcome: the path ran and TensorRT was not
    # involved in running it.
    record["tensorrt_engaged"] = (
        None
        if backend != "tensorrt" or not record["engine_counter"]
        else bool(_ENGINE_BUILDS[0]) or bool(record.get("engines_in_artifact"))
    )
    compilations_before = _compilation_count()

    if (arguments.dynamic or arguments.sweep_shapes) and arguments.model in SEQUENCE_MODELS:
        record["shapes"] = _sweep_shapes(
            callable_model,
            arguments,
            target,
            compilations_before=compilations_before,
        )
    record["works"] = True
    return record


def _sweep_shapes(
    callable_model: Any,
    arguments: argparse.Namespace,
    target: Any,
    *,
    compilations_before: int | None,
) -> list[dict[str, Any]]:
    """Call one compiled module at several sequence lengths, in order.

    What matters is not only that each length produces the right answer but what
    reaching it cost. `recompiled` reports the Dynamo graph-count delta and
    `engines_built` the TensorRT delta, so "absorbed the shape" is separable from
    "quietly built a second engine for it" -- which have very different
    consequences for a server whose sequence lengths vary per request.
    """
    model_id = HF_MODELS[arguments.model]
    results = []
    running = compilations_before
    engines = _ENGINE_BUILDS[0]
    for length in DYNAMIC_SEQUENCE_LENGTHS:
        entry: dict[str, Any] = {"seq_len": length}
        try:
            inputs = _token_inputs(model_id, arguments.batch_size, length)
            inputs = tuple(tensor.to(torch_device(target)) for tensor in inputs)
            measured, _ = _measure(
                callable_model, inputs, target, warmup=2, repeats=max(arguments.repeats // 2, 5)
            )
            after = _compilation_count()
            entry.update(
                {
                    "works": True,
                    "first_call_ms": measured["first_call_ms"],
                    "latency_median_ms": measured["latency_median_ms"],
                    "recompiled": (
                        None if after is None or running is None else max(after - running, 0)
                    ),
                    "engines_built": _ENGINE_BUILDS[0] - engines,
                }
            )
            running = after
            engines = _ENGINE_BUILDS[0]
        except Exception as error:  # noqa: BLE001 - a shape that fails is a result
            entry.update(
                {"works": False, "error_type": type(error).__name__, "error": str(error)[:400]}
            )
        results.append(entry)
    return results


def _inputs_from_signature(signature: Any) -> tuple[torch.Tensor, ...]:
    """Rebuild callable inputs from the manifest, so reload needs no source model.

    The manifest records `("tuple", (("tensor", shape, dtype, stride, device), ...))`
    for the positional arguments. Only shape and dtype matter here -- this cell
    times loading and first inference, and checks parity nowhere -- but token ids
    and masks are filled with ones rather than zeros so a transformer is not
    handed an all-masked batch.
    """
    entries = signature[0][1]
    inputs = []
    for entry in entries:
        _kind, shape, dtype_name, *_ = entry
        dtype = getattr(torch, str(dtype_name).split(".")[-1])
        filler = torch.zeros if dtype.is_floating_point else torch.ones
        inputs.append(filler(*shape, dtype=dtype).to("cuda"))
    return tuple(inputs)


def reload_cell(arguments: argparse.Namespace) -> dict[str, Any]:
    """Load an artifact built by an earlier process and time first inference.

    This is the number that decides whether serializing an engine was worth it,
    and it cannot be measured in the process that built it: that process has the
    engine in memory, a warm CUDA context and a populated allocator.
    """
    target = resolve_target(arguments.target)
    path = Path(arguments.reload)
    record: dict[str, Any] = {
        "cell": f"{path.stem}__reload",
        "mode": "reload",
        "artifact_path": str(path),
        "host": platform.node(),
        "versions": _versions(),
        **_device_facts(),
    }
    started = time.perf_counter()
    loaded = lm7.load_artifact(path)
    record["load_seconds"] = time.perf_counter() - started
    record["backend"] = loaded.manifest.backend
    record["artifact_bytes"] = _artifact_bytes(path)
    coverage = _graph_coverage(loaded.module()) or {}
    record["engines_in_artifact"] = coverage.get("engines")
    record["fallback_ops_in_artifact"] = coverage.get("fallback_ops")

    inputs = _inputs_from_signature(loaded.manifest.input_signature)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        loaded(*inputs)
    synchronize(target)
    record["first_call_ms"] = (time.perf_counter() - started) * 1000.0
    record["time_to_first_inference_seconds"] = (
        record["load_seconds"] + record["first_call_ms"] / 1000.0
    )
    steady, _ = _measure(loaded, inputs, target, warmup=arguments.warmup, repeats=arguments.repeats)
    # The cold first call above is the point of this cell; `_measure` re-times it
    # warm, so keep the cold number and take only the steady state from here.
    record["latency_median_ms"] = steady["latency_median_ms"]
    record["latency_min_ms"] = steady["latency_min_ms"]
    record["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    record["works"] = True
    return record


def cell_name(arguments: argparse.Namespace) -> str:
    parts = [arguments.model, arguments.dtype, f"b{arguments.batch_size}"]
    if arguments.model in SEQUENCE_MODELS:
        parts.append(f"s{arguments.seq_len}")
    if arguments.quantize != "none":
        parts.append(arguments.quantize)
    if arguments.dynamic:
        parts.append("dyn")
    elif arguments.sweep_shapes:
        parts.append("static-sweep")
    parts += [item.replace("=", "") for item in arguments.options or []]
    parts.append(arguments.path)
    return "__".join(parts)


def _plan(name: str) -> list[list[str]]:
    """Cell lists, as argument vectors the driver can hand straight to the script."""

    def cell(model: str, path: str, **overrides: Any) -> list[str]:
        arguments = ["--model", model, "--path", path]
        for key, value in overrides.items():
            flag = "--" + key.replace("_", "-")
            arguments += [flag] if value is True else [flag, str(value)]
        return arguments

    def with_options(model: str, path: str, *options: str, **overrides: Any) -> list[str]:
        return cell(model, path, **overrides) + ["--options", *options]

    compared = ("inductor", "tensorrt", "tensorrt-export")
    if name == "core":
        # Every model at every precision, against Inductor. Batch and sequence
        # are held at one value so this axis is the only one moving.
        return [
            cell(model, path, dtype=dtype)
            for model in ("mlp", "resnet18", "bert", "smollm2")
            for dtype in ("float32", "float16", "bfloat16")
            for path in ("eager", *compared)
        ]
    if name == "fp8":
        return [
            cell("fp8-matmul", path, dtype=dtype)
            for dtype in ("float16", "bfloat16")
            for path in ("eager", "inductor", "tensorrt", "tensorrt-export")
        ] + [
            cell("smollm2", path, dtype="bfloat16", quantize=mode)
            for mode in ("fp8", "fp8-dynamic")
            for path in ("eager", "inductor", "tensorrt", "tensorrt-export")
        ]
    if name == "batch":
        return [
            cell(model, path, dtype="float16", batch_size=batch)
            for model in ("mlp", "resnet18", "bert", "smollm2")
            for batch in (1, 8, 32)
            for path in ("eager", "inductor", "tensorrt-export")
        ]
    if name == "dynamic":
        return [
            cell(model, path, dtype="float16", dynamic=True, sweep_shapes=True)
            for model in ("bert", "smollm2")
            for path in ("inductor", "tensorrt", "tensorrt-export")
        ] + [
            # The static build, called at the same four lengths, is the control:
            # without it "the dynamic one absorbed every shape" cannot be told
            # from "this model never minded the shape in the first place".
            cell(model, path, dtype="float16", sweep_shapes=True)
            for model in ("bert", "smollm2")
            for path in ("tensorrt", "tensorrt-export")
        ]
    if name == "min-block":
        # The partitioner declines graphs below `min_block_size` and returns them
        # unconverted without raising. These cells lower the floor to 1 on the
        # models the default silently skipped, so the workaround is measured
        # rather than assumed.
        return [
            with_options(model, path, "min_block_size=1", dtype="float16")
            for model in ("mlp", "resnet18", "bert", "smollm2")
            for path in ("tensorrt", "tensorrt-export")
        ]
    raise SystemExit(f"Unknown plan {name!r}; choose core, fp8, batch, dynamic or min-block.")


def _summarize(directory: Path) -> str:
    records = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))
    ]
    lines = [f"{len(records)} cells from {directory}", ""]
    header = (
        f"{'cell':<58} {'ok':<4} {'trt':<5} {'build s':>9} {'median ms':>10} "
        f"{'vram GB':>8} {'max diff':>10} {'argmax':>7}"
    )
    lines += [header, "-" * len(header)]
    for record in records:
        vram = record.get("peak_vram_bytes")
        difference = record.get("max_abs_diff")
        build = record.get("build_seconds") or record.get("load_seconds")
        latency = record.get("latency_median_ms") or record.get("first_call_ms")
        engaged = record.get("tensorrt_engaged")
        lines.append(
            f"{record.get('cell', '?'):<58} "
            f"{'ok' if record.get('works') else 'FAIL':<4} "
            f"{'-' if engaged is None else ('yes' if engaged else 'NO'):<5} "
            f"{build if build is not None else float('nan'):>9.2f} "
            f"{latency if latency is not None else float('nan'):>10.3f} "
            f"{vram / 1e9 if vram else float('nan'):>8.2f} "
            f"{difference if difference is not None else float('nan'):>10.2e} "
            f"{record.get('argmax_agrees')!s:>7}"
        )
        if not record.get("works"):
            lines.append(f"    {record.get('error_type')}: {str(record.get('error'))[:200]}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LM7's TensorRT path on one GPU.")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--path", choices=sorted(PATHS))
    parser.add_argument("--target", default="nvidia")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    parser.add_argument("--quantize", choices=("none", "fp8", "fp8-dynamic"), default="none")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--dynamic", action="store_true", help="Compile for dynamic shapes.")
    parser.add_argument(
        "--sweep-shapes",
        action="store_true",
        help="Call the compiled model at several sequence lengths, dynamic or not.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="Backend options as key=value, e.g. min_block_size=1.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts/trt-matrix"))
    parser.add_argument("--reload", help="Load this .lm7 artifact instead of building a cell.")
    parser.add_argument("--plan", help="Print the cell list for a plan and exit.")
    parser.add_argument("--summarize", type=Path, help="Summarize a results directory and exit.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a cell whose result file already records success, so an "
        "interrupted sweep resumes instead of rebuilding every engine.",
    )
    arguments = parser.parse_args()

    if arguments.plan:
        for cell in _plan(arguments.plan):
            print(" ".join(cell))
        return
    if arguments.summarize:
        print(_summarize(arguments.summarize))
        return

    arguments.results_dir.mkdir(parents=True, exist_ok=True)
    if arguments.reload:
        name = f"{Path(arguments.reload).stem}__reload"
        runner: Any = reload_cell
    else:
        if not arguments.model or not arguments.path:
            raise SystemExit("--model and --path are required unless --reload is given.")
        name = cell_name(arguments)
        runner = run_cell
    destination = arguments.results_dir / f"{name}.json"
    if arguments.skip_existing and destination.exists():
        # Only a *successful* cell is skipped. A failed one is re-run, because
        # the usual reason a sweep is resumed is that something was fixed.
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous.get("works"):
            print(f"{name:<58} skipped (already ok)")
            return

    try:
        record = runner(arguments)
    except BaseException as error:  # noqa: BLE001 - a failed cell is a result
        record = {
            "cell": name,
            "model": arguments.model,
            "path": arguments.path,
            "dtype": arguments.dtype,
            "quantize": arguments.quantize,
            "batch_size": arguments.batch_size,
            "seq_len": arguments.seq_len,
            "dynamic": bool(arguments.dynamic),
            "host": platform.node(),
            "versions": _versions(),
            **_device_facts(),
            "works": False,
            "error_type": type(error).__name__,
            "error": str(error)[:800],
            "traceback": traceback.format_exc()[-1500:],
        }

    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = "ok" if record.get("works") else f"FAIL {record.get('error_type')}"
    latency = record.get("latency_median_ms")
    build = record.get("build_seconds")
    print(
        f"{name:<58} {status:<26}"
        + (f"  median={latency:9.3f} ms" if latency is not None else "")
        + (f"  build={build:7.1f} s" if build is not None else "")
        + (f"  diff={record['max_abs_diff']:.2e}" if record.get("max_abs_diff") is not None else "")
    )


if __name__ == "__main__":
    main()
