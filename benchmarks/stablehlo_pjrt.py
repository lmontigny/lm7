"""Evaluate the torch.export -> StableHLO -> PJRT path for LM7 artifacts.

This is the first implementation slice of the evaluation in
``docs/stablehlo-pjrt-evaluation.md``. It does not add an LM7 backend. It
answers one question with measurements: can an LM7-style captured graph be
lowered to StableHLO and executed by a PJRT client, and does the result still
match eager PyTorch?

The interesting property is what the second half needs. An AOTInductor package
needs PyTorch to load, and OpenVINO IR runs without PyTorch but only on Intel
hardware. StableHLO plus a PJRT plugin is the only combination LM7 has looked at
that is both PyTorch-free and vendor-neutral, so the harness deliberately splits
into two commands that run in *different* environments:

    # 1. Capture and lower. Needs torch + torch_xla, ABI-matched.
    PJRT_DEVICE=CPU python benchmarks/stablehlo_pjrt.py export \\
      --model mlp --output artifacts/stablehlo/mlp

    # 2. Execute. Needs a PJRT client only; torch must NOT be installed.
    PJRT_DEVICE=CPU python benchmarks/stablehlo_pjrt.py execute \\
      artifacts/stablehlo/mlp --reference artifacts/stablehlo/mlp/reference.npz

``execute`` asserts that ``torch`` is absent rather than merely unimported, so a
passing run is evidence about the artifact and not about the environment that
happened to produce it. See the evaluation doc for why the two environments
cannot currently be one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_MODELS = ("mlp", "smollm2")
_DEFAULT_PROMPT = "The capital of France is"


def _mlp(dtype: Any) -> tuple[Any, tuple[Any, ...]]:
    import torch

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    ).eval()
    return model.to(dtype), (torch.randn(8, 16, dtype=dtype),)


def _smollm2(dtype: Any, prompt: str) -> tuple[Any, tuple[Any, ...]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Eager attention for the same reason lm7.export uses it for a dynamic
    # capture: the blocked-mask path emits guards torch.export cannot discharge.
    source = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation="eager"
    ).eval()

    class LogitsOnly(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.model = wrapped

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits

    encoded = tokenizer(prompt, return_tensors="pt")
    return LogitsOnly(source).eval(), (encoded["input_ids"], encoded["attention_mask"])


def _build(model_name: str, dtype_name: str, prompt: str) -> tuple[Any, tuple[Any, ...]]:
    import torch

    dtype = getattr(torch, dtype_name)
    if model_name == "mlp":
        return _mlp(dtype)
    return _smollm2(dtype, prompt)


def run_export(arguments: argparse.Namespace) -> int:
    import numpy
    import torch
    from torch_xla.stablehlo import exported_program_to_stablehlo, save_as_stablehlo

    output = arguments.output.resolve()
    model, example = _build(arguments.model, arguments.dtype, arguments.prompt)
    with torch.no_grad():
        expected = model(*example)

    started = time.perf_counter()
    exported = torch.export.export(model, example)
    export_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    lowered = exported_program_to_stablehlo(exported)
    lower_ms = (time.perf_counter() - started) * 1000
    stablehlo_text = lowered.get_stablehlo_text()

    # Executing through torch_xla's own runtime is the control: it proves the
    # lowering is correct before any PyTorch-free claim is made.
    started = time.perf_counter()
    through_xla = lowered(*example)
    xla_first_call_ms = (time.perf_counter() - started) * 1000
    xla_max_abs_diff = float((through_xla.cpu() - expected).abs().max())

    shutil.rmtree(output, ignore_errors=True)
    started = time.perf_counter()
    save_as_stablehlo(exported, str(output))
    save_ms = (time.perf_counter() - started) * 1000

    numpy.savez(
        output / "reference.npz",
        expected=expected.detach().cpu().numpy(),
        **{f"input_{index}": tensor.cpu().numpy() for index, tensor in enumerate(example)},
    )
    files = [path for path in output.rglob("*") if path.is_file()]
    report = {
        "stage": "export",
        "model": arguments.model,
        "dtype": arguments.dtype,
        "torch_version": torch.__version__,
        "export_ms": export_ms,
        "lower_ms": lower_ms,
        "save_ms": save_ms,
        "stablehlo_chars": len(stablehlo_text),
        "artifact_files": len(files),
        "artifact_bytes": sum(path.stat().st_size for path in files),
        "xla_first_call_ms": xla_first_call_ms,
        "xla_max_abs_diff": xla_max_abs_diff,
    }
    _emit(report, arguments.report)
    return 0


def _artifact_inputs(root: Path, meta: dict[str, Any], reference: Any) -> list[Any]:
    """Rebuild the flat input list the executable expects.

    ``input_locations`` is the whole reason a StableHLO artifact is usable
    without the framework that produced it: every position is labelled as a
    baked parameter, a baked constant, or a runtime argument, so a loader needs
    no model definition to know what to pass.
    """
    import numpy

    inputs = []
    for location, signature in zip(meta["input_locations"], meta["input_signature"]):
        kind = location["type_"]
        if kind == "parameter":
            value = numpy.load(root / "data" / location["name"])
        elif kind == "constant":
            value = numpy.load(root / "constants" / str(location["position"]))
        else:
            value = reference[f"input_{location['position']}"]
        inputs.append(numpy.asarray(value, dtype=signature["dtype"]))
    return inputs


def run_execute(arguments: argparse.Namespace) -> int:
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        raise SystemExit(
            "execute must run in an environment without PyTorch installed; that is the "
            "property under evaluation. Use a separate virtual environment holding only "
            "a PJRT client."
        )
    import jax

    # The captured graphs use int64 token ids, which a default JAX build would
    # silently narrow to int32 and fail against the executable's signature.
    jax.config.update("jax_enable_x64", True)
    import jax.extend.backend
    import jaxlib._jax as jax_internal
    import numpy
    from jax._src.lib import xla_client

    root = arguments.artifact.resolve()
    meta = json.loads((root / "functions" / "forward.meta").read_text(encoding="utf-8"))
    backend = jax.extend.backend.get_backend()
    devices = xla_client.DeviceList(tuple(backend.devices()[:1]))

    # torch_xla writes bytecode; the torch-mlir path in
    # benchmarks/torch_mlir_lowering.py must emit text, because inlining its
    # resource constants is a textual rewrite. make_hlo_program takes either.
    bytecode = root / "functions" / "forward.bytecode"
    assembly = root / "functions" / "forward.mlir"
    if bytecode.is_file():
        program_source: bytes | str = bytecode.read_bytes()
    elif assembly.is_file():
        program_source = assembly.read_text(encoding="utf-8")
    else:
        raise SystemExit(f"{root} holds neither functions/forward.bytecode nor forward.mlir")

    started = time.perf_counter()
    program = jax_internal.ifrt_programs.make_hlo_program(program_source)
    options = jax_internal.ifrt_programs.make_xla_compile_options(
        xla_client.CompileOptions(), devices, []
    )
    executable = backend.compile_and_load_ifrt_program(program, options)
    compile_ms = (time.perf_counter() - started) * 1000

    reference_path = arguments.reference or (root / "reference.npz")
    reference = numpy.load(reference_path)

    started = time.perf_counter()
    inputs = _artifact_inputs(root, meta, reference)
    load_ms = (time.perf_counter() - started) * 1000

    buffers = [jax.device_put(value) for value in inputs]
    latencies = []
    result = None
    for _ in range(arguments.repeats):
        started = time.perf_counter()
        result = numpy.asarray(executable.execute(buffers)[0])
        latencies.append((time.perf_counter() - started) * 1000)

    expected = reference["expected"]
    max_abs_diff = float(numpy.abs(result - expected).max())
    report = {
        "stage": "execute",
        "artifact": str(root),
        "pjrt_platform": backend.platform,
        "pjrt_devices": len(backend.devices()),
        "torch_installed": False,
        "compile_ms": compile_ms,
        "weight_load_ms": load_ms,
        "first_call_ms": latencies[0],
        "median_ms": sorted(latencies)[len(latencies) // 2],
        "output_shape": list(result.shape),
        "max_abs_diff": max_abs_diff,
        "argmax_matches": bool(
            result.reshape(-1, result.shape[-1])[-1].argmax()
            == expected.reshape(-1, expected.shape[-1])[-1].argmax()
        ),
    }
    _emit(report, arguments.report)
    return 0


def _emit(report: dict[str, Any], destination: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate torch.export -> StableHLO -> PJRT for LM7 artifacts.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    export_parser = subparsers.add_parser("export", help="capture and lower to StableHLO")
    export_parser.add_argument("--model", choices=_MODELS, default="mlp")
    export_parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="float32"
    )
    export_parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--report", type=Path)
    export_parser.set_defaults(handler=run_export)

    execute_parser = subparsers.add_parser(
        "execute", help="run the artifact through PJRT with no PyTorch installed"
    )
    execute_parser.add_argument("artifact", type=Path)
    execute_parser.add_argument("--reference", type=Path)
    execute_parser.add_argument("--repeats", type=int, default=10)
    execute_parser.add_argument("--report", type=Path)
    execute_parser.set_defaults(handler=run_execute)

    arguments = parser.parse_args()
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    sys.exit(main())
