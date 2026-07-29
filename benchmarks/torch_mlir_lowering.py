"""Lower a captured model to StableHLO with torch-mlir instead of torch_xla.

This is the measurement harness for ``docs/torch-mlir-lowering-evaluation.md``.
It adds no LM7 backend. It answers whether torch-mlir can replace ``torch_xla``
as the ``stablehlo`` backend's lowering path, which matters because torch_xla is
ABI-tied to a matching PyTorch and torch-mlir is not.

    python benchmarks/torch_mlir_lowering.py lower \\
      --model smollm2 --output artifacts/torch-mlir/smollm2

The output directory is deliberately shaped like the one
``benchmarks/stablehlo_pjrt.py execute`` already consumes, so the same
PyTorch-free PJRT runner validates both lowering paths:

    .venv-pjrt/bin/python benchmarks/stablehlo_pjrt.py execute \\
      artifacts/torch-mlir/smollm2

Two behaviours of torch-mlir shape this script, and both are load-bearing:

* Every multi-element weight is emitted as a ``DenseResourceElementsAttr``, and
  XLA rejects those outright with "Only dense elements attr are supported". The
  choice is hardcoded in the importer, so the blobs are inlined afterwards from
  the printed ``{-# dialect_resources #-}`` section. Resource names contain
  dots, and each blob carries a four-byte alignment prefix the ``dense<>`` form
  omits -- miss either and the rewrite silently does nothing.
* ``torch.export`` lifts some buffers to inputs rather than constants, so the
  lowered ``@main`` can take more arguments than the model does. torch-mlir
  writes no metadata describing them, so this script emits the ``forward.meta``
  that the PJRT runner expects, built from the ExportedProgram's own signature.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

_MODELS = ("mlp", "smollm2")
_DEFAULT_PROMPT = "The capital of France is"

# A resource name is `torch_tensor_<shape>_torch.<dtype>`; the dot is why a
# `\w+` pattern matches nothing and the rewrite appears to succeed while
# leaving every constant untouched.
_RESOURCE_NAME = r"[\w.]+"
_BLOB_PATTERN = re.compile(rf'({_RESOURCE_NAME}):\s*"(0x[0-9a-fA-F]+)"')
_CONSTANT_PATTERN = re.compile(rf"dense_resource<({_RESOURCE_NAME})>\s*:\s*(tensor<[^>]*>)")
# MLIR blobs are prefixed with a four-byte alignment word, i.e. eight hex chars.
_BLOB_PREFIX_CHARS = 8


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


def inline_resources(module_text: str) -> tuple[str, int]:
    """Rewrite `dense_resource<name>` constants as inline `dense<"0x...">`.

    Returns the rewritten assembly and how many constants were replaced, so a
    caller can fail loudly rather than hand XLA a module it will reject.
    """
    marker = module_text.find("{-#")
    if marker < 0:
        return module_text, 0
    blobs = dict(_BLOB_PATTERN.findall(module_text[marker:]))
    body = module_text[:marker]
    replaced = 0

    def substitute(match: re.Match[str]) -> str:
        nonlocal replaced
        name, tensor_type = match.group(1), match.group(2)
        blob = blobs.get(name)
        if blob is None:
            return match.group(0)
        replaced += 1
        return f'dense<"0x{blob[2:][_BLOB_PREFIX_CHARS:]}"> : {tensor_type}'

    return _CONSTANT_PATTERN.sub(substitute, body), replaced


_MAIN_SIGNATURE = re.compile(r"func\.func @main\(([^)]*)\)")


def main_argument_count(module_text: str) -> int:
    """Count the arguments the lowered @main actually takes."""
    match = _MAIN_SIGNATURE.search(module_text)
    if match is None:
        raise SystemExit("the lowered module has no func.func @main to inspect")
    body = match.group(1).strip()
    return len([part for part in body.split("%arg") if part]) if body else 0


def payload_metadata(
    exported_program: Any, runtime_arguments: tuple[Any, ...], module_text: str, output: Path
) -> dict[str, Any]:
    """Describe every `@main` position, the way torch_xla's forward.meta does.

    torch-mlir bakes parameters and tensor constants into the module but leaves
    buffers lifted as arguments, so `@main` takes the buffers followed by the
    user inputs. It writes nothing describing that, so the buffers are saved
    beside the module and the ordering is recorded here. The count is checked
    against the real signature rather than assumed, because that split is
    torch-mlir's choice and not something the ExportedProgram states.
    """
    import numpy
    import torch

    state = {**exported_program.state_dict, **getattr(exported_program, "constants", {})}
    locations: list[dict[str, Any]] = []
    signatures: list[dict[str, Any]] = []
    data = output / "data"
    user_input_index = 0

    for spec in exported_program.graph_signature.input_specs:
        kind = str(spec.kind).rsplit(".", 1)[-1]
        if kind == "BUFFER":
            target = str(spec.target)
            value = state.get(target)
            if value is None:
                continue
            data.mkdir(exist_ok=True)
            numpy.save(data / f"{target}.npy", value.detach().cpu().numpy())
            locations.append({"type_": "parameter", "position": -1, "name": f"{target}.npy"})
        elif kind == "USER_INPUT":
            value = runtime_arguments[user_input_index]
            locations.append({"type_": "input_arg", "position": user_input_index, "name": ""})
            user_input_index += 1
        else:
            continue  # baked into the module as a constant
        if not isinstance(value, torch.Tensor):
            continue
        signatures.append(
            {
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "dynamic_dims": [],
            }
        )

    expected = main_argument_count(module_text)
    if len(locations) != expected:
        raise SystemExit(
            f"@main takes {expected} arguments but the signature accounts for "
            f"{len(locations)}; torch-mlir's constant-baking rules have changed and "
            "this harness would emit a payload no loader could call."
        )
    return {"input_locations": locations, "input_signature": signatures}


def run_lower(arguments: argparse.Namespace) -> int:
    import numpy
    import torch
    from torch_mlir import fx

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, arguments.dtype)
    model, example = _mlp(dtype) if arguments.model == "mlp" else _smollm2(dtype, arguments.prompt)
    with torch.no_grad():
        expected = model(*example)

    started = time.perf_counter()
    exported = torch.export.export(model, example)
    export_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    module = fx.export_and_import(exported, output_type="stablehlo")
    lower_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    text = str(module)
    inlined, replaced = inline_resources(text)
    inline_ms = (time.perf_counter() - started) * 1000
    if "dense_resource" in inlined:
        raise SystemExit(
            "dense_resource constants survived the rewrite; XLA will reject this module. "
            "The resource-name or blob-prefix assumptions in this script need revisiting."
        )

    functions = output / "functions"
    functions.mkdir(exist_ok=True)
    (functions / "forward.mlir").write_text(inlined, encoding="utf-8")
    metadata = payload_metadata(exported, example, inlined, output)
    (functions / "forward.meta").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    numpy.savez(
        output / "reference.npz",
        expected=expected.detach().cpu().numpy(),
        **{f"input_{index}": tensor.cpu().numpy() for index, tensor in enumerate(example)},
    )

    report = {
        "stage": "lower",
        "lowering": "torch-mlir",
        "model": arguments.model,
        "dtype": arguments.dtype,
        "torch_version": torch.__version__,
        "export_ms": export_ms,
        "lower_ms": lower_ms,
        "inline_ms": inline_ms,
        "constants_inlined": replaced,
        "module_bytes": len(inlined),
        "main_arguments": len(metadata["input_locations"]),
    }
    text_report = json.dumps(report, indent=2, sort_keys=True)
    print(text_report)
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(text_report + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lower a model to StableHLO with torch-mlir and measure the result.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    lower = subparsers.add_parser("lower", help="capture and lower through torch-mlir")
    lower.add_argument("--model", choices=_MODELS, default="mlp")
    lower.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    lower.add_argument("--prompt", default=_DEFAULT_PROMPT)
    lower.add_argument("--output", type=Path, required=True)
    lower.add_argument("--report", type=Path)
    lower.set_defaults(handler=run_lower)
    arguments = parser.parse_args()
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    sys.exit(main())
