"""Model-source plumbing shared by every task LM7 can load, not just causal LMs.

``huggingface.py`` grew these helpers while it was the only module that loaded a
model, so they read as causal-LM code by association rather than by content:
parsing an ``hf://`` URI, picking a compute dtype for a target, importing an
optional dependency by name, recording what an artifact was built from. None of
that is about text.

They live here so a second modality can reuse them without importing a module
whose every other function assumes ``AutoModelForCausalLM``. ``huggingface.py``
re-exports them under their original private names, so its call sites, its tests'
monkeypatches, and the benchmark scripts that import them are all unchanged.

What deliberately did *not* move is anything that consults the quantization
tables. ``resolve_dtype`` takes the quantized compute dtype as an argument rather
than looking it up, because the tables are a property of the TorchAO path and a
diffusion pipeline has no business importing them -- see the note on
``_QUANTIZED_COMPUTE_DTYPE`` in ``huggingface.py``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import torch

from .detection import compute_capability
from .errors import UnsupportedModelError
from .targets import TargetSpec

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Weight-only quantization needs native BF16 arithmetic, which starts at Ampere.
# The measurements behind this number, and why LM7 refuses rather than emulating,
# are recorded beside the quantization gates in `huggingface.py`.
BF16_MINIMUM_CAPABILITY = 80


def parse_model_uri(model_uri: str) -> str:
    """Turn ``hf://owner/model`` into ``owner/model``, or say why it is not one."""
    if not isinstance(model_uri, str) or not model_uri.startswith("hf://"):
        raise UnsupportedModelError(
            f"Unsupported model {model_uri!r}; expected a Hugging Face URI such as "
            "'hf://HuggingFaceTB/SmolLM2-135M-Instruct'."
        )
    model_id = model_uri.removeprefix("hf://").strip("/")
    if not model_id or "/" not in model_id:
        raise UnsupportedModelError(
            f"Invalid Hugging Face model URI {model_uri!r}; expected 'hf://owner/model'."
        )
    return model_id


def resolve_dtype(
    value: str, target: TargetSpec, quantized_default: str | None = None
) -> torch.dtype:
    """Pick a compute dtype, honouring an explicit request over any default.

    ``quantized_default`` is the name of the dtype quantization pins for this
    target, or None when nothing is being quantized. It is passed in rather than
    derived so this function stays free of the quantization tables.
    """
    if value == "auto":
        if quantized_default is not None:
            return _DTYPES[quantized_default]
        if target.vendor == "cpu":
            return torch.float32
        if target.vendor == "tpu":
            return torch.bfloat16
        return torch.float16
    if value not in _DTYPES:
        raise ValueError(
            f"Unsupported dtype {value!r}; expected auto, float32, float16, or bfloat16."
        )
    return _DTYPES[value]


def supports_native_bf16(target: TargetSpec) -> bool:
    """Whether this target has native BF16 arithmetic rather than emulation.

    Only meaningful for NVIDIA, where LM7 knows the capability number. Everything
    else answers True, because the compute dtype for those targets is decided by
    the caller and not by this.
    """
    if target.vendor != "nvidia":
        return True
    capability = compute_capability(target)
    return capability is None or capability >= BF16_MINIMUM_CAPABILITY


def source_metadata(
    model_uri: str, model_id: str, torch_dtype: torch.dtype, **extra: Any
) -> dict[str, Any]:
    """What an artifact was built from, for whoever loads it later.

    A lowered graph has weights and no name, so an artifact cannot say which
    checkpoint it is unless the export records it. ``extra`` carries whatever the
    modality needs on top -- a causal LM adds the tokenizer it decodes with, a
    diffusion component adds which component of which pipeline it is -- because
    the failure those fields prevent is silent in both cases.
    """
    return {
        "model_uri": model_uri,
        "model_id": model_id,
        "dtype": str(torch_dtype).removeprefix("torch."),
        **extra,
    }


def reset_peak_memory(target: TargetSpec) -> None:
    if target.vendor in {"nvidia", "amd"}:
        torch.cuda.reset_peak_memory_stats(target.ordinal or 0)


def peak_memory(target: TargetSpec) -> int | None:
    if target.vendor not in {"nvidia", "amd"}:
        return None
    return int(torch.cuda.max_memory_allocated(target.ordinal or 0))


def load_transformers() -> ModuleType:
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:
        raise UnsupportedModelError(
            'Hugging Face support is not installed. Install it with: pip install "lm7[hf]".'
        ) from exc


def load_diffusers() -> ModuleType:
    """Import ``diffusers``, or explain which extra provides it.

    Imported by name rather than with a plain ``import`` for the reason every
    optional dependency here is: CI type-checks a ``[dev]`` install, where the
    package is absent, and a plain import would need a mypy override instead of
    simply not being resolved.
    """
    try:
        return importlib.import_module("diffusers")
    except ImportError as exc:
        raise UnsupportedModelError(
            'Diffusion support is not installed. Install it with: pip install "lm7[diffusion]".'
        ) from exc
