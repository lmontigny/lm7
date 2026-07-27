from __future__ import annotations

import importlib
import time
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any

import torch

from .api import compile
from .detection import resolve_target
from .errors import UnsupportedModelError
from .targets import TargetSpec


@dataclass(frozen=True)
class HuggingFaceRunResult:
    model_uri: str
    model_id: str
    prompt: str
    target: str
    backend: str
    dtype: str
    parameter_count: int
    input_tokens: int
    output_shape: tuple[int, ...]
    first_call_ms: float
    next_token_id: int
    next_token: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_hf_model(
    model_uri: str,
    *,
    prompt: str,
    target: str | TargetSpec = "auto",
    backend: str = "auto",
    dtype: str = "auto",
) -> HuggingFaceRunResult:
    """Load and run one compiled causal-LM forward pass from Hugging Face."""
    model_id = _model_id(model_uri)
    resolved_target = resolve_target(target)
    torch_dtype = _resolve_dtype(dtype, resolved_target)
    transformers = _load_transformers()

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
        ).eval()
        inputs = dict(tokenizer(prompt, return_tensors="pt"))
    except Exception as exc:
        raise UnsupportedModelError(
            f"Hugging Face load stage failed for {model_uri}: {exc}."
        ) from exc

    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
        raise UnsupportedModelError(
            f"Hugging Face tokenization stage failed for {model_uri}: "
            "the tokenizer did not return batched input_ids."
        )
    wrapped = compile(
        model,
        target=resolved_target,
        backend=backend,
        transfers="automatic",
        fallback="error",
        cache=False,
    )
    _synchronize(resolved_target)
    started = time.perf_counter()
    output = wrapped(**inputs, use_cache=False)
    _synchronize(wrapped.target)
    first_call_ms = (time.perf_counter() - started) * 1000

    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim < 2:
        raise UnsupportedModelError(
            f"Hugging Face execution stage failed for {model_uri}: "
            "the model did not return tensor logits."
        )

    next_token_id = int(logits[0, -1].argmax().item())
    next_token = tokenizer.decode([next_token_id], skip_special_tokens=False)
    assert wrapped.target is not None
    assert wrapped.selected_backend is not None
    return HuggingFaceRunResult(
        model_uri=model_uri,
        model_id=model_id,
        prompt=prompt,
        target=str(wrapped.target),
        backend=wrapped.selected_backend,
        dtype=str(torch_dtype).removeprefix("torch."),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        input_tokens=int(input_ids.shape[-1]),
        output_shape=tuple(logits.shape),
        first_call_ms=first_call_ms,
        next_token_id=next_token_id,
        next_token=next_token,
    )


def _model_id(model_uri: str) -> str:
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


def _resolve_dtype(value: str, target: TargetSpec) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if value == "auto":
        if target.vendor == "cpu":
            return torch.float32
        if target.vendor == "tpu":
            return torch.bfloat16
        return torch.float16
    if value not in values:
        raise ValueError(
            f"Unsupported dtype {value!r}; expected auto, float32, float16, or bfloat16."
        )
    return values[value]


def _load_transformers() -> ModuleType:
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:
        raise UnsupportedModelError(
            'Hugging Face support is not installed. Install it with: pip install "lm7[hf]".'
        ) from exc


def _synchronize(target: TargetSpec | None) -> None:
    if target is not None and target.vendor in {"nvidia", "amd"}:
        torch.cuda.synchronize(target.ordinal or 0)
