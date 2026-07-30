from __future__ import annotations

import importlib
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any

import torch

from .api import compile
from .detection import resolve_target, torch_device
from .errors import UnsupportedModelError
from .exporting import DynamicDimension, ShapeProfile
from .targets import TargetSpec

# The tensors _LogitsOnly forwards, and therefore the only ones captured.
_CAPTURED_INPUTS = ("input_ids", "attention_mask")

INT8 = "int8"
FP8 = "fp8"
NVFP4 = "nvfp4"
WEIGHT_ONLY_QUANTIZATIONS = frozenset({INT8, FP8, NVFP4})
NO_QUANTIZATION = "none"

# The pre-0.2 spellings. `--quantize` advertises the short names; these keep
# existing scripts and `--quantization` working.
QUANTIZATION_ALIASES: dict[str, str] = {
    "int8-weight-only": INT8,
    "fp8-weight-only": FP8,
}

_QUANTIZATION_LABELS = {INT8: "INT8", FP8: "FP8", NVFP4: "NVFP4"}

# Validated per model *and* per mode, because the two are not interchangeable:
# LFM2.5-230M keeps its top-1 token under FP8 but diverges completely under INT8
# (0/4 prompts agreed with BF16, max logit difference 22.4 on NVIDIA sm89). A
# model earns an entry here only after its outputs have been compared against an
# unquantized baseline on real hardware. See docs/quantization.md.
VALIDATED_WEIGHT_ONLY: dict[str, frozenset[str]] = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": frozenset({INT8, FP8}),
    "unsloth/Llama-3.2-1B-Instruct": frozenset({INT8, FP8, NVFP4}),
}
WEIGHT_ONLY_MODEL_IDS = frozenset(VALIDATED_WEIGHT_ONLY)


def normalize_quantization(value: str) -> str:
    """Map a deprecated long-form quantization name onto its short name."""
    return QUANTIZATION_ALIASES.get(value, value)


@dataclass(frozen=True)
class HuggingFaceRunResult:
    model_uri: str
    model_id: str
    prompt: str
    target: str
    backend: str
    dtype: str
    quantization: str
    parameter_count: int
    baseline_model_storage_bytes: int
    model_storage_bytes: int
    quantized_modules: int
    input_tokens: int
    output_shape: tuple[int, ...]
    quantization_ms: float
    first_call_ms: float
    latency_ms: float
    peak_memory_bytes: int | None
    next_token_id: int
    next_token: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HuggingFaceGenerateResult:
    model_uri: str
    model_id: str
    prompt: str
    target: str
    backend: str
    dtype: str
    parameter_count: int
    input_tokens: int
    generated_tokens: int
    max_new_tokens: int
    cache_implementation: str
    first_call_ms: float
    latency_ms: float
    peak_memory_bytes: int | None
    generated_token_ids: tuple[int, ...]
    generated_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_hf_model(
    model_uri: str,
    *,
    prompt: str,
    max_new_tokens: int = 32,
    target: str | TargetSpec = "auto",
    backend: str = "auto",
    dtype: str = "auto",
) -> HuggingFaceGenerateResult:
    """Greedily generate with an eager prefill and compiled static-cache decode."""
    if max_new_tokens < 2:
        raise UnsupportedModelError(
            "Compiled generation requires max_new_tokens >= 2: the first token "
            "comes from prefill and the fixed-shape decode graph starts with the second."
        )
    if backend not in {"auto", "inductor"}:
        raise UnsupportedModelError(
            "Compiled Hugging Face generation currently requires "
            "backend='auto' or backend='inductor'."
        )

    model_id = _model_id(model_uri)
    resolved_target = resolve_target(target)
    torch_dtype = _resolve_dtype(dtype, resolved_target)
    transformers = _load_transformers()
    compile_config_type = getattr(transformers, "CompileConfig", None)
    if compile_config_type is None:
        raise UnsupportedModelError(
            "This Transformers version does not expose compiled generation. "
            'Upgrade the Hugging Face extra with: pip install -U "lm7[hf]".'
        )

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

    device = torch_device(resolved_target)
    model = model.to(device)
    inputs = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    compile_config = compile_config_type(
        backend="inductor",
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=None,
    )
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "cache_implementation": "static",
        "compile_config": compile_config,
    }

    _reset_peak_memory(resolved_target)
    try:
        with torch.inference_mode():
            _synchronize(resolved_target)
            started = time.perf_counter()
            generated = model.generate(**generation_kwargs)
            _synchronize(resolved_target)
            first_call_ms = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            generated = model.generate(**generation_kwargs)
            _synchronize(resolved_target)
            latency_ms = (time.perf_counter() - started) * 1000
    except Exception as exc:
        raise UnsupportedModelError(
            f"Hugging Face compiled generation failed for {model_uri}: {exc}."
        ) from exc

    prompt_tokens = int(input_ids.shape[-1])
    generated_ids = generated[0, prompt_tokens:].detach().cpu().tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return HuggingFaceGenerateResult(
        model_uri=model_uri,
        model_id=model_id,
        prompt=prompt,
        target=str(resolved_target),
        backend="inductor",
        dtype=str(torch_dtype).removeprefix("torch."),
        parameter_count=parameter_count,
        input_tokens=prompt_tokens,
        generated_tokens=len(generated_ids),
        max_new_tokens=max_new_tokens,
        cache_implementation="static",
        first_call_ms=first_call_ms,
        latency_ms=latency_ms,
        peak_memory_bytes=_peak_memory(resolved_target),
        generated_token_ids=tuple(int(token_id) for token_id in generated_ids),
        generated_text=generated_text,
    )


def run_hf_model(
    model_uri: str,
    *,
    prompt: str,
    target: str | TargetSpec = "auto",
    backend: str = "auto",
    dtype: str = "auto",
    quantization: str = NO_QUANTIZATION,
) -> HuggingFaceRunResult:
    """Load and run one compiled causal-LM forward pass from Hugging Face."""
    model_id = _model_id(model_uri)
    resolved_target = resolve_target(target)
    quantization = normalize_quantization(quantization)
    _validate_quantization(quantization, resolved_target, backend, dtype, model_id)
    torch_dtype = _resolve_dtype(dtype, resolved_target, quantization)
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
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    baseline_model_storage_bytes = _model_storage_bytes(model)
    _reset_peak_memory(resolved_target)
    quantization_ms, quantized_modules = _apply_quantization(model, resolved_target, quantization)
    model_storage_bytes = _model_storage_bytes(model)
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
    started = time.perf_counter()
    output = wrapped(**inputs, use_cache=False)
    _synchronize(wrapped.target)
    latency_ms = (time.perf_counter() - started) * 1000

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
        quantization=quantization,
        parameter_count=parameter_count,
        baseline_model_storage_bytes=baseline_model_storage_bytes,
        model_storage_bytes=model_storage_bytes,
        quantized_modules=quantized_modules,
        input_tokens=int(input_ids.shape[-1]),
        output_shape=tuple(logits.shape),
        quantization_ms=quantization_ms,
        first_call_ms=first_call_ms,
        latency_ms=latency_ms,
        peak_memory_bytes=_peak_memory(wrapped.target),
        next_token_id=next_token_id,
        next_token=next_token,
    )


class _LogitsOnly(torch.nn.Module):
    """Expose a causal LM as tensors in, one logits tensor out.

    Hugging Face models return a ``CausalLMOutputWithPast`` dataclass. torch.export
    captures that in the output pytree, and ``torch.export.load`` then fails with
    "Deserializing transformers.modeling_outputs.CausalLMOutputWithPast in pytree
    is not registered" -- so the artifact would save but never reload. Capturing a
    plain tensor keeps the artifact loadable by anything that can read a ``.pt2``.

    The signature names its inputs rather than taking ``**inputs``. A shape
    profile is bound with ``inspect.signature(...).bind``, and a ``VAR_KEYWORD``
    parameter collects every tensor under one argument name, leaving no per-input
    dimension for a profile to constrain.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits


@dataclass(frozen=True)
class HuggingFaceExportResult:
    model_uri: str
    model_id: str
    target: str
    backend: str
    dtype: str
    output: str
    prompt: str
    input_tokens: int
    parameter_count: int
    export_ms: float
    artifact_bytes: int
    files: tuple[str, ...]
    sequence_bounds: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Captured graphs are prefill-only, and a causal LM's own limit is the ceiling
# worth offering; this caps the default when a config does not report one.
_DEFAULT_MAX_SEQUENCE = 2048


def _sequence_bounds(model: torch.nn.Module, requested: tuple[int, int] | None) -> tuple[int, int]:
    if requested is not None:
        minimum, maximum = int(requested[0]), int(requested[1])
        if minimum < 1:
            raise ValueError("Dynamic sequence minimum must be at least 1.")
        if maximum < minimum:
            raise ValueError("Dynamic sequence maximum must be at least its minimum.")
        return minimum, maximum
    positions = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    maximum = int(positions) if isinstance(positions, int) and positions > 0 else None
    return 1, min(maximum or _DEFAULT_MAX_SEQUENCE, _DEFAULT_MAX_SEQUENCE)


def _sequence_shape_profile(
    inputs: Mapping[str, torch.Tensor], bounds: tuple[int, int]
) -> ShapeProfile:
    """Mark dimension 1 of every captured tensor as one shared sequence length."""
    minimum, maximum = bounds
    dimension = DynamicDimension("sequence", min=minimum, max=maximum)
    return ShapeProfile(inputs={name: {1: dimension} for name in inputs})


def export_hf_model(
    model_uri: str,
    *,
    output: str,
    prompt: str = "The capital of France is",
    target: str | TargetSpec = "auto",
    backend: str = "export",
    dtype: str = "auto",
    dynamic_sequence: bool | tuple[int, int] = False,
) -> HuggingFaceExportResult:
    """Capture a Hugging Face causal LM into an LM7 artifact.

    The example inputs come from tokenizing ``prompt``. By default the artifact
    is fixed to that input signature. Pass ``dynamic_sequence`` to capture the
    sequence length as a bounded dynamic dimension instead, so one artifact
    serves prompts of any length inside those bounds — either ``True`` for
    bounds derived from the model config, or an explicit ``(min, max)``.
    """
    from .exporting import export as export_artifact

    model_id = _model_id(model_uri)
    resolved_target = resolve_target(target)
    torch_dtype = _resolve_dtype(dtype, resolved_target)
    transformers = _load_transformers()
    requested_bounds = dynamic_sequence if isinstance(dynamic_sequence, tuple) else None
    is_dynamic = dynamic_sequence is not False and dynamic_sequence is not None

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        # The default attention path builds its mask in blocks, which makes
        # torch.export emit a `sequence % 8` guard it cannot prove over a range:
        # "Not all values of sequence ... satisfy the generated guard". The eager
        # attention path has no such guard. Only a dynamic capture asks for it,
        # so a fixed export keeps the faster default.
        attention = {"attn_implementation": "eager"} if is_dynamic else {}
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            **attention,
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

    # _LogitsOnly names the two tensors it forwards, so anything else the
    # tokenizer produced (token_type_ids, offsets) is not part of the graph.
    inputs = {name: value for name, value in inputs.items() if name in _CAPTURED_INPUTS}
    bounds = _sequence_bounds(model, requested_bounds) if is_dynamic else None
    if bounds is not None and not bounds[0] <= int(input_ids.shape[-1]) <= bounds[1]:
        raise UnsupportedModelError(
            f"Hugging Face export stage failed for {model_uri}: the prompt tokenizes to "
            f"{int(input_ids.shape[-1])} tokens, outside the requested sequence bounds "
            f"[{bounds[0]}, {bounds[1]}]. torch.export traces the example input, so it "
            "has to sit inside the range the artifact accepts."
        )

    # PyTorch/XLA cannot lower a program captured with keyword inputs to
    # StableHLO, so that backend is fed the same tensors positionally. The order
    # is _LogitsOnly.forward's, and the reloaded artifact takes them the same way.
    positional = backend == "stablehlo"
    capture_args = tuple(inputs[name] for name in _CAPTURED_INPUTS if name in inputs)

    started = time.perf_counter()
    artifact = export_artifact(
        # _LogitsOnly pins use_cache=False, so the captured graph is a single
        # prefill forward pass; a KV-cache decode loop is a different graph and is
        # not supported here.
        _LogitsOnly(model).eval(),
        args=capture_args if positional else (),
        kwargs={} if positional else inputs,
        target=resolved_target,
        backend=backend,
        output=output,
        shape_profile=_sequence_shape_profile(inputs, bounds) if bounds else None,
    )
    export_ms = (time.perf_counter() - started) * 1000

    files = tuple(sorted(item.name for item in artifact.path.iterdir() if item.is_file()))
    artifact_bytes = sum(item.stat().st_size for item in artifact.path.rglob("*") if item.is_file())
    return HuggingFaceExportResult(
        model_uri=model_uri,
        model_id=model_id,
        target=str(resolved_target),
        backend=backend,
        dtype=str(torch_dtype).removeprefix("torch."),
        output=str(artifact.path),
        prompt=prompt,
        input_tokens=int(input_ids.shape[-1]),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        export_ms=export_ms,
        artifact_bytes=artifact_bytes,
        files=files,
        sequence_bounds=bounds,
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


def _resolve_dtype(
    value: str, target: TargetSpec, quantization: str = NO_QUANTIZATION
) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if value == "auto":
        if quantization in WEIGHT_ONLY_QUANTIZATIONS:
            return torch.bfloat16
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


def _validate_quantization(
    quantization: str,
    target: TargetSpec,
    backend: str,
    dtype: str,
    model_id: str | None = None,
) -> None:
    if quantization == NO_QUANTIZATION:
        return
    if quantization not in WEIGHT_ONLY_QUANTIZATIONS:
        choices = ", ".join([NO_QUANTIZATION, *sorted(WEIGHT_ONLY_QUANTIZATIONS)])
        raise UnsupportedModelError(
            f"Unsupported quantization {quantization!r}; expected one of: {choices}."
        )
    label = _QUANTIZATION_LABELS[quantization]
    if target.vendor != "nvidia":
        raise UnsupportedModelError(
            f"{label} weight-only quantization is supported only on detected NVIDIA GPUs."
        )
    if backend not in {"auto", "inductor"}:
        raise UnsupportedModelError(
            f"{label} weight-only quantization requires backend='auto' or backend='inductor'."
        )
    if dtype not in {"auto", "bfloat16"}:
        raise UnsupportedModelError(
            f"{label} weight-only quantization requires dtype='auto' or dtype='bfloat16'."
        )
    if quantization == FP8 and not _supports_fp8(target):
        raise UnsupportedModelError(
            "FP8 weight-only quantization requires NVIDIA Ada (sm89), Hopper (sm90), "
            "or newer hardware."
        )
    if model_id is not None and quantization not in VALIDATED_WEIGHT_ONLY.get(
        model_id, frozenset()
    ):
        validated = ", ".join(
            f"{name} ({', '.join(sorted(modes))})"
            for name, modes in sorted(VALIDATED_WEIGHT_ONLY.items())
        )
        raise UnsupportedModelError(
            f"{label} weight-only quantization is not validated for {model_id!r}. "
            f"Currently validated: {validated}. Use quantization='none' for this model."
        )


def _apply_quantization(
    model: torch.nn.Module,
    target: TargetSpec,
    quantization: str,
) -> tuple[float, int]:
    """Quantize in place, returning elapsed milliseconds and converted layer count."""
    if quantization == NO_QUANTIZATION:
        return 0.0, 0
    torchao_quantization = _load_torchao_quantization()
    started = time.perf_counter()
    config = _quantization_config(torchao_quantization, quantization)
    filter_fn = _QUANTIZATION_FILTERS[quantization]
    # torchao silently does nothing when the filter matches no module, so an
    # unmatched filter would report a successful quantization that left the model
    # untouched. LFM2.5-230M hits this with the FP8 filter: it has no ".mlp."
    # linears, so the run reported 1.00x storage reduction and byte-identical
    # logits.
    matched = sum(1 for fqn, module in model.named_modules() if filter_fn(module, fqn))
    if matched == 0:
        alternative = (
            f" Try {INT8}, which selects every linear except lm_head."
            if quantization != INT8
            else ""
        )
        raise UnsupportedModelError(
            f"{quantization} matched no quantizable layers in this model, so quantization "
            f"would silently do nothing. It selects {_QUANTIZATION_SELECTS[quantization]}, "
            f"and this model has none. Use quantization='none'.{alternative}"
        )
    torchao_quantization.quantize_(
        model,
        config,
        filter_fn=filter_fn,
        device=torch_device(target),
    )
    _synchronize(target)
    return (time.perf_counter() - started) * 1000, matched


def _quantization_config(torchao_quantization: ModuleType, quantization: str) -> Any:
    if quantization == FP8:
        return torchao_quantization.Float8WeightOnlyConfig(version=2)
    if quantization == NVFP4:
        return _load_torchao_nvfp4().NVFP4WeightOnlyConfig()
    return torchao_quantization.Int8WeightOnlyConfig(version=2)


def _is_lm_head(fqn: str) -> bool:
    return fqn == "lm_head" or fqn.endswith(".lm_head")


def _is_quantizable_linear(module: torch.nn.Module, fqn: str) -> bool:
    return isinstance(module, torch.nn.Linear) and not _is_lm_head(fqn)


def _is_fp8_quantizable_linear(module: torch.nn.Module, fqn: str) -> bool:
    return isinstance(module, torch.nn.Linear) and ".mlp." in fqn


def _is_nvfp4_quantizable_linear(module: torch.nn.Module, fqn: str) -> bool:
    # NVFP4 packs two 4-bit values per byte against a scale for every block of
    # 16 elements, so torchao raises unless both trailing weight dimensions are
    # multiples of 16. Skipping those layers keeps a model with one odd-shaped
    # projection usable instead of failing the whole run.
    if not isinstance(module, torch.nn.Linear) or _is_lm_head(fqn):
        return False
    out_features, in_features = module.weight.shape[-2:]
    return out_features % 16 == 0 and in_features % 16 == 0


_QUANTIZATION_FILTERS = {
    INT8: _is_quantizable_linear,
    FP8: _is_fp8_quantizable_linear,
    NVFP4: _is_nvfp4_quantizable_linear,
}

_QUANTIZATION_SELECTS = {
    INT8: "every linear except lm_head",
    FP8: "linears whose module path contains '.mlp.'",
    NVFP4: "every linear except lm_head whose last two dimensions are multiples of 16",
}


def _supports_fp8(target: TargetSpec) -> bool:
    architecture = target.architecture
    if not architecture or not architecture.startswith("sm"):
        return True
    try:
        capability = int(architecture.removeprefix("sm"))
    except ValueError:
        return True
    return capability >= 89


def _model_storage_bytes(model: torch.nn.Module) -> int:
    seen: set[int] = set()

    def tensor_bytes(tensor: torch.Tensor) -> int:
        identity = id(tensor)
        if identity in seen:
            return 0
        seen.add(identity)
        flatten = getattr(tensor, "__tensor_flatten__", None)
        if flatten is not None:
            names, _ = flatten()
            return sum(tensor_bytes(getattr(tensor, name)) for name in names)
        return tensor.numel() * tensor.element_size()

    tensors = (*model.parameters(), *model.buffers())
    return sum(tensor_bytes(tensor) for tensor in tensors)


def _load_transformers() -> ModuleType:
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:
        raise UnsupportedModelError(
            'Hugging Face support is not installed. Install it with: pip install "lm7[hf]".'
        ) from exc


def _load_torchao_quantization() -> ModuleType:
    try:
        return importlib.import_module("torchao.quantization")
    except ImportError as exc:
        raise UnsupportedModelError(
            'TorchAO quantization is not installed. Install it with: pip install "lm7[hf,torchao]".'
        ) from exc


def _load_torchao_nvfp4() -> ModuleType:
    # NVFP4 lives under torchao.prototype, so it carries no API stability promise
    # and moved module in recent releases. Fail with the pin rather than an
    # ImportError from inside torchao.
    try:
        return importlib.import_module("torchao.prototype.mx_formats")
    except ImportError as exc:
        raise UnsupportedModelError(
            "NVFP4 quantization needs torchao.prototype.mx_formats, which this torchao "
            'build does not provide. Install the pinned version with: pip install "lm7[hf,torchao]".'
        ) from exc


def _reset_peak_memory(target: TargetSpec) -> None:
    if target.vendor in {"nvidia", "amd"}:
        torch.cuda.reset_peak_memory_stats(target.ordinal or 0)


def _peak_memory(target: TargetSpec) -> int | None:
    if target.vendor not in {"nvidia", "amd"}:
        return None
    return torch.cuda.max_memory_allocated(target.ordinal or 0)


def _synchronize(target: TargetSpec | None) -> None:
    if target is None:
        return
    if target.vendor in {"nvidia", "amd"}:
        torch.cuda.synchronize(target.ordinal or 0)
    elif target.vendor == "apple":
        torch.mps.synchronize()
