from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .backends import registry
from .backends.base import CompileRequest
from .detection import resolve_target, torch_device
from .errors import BackendUnavailableError, UnsupportedModelError
from .huggingface import (
    FP8,
    INT8,
    NVFP4,
    _load_transformers,
    _model_id,
    _validate_quantization,
    compiles_decode,
)
from .targets import TargetSpec


@dataclass(frozen=True)
class CompatibilityCheck:
    name: str
    status: str
    reason: str


@dataclass(frozen=True)
class BackendCompatibility:
    name: str
    supported: bool
    priority: int
    reason: str


@dataclass(frozen=True)
class ModelCompatibilityResult:
    model_uri: str
    model_id: str
    status: str
    model_type: str | None
    architectures: tuple[str, ...]
    task: str
    config_class: str | None
    dtype: str | None
    context_length: int | None
    vocab_size: int | None
    hidden_size: int | None
    num_hidden_layers: int | None
    is_encoder_decoder: bool
    is_multimodal: bool
    requires_remote_code: bool
    target: str
    requested_backend: str
    selected_backend: str | None
    workflows: tuple[CompatibilityCheck, ...]
    quantization: tuple[CompatibilityCheck, ...]
    backend_candidates: tuple[BackendCompatibility, ...]
    notes: tuple[str, ...]
    config_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_hf_model(
    model_uri: str,
    *,
    target: str | TargetSpec = "auto",
    backend: str = "auto",
) -> ModelCompatibilityResult:
    """Inspect a Hugging Face config without downloading model weights."""
    model_id = _model_id(model_uri)
    resolved_target = resolve_target(target)
    selected_backend, candidates = _backend_compatibility(resolved_target, backend)
    transformers = _load_transformers()

    try:
        config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    except Exception as exc:
        message = str(exc)
        if _remote_code_error(message):
            return _remote_code_result(
                model_uri,
                model_id,
                resolved_target,
                backend,
                selected_backend,
                candidates,
                message,
            )
        raise UnsupportedModelError(
            f"Hugging Face config inspection failed for {model_uri}: {exc}."
        ) from exc

    architectures = tuple(str(item) for item in (getattr(config, "architectures", None) or ()))
    auto_map = getattr(config, "auto_map", None) or {}
    requires_remote_code = "AutoModelForCausalLM" in auto_map
    is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
    is_multimodal = any(
        getattr(config, name, None) is not None for name in ("vision_config", "audio_config")
    )
    causal_registered = _causal_lm_registered(transformers, config)
    looks_causal = any(
        name.endswith(("ForCausalLM", "CausalLM", "LMHeadModel")) for name in architectures
    )
    model_status, task, model_reason = _model_status(
        causal_registered=causal_registered,
        looks_causal=looks_causal,
        is_encoder_decoder=is_encoder_decoder,
        is_multimodal=is_multimodal,
        requires_remote_code=requires_remote_code,
    )
    workflows = _workflow_checks(
        model_status,
        model_reason,
        resolved_target,
        backend,
        selected_backend,
    )
    quantization = _quantization_checks(
        model_id,
        model_status,
        model_reason,
        resolved_target,
        selected_backend or backend,
    )
    overall = model_status
    if model_status == "compatible" and selected_backend is None:
        overall = "incompatible"
    notes = (
        "Configuration-only preflight: no model weights were downloaded.",
        (
            "A compatible config does not prove torch.export or compiler operator coverage; "
            "the first real run/export is the definitive check."
        ),
    )
    return ModelCompatibilityResult(
        model_uri=model_uri,
        model_id=model_id,
        status=overall,
        model_type=_optional_string(getattr(config, "model_type", None)),
        architectures=architectures,
        task=task,
        config_class=type(config).__name__,
        dtype=_dtype_name(config),
        context_length=_positive_int(
            config,
            "max_position_embeddings",
            "n_positions",
            "max_sequence_length",
            "seq_length",
        ),
        vocab_size=_positive_int(config, "vocab_size"),
        hidden_size=_positive_int(config, "hidden_size", "n_embd", "d_model"),
        num_hidden_layers=_positive_int(config, "num_hidden_layers", "n_layer", "num_layers"),
        is_encoder_decoder=is_encoder_decoder,
        is_multimodal=is_multimodal,
        requires_remote_code=requires_remote_code,
        target=str(resolved_target),
        requested_backend=backend,
        selected_backend=selected_backend,
        workflows=workflows,
        quantization=quantization,
        backend_candidates=candidates,
        notes=notes,
    )


def _backend_compatibility(
    target: TargetSpec, requested_backend: str
) -> tuple[str | None, tuple[BackendCompatibility, ...]]:
    request = CompileRequest(torch.nn.Identity(), target, "lazy", "automatic", "error", {})
    candidates = tuple(
        BackendCompatibility(
            candidate.name,
            (support := candidate.supports(request)).supported,
            support.priority,
            support.reason,
        )
        for candidate in registry.all()
    )
    by_name = {candidate.name: candidate for candidate in candidates}
    if requested_backend != "auto":
        if requested_backend not in by_name:
            raise BackendUnavailableError(
                f"Requested backend {requested_backend!r} is not registered. "
                f"Available: {', '.join(by_name)}."
            )
        selected = requested_backend if by_name[requested_backend].supported else None
        return selected, candidates
    supported = [candidate for candidate in candidates if candidate.supported]
    selected = (
        min(supported, key=lambda item: (-item.priority, item.name)).name if supported else None
    )
    return selected, candidates


def _model_status(
    *,
    causal_registered: bool,
    looks_causal: bool,
    is_encoder_decoder: bool,
    is_multimodal: bool,
    requires_remote_code: bool,
) -> tuple[str, str, str]:
    if requires_remote_code:
        return (
            "incompatible",
            "causal-lm" if looks_causal else "unknown",
            (
                "The checkpoint requires custom AutoModelForCausalLM code; LM7 does not enable "
                "trust_remote_code."
            ),
        )
    if is_encoder_decoder:
        return (
            "incompatible",
            "seq2seq",
            (
                "LM7 model commands currently support decoder-only causal language models, "
                "not encoder-decoder generation."
            ),
        )
    if is_multimodal:
        return (
            "incompatible",
            "multimodal",
            "LM7 model commands currently tokenize and capture text tensors only.",
        )
    if causal_registered:
        return (
            "compatible",
            "causal-lm",
            "The installed Transformers build registers this config for AutoModelForCausalLM.",
        )
    if looks_causal:
        return (
            "unknown",
            "causal-lm",
            (
                "The declared architecture looks causal, but the installed Transformers build "
                "does not register its config for AutoModelForCausalLM."
            ),
        )
    return (
        "incompatible",
        "unknown",
        "The config is not registered for AutoModelForCausalLM.",
    )


def _workflow_checks(
    model_status: str,
    model_reason: str,
    target: TargetSpec,
    requested_backend: str,
    selected_backend: str | None,
) -> tuple[CompatibilityCheck, ...]:
    if model_status != "compatible":
        status = "unknown" if model_status == "unknown" else "unsupported"
        return tuple(
            CompatibilityCheck(name, status, model_reason) for name in ("run", "generate", "export")
        )
    if selected_backend is None:
        run = CompatibilityCheck(
            "run",
            "unsupported",
            f"Backend {requested_backend!r} is unavailable for target {target}.",
        )
    else:
        run = CompatibilityCheck(
            "run",
            "compatible",
            f"Configuration is compatible and backend {selected_backend!r} supports {target}; "
            "the first call still validates model operators.",
        )
    if requested_backend not in {"auto", "inductor"}:
        generate = CompatibilityCheck(
            "generate",
            "unsupported",
            "Compiled generation accepts backend='auto' or backend='inductor' only.",
        )
    elif compiles_decode(torch_device(target)):
        generate = CompatibilityCheck(
            "generate",
            "conditional",
            "Transformers can compile static-cache decode on this device; the model must "
            "support the static cache implementation at runtime.",
        )
    else:
        generate = CompatibilityCheck(
            "generate",
            "conditional",
            "Generation is available, but Transformers will decode eagerly on this device; "
            "the model must support the static cache implementation.",
        )
    export = CompatibilityCheck(
        "export",
        "conditional",
        "The config is compatible, but torch.export and the selected export backend must "
        "validate the model's operators with representative inputs.",
    )
    return run, generate, export


def _quantization_checks(
    model_id: str,
    model_status: str,
    model_reason: str,
    target: TargetSpec,
    backend: str,
) -> tuple[CompatibilityCheck, ...]:
    checks = []
    for mode in (INT8, FP8, NVFP4):
        if model_status != "compatible":
            checks.append(CompatibilityCheck(mode, "unsupported", model_reason))
            continue
        try:
            _validate_quantization(mode, target, backend, "auto", model_id)
        except UnsupportedModelError as exc:
            checks.append(CompatibilityCheck(mode, "unsupported", str(exc)))
        else:
            checks.append(
                CompatibilityCheck(
                    mode,
                    "compatible",
                    f"This model/target/backend combination passes LM7's {mode} validation gate.",
                )
            )
    return tuple(checks)


def _causal_lm_registered(transformers: Any, config: Any) -> bool:
    mapping = getattr(getattr(transformers, "AutoModelForCausalLM", None), "_model_mapping", ())
    try:
        return type(config) in mapping
    except (KeyError, TypeError):
        return False


def _positive_int(config: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _dtype_name(config: Any) -> str | None:
    # Read instance storage directly so older Transformers configs do not emit
    # a torch_dtype deprecation warning from their compatibility property.
    values = vars(config)
    value = values.get("dtype", values.get("torch_dtype"))
    if value is None:
        return None
    return str(value).removeprefix("torch.")


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _remote_code_error(message: str) -> bool:
    lowered = message.lower()
    return "trust_remote_code" in lowered or "execute the configuration file" in lowered


def _remote_code_result(
    model_uri: str,
    model_id: str,
    target: TargetSpec,
    requested_backend: str,
    selected_backend: str | None,
    candidates: tuple[BackendCompatibility, ...],
    message: str,
) -> ModelCompatibilityResult:
    reason = (
        "The checkpoint requires custom configuration code; LM7 does not enable trust_remote_code."
    )
    checks = tuple(
        CompatibilityCheck(name, "unsupported", reason) for name in ("run", "generate", "export")
    )
    quantization = tuple(
        CompatibilityCheck(name, "unsupported", reason) for name in (INT8, FP8, NVFP4)
    )
    return ModelCompatibilityResult(
        model_uri=model_uri,
        model_id=model_id,
        status="incompatible",
        model_type=None,
        architectures=(),
        task="unknown",
        config_class=None,
        dtype=None,
        context_length=None,
        vocab_size=None,
        hidden_size=None,
        num_hidden_layers=None,
        is_encoder_decoder=False,
        is_multimodal=False,
        requires_remote_code=True,
        target=str(target),
        requested_backend=requested_backend,
        selected_backend=selected_backend,
        workflows=checks,
        quantization=quantization,
        backend_candidates=candidates,
        notes=(
            "Configuration-only preflight: no model weights were downloaded.",
            f"Transformers refused the config without remote code: {message}",
        ),
    )
