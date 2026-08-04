from __future__ import annotations

import importlib
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from .api import compile
from .detection import inference_context, resolve_target, synchronize, torch_device
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

# Which targets each mode is allowed on. INT8 is the only mode measured off
# NVIDIA: FP8 needs Ada-class tensor cores, and NVFP4 on CPU kept only 2 of 4
# top-1 tokens and ran 8.5x slower than compiled FP32, so both stay NVIDIA-only.
_QUANTIZATION_VENDORS = {
    INT8: frozenset({"nvidia", "cpu"}),
    FP8: frozenset({"nvidia"}),
    NVFP4: frozenset({"nvidia"}),
}

_QUANTIZATION_VENDOR_TEXT = {
    INT8: "detected NVIDIA GPUs and CPU targets",
    FP8: "detected NVIDIA GPUs",
    NVFP4: "detected NVIDIA GPUs",
}

# Quantization pins compute dtype so the (model, mode) measurements stay
# comparable. CPU uses FP32 rather than BF16: x86 without AVX-512 has no native
# BF16 path, so forcing BF16 there would measure emulation. Read
# `quantized_compute_dtype` rather than this table directly -- NVIDIA's answer
# depends on architecture as well as vendor.
_QUANTIZED_COMPUTE_DTYPE = {"nvidia": "bfloat16", "cpu": "float32"}

# FP8 needs Ada-class tensor cores.
_FP8_MINIMUM_CAPABILITY = 89

# Weight-only quantization needs native BF16 arithmetic, which starts at Ampere.
# Turing and older report torch.cuda.is_bf16_supported() as True but emulate it, and
# neither compute dtype is usable there. Measured on a Tesla T4 (sm75) with
# SmolLM2-135M:
#
#   unquantized FP16   17.5 ms   4/4 top-1
#   INT8 + BF16        58.0 ms   3/4 top-1   emulated, 3.3x slower than not
#                                            quantizing, and below the 4/4 gate
#                                            this model clears on sm89
#   INT8 + FP16        13.5 ms   0/4 top-1   NaN logits: dequantized products
#                                            leave FP16's 5-bit exponent range
#
# So INT8 on Turing is both slower and less accurate than plain FP16, and FP16
# compute is numerically broken. LM7 rejects it rather than offering a mode whose
# best case is a regression. See docs/quantization.md.
_BF16_MINIMUM_CAPABILITY = 80

# Validated per model *and* per mode, because the two are not interchangeable:
# LFM2.5-230M keeps its top-1 token under FP8 but diverges completely under INT8
# (0/4 prompts agreed with BF16, max logit difference 22.4 on NVIDIA sm89). A
# model earns an entry here only after its outputs have been compared against an
# unquantized baseline on real hardware. The narrower formats are NVIDIA-only, so
# _QUANTIZATION_VENDORS carries the target half of the gate. See
# docs/quantization.md.
#
# Every INT8 entry below 8B was measured on NVIDIA sm89 *and* on x86-64 CPU.
# Llama-3.1-8B was the exception, admitted on CPU evidence alone because no GPU
# here could hold it. That has now been measured on a Blackwell sm120 (96 GB),
# where the BF16 GPU baseline is 16.1 GB rather than the 30 GiB the CPU FP32 path
# needs: INT8 keeps 4/4 top-1 with a maximum logit difference of 0.39, so the
# NVIDIA half of that pair passes rather than being unmeasured.
#
# The same run measured FP8 and NVFP4 against it for the first time, and both are
# rejected -- FP8 at 3/4 and NVFP4 at 2/4. So this entry's value is unchanged and
# its evidence is not. See docs/quantization.md.
VALIDATED_WEIGHT_ONLY: dict[str, frozenset[str]] = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": frozenset({INT8, FP8}),
    "unsloth/Llama-3.2-1B-Instruct": frozenset({INT8, FP8, NVFP4}),
    "deepseek-ai/deepseek-coder-1.3b-instruct": frozenset({INT8, FP8}),
    "unsloth/Llama-3.1-8B-Instruct": frozenset({INT8}),
}
WEIGHT_ONLY_MODEL_IDS = frozenset(VALIDATED_WEIGHT_ONLY)

# Export backends that accept `quantization`. The two are unrelated mechanisms:
# ExecuTorch runs calibrated XNNPACK PTQ over the lowered graph, while OpenVINO
# compresses the IR's weights with NNCF and needs no calibration.
QUANTIZING_EXPORT_BACKENDS = frozenset({"executorch", "openvino"})

# Backends `lm7 model run` reaches through torch.export rather than through a
# torch-level compile. They need the _LogitsOnly wrapper, because a captured graph
# cannot take `use_cache=False` as a call kwarg -- see run_hf_model.
_EXPORTING_RUN_BACKENDS = frozenset({"openvino"})

# Transformers decides for itself whether to compile a decode step, and only these
# torch device types satisfy it -- see `_valid_auto_compile_criteria` in
# transformers/generation/utils.py. Anywhere else it logs "Compilation will be
# skipped" and decodes eagerly, so a passed `compile_config` is ignored. LM7 mirrors
# the upstream set rather than second-guessing it, and reports the backend that
# actually ran instead of the one it asked for. Measured locally: forcing the CPU
# path to compile anyway (the private `_compile_all_devices` flag) bought 1.06x for
# a 43 s compile, so there is nothing to reclaim here. See
# docs/huggingface-generation.md.
_COMPILED_DECODE_DEVICE_TYPES = frozenset({"cuda", "xpu", "neuron", "tpu"})


def compiles_decode(device: torch.device) -> bool:
    """Whether Transformers will compile a decode step on this torch device.

    LM7's own target mapping decides the answer: `nvidia` and `amd` land on
    `cuda` and an Intel GPU on `xpu`, so those compile, while `apple` (`mps`),
    `tpu`/`tenstorrent` (`xla`), `intel:npu` (`cpu`) and plain `cpu` do not.
    """
    return device.type in _COMPILED_DECODE_DEVICE_TYPES


# NNCF compresses every eligible layer, the vocabulary projection included, so
# it is checked per model like the runtime path. SmolLM2-135M held 4/4 top-1
# tokens at a 1.20 max logit difference and DeepSeek-Coder-1.3B 4/4 at 0.79;
# Llama-3.2-1B managed only 3/4, and excluding lm_head did not recover it because
# its embedding is tied. See docs/quantization.md.
VALIDATED_OPENVINO_INT8 = frozenset(
    {
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "deepseek-ai/deepseek-coder-1.3b-instruct",
    }
)


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
    # Set only when the backend owns the quantization rather than TorchAO. The
    # OpenVINO path leaves the torch module alone and compresses its own IR, so
    # `model_storage_bytes` cannot show the saving and this reports the weights the
    # backend actually executes.
    compiled_weight_bytes: int | None = None

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
    """Greedily generate with an eager prefill and a static-cache decode loop.

    The decode loop is compiled only where Transformers agrees to compile it --
    see ``_COMPILED_DECODE_DEVICE_TYPES``. On other targets generation still runs,
    and the returned ``backend`` says ``eager`` because that is what executed.
    """
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
    decode_is_compiled = compiles_decode(device)
    # These are also CompileConfig's own defaults, so this pins the behaviour
    # against a future change in that default rather than requesting anything new.
    compile_config = compile_config_type(
        backend="inductor",
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=None,
    )
    generation_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "cache_implementation": "static",
    }
    if decode_is_compiled:
        # Sending it anyway is not harmless: Transformers logs "You have set
        # `compile_config`, but we are unable to meet the criteria for
        # compilation. Compilation will be skipped." which reads like an LM7
        # fault on every device it does not compile on -- xla included. LM7
        # already knows the answer, so it asks only when the answer is yes.
        generation_kwargs["compile_config"] = compile_config

    _reset_peak_memory(resolved_target)
    try:
        with inference_context(resolved_target):
            synchronize(resolved_target)
            started = time.perf_counter()
            generated = model.generate(**generation_kwargs)
            synchronize(resolved_target)
            first_call_ms = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            generated = model.generate(**generation_kwargs)
            synchronize(resolved_target)
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
        backend="inductor" if decode_is_compiled else "eager",
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
    # Who applies the quantization depends on the backend. TorchAO converts torch
    # modules before compiling; NNCF compresses the OpenVINO IR during the compile,
    # so that request is forwarded as a backend option and the torch module is left
    # untouched. The saving then lives in the IR, not in `_model_storage_bytes`.
    backend_quantizes = backend == "openvino" and quantization != NO_QUANTIZATION
    quantization_ms, quantized_modules = (
        (0.0, 0) if backend_quantizes else _apply_quantization(model, resolved_target, quantization)
    )
    model_storage_bytes = _model_storage_bytes(model)
    # A backend that runs through torch.export cannot take `use_cache=False` as a
    # call kwarg: it becomes a graph input, and OpenVINO's CPU plugin rejects it
    # with "Parameter operation with dynamic rank. Operation name: use_cache".
    # _LogitsOnly pins use_cache internally and takes tensors positionally, which is
    # why the export path already wraps in it.
    wrap_for_export = backend in _EXPORTING_RUN_BACKENDS
    call_args: tuple[torch.Tensor, ...] = ()
    call_kwargs: dict[str, Any] = {**inputs, "use_cache": False}
    if wrap_for_export:
        call_args = tuple(inputs[name] for name in _CAPTURED_INPUTS if name in inputs)
        call_kwargs = {}
    wrapped = compile(
        _LogitsOnly(model).eval() if wrap_for_export else model,
        target=resolved_target,
        backend=backend,
        transfers="automatic",
        fallback="error",
        cache=False,
        options={"quantization": quantization} if backend_quantizes else None,
    )
    synchronize(resolved_target)
    started = time.perf_counter()
    output = wrapped(*call_args, **call_kwargs)
    synchronize(wrapped.target)
    first_call_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    output = wrapped(*call_args, **call_kwargs)
    synchronize(wrapped.target)
    latency_ms = (time.perf_counter() - started) * 1000

    # _LogitsOnly returns the tensor directly; a bare model returns a dataclass.
    if isinstance(output, tuple):
        output = output[0]
    logits = output if isinstance(output, torch.Tensor) else getattr(output, "logits", None)
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
        compiled_weight_bytes=_compiled_weight_bytes(wrapped),
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
    quantization: str = NO_QUANTIZATION
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
    configured_maximum = int(positions) if isinstance(positions, int) and positions > 0 else None
    return 1, min(configured_maximum or _DEFAULT_MAX_SEQUENCE, _DEFAULT_MAX_SEQUENCE)


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
    quantization: str = NO_QUANTIZATION,
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
    if quantization not in {NO_QUANTIZATION, "int8"}:
        raise UnsupportedModelError(
            f"Unsupported export quantization {quantization!r}; expected 'none' or 'int8'."
        )
    if quantization != NO_QUANTIZATION and backend not in QUANTIZING_EXPORT_BACKENDS:
        backends = ", ".join(sorted(QUANTIZING_EXPORT_BACKENDS))
        raise UnsupportedModelError(
            f"Export quantization is currently supported only by these backends: {backends}."
        )
    if quantization != NO_QUANTIZATION and backend == "executorch" and dynamic_sequence:
        raise UnsupportedModelError(
            "ExecuTorch INT8 export currently requires a fixed input shape because the "
            "captured example is also its calibration sample."
        )
    if (
        quantization != NO_QUANTIZATION
        and backend == "openvino"
        and model_id not in VALIDATED_OPENVINO_INT8
    ):
        validated = ", ".join(sorted(VALIDATED_OPENVINO_INT8))
        raise UnsupportedModelError(
            f"INT8 OpenVINO export is not validated for {model_id!r}. Currently "
            f"validated: {validated}. Llama-3.2-1B loses its top-1 token on one prompt "
            "in four under this path, which is why the gate is per model."
        )
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
    positional = backend in {"qnn", "stablehlo"}
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
        options=({"quantization": quantization} if backend in QUANTIZING_EXPORT_BACKENDS else None),
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
        quantization=quantization,
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
            return values[_QUANTIZED_COMPUTE_DTYPE.get(target.vendor, "bfloat16")]
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


def _validate_openvino_quantization(quantization: str, model_id: str | None) -> None:
    """Gate the NNCF path, which is a different mechanism from TorchAO's.

    Narrower than the TorchAO gate on purpose: NNCF only implements INT8 here, and
    the per-model list is the export path's, because it is the same compression
    applied to the same IR.
    """
    if quantization != INT8:
        raise UnsupportedModelError(
            f"The openvino backend implements {INT8!r} only; got {quantization!r}. "
            "The narrower formats are TorchAO's and need backend='inductor'."
        )
    if model_id is not None and model_id not in VALIDATED_OPENVINO_INT8:
        validated = ", ".join(sorted(VALIDATED_OPENVINO_INT8))
        raise UnsupportedModelError(
            f"INT8 OpenVINO quantization is not validated for {model_id!r}. Currently "
            f"validated: {validated}. NNCF compresses the vocabulary projection too, so "
            "the gate is per model exactly as it is for export."
        )


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
    # The OpenVINO backend compresses the IR with NNCF instead of converting torch
    # modules with TorchAO, so it answers to its own gate and skips the checks below
    # -- vendor, dtype and FP8 hardware are all TorchAO's constraints.
    if backend == "openvino":
        if target.vendor not in {"cpu", "intel"}:
            raise UnsupportedModelError(
                "The openvino backend quantizes for Intel CPU and NPU targets only."
            )
        _validate_openvino_quantization(quantization, model_id)
        return
    label = _QUANTIZATION_LABELS[quantization]
    if target.vendor not in _QUANTIZATION_VENDORS[quantization]:
        raise UnsupportedModelError(
            f"{label} weight-only quantization is supported only on "
            f"{_QUANTIZATION_VENDOR_TEXT[quantization]}."
        )
    if backend not in {"auto", "inductor"}:
        raise UnsupportedModelError(
            f"{label} weight-only quantization requires backend='auto' or backend='inductor'."
        )
    if not supports_native_bf16(target):
        raise UnsupportedModelError(
            f"{label} weight-only quantization requires NVIDIA Ampere (sm80) or newer; "
            f"{target.architecture} emulates bfloat16. Measured on a Tesla T4 (sm75), "
            "INT8 ran 3.3x slower than unquantized float16 and lost a top-1 token, and "
            "float16 compute produces NaN logits. Use quantization='none' on this GPU."
        )
    expected_dtype = _QUANTIZED_COMPUTE_DTYPE[target.vendor]
    if dtype not in {"auto", expected_dtype}:
        raise UnsupportedModelError(
            f"{label} weight-only quantization on {target.vendor} requires dtype='auto' "
            f"or dtype='{expected_dtype}'."
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
    synchronize(target)
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


def _compute_capability(target: TargetSpec) -> int | None:
    """The ``smXX`` number for a CUDA target, or None when it is not stated.

    None means "do not gate on architecture". An unqualified ``nvidia`` target has
    no architecture until it is resolved against real hardware, so refusing on a
    missing value would reject the common case.
    """
    architecture = target.architecture
    if not architecture or not architecture.startswith("sm"):
        return None
    try:
        return int(architecture.removeprefix("sm"))
    except ValueError:
        return None


def _supports_fp8(target: TargetSpec) -> bool:
    capability = _compute_capability(target)
    return capability is None or capability >= _FP8_MINIMUM_CAPABILITY


def supports_native_bf16(target: TargetSpec) -> bool:
    """Whether this target has native BF16 arithmetic rather than emulation.

    Only meaningful for NVIDIA, where LM7 knows the capability number. Everything
    else answers True, because the compute dtype for those targets is decided by
    ``_QUANTIZED_COMPUTE_DTYPE`` and not by this.
    """
    if target.vendor != "nvidia":
        return True
    capability = _compute_capability(target)
    return capability is None or capability >= _BF16_MINIMUM_CAPABILITY


def _compiled_weight_bytes(wrapped: Any) -> int | None:
    """Weight bytes of a backend-owned artifact, when the backend wrote one.

    OpenVINO saves an ``.xml`` graph beside a ``.bin`` of weights, and the ``.bin``
    is where NNCF compression shows up. Returns None for backends that keep their
    weights in the torch module, where ``_model_storage_bytes`` is already the
    answer.
    """
    artifact = getattr(wrapped, "artifact", None)
    path = getattr(artifact, "path", None)
    if path is None:
        return None
    weights = Path(path).with_suffix(".bin")
    return weights.stat().st_size if weights.is_file() else None


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
