from __future__ import annotations

import importlib
import importlib.util
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from .api import compile
from .detection import (
    compute_capability,
    inference_context,
    resolve_target,
    synchronize,
    torch_device,
)
from .errors import UnsupportedModelError
from .exporting import DynamicDimension, ShapeProfile
from .hub import load_transformers as _load_transformers
from .hub import parse_model_uri as _model_id
from .hub import peak_memory as _peak_memory
from .hub import reset_peak_memory as _reset_peak_memory
from .hub import resolve_dtype, source_metadata, supports_native_bf16
from .targets import TargetSpec

# The tensors _LogitsOnly forwards, and therefore the only ones captured.
_CAPTURED_INPUTS = ("input_ids", "attention_mask")

# Tokens of KV cache a decode artifact carries when nobody says otherwise. Matches
# `compile_generation`'s own default, so the JIT and AOT paths do not disagree
# about how long a sequence is by default. It is a hard bound, not a hint: the
# cache is buffers inside the artifact and cannot grow after export.
DEFAULT_MAX_CACHE_LEN = 2048

# The same, for _DecodeStep. No attention mask: the decode graph's mask is built
# against the whole static cache from `cache_position`, so there is nothing for a
# caller to pass and nothing to capture. A padded batch is out of scope here for
# exactly that reason -- see docs/exported-decode.md.
_DECODE_CAPTURED_INPUTS = ("input_ids", "cache_position")

# How many tokens a decode artifact's graph accepts per call.
#
# `dynamic` captures the sequence length as a bounded dynamic dimension, so the
# *same* graph and the *same* cache serve a whole prompt in one call and then one
# token per call after it. `single-token` fixes it at one, which makes a prompt
# cost a forward pass per token.
#
# Two graphs would be the obvious alternative and cannot work here: each exported
# program carries its own cache buffers, so a separate prefill artifact would
# fill a cache the decode artifact never sees. Sharing a cache means sharing a
# graph.
#
# Which is faster is a property of the workload, not of the capture, and both
# halves were measured -- `dynamic` wins the prompt by 10-34x and loses every
# decoded token by 1.38x. See docs/exported-decode.md#prefill-in-one-call.
DECODE_SHAPES = ("dynamic", "single-token")
DEFAULT_DECODE_SHAPE = "dynamic"

# A dynamic capture needs a range to bind, and `torch.export.Dim` wants a real
# one. The cache has to keep a slot for the token being decoded, so the longest
# prompt is one below the cache; below this a range is degenerate and the fixed
# capture is what the caller wanted anyway.
_MINIMUM_DYNAMIC_CACHE_LEN = 4

INT8 = "int8"
FP8 = "fp8"
NVFP4 = "nvfp4"
WEIGHT_ONLY_QUANTIZATIONS = frozenset({INT8, FP8, NVFP4})

# Dynamic activation quantization: the activations are quantized at runtime as
# well as the weights, so the matmul itself executes in the narrow format rather
# than dequantizing to BF16 first. This is the only family here that can cut
# arithmetic work instead of only bytes moved, and therefore the only one that
# has ever come out faster than its BF16 baseline -- see docs/quantization.md.
FP8_DYNAMIC = "fp8-dynamic"
NVFP4_DYNAMIC = "nvfp4-dynamic"

# Same arithmetic as FP8_DYNAMIC, different scale granularity. `fp8-dynamic`
# passes no `granularity` to TorchAO, which resolves to PerTensor: one scale for
# the whole activation tensor and one for the whole weight. This mode asks for
# PerRow instead -- a scale per weight output row and per activation token, which
# is the granularity TorchAO's own H100 numbers are quoted at.
#
# It is a separate mode rather than a change to `fp8-dynamic` for the reason the
# aliases below record: an existing command must keep doing what it did. Which
# one is better is a measurement, not a default -- see docs/quantization.md.
FP8_DYNAMIC_ROWWISE = "fp8-dynamic-rowwise"
DYNAMIC_ACTIVATION_QUANTIZATIONS = frozenset({FP8_DYNAMIC, FP8_DYNAMIC_ROWWISE, NVFP4_DYNAMIC})
NO_QUANTIZATION = "none"

# The pre-0.2 spellings, plus explicit long forms for the weight-only modes. The
# short names keep their existing weight-only meaning: `nvfp4` did not silently
# become an activation mode when `nvfp4-dynamic` was added, because that would
# change what an existing command does.
QUANTIZATION_ALIASES: dict[str, str] = {
    "int8-weight-only": INT8,
    "fp8-weight-only": FP8,
    "nvfp4-weight-only": NVFP4,
}

_QUANTIZATION_LABELS = {
    INT8: "INT8",
    FP8: "FP8",
    NVFP4: "NVFP4",
    FP8_DYNAMIC: "FP8 dynamic activation + FP8 weight, per-tensor scales",
    FP8_DYNAMIC_ROWWISE: "FP8 dynamic activation + FP8 weight, per-row scales",
    NVFP4_DYNAMIC: "NVFP4 dynamic activation + NVFP4 weight",
}

# Which targets each mode is allowed on. INT8 is the only mode measured off
# NVIDIA: FP8 needs Ada-class tensor cores, and NVFP4 on CPU kept only 2 of 4
# top-1 tokens and ran 8.5x slower than compiled FP32, so both stay NVIDIA-only.
_QUANTIZATION_VENDORS = {
    INT8: frozenset({"nvidia", "cpu"}),
    FP8: frozenset({"nvidia"}),
    NVFP4: frozenset({"nvidia"}),
    FP8_DYNAMIC: frozenset({"nvidia"}),
    FP8_DYNAMIC_ROWWISE: frozenset({"nvidia"}),
    NVFP4_DYNAMIC: frozenset({"nvidia"}),
}

_QUANTIZATION_VENDOR_TEXT = {
    INT8: "detected NVIDIA GPUs and CPU targets",
    FP8: "detected NVIDIA GPUs",
    NVFP4: "detected NVIDIA GPUs",
    FP8_DYNAMIC: "detected NVIDIA GPUs",
    FP8_DYNAMIC_ROWWISE: "detected NVIDIA GPUs",
    NVFP4_DYNAMIC: "detected NVIDIA GPUs",
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

# FP4 arithmetic exists only on Blackwell.
_NVFP4_DYNAMIC_MINIMUM_CAPABILITY = 100

# The capability floor per mode, above the BF16 floor every mode shares. Note
# what is *not* here: weight-only NVFP4 never issues an FP4 matmul -- it unpacks
# to BF16 inside the kernel -- so it runs on anything Ampere or newer. The
# dynamic mode asks the tensor cores to multiply in FP4 and genuinely needs the
# silicon, which is why the two NVFP4 modes have different floors.
_MODE_MINIMUM_CAPABILITY = {
    FP8: _FP8_MINIMUM_CAPABILITY,
    FP8_DYNAMIC: _FP8_MINIMUM_CAPABILITY,
    FP8_DYNAMIC_ROWWISE: _FP8_MINIMUM_CAPABILITY,
    NVFP4_DYNAMIC: _NVFP4_DYNAMIC_MINIMUM_CAPABILITY,
}

_MODE_CAPABILITY_TEXT = {
    FP8: "NVIDIA Ada (sm89), Hopper (sm90), or newer",
    FP8_DYNAMIC: "NVIDIA Ada (sm89), Hopper (sm90), or newer",
    FP8_DYNAMIC_ROWWISE: "NVIDIA Ada (sm89), Hopper (sm90), or newer",
    NVFP4_DYNAMIC: "NVIDIA Blackwell (sm100, sm120) or newer, where FP4 arithmetic exists",
}

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

# The same per-(model, mode) gate for the activation modes, kept separate because
# the two families fail differently: a weight-only mode changes stored weights and
# a dynamic mode also changes what the matmul executes in, so a model can pass one
# and fail the other. Populated only from measurements on real hardware.
#
# Measured on a Blackwell sm120 against a BF16 baseline, Llama-3.2-1B:
#
#   fp8-dynamic     4/4 top-1, max logit difference 1.59, 0.97x baseline latency
#   nvfp4-dynamic   3/4 top-1, max logit difference 5.03, 1.48x baseline latency
#
# So FP8 dynamic is admitted and NVFP4 dynamic is not. NVFP4's 3/4 is the same bar
# its weight-only counterpart fails on a second prompt set, and 4 bits of
# activation on top of 4 bits of weight is where this model stops holding its
# token. See docs/quantization.md.
#
# Measured on a Hopper sm90 (H100 80GB) against a BF16 baseline, adding the
# per-row mode and the 8B pair, which had no activation-mode evidence at all:
#
#   Llama-3.2-1B  fp8-dynamic           4/4, max logit difference 1.33, 1.02x
#                 fp8-dynamic-rowwise   4/4, max logit difference 1.09, 0.94x
#   Llama-3.1-8B  fp8-dynamic           4/4, max logit difference 0.81, 1.13x
#                 fp8-dynamic-rowwise   4/4, max logit difference 0.78, 1.08x
#
# All four clear the 4/4 bar, and per-row is both faster and closer to the
# baseline than per-tensor on each model -- which is the expected direction, since
# a scale per row fits the data better than one scale for a whole tensor.
#
# Two things worth not glossing. Weight-only FP8 scored 3/4 on the 8B here,
# reproducing the sm120 rejection recorded above on a second card -- so on that
# model the *dynamic* modes are more accurate than the weight-only one, not less.
# And only rowwise-on-1B is actually faster than not quantizing (0.94x); the other
# three are admitted on accuracy while costing latency.
VALIDATED_ACTIVATION: dict[str, frozenset[str]] = {
    "unsloth/Llama-3.2-1B-Instruct": frozenset({FP8_DYNAMIC, FP8_DYNAMIC_ROWWISE}),
    "unsloth/Llama-3.1-8B-Instruct": frozenset({FP8_DYNAMIC, FP8_DYNAMIC_ROWWISE}),
}

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


class _DecodeStep(torch.nn.Module):
    """One token in, one position of logits out, against a cache the graph owns.

    The reason a decode loop was never exportable was recorded as the pytree
    problem ``_LogitsOnly`` solves -- ``CausalLMOutputWithPast`` cannot be
    deserialized by ``torch.export.load``. That is true, and it is about the
    *output*: it is not what stopped the cache from being captured. A ``Cache``
    is not a tensor and cannot be a graph input, so the question was always where
    to put the state, and the answer is Transformers' own: hold the cache as
    **buffers on the exported module**. ``torch.export`` lifts buffers, and the
    writes survive as ``index_copy_`` on them.

    LM7 still owns no cache. ``TorchExportableModuleWithStaticCache`` is
    Transformers', it is the module ExecuTorch's LLM export already uses, and the
    reason it works for a stateful graph is one line inside it:

        layer.cumulative_length.copy_(cache_position[0])

    The write position is re-derived from an *input* on every call rather than
    advanced by one per execution. So unlike the JIT path -- where an extra
    execution silently spends a cache slot, which is what ``warmup: False``
    exists to prevent -- calling this graph twice at the same ``cache_position``
    writes the same slot twice and stays correct.

    The signature names its two tensors for the same reason ``_LogitsOnly`` does,
    and takes no ``inputs_embeds``: the wrapped module accepts one or the other,
    and an optional input that is always ``None`` is one more thing for a shape
    profile and an artifact signature to carry for no benefit.
    """

    def __init__(self, static_cache_module: torch.nn.Module) -> None:
        super().__init__()
        self.model = static_cache_module

    def forward(self, input_ids: torch.Tensor, cache_position: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, cache_position=cache_position)


def _source_metadata(model_uri: str, model_id: str, torch_dtype: torch.dtype) -> dict[str, Any]:
    """What this artifact was built from, for whoever loads it later.

    A lowered graph has weights and no name, so an artifact cannot say which
    checkpoint it is unless the export records it. That matters most for a causal
    LM: its outputs are token ids, and ids only mean words under the tokenizer
    they were trained with. Pair an artifact with the wrong one and the result is
    fluent text made of the wrong words -- no exception, no warning.

    ``tokenizer_id`` is stored separately from ``model_id`` even though they are
    the same string for every Hugging Face model LM7 exports today, because they
    are different questions and a caller reading the manifest should not have to
    know they happen to coincide.
    """
    return source_metadata(model_uri, model_id, torch_dtype, tokenizer_id=model_id)


def _decode_module(model: torch.nn.Module, *, batch_size: int, max_cache_len: int) -> _DecodeStep:
    """Wrap a causal LM into an exportable decode step, or say why it cannot be.

    The two ``generation_config`` fields are set here rather than demanded of the
    caller. Transformers reads them off the config instead of the call and raises
    a bare ``AssertionError`` when they disagree, which is a confusing thing to
    hand someone who asked LM7 to export a model, not to configure generation.
    """
    # Imported by name rather than with a plain `import`, like every other
    # optional dependency here: CI type-checks a `[dev]` install, where
    # Transformers is absent, and a plain import would have to be excused with a
    # mypy override instead of simply not being resolved.
    try:
        integration = importlib.import_module("transformers.integrations.executorch")
        exportable_static_cache = integration.TorchExportableModuleWithStaticCache
    except (ImportError, AttributeError) as exc:
        raise UnsupportedModelError(
            "Exporting a decode step needs Transformers' static-cache export wrapper "
            "(transformers.integrations.executorch). Install or upgrade the Hugging Face "
            'extra with: pip install -U "lm7[hf]".'
        ) from exc

    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        raise UnsupportedModelError(
            "Exporting a decode step needs model.generation_config, which this model has none of."
        )
    generation_config.use_cache = True
    generation_config.cache_implementation = "static"
    try:
        wrapped = exportable_static_cache(model, batch_size=batch_size, max_cache_len=max_cache_len)
    except Exception as exc:
        raise UnsupportedModelError(
            f"Could not build an exportable static-cache decode step for this model: {exc}."
        ) from exc
    return _DecodeStep(wrapped).eval()


def _decode_cache_bytes(module: torch.nn.Module) -> int:
    """Bytes of KV cache the artifact carries, which is bytes it will not reload.

    Worth reporting because it is the part of the artifact's size that is not
    weights, and the part a caller chose with ``max_cache_len``.
    """
    return sum(
        buffer.numel() * buffer.element_size()
        for name, buffer in module.named_buffers()
        if "key_cache" in name or "value_cache" in name
    )


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
    # For a decode artifact `input_tokens` is the length actually traced, which a
    # dynamic capture is not bound by: `max_tokens_per_call` is what one call may
    # carry, and `max_cache_len` is what a whole sequence may.
    decode: bool = False
    max_cache_len: int | None = None
    decode_shape: str | None = None
    max_tokens_per_call: int | None = None

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
    decode: bool = False,
    max_cache_len: int = DEFAULT_MAX_CACHE_LEN,
    decode_shape: str = DEFAULT_DECODE_SHAPE,
) -> HuggingFaceExportResult:
    """Capture a Hugging Face causal LM into an LM7 artifact.

    The example inputs come from tokenizing ``prompt``. By default the artifact
    is fixed to that input signature. Pass ``dynamic_sequence`` to capture the
    sequence length as a bounded dynamic dimension instead, so one artifact
    serves prompts of any length inside those bounds — either ``True`` for
    bounds derived from the model config, or an explicit ``(min, max)``.

    ``decode=True`` captures a **KV-cache decode step** instead of a prefill
    forward pass, against a static cache the artifact carries as buffers and
    writes into. ``max_cache_len`` sizes that cache, is fixed at export, and
    bounds prompt plus completion together. The artifact is stateful, which no
    other artifact LM7 writes is — see ``docs/exported-decode.md`` for what that
    costs a caller.

    ``decode_shape`` decides how many tokens that graph takes per call.
    ``"dynamic"`` captures the sequence length as a bounded dimension, so one
    graph prefills a whole prompt in a single call and then decodes a token at a
    time against the same cache. ``"single-token"`` fixes it at one, which costs
    a forward pass per prompt token and buys a faster decode step; the trade is
    measured in ``docs/exported-decode.md``.
    """
    from .exporting import export as export_artifact

    model_id = _model_id(model_uri)
    if decode:
        from .exporting import DECODE_BACKENDS

        if backend not in DECODE_BACKENDS:
            choices = ", ".join(sorted(DECODE_BACKENDS))
            raise UnsupportedModelError(
                f"Exporting a decode step is currently validated for these backends: {choices}. "
                f"Backend {backend!r} has never been run against a graph that writes into a KV "
                "cache, and a backend that drops those writes returns a correct first token and "
                "then diverges without raising."
            )
        if dynamic_sequence is not False and dynamic_sequence is not None:
            raise UnsupportedModelError(
                "A decode artifact takes exactly one token per call, so there is no sequence "
                "dimension for --dynamic-seq to vary. The cache length is what bounds a "
                "sequence here; set it with max_cache_len."
            )
        if quantization != NO_QUANTIZATION:
            raise UnsupportedModelError(
                "Quantized decode export is not implemented. The quantizing export backends are "
                f"{', '.join(sorted(QUANTIZING_EXPORT_BACKENDS))}, neither of which is validated "
                "for a stateful graph."
            )
        if max_cache_len < 2:
            raise UnsupportedModelError(
                "max_cache_len must be at least 2: one slot for a prompt token and one for a "
                "decoded token, or there is no decode step to export."
            )
        if decode_shape not in DECODE_SHAPES:
            choices = ", ".join(repr(name) for name in DECODE_SHAPES)
            raise UnsupportedModelError(
                f"decode_shape must be one of {choices}; got {decode_shape!r}."
            )
        if decode_shape == "dynamic" and max_cache_len < _MINIMUM_DYNAMIC_CACHE_LEN:
            raise UnsupportedModelError(
                f"A dynamic decode capture needs max_cache_len of at least "
                f"{_MINIMUM_DYNAMIC_CACHE_LEN}, because the prompt dimension is bounded at one "
                "below the cache and a range that small is degenerate. Use "
                "decode_shape='single-token' for a cache this size."
            )
    elif decode_shape != DEFAULT_DECODE_SHAPE:
        raise UnsupportedModelError(
            "decode_shape only describes a decode capture; pass decode=True as well, or leave "
            "it at its default."
        )
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

    if decode:
        decode_module = _decode_module(model, batch_size=1, max_cache_len=max_cache_len)
        # Real token ids rather than arbitrary ones, because torch.export traces
        # the example it is given. The cache position is a tensor whose values
        # change per call and whose shape tracks the tokens it accompanies -- one
        # position per token, which is what lets a single graph write a whole
        # prompt and then one token at a time.
        maximum_prompt = max_cache_len - 1 if decode_shape == "dynamic" else 1
        # A prompt longer than one call may carry is truncated for the *trace*
        # only; the artifact still serves anything inside the bound.
        example_tokens = input_ids[:, :maximum_prompt]
        decode_inputs = {
            "input_ids": example_tokens,
            "cache_position": torch.arange(example_tokens.shape[-1], dtype=torch.long),
        }
        dynamic_shapes = None
        if decode_shape == "dynamic":
            # One dimension shared by both tensors: a call carries N tokens and
            # exactly N positions, so binding them separately would let a caller
            # describe a state that cannot exist. The cache keeps a slot for the
            # token being decoded, hence one below its length.
            sequence = torch.export.Dim("sequence", min=1, max=max_cache_len - 1)
            dynamic_shapes = {"input_ids": {1: sequence}, "cache_position": {0: sequence}}
        decode_metadata: dict[str, Any] | None = {
            "batch_size": 1,
            "max_cache_len": max_cache_len,
            "cache_bytes": _decode_cache_bytes(decode_module),
            "inputs": list(_DECODE_CAPTURED_INPUTS),
            "shape": decode_shape,
            # What one call may carry. The cache length bounds a whole sequence;
            # this bounds a single call, and the two differ by the slot the next
            # token needs.
            "max_tokens_per_call": maximum_prompt,
        }
        started = time.perf_counter()
        artifact = export_artifact(
            decode_module,
            args=(),
            kwargs=decode_inputs,
            target=resolved_target,
            backend=backend,
            output=output,
            decode=decode_metadata,
            source=_source_metadata(model_uri, model_id, torch_dtype),
            dynamic_shapes=dynamic_shapes,
            # Strict, where the prefill path is not, and not a preference. Under
            # non-strict export the cache tensors arrive as *lifted constants*
            # rather than buffers, and lowering one then dies inside
            # functionalization with
            #
            #     mutating a non-functional tensor with a functional tensor
            #     is not allowed
            #
            # from `cumulative_length.copy_`. Dynamo's tracing keeps them as
            # buffers, which is what makes the writes survive. Measured both
            # ways: `export` tolerates non-strict and `aot_inductor` does not, so
            # capturing strictly for both keeps the artifact independent of which
            # backend was asked for. It is also what Transformers' own
            # `TorchExportableModuleForDecoderOnlyLM.export` defaults to.
            strict=True,
        )
        export_ms = (time.perf_counter() - started) * 1000
        return _export_result(
            artifact,
            model_uri=model_uri,
            model_id=model_id,
            target=resolved_target,
            backend=backend,
            torch_dtype=torch_dtype,
            prompt=prompt,
            input_tokens=int(example_tokens.shape[-1]),
            model=model,
            export_ms=export_ms,
            quantization=quantization,
            sequence_bounds=None,
            decode=True,
            max_cache_len=max_cache_len,
            decode_shape=decode_shape,
            max_tokens_per_call=maximum_prompt,
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
        source=_source_metadata(model_uri, model_id, torch_dtype),
        options=({"quantization": quantization} if backend in QUANTIZING_EXPORT_BACKENDS else None),
    )
    export_ms = (time.perf_counter() - started) * 1000
    return _export_result(
        artifact,
        model_uri=model_uri,
        model_id=model_id,
        target=resolved_target,
        backend=backend,
        torch_dtype=torch_dtype,
        prompt=prompt,
        input_tokens=int(input_ids.shape[-1]),
        model=model,
        export_ms=export_ms,
        quantization=quantization,
        sequence_bounds=bounds,
    )


def _export_result(
    artifact: Any,
    *,
    model_uri: str,
    model_id: str,
    target: TargetSpec,
    backend: str,
    torch_dtype: torch.dtype,
    prompt: str,
    input_tokens: int,
    model: torch.nn.Module,
    export_ms: float,
    quantization: str,
    sequence_bounds: tuple[int, int] | None,
    decode: bool = False,
    max_cache_len: int | None = None,
    decode_shape: str | None = None,
    max_tokens_per_call: int | None = None,
) -> HuggingFaceExportResult:
    """Measure what landed on disk and describe it, for either kind of artifact."""
    files = tuple(sorted(item.name for item in artifact.path.iterdir() if item.is_file()))
    artifact_bytes = sum(item.stat().st_size for item in artifact.path.rglob("*") if item.is_file())
    return HuggingFaceExportResult(
        model_uri=model_uri,
        model_id=model_id,
        target=str(target),
        backend=backend,
        dtype=str(torch_dtype).removeprefix("torch."),
        output=str(artifact.path),
        prompt=prompt,
        input_tokens=input_tokens,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        export_ms=export_ms,
        artifact_bytes=artifact_bytes,
        files=files,
        quantization=quantization,
        sequence_bounds=sequence_bounds,
        decode=decode,
        max_cache_len=max_cache_len,
        decode_shape=decode_shape,
        max_tokens_per_call=max_tokens_per_call,
    )


def _resolve_dtype(
    value: str, target: TargetSpec, quantization: str = NO_QUANTIZATION
) -> torch.dtype:
    """The generic dtype choice, plus the one part of it quantization owns."""
    quantized_default = (
        _QUANTIZED_COMPUTE_DTYPE.get(target.vendor, "bfloat16")
        if quantization in WEIGHT_ONLY_QUANTIZATIONS
        else None
    )
    return resolve_dtype(value, target, quantized_default)


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
    if quantization not in WEIGHT_ONLY_QUANTIZATIONS | DYNAMIC_ACTIVATION_QUANTIZATIONS:
        choices = ", ".join(
            [
                NO_QUANTIZATION,
                *sorted(WEIGHT_ONLY_QUANTIZATIONS | DYNAMIC_ACTIVATION_QUANTIZATIONS),
            ]
        )
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
    family = "activation" if quantization in DYNAMIC_ACTIVATION_QUANTIZATIONS else "weight-only"
    if target.vendor not in _QUANTIZATION_VENDORS[quantization]:
        raise UnsupportedModelError(
            f"{label} {family} quantization is supported only on "
            f"{_QUANTIZATION_VENDOR_TEXT[quantization]}."
        )
    if backend not in {"auto", "inductor"}:
        raise UnsupportedModelError(
            f"{label} {family} quantization requires backend='auto' or backend='inductor'."
        )
    if not supports_native_bf16(target):
        raise UnsupportedModelError(
            f"{label} {family} quantization requires NVIDIA Ampere (sm80) or newer; "
            f"{target.architecture} emulates bfloat16. Measured on a Tesla T4 (sm75), "
            "INT8 ran 3.3x slower than unquantized float16 and lost a top-1 token, and "
            "float16 compute produces NaN logits. Use quantization='none' on this GPU."
        )
    expected_dtype = _QUANTIZED_COMPUTE_DTYPE[target.vendor]
    if dtype not in {"auto", expected_dtype}:
        raise UnsupportedModelError(
            f"{label} {family} quantization on {target.vendor} requires dtype='auto' "
            f"or dtype='{expected_dtype}'."
        )
    minimum = _MODE_MINIMUM_CAPABILITY.get(quantization)
    capability = _compute_capability(target)
    if minimum is not None and capability is not None and capability < minimum:
        raise UnsupportedModelError(
            f"{label} {family} quantization requires {_MODE_CAPABILITY_TEXT[quantization]}; "
            f"{target.architecture} is below that."
        )
    validated = (
        VALIDATED_ACTIVATION
        if quantization in DYNAMIC_ACTIVATION_QUANTIZATIONS
        else VALIDATED_WEIGHT_ONLY
    )
    if model_id is not None and quantization not in validated.get(model_id, frozenset()):
        listed = ", ".join(
            f"{name} ({', '.join(sorted(modes))})" for name, modes in sorted(validated.items())
        )
        raise UnsupportedModelError(
            f"{label} {family} quantization is not validated for {model_id!r}. "
            f"Currently validated: {listed or 'nothing'}. "
            "Use quantization='none' for this model."
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
    if quantization == FP8_DYNAMIC:
        # No `granularity` argument. TorchAO resolves that to PerTensor, which is
        # what this mode has always meant; FP8_DYNAMIC_ROWWISE is the per-row one.
        return torchao_quantization.Float8DynamicActivationFloat8WeightConfig()
    if quantization == FP8_DYNAMIC_ROWWISE:
        return torchao_quantization.Float8DynamicActivationFloat8WeightConfig(
            granularity=torchao_quantization.PerRow(),
        )
    if quantization == NVFP4_DYNAMIC:
        # use_triton_kernel=False deliberately. The fused Triton activation-scaling
        # path needs MSLK (github.com/pytorch/MSLK), which is not on PyPI -- the
        # name there resolves to an empty 0.0.0 placeholder that installs nothing
        # importable. Requesting the Triton kernel without it raises
        # "mslk is required for NVFP4 triton quantization" at the first call, so
        # LM7 asks for the path that actually runs and reports which one it got.
        # See nvfp4_dynamic_kernel().
        return _load_torchao_nvfp4().NVFP4DynamicActivationNVFP4WeightConfig(
            use_triton_kernel=_mslk_available(),
            use_dynamic_per_tensor_scale=True,
        )
    return torchao_quantization.Int8WeightOnlyConfig(version=2)


def _mslk_available() -> bool:
    """Whether TorchAO's fused Triton NVFP4 activation kernel can be used."""
    return importlib.util.find_spec("mslk") is not None


def nvfp4_dynamic_kernel() -> str:
    """Which NVFP4 activation-scaling implementation this environment will use.

    Reported rather than assumed, because "NVFP4 dynamic ran" and "the fused
    Triton kernel ran" are different claims and only one of them is checkable
    from Python.
    """
    return "triton-mslk" if _mslk_available() else "torch-fallback"


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


# A dynamic mode converts exactly the layers its weight-only counterpart does, so
# the only difference between `fp8` and `fp8-dynamic` is whether activations are
# quantized. Giving the new modes a different layer selection would have made the
# pair incomparable, and the comparison is the point.
_QUANTIZATION_FILTERS = {
    INT8: _is_quantizable_linear,
    FP8: _is_fp8_quantizable_linear,
    NVFP4: _is_nvfp4_quantizable_linear,
    FP8_DYNAMIC: _is_fp8_quantizable_linear,
    FP8_DYNAMIC_ROWWISE: _is_fp8_quantizable_linear,
    NVFP4_DYNAMIC: _is_nvfp4_quantizable_linear,
}

_QUANTIZATION_SELECTS = {
    INT8: "every linear except lm_head",
    FP8: "linears whose module path contains '.mlp.'",
    NVFP4: "every linear except lm_head whose last two dimensions are multiples of 16",
    FP8_DYNAMIC: "linears whose module path contains '.mlp.'",
    FP8_DYNAMIC_ROWWISE: "linears whose module path contains '.mlp.'",
    NVFP4_DYNAMIC: "every linear except lm_head whose last two dimensions are multiples of 16",
}


def fp8_scale_granularity(quantization: str) -> str | None:
    """Which FP8 scale granularity a mode requests, or None if it is not FP8 dynamic.

    Reported rather than inferred, for the same reason `nvfp4_dynamic_kernel()`
    exists: "FP8 dynamic ran" and "it scaled per row" are different claims, and
    the second is the one the H100 numbers turn on. The scale tensor's shape is
    the ground truth -- (1, 1) per-tensor against (out_features, 1) per-row --
    and `benchmarks/activation_quant.py` checks it rather than trusting this.
    """
    if quantization == FP8_DYNAMIC:
        return "per-tensor"
    if quantization == FP8_DYNAMIC_ROWWISE:
        return "per-row"
    return None


def _compute_capability(target: TargetSpec) -> int | None:
    """The ``smXX`` number for a CUDA target, or None when it is not stated.

    Kept as an alias so the quantization gates and `lm7 targets` cannot drift
    apart on what `sm120` means. See `detection.compute_capability`.
    """
    return compute_capability(target)


def _supports_fp8(target: TargetSpec) -> bool:
    capability = _compute_capability(target)
    return capability is None or capability >= _MODE_MINIMUM_CAPABILITY[FP8]


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
