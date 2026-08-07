"""TensorRT-LLM serving runtime -- an experimental adapter.

**Deliberately not part of the `tensorrt` backend.** That one compiles a module
through Torch-TensorRT and returns something callable. This drives a serving
runtime that owns its own decode loop, paged KV cache, and batch scheduler, and
those cannot be expressed as a compiled callable. Merging them would mean one
name with two behaviours.

What LM7 does here is decide *whether and how* to hand over: resolve the target,
check the dependency set, validate the configuration against the hardware, and
compute the engine cache identity. Everything after `prepare` is TensorRT-LLM's.

TensorRT-LLM pins a version set that does not overlap this repo's other
environments -- `torch>=2.9.1,<=2.10.0a0` against 2.13 in the CUDA venv and
2.12.1 in the TensorRT one, `transformers==4.57.3` exactly against 5.14.1, and
`tensorrt~=10.14.1` against 10.16.1. It therefore needs its own environment, and
the `pinned` versions it reports are what an engine gets keyed on.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterator
from typing import Any

from ..detection import compute_capability
from ..errors import UnsupportedModelError
from ..targets import TargetSpec
from .base import GenerationChunk, RuntimeInfo, RuntimeSupport, ServeConfig

# TensorRT-LLM's kernels target Ampere and newer. Below that the runtime either
# refuses or silently falls back, and neither is a good experience to pass on.
_MINIMUM_CAPABILITY = 80

_REQUIRED = ("tensorrt_llm",)

DTYPES = ("bfloat16", "float16")
QUANTIZATIONS = ("none", "fp8")


def _module_version(name: str) -> str | None:
    try:
        return str(importlib.import_module(name).__version__)
    except Exception:  # noqa: BLE001 - an absent or version-less module is not a failure
        return None


class TensorRTLLMRuntime:
    name = "tensorrt-llm"

    def probe(self) -> RuntimeInfo:
        missing = [name for name in _REQUIRED if importlib.util.find_spec(name) is None]
        pinned = {
            "tensorrt_llm": _module_version("tensorrt_llm"),
            "torch": _module_version("torch"),
            "tensorrt": _module_version("tensorrt"),
            "transformers": _module_version("transformers"),
        }
        if missing:
            return RuntimeInfo(
                name=self.name,
                version=None,
                available=False,
                reason=(
                    "TensorRT-LLM is not installed. It pins torch >=2.9.1,<=2.10.0a0, "
                    "transformers==4.57.3 and tensorrt~=10.14.1, which conflict with LM7's "
                    "other CUDA environments, so install it into its own venv: "
                    'uv pip install --extra-index-url https://pypi.nvidia.com "tensorrt-llm". '
                    f"Missing: {', '.join(missing)}."
                ),
                pinned=pinned,
            )
        return RuntimeInfo(
            name=self.name,
            version=pinned["tensorrt_llm"],
            available=True,
            reason="TensorRT-LLM is installed.",
            pinned=pinned,
        )

    def supports(self, target: TargetSpec, model_id: str, config: ServeConfig) -> RuntimeSupport:
        if target.vendor != "nvidia":
            return RuntimeSupport(False, f"TensorRT-LLM is NVIDIA only; got {target.vendor!r}.")
        capability = compute_capability(target)
        if capability is not None and capability < _MINIMUM_CAPABILITY:
            return RuntimeSupport(
                False,
                f"TensorRT-LLM needs NVIDIA Ampere (sm80) or newer; {target.architecture} is below "
                "that.",
            )
        if config.dtype not in DTYPES:
            return RuntimeSupport(
                False, f"dtype must be one of {', '.join(DTYPES)}; got {config.dtype!r}."
            )
        if config.quantization not in QUANTIZATIONS:
            return RuntimeSupport(
                False,
                f"quantization must be one of {', '.join(QUANTIZATIONS)}; "
                f"got {config.quantization!r}.",
            )
        if config.quantization == "fp8" and capability is not None and capability < 89:
            return RuntimeSupport(
                False,
                f"FP8 needs NVIDIA Ada (sm89) or newer; {target.architecture} is below that.",
            )
        if not 0.0 < config.kv_cache_free_gpu_memory_fraction <= 1.0:
            return RuntimeSupport(
                False,
                "kv_cache_free_gpu_memory_fraction must be in (0, 1]; got "
                f"{config.kv_cache_free_gpu_memory_fraction}.",
            )
        return RuntimeSupport(True, "TensorRT-LLM supports this target and configuration.")

    def prepare(self, target: TargetSpec, model_id: str, config: ServeConfig) -> Any:
        """Hand over to TensorRT-LLM's own engine build and runtime.

        LM7 stops here. `LLM(...)` owns the engine build, the paged KV cache, the
        scheduler and the decode loop; re-implementing any of it in this file
        would be the failure mode this adapter exists to avoid.
        """
        info = self.probe()
        if not info.available:
            raise UnsupportedModelError(info.reason)
        support = self.supports(target, model_id, config)
        if not support.supported:
            raise UnsupportedModelError(support.reason)

        from tensorrt_llm import BuildConfig  # type: ignore[import-not-found]

        # `tensorrt_llm.LLM` is *not* the TensorRT engine path on 1.2.x. The
        # public class became the PyTorch backend, which rejects `build_config`
        # with "specific to TensorRT backend and cannot be used with PyTorch
        # backend" and points at `_tensorrt_engine` instead. Engine execution is
        # the reason this runtime exists, so that is the one imported here --
        # accepting that a leading-underscore module is a version risk, which is
        # exactly why the engine cache is keyed on the TensorRT-LLM version.
        try:
            from tensorrt_llm._tensorrt_engine import LLM  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: no cover - depends on the installed version
            raise UnsupportedModelError(
                "This TensorRT-LLM build does not expose `tensorrt_llm._tensorrt_engine.LLM`, "
                "which is the TensorRT engine path on 1.2.x. Its public `LLM` is the PyTorch "
                f"backend and does not build engines. Installed: {info.version}. ({error})"
            ) from error

        build = BuildConfig(
            max_batch_size=config.max_batch_size,
            max_input_len=config.max_input_len,
            max_seq_len=config.max_input_len + config.max_output_len,
        )
        return LLM(
            model=model_id,
            dtype=config.dtype,
            build_config=build,
            kv_cache_config={"free_gpu_memory_fraction": config.kv_cache_free_gpu_memory_fraction},
        )

    def generate(
        self, prepared: Any, prompt: str, *, max_new_tokens: int
    ) -> Iterator[GenerationChunk]:
        """Stream deltas out of TensorRT-LLM's generator.

        The runtime yields a cumulative string per step, so the delta is computed
        here -- `GenerationChunk.text` is documented as a delta and a caller that
        wants the total can accumulate.
        """
        from tensorrt_llm import SamplingParams  # type: ignore[import-not-found]

        emitted = ""
        for output in prepared.generate_async(
            prompt,
            sampling_params=SamplingParams(max_tokens=max_new_tokens),
            streaming=True,
        ):
            text = output.outputs[0].text
            delta, emitted = text[len(emitted) :], text
            if delta:
                yield GenerationChunk(text=delta)
        yield GenerationChunk(text="", finished=True)
