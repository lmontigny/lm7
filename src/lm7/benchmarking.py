from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch

from .api import compile
from .detection import (
    intel_npu_device_nodes,
    resolve_target,
    synchronize,
    tpu_accelerator_type,
)
from .targets import TargetSpec


@dataclass(frozen=True)
class BenchmarkResult:
    target: str
    backend: str
    first_call_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    samples_per_second: float
    peak_memory_bytes: int | None
    warmup: int
    repeats: int
    batch_size: int
    environment: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark(
    model: torch.nn.Module,
    *,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
    target: str | TargetSpec = "auto",
    backend: str = "auto",
    warmup: int = 5,
    repeats: int = 30,
    options: Mapping[str, Any] | None = None,
) -> BenchmarkResult:
    """Measure first-call cost and steady-state inference latency through LM7."""
    kwargs = dict(kwargs or {})
    wrapped = compile(
        model,
        target=target,
        backend=backend,
        transfers="automatic",
        fallback="error",
        cache=False,
        options=options,
    )
    # Resolved here rather than read off `wrapped` afterwards because the timing
    # loop has to synchronize the right device from the first call onwards, and
    # `compile` does not resolve until that call happens. Detection is
    # deterministic, so this is the same answer `wrapped.target` reports below.
    resolved = target if isinstance(target, TargetSpec) else resolve_target(target)
    result = benchmark_callable(
        lambda: wrapped(*args, **kwargs),
        target=resolved,
        backend=backend,
        warmup=warmup,
        repeats=repeats,
        batch_size=_batch_size((args, kwargs)),
    )

    assert wrapped.target is not None
    assert wrapped.selected_backend is not None
    # The backend that was *selected* rather than the one that was asked for --
    # `backend="auto"` is a request, not an answer.
    return replace(
        result,
        target=str(wrapped.target),
        backend=wrapped.selected_backend,
        peak_memory_bytes=_peak_memory(wrapped.target),
        environment=_environment(wrapped.target, wrapped.selected_backend),
    )


def benchmark_callable(
    call: Callable[[], Any],
    *,
    target: TargetSpec,
    backend: str,
    warmup: int = 5,
    repeats: int = 30,
    batch_size: int = 1,
) -> BenchmarkResult:
    """Time an arbitrary callable the way :func:`benchmark` times an LM7 one.

    Exists so that a baseline which deliberately does *not* go through LM7 --
    a direct ``torch.compile``, say, which is what a user writing against one
    vendor's stack would call -- can be compared against one that does, without
    the two being timed by different code. Two harnesses in this repo already
    disagree by 2.3x on the same card purely from building inputs differently,
    and "how much does LM7 cost over the toolchain underneath it" is a question
    where that much drift is the whole answer.

    The caller owns everything outside the timing: this moves no tensors, sets
    no inference mode, and compiles nothing. A baseline that means anything has
    to place its own model and inputs, because doing that per call is one of the
    things being measured.
    """
    if warmup < 0:
        raise ValueError("warmup cannot be negative.")
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if _uses_cuda_runtime(target) and torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    call()
    synchronize(target)
    first_call_ms = (time.perf_counter() - started) * 1000

    for _ in range(warmup):
        call()
    synchronize(target)

    latencies_ms = []
    for _ in range(repeats):
        synchronize(target)
        started = time.perf_counter()
        call()
        synchronize(target)
        latencies_ms.append((time.perf_counter() - started) * 1000)

    median_ms = statistics.median(latencies_ms)
    return BenchmarkResult(
        target=str(target),
        backend=backend,
        first_call_ms=first_call_ms,
        latency_median_ms=median_ms,
        latency_p95_ms=_percentile(latencies_ms, 0.95),
        samples_per_second=batch_size / (median_ms / 1000),
        peak_memory_bytes=_peak_memory(target),
        warmup=warmup,
        repeats=repeats,
        batch_size=batch_size,
        environment=_environment(target, backend),
    )


def _peak_memory(target: TargetSpec) -> int | None:
    if target.vendor not in {"nvidia", "amd"}:
        return None
    return torch.cuda.max_memory_allocated(target.ordinal or 0)


def _environment(target: TargetSpec, backend: str | None = None) -> Mapping[str, Any]:
    value: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    if target.vendor == "cpu":
        cpu_backend = getattr(getattr(torch, "backends", None), "cpu", None)
        get_capability = getattr(cpu_backend, "get_cpu_capability", None)
        value.update(
            {
                "device_name": platform.processor() or platform.machine() or "CPU",
                "logical_cpu_count": os.cpu_count(),
                "torch_threads": torch.get_num_threads(),
                "cpu_capability": get_capability() if callable(get_capability) else None,
            }
        )
    elif target.vendor in {"nvidia", "amd"}:
        ordinal = target.ordinal or 0
        value.update(
            {
                "device_name": torch.cuda.get_device_name(ordinal),
                "cuda": torch.version.cuda,
                "hip": torch.version.hip,
                "architecture": target.architecture,
            }
        )
        # NVIDIA-only, and deliberately not filled in for AMD. ROCm answers
        # `get_device_capability` too -- it returns (9, 4) on a gfx942 -- so
        # calling it for both vendors put a number in every AMD report that
        # looks like a CUDA compute capability, is not one, and does not
        # correspond to any NVIDIA part. `architecture` above carries the answer
        # for both; this stays where it means something.
        if target.vendor == "nvidia":
            value["compute_capability"] = list(torch.cuda.get_device_capability(ordinal))
    elif target.vendor == "apple":
        value.update({"device_name": "Apple Metal GPU", "mps_built": torch.backends.mps.is_built()})
    elif target.vendor == "intel" and target.kind == "npu":
        # The NPU has no torch device to interrogate, so the runtime that owns
        # it -- and the driver nodes under it -- are the environment.
        value.update(
            {
                "device_name": "Intel NPU",
                "openvino": _package_version("openvino"),
                "npu_device_nodes": intel_npu_device_nodes(),
            }
        )
    elif target.vendor in {"tpu", "tenstorrent"}:
        device_name = "Google TPU" if target.vendor == "tpu" else "Tenstorrent device"
        try:
            xr = importlib.import_module("torch_xla.runtime")
            value.update(
                {
                    "device_name": device_name,
                    "pjrt_device": xr.device_type(),
                    "addressable_device_count": xr.addressable_device_count(),
                }
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
            value.update({"device_name": device_name, "pjrt_error": str(exc)})
        if target.vendor == "tpu":
            # Without this a committed report cannot say which TPU generation
            # produced it, and the numbers do not transfer between them.
            value.update(
                {
                    "accelerator_type": tpu_accelerator_type(),
                    "torch_xla": _package_version("torch-xla"),
                    "libtpu": _package_version("libtpu"),
                }
            )
        if target.vendor == "tenstorrent":
            value.update(
                {
                    "pjrt_plugin_tt": _package_version("pjrt-plugin-tt"),
                    "torch_xla": _package_version("torch-xla"),
                }
            )
    if backend == "tensorrt":
        value.update(
            {
                "torch_tensorrt": _package_version("torch-tensorrt"),
                "tensorrt": _package_version("tensorrt"),
            }
        )

    return value


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _uses_cuda_runtime(target: str | TargetSpec) -> bool:
    vendor = target.vendor if isinstance(target, TargetSpec) else target.split(":", 1)[0].lower()
    return vendor in {"auto", "nvidia", "amd"}


def _batch_size(value: Any) -> int:
    return _find_batch_size(value) or 1


def _find_batch_size(value: Any) -> int | None:
    if isinstance(value, torch.Tensor) and value.dim() > 0:
        return value.shape[0]
    if isinstance(value, Mapping):
        for item in value.values():
            size = _find_batch_size(item)
            if size is not None:
                return size
    if isinstance(value, (tuple, list)):
        for item in value:
            size = _find_batch_size(item)
            if size is not None:
                return size
    return None


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
