from __future__ import annotations

import importlib
import os
import platform
import statistics
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .api import compile
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
    if warmup < 0:
        raise ValueError("warmup cannot be negative.")
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
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
    if _uses_cuda_runtime(target) and torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    wrapped(*args, **kwargs)
    _synchronize(wrapped.target)
    first_call_ms = (time.perf_counter() - started) * 1000

    for _ in range(warmup):
        wrapped(*args, **kwargs)
    _synchronize(wrapped.target)

    latencies_ms = []
    for _ in range(repeats):
        _synchronize(wrapped.target)
        started = time.perf_counter()
        wrapped(*args, **kwargs)
        _synchronize(wrapped.target)
        latencies_ms.append((time.perf_counter() - started) * 1000)

    assert wrapped.target is not None
    assert wrapped.selected_backend is not None
    median_ms = statistics.median(latencies_ms)
    batch_size = _batch_size((args, kwargs))
    return BenchmarkResult(
        target=str(wrapped.target),
        backend=wrapped.selected_backend,
        first_call_ms=first_call_ms,
        latency_median_ms=median_ms,
        latency_p95_ms=_percentile(latencies_ms, 0.95),
        samples_per_second=batch_size / (median_ms / 1000),
        peak_memory_bytes=_peak_memory(wrapped.target),
        warmup=warmup,
        repeats=repeats,
        batch_size=batch_size,
        environment=_environment(wrapped.target),
    )


def _synchronize(target: TargetSpec | None) -> None:
    if target is None:
        return
    if target.vendor in {"nvidia", "amd"}:
        torch.cuda.synchronize(target.ordinal or 0)
    elif target.vendor == "apple":
        torch.mps.synchronize()
    elif target.vendor == "tpu":
        torch_xla = importlib.import_module("torch_xla")
        torch_xla.sync(wait=True)


def _peak_memory(target: TargetSpec) -> int | None:
    if target.vendor not in {"nvidia", "amd"}:
        return None
    return torch.cuda.max_memory_allocated(target.ordinal or 0)


def _environment(target: TargetSpec) -> Mapping[str, Any]:
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
                "compute_capability": list(torch.cuda.get_device_capability(ordinal)),
            }
        )
    elif target.vendor == "apple":
        value.update({"device_name": "Apple Metal GPU", "mps_built": torch.backends.mps.is_built()})
    elif target.vendor == "tpu":
        try:
            xr = importlib.import_module("torch_xla.runtime")
            value.update(
                {
                    "device_name": "Google TPU",
                    "pjrt_device": xr.device_type(),
                    "addressable_device_count": xr.addressable_device_count(),
                }
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
            value.update({"device_name": "Google TPU", "pjrt_error": str(exc)})
    return value


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
