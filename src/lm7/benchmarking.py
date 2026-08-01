from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import os
import platform
import statistics
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .api import compile
from .detection import detect_cpu_target, intel_npu_device_nodes
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
    threads: int | None = None,
    options: Mapping[str, Any] | None = None,
) -> BenchmarkResult:
    """Measure first-call cost and steady-state inference latency through LM7.

    ``threads`` pins torch's intra-op pool for the duration of the measurement
    and restores it afterwards. It is a benchmark parameter rather than a
    ``compile`` one on purpose: the thread count is process-global state, so a
    compiled module cannot own one without silently changing every other module
    in the process. See docs/cpu.md.
    """
    if warmup < 0:
        raise ValueError("warmup cannot be negative.")
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if threads is not None and threads < 1:
        raise ValueError("threads must be at least 1.")
    kwargs = dict(kwargs or {})
    with _intra_op_threads(threads):
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
            environment=_environment(wrapped.target, wrapped.selected_backend),
        )


@contextlib.contextmanager
def _intra_op_threads(threads: int | None) -> Iterator[None]:
    """Pin torch's intra-op thread count, then put back what was there.

    ``torch.set_num_threads`` is process-global. Restoring it is what keeps a
    sweep honest: without this, the first measurement in a
    ``for threads in (1, 2, 4, ...)`` loop would set the thread count for every
    measurement after it, and the sweep would report one number repeatedly.

    Only the intra-op pool is touched. ``set_num_interop_threads`` raises once
    inter-op work has started, so it cannot be set per measurement and is left
    alone.
    """
    if threads is None:
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _synchronize(target: TargetSpec | None) -> None:
    if target is None:
        return
    if target.vendor in {"nvidia", "amd"}:
        torch.cuda.synchronize(target.ordinal or 0)
    elif target.vendor == "apple":
        torch.mps.synchronize()
    elif target.vendor in {"tpu", "tenstorrent"}:
        torch_xla = importlib.import_module("torch_xla")
        torch_xla.sync(wait=True)


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
        # A CPU latency number is uninterpretable without the part that produced
        # it: physical cores explain the thread count, and the ISA flags explain
        # whether an INT8 or BF16 result had a native instruction behind it. Both
        # come from the same detection the target list uses.
        cpu = detect_cpu_target()
        value.update(
            {
                "device_name": cpu.name,
                "vendor_id": cpu.capabilities.get("vendor_id"),
                "logical_cpu_count": cpu.capabilities.get("logical_cores") or os.cpu_count(),
                "physical_cpu_count": cpu.capabilities.get("physical_cores"),
                "isa_extensions": list(cpu.capabilities.get("isa_extensions", ())),
                "total_memory_bytes": cpu.total_memory_bytes,
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
