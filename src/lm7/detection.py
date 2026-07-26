from __future__ import annotations

import importlib
import importlib.util
import platform
import re
from typing import Any

import torch

from .errors import TargetNotFoundError
from .targets import DeviceInfo, TargetSpec, parse_target


def detect_targets() -> list[DeviceInfo]:
    devices: list[DeviceInfo] = []
    try:
        if torch.cuda.is_available():
            is_rocm = bool(getattr(torch.version, "hip", None))
            vendor = "amd" if is_rocm else "nvidia"
            for ordinal in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(ordinal)
                architecture = None
                capabilities: dict[str, Any] = {}
                if is_rocm:
                    gcn_arch_name = getattr(props, "gcnArchName", None)
                    architecture = str(gcn_arch_name).split(":", 1)[0] if gcn_arch_name else None
                    capabilities["hip"] = torch.version.hip
                    capabilities["gcn_arch_name"] = gcn_arch_name
                else:
                    major, minor = torch.cuda.get_device_capability(ordinal)
                    architecture = f"sm{major}{minor}"
                    capabilities["compute_capability"] = (major, minor)
                devices.append(
                    DeviceInfo(
                        TargetSpec(vendor, "gpu", architecture, ordinal=ordinal),
                        props.name,
                        getattr(props, "total_memory", None),
                        capabilities,
                    )
                )
    except (AttributeError, RuntimeError):
        pass

    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            for ordinal in range(xpu.device_count()):
                props = xpu.get_device_properties(ordinal)
                devices.append(
                    DeviceInfo(
                        TargetSpec("intel", "gpu", ordinal=ordinal),
                        getattr(props, "name", f"Intel XPU {ordinal}"),
                        getattr(props, "total_memory", None),
                    )
                )
    except (AttributeError, RuntimeError):
        pass

    try:
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            devices.append(
                DeviceInfo(TargetSpec("apple", "gpu", architecture="metal"), "Apple Metal GPU")
            )
    except (AttributeError, RuntimeError):
        pass

    devices.extend(_detect_tpu_targets())

    cpu_arch = platform.machine().lower() or None
    devices.append(
        DeviceInfo(TargetSpec("cpu", "cpu", architecture=cpu_arch), platform.processor() or "CPU")
    )
    return devices


def resolve_target(requested: str | TargetSpec) -> TargetSpec:
    parsed = parse_target(requested)
    devices = detect_targets()
    if parsed.vendor == "auto":
        return next(
            (d.target for d in devices if d.target.kind in {"gpu", "accelerator"}),
            devices[-1].target,
        )
    for device in devices:
        target = device.target
        if parsed.vendor != target.vendor or parsed.kind != target.kind:
            continue
        if parsed.ordinal is not None and parsed.ordinal != target.ordinal:
            continue
        if parsed.architecture and parsed.architecture != target.architecture:
            continue
        if parsed.model and parsed.model not in device.name.lower().replace(" ", ""):
            continue
        return TargetSpec(
            target.vendor,
            target.kind,
            parsed.architecture or target.architecture,
            parsed.model or target.model,
            parsed.ordinal if parsed.ordinal is not None else target.ordinal,
        )
    raise TargetNotFoundError(
        f"Requested target {parsed} was not found locally. "
        f"Detected: {', '.join(str(d.target) for d in devices)}."
    )


def torch_device(target: TargetSpec) -> torch.device:
    ordinal = target.ordinal or 0
    if target.vendor == "nvidia" or target.vendor == "amd":
        return torch.device("cuda", ordinal)
    if target.vendor == "intel":
        return torch.device("xpu", ordinal)
    if target.vendor == "apple":
        return torch.device("mps")
    if target.vendor == "tpu":
        return torch.device("xla", ordinal)
    return torch.device("cpu")


def _detect_tpu_targets() -> list[DeviceInfo]:
    try:
        if importlib.util.find_spec("torch_xla") is None:
            return []
        torch_xla = importlib.import_module("torch_xla")
        runtime = importlib.import_module("torch_xla.runtime")
        if runtime.device_type() != "TPU":
            return []
        count = runtime.addressable_device_count()
        attributes = runtime.global_runtime_device_attributes()
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError):
        # Optional runtime discovery must not make CPU/GPU detection fail when
        # libtpu is missing or the current host cannot initialize PJRT.
        return []

    first_attributes = dict(attributes[0]) if attributes else {}
    device_kind = str(first_attributes.get("device_kind", "Google TPU"))
    model_match = re.search(r"\bv\d+[a-z]?\b", device_kind.lower())
    model = model_match.group(0) if model_match else None
    capabilities = {
        "openxla": getattr(torch_xla, "__version__", None),
        "pjrt_device": "TPU",
        "addressable_device_count": count,
        "runtime_attributes": first_attributes,
    }
    return [
        DeviceInfo(
            TargetSpec("tpu", "accelerator", model=model, ordinal=ordinal),
            device_kind,
            capabilities=capabilities,
        )
        for ordinal in range(count)
    ]
