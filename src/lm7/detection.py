from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import re
from pathlib import Path
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
    devices.extend(_detect_tenstorrent_targets())
    devices.extend(_detect_intel_npu_targets())

    cpu_arch = platform.machine().lower() or None
    devices.append(
        DeviceInfo(TargetSpec("cpu", "cpu", architecture=cpu_arch), platform.processor() or "CPU")
    )
    return devices


def resolve_target(requested: str | TargetSpec) -> TargetSpec:
    parsed = parse_target(requested)
    devices = detect_targets()
    if parsed.vendor == "auto":
        # "npu" is deliberately absent: an Intel NPU is a low-power accelerator
        # that wants INT8 weights and rejects dynamic shapes, so it is never a
        # silent substitute for the CPU. Ask for it with target="intel:npu".
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
        f"Detected: {', '.join(str(d.target) for d in devices)}.{_missing_target_hint(parsed)}"
    )


def _missing_target_hint(parsed: TargetSpec) -> str:
    """Separate "no such hardware" from "no runtime for it" where LM7 can tell.

    Only the Intel NPU needs this: it is discovered through OpenVINO rather than
    torch, so a missing plugin and a missing device look identical from the
    target list alone. The kernel driver's node tells them apart.
    """
    if parsed.kind != "npu":
        return ""
    nodes = intel_npu_device_nodes()
    if nodes:
        return (
            f" The intel_vpu driver publishes /dev/accel/{nodes[0]}, so the NPU is "
            'present and the OpenVINO NPU plugin is what is missing: install ".[openvino]".'
        )
    return (
        " Neither an OpenVINO NPU device nor an intel_vpu driver node was found; "
        "see docs/intel-npu.md."
    )


def torch_device(target: TargetSpec) -> torch.device:
    ordinal = target.ordinal or 0
    if target.vendor == "nvidia" or target.vendor == "amd":
        return torch.device("cuda", ordinal)
    if target.vendor == "intel":
        if target.kind == "npu":
            # There is no torch NPU device. The OpenVINO plugin owns the NPU and
            # takes host tensors, so LM7 leaves inputs on the CPU exactly as it
            # does for the OpenVINO CPU path.
            return torch.device("cpu")
        return torch.device("xpu", ordinal)
    if target.vendor == "apple":
        return torch.device("mps")
    if target.vendor in {"tpu", "tenstorrent"}:
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


TENSTORRENT_DEVICE_ROOT = Path("/dev/tenstorrent")


def tenstorrent_device_nodes() -> list[str]:
    """Character devices published by tt-kmd, one per Tenstorrent PCIe card.

    This is the driver-level view, independent of any Python package, so it
    tells a missing card apart from a missing runtime in diagnostics.
    """
    try:
        return sorted(node.name for node in TENSTORRENT_DEVICE_ROOT.iterdir())
    except OSError:
        return []


def activate_tenstorrent_pjrt(runtime: Any) -> str | None:
    """Select the TT PJRT device when that is safe, returning the resulting type.

    torch_plugin_tt registers the plugin through torch_xla's entry point, but the
    device type still has to be selected — this is the call tt-xla's own demos
    make. PJRT serves one device type per process, so an explicit `PJRT_DEVICE`
    and a runtime that has already come up elsewhere both win over Tenstorrent.
    """
    device_type = runtime.device_type()
    if device_type == "TT" or device_type == "TPU":
        return device_type
    requested = os.environ.get("PJRT_DEVICE")
    if requested not in {None, "", "TT"}:
        return device_type
    runtime.set_device_type("TT")
    return runtime.device_type()


def _detect_tenstorrent_targets() -> list[DeviceInfo]:
    try:
        if importlib.util.find_spec("torch_plugin_tt") is None:
            return []
        torch_xla = importlib.import_module("torch_xla")
        runtime = importlib.import_module("torch_xla.runtime")
        if activate_tenstorrent_pjrt(runtime) != "TT":
            return []
        count = runtime.addressable_device_count()
        attributes = runtime.global_runtime_device_attributes()
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError):
        # Optional runtime discovery must not make CPU/GPU detection fail when
        # the card, the tt-kmd driver, or the tt-metal runtime is absent.
        return []
    if count < 1:
        return []

    first_attributes = dict(attributes[0]) if attributes else {}
    device_kind = str(first_attributes.get("device_kind", "Tenstorrent device"))
    architecture = _tenstorrent_architecture(device_kind)
    capabilities = {
        "torch_xla": getattr(torch_xla, "__version__", None),
        "pjrt_device": "TT",
        "addressable_device_count": count,
        "device_nodes": tenstorrent_device_nodes(),
        "runtime_attributes": first_attributes,
    }
    return [
        DeviceInfo(
            TargetSpec("tenstorrent", "accelerator", architecture=architecture, ordinal=ordinal),
            device_kind,
            capabilities=capabilities,
        )
        for ordinal in range(count)
    ]


def _tenstorrent_architecture(device_kind: str) -> str | None:
    lowered = device_kind.lower()
    return next((arch for arch in ("blackhole", "wormhole") if arch in lowered), None)


INTEL_NPU_DEVICE_ROOT = Path("/dev/accel")


def intel_npu_device_nodes() -> list[str]:
    """Accel character devices published by the Linux ``intel_vpu`` driver.

    This is the driver-level view, independent of OpenVINO, so diagnostics can
    tell a host with no NPU apart from one whose OpenVINO NPU plugin is missing.
    Empty on Windows, where the NPU is only visible through the driver stack
    OpenVINO itself talks to.
    """
    try:
        return sorted(
            node.name for node in INTEL_NPU_DEVICE_ROOT.iterdir() if node.name.startswith("accel")
        )
    except OSError:
        return []


def _detect_intel_npu_targets() -> list[DeviceInfo]:
    """Intel NPUs, discovered through the OpenVINO runtime rather than torch.

    OpenVINO is the only local view of this device: it reports one entry per NPU
    in ``Core().available_devices``, named ``NPU`` or ``NPU.<ordinal>``.
    """
    try:
        if importlib.util.find_spec("openvino") is None:
            return []
        openvino = importlib.import_module("openvino")
        core = openvino.Core()
        names = [name for name in core.available_devices if name.split(".", 1)[0] == "NPU"]
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError):
        # Optional runtime discovery must not make CPU/GPU detection fail when
        # the NPU plugin is absent or its driver cannot be opened.
        return []

    device_nodes = intel_npu_device_nodes()
    devices: list[DeviceInfo] = []
    for name in names:
        devices.append(
            DeviceInfo(
                TargetSpec("intel", "npu", ordinal=_intel_npu_ordinal(name)),
                str(_openvino_property(core, name, "FULL_DEVICE_NAME") or "Intel NPU"),
                capabilities={
                    "openvino": getattr(openvino, "__version__", None),
                    "openvino_device": name,
                    # "3720" is Meteor Lake, "4000" Lunar Lake; the plugin
                    # reports it as an opaque string, so LM7 records rather
                    # than interprets it.
                    "device_architecture": _openvino_property(core, name, "DEVICE_ARCHITECTURE"),
                    "driver_version": _openvino_property(core, name, "NPU_DRIVER_VERSION"),
                    "device_nodes": device_nodes,
                },
            )
        )
    return devices


def _intel_npu_ordinal(device_name: str) -> int | None:
    _, _, suffix = device_name.partition(".")
    return int(suffix) if suffix.isdigit() else None


def _openvino_property(core: Any, device_name: str, key: str) -> str | None:
    try:
        return str(core.get_property(device_name, key))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # Property support varies by plugin version; a missing one is not a
        # reason to hide a device that available_devices already reported.
        return None
