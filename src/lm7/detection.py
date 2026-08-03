from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import re
from contextlib import AbstractContextManager
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

    devices.append(detect_cpu_target())
    return devices


def resolve_target(requested: str | TargetSpec) -> TargetSpec:
    parsed = parse_target(requested)
    if parsed.remote:
        # AOT-only targets describe the deployment device, not hardware attached
        # to this compiler host. Their backend performs the dependency checks.
        return parsed
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


def synchronize(target: TargetSpec | None) -> None:
    """Block until the target's accelerator has finished the work queued on it.

    Every timing LM7 reports depends on this being a real barrier.

    The XLA branch needs both calls. ``torch_xla.sync(wait=True)`` flushes the
    pending lazy graph and returns once the work is *dispatched* -- its ``wait``
    is the lazy-tensor barrier, not the accelerator -- so on its own it reported
    a constant 0.08 ms for a matmul whose cost grows 32x across batch sizes,
    i.e. 14,538 TFLOP/s on a chip that peaks near 918.
    ``wait_device_ops()`` is the one that blocks.
    """
    if target is None:
        return
    if target.vendor in {"nvidia", "amd"}:
        torch.cuda.synchronize(target.ordinal or 0)
    elif target.vendor == "apple":
        torch.mps.synchronize()
    elif target.vendor in {"tpu", "tenstorrent"}:
        torch_xla = importlib.import_module("torch_xla")
        torch_xla.sync(wait=True)
        xla_model = importlib.import_module("torch_xla.core.xla_model")
        xla_model.wait_device_ops()


def inference_context(target: TargetSpec | None) -> AbstractContextManager[None]:
    """Return the strongest no-grad context the target's runtime tolerates.

    ``torch.inference_mode()`` is the faster choice and the default. PyTorch/XLA
    -- the route to both TPU and Tenstorrent -- cannot use it: it needs the
    tensor version counters inference mode disables, and fails with "Cannot set
    version_counter for inference tensor" partway through a call that has
    already done real work. ``no_grad()`` is inference-safe, so those two get it.
    """
    if target is not None and target.vendor in {"tpu", "tenstorrent"}:
        return torch.no_grad()
    return torch.inference_mode()


def tpu_accelerator_type() -> str | None:
    """Report the TPU generation and slice, for example ``v6e-1``.

    ``global_runtime_device_attributes()`` carries no ``device_kind`` on a TPU
    VM -- it reports coords, ``core_on_chip``, ``num_cores`` and a name -- so
    every TPU otherwise detects as an unqualified "Google TPU" and a benchmark
    report cannot say which generation produced it. The generation lives in the
    runtime environment torch_xla reads at startup, behind a private module,
    hence the defensive import: losing it costs the label, not the detection.

    Only this one key is read. The same mapping also carries the project, node
    and zone the VM belongs to, which has no business in an LM7 device record.
    """
    try:
        tpu = importlib.import_module("torch_xla._internal.tpu")
        value = tpu.get_tpu_env().get("ACCELERATOR_TYPE")
    except (ImportError, AttributeError, KeyError, RuntimeError, OSError, ValueError):
        return None
    return str(value) if value else None


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
    accelerator_type = tpu_accelerator_type()
    device_kind = str(
        first_attributes.get("device_kind")
        or (f"Google TPU {accelerator_type}" if accelerator_type else "Google TPU")
    )
    model_match = re.search(r"\bv\d+[a-z]?\b", device_kind.lower())
    model = model_match.group(0) if model_match else None
    capabilities = {
        "openxla": getattr(torch_xla, "__version__", None),
        "pjrt_device": "TPU",
        "accelerator_type": accelerator_type,
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


CPU_INFO_PATH = Path("/proc/cpuinfo")
MEMORY_INFO_PATH = Path("/proc/meminfo")

# Instruction-set extensions that change what LM7 should compile or quantize for,
# named exactly as Linux prints them. AVX-512 and AMX decide whether BF16 is
# native or emulated; VNNI and AMX-INT8 decide whether an INT8 GEMM has a
# dot-product instruction behind it at all.
#
# LM7 records this set and interprets none of it here. The flags are the
# evidence; each caller applies its own rule, and an absent flag means "this host
# did not report it", which on a kernel without /proc is not the same as "the
# silicon lacks it".
_X86_ISA_FLAGS = frozenset(
    {
        "avx",
        "avx2",
        "f16c",
        "avx_vnni",
        "avx512f",
        "avx512bw",
        "avx512vl",
        "avx512_vnni",
        "avx512_bf16",
        "avx512_fp16",
        "amx_tile",
        "amx_bf16",
        "amx_int8",
    }
)

# The AArch64 equivalents, which the kernel prints under "Features" rather than
# "flags". Apple Silicon and Graviton are both LM7 targets, so an x86-only
# vocabulary would report them as having no vector ISA at all.
_ARM_ISA_FLAGS = frozenset({"asimd", "asimdhp", "asimddp", "bf16", "i8mm", "sve", "sve2"})

_RECORDED_ISA_FLAGS = _X86_ISA_FLAGS | _ARM_ISA_FLAGS


def detect_cpu_target() -> DeviceInfo:
    """The host CPU, described well enough to compile and quantize for it.

    Every other detector here can return an empty list; this one cannot, because
    the CPU is LM7's fallback target and must always resolve. So it degrades
    instead: on a host without ``/proc`` the topology and ISA flags are simply
    absent, and the device still carries a usable name.
    """
    info = _read_cpu_info()
    isa_extensions = info.get("isa_extensions", ())
    capabilities: dict[str, Any] = {
        "vendor_id": info.get("vendor_id"),
        "physical_cores": info.get("physical_cores"),
        # os.cpu_count() is the fallback rather than the source: it counts the
        # CPUs the OS exposes, which is what /proc/cpuinfo lists anyway, and it
        # is all that is available off Linux.
        "logical_cores": info.get("logical_cores") or os.cpu_count(),
        "isa_extensions": isa_extensions,
    }
    return DeviceInfo(
        TargetSpec("cpu", "cpu", architecture=platform.machine().lower() or None),
        info.get("model_name") or platform.processor() or "CPU",
        _read_total_memory_bytes(),
        capabilities,
    )


def _read_cpu_info() -> dict[str, Any]:
    try:
        return parse_cpu_info(CPU_INFO_PATH.read_text())
    except OSError:
        return {}


def parse_cpu_info(text: str) -> dict[str, Any]:
    """Read vendor, model, topology, and ISA flags out of ``/proc/cpuinfo`` text.

    Kept separate from the file read so it can be tested against captured
    ``/proc/cpuinfo`` from hosts LM7 has no access to — an AMX Xeon, an ARM
    server — which is the only way to cover the flags that decide the BF16 and
    INT8 questions.
    """
    vendor_id: str | None = None
    model_name: str | None = None
    flags: set[str] = set()
    # A physical core is one (socket, core) pair. Counting these rather than
    # reading "cpu cores" gets the answer right on multi-socket hosts and folds
    # SMT siblings together, which is what a thread count wants to be based on.
    cores: set[tuple[str, str]] = set()
    logical_cores = 0

    for block in re.split(r"\n\s*\n", text):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip()] = value.strip()
        if not fields:
            continue
        logical_cores += 1
        vendor_id = vendor_id or fields.get("vendor_id")
        model_name = model_name or fields.get("model name")
        # "flags" on x86, "Features" on AArch64; the two never co-occur.
        flags.update(fields.get("flags", fields.get("Features", "")).split())
        socket = fields.get("physical id")
        core = fields.get("core id")
        if socket is not None and core is not None:
            cores.add((socket, core))

    return {
        "vendor_id": vendor_id,
        "model_name": model_name,
        "logical_cores": logical_cores or None,
        "physical_cores": len(cores) or None,
        "isa_extensions": tuple(sorted(flags & _RECORDED_ISA_FLAGS)),
    }


def _read_total_memory_bytes() -> int | None:
    try:
        return parse_total_memory_bytes(MEMORY_INFO_PATH.read_text())
    except OSError:
        return None


def parse_total_memory_bytes(text: str) -> int | None:
    """Installed host RAM from ``/proc/meminfo`` text, in bytes.

    ``MemTotal`` is what is installed, not what is free now — the CPU analogue of
    a GPU's total memory, and the number that decides whether a model's weights
    can fit at all.
    """
    match = re.search(r"^MemTotal:\s+(\d+)\s*kB", text, re.MULTILINE)
    return int(match.group(1)) * 1024 if match else None
