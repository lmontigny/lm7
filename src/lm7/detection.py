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

# NVIDIA names its silicon by generation and torch reports only the compute
# capability, so `sm120` says nothing to a reader who has not memorized the
# table. The TPU path already names its generation -- see
# `tpu_accelerator_type` -- and this is the NVIDIA counterpart.
#
# Thresholds are descending and inclusive, which is what makes the table short:
# 86 and 87 fall through to Ampere, 89 is its own entry because Ada sits between
# Ampere and Hopper, and both 100 (datacenter) and 120 (RTX/workstation) are
# Blackwell. An unrecognized number returns None rather than guessing, so a
# generation newer than this table costs the label and nothing else.
_NVIDIA_GENERATIONS: tuple[tuple[int, str], ...] = (
    (100, "Blackwell"),
    (90, "Hopper"),
    (89, "Ada Lovelace"),
    (80, "Ampere"),
    (75, "Turing"),
    (70, "Volta"),
    (60, "Pascal"),
    (50, "Maxwell"),
)

# The AMD counterpart, and it cannot be a threshold table like the one above.
# CUDA capabilities sort as concatenated digits, so `>=` orders them correctly.
# `gfx` numbers do not: `gfx1100` is a consumer RDNA 3 part and `gfx942` is a
# datacenter CDNA 3 one, so the larger number is the *less* capable chip for
# every format this file reports. The two lines are separate product families
# that happen to share a numbering space, so the mapping is exact and an
# unrecognized string returns None rather than falling through to a neighbour.
#
# `gfx90a` is why the key is the string and not an int: the last position is
# hexadecimal, so CDNA 2 does not parse as a number at all.
#
# Every value here is read from AMD's ISA documentation, and none of it has been
# confirmed against hardware -- LM7 has never run on an AMD GPU. See
# docs/limitations.md#hardware-validation.
_AMD_GENERATIONS: dict[str, str] = {
    "gfx906": "Vega 20",
    "gfx908": "CDNA 1",
    "gfx90a": "CDNA 2",
    "gfx942": "CDNA 3",
    "gfx950": "CDNA 4",
    "gfx1030": "RDNA 2",
    "gfx1100": "RDNA 3",
    "gfx1101": "RDNA 3",
    "gfx1102": "RDNA 3",
    "gfx1200": "RDNA 4",
    "gfx1201": "RDNA 4",
}

# Which `gfx` architectures compute each format, as the sets that differ. Formats
# every entry in `_AMD_GENERATIONS` handles natively -- fp32, fp16 -- are not
# listed, and neither is fp4, which no shipping AMD part in this table has.
#
# The CDNA line gained bf16 matrix instructions with CDNA 1 and FP8 with CDNA 3;
# the RDNA line gained bf16 with RDNA 3 (WMMA) and FP8 with RDNA 4. Vega 20 has
# neither and is here because ROCm still supports it.
_AMD_BF16 = frozenset(
    {"gfx908", "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"}
)
_AMD_INT8 = frozenset(_AMD_GENERATIONS) - {"gfx906"}
_AMD_FP8 = frozenset({"gfx942", "gfx950", "gfx1200", "gfx1201"})

# FP8 is one name for two incompatible encodings, and this is the difference
# between "AMD has fp8 too" and a comparable measurement. CDNA 3 implements the
# `fnuz` variants -- no infinities, one NaN, and an exponent bias one greater
# than OCP's -- so `torch.float8_e4m3fnuz` is the dtype that exists there, not
# the `torch.float8_e4m3fn` that `sm89`+ implements. CDNA 4 and RDNA 4 moved to
# the OCP encoding NVIDIA already used.
#
# Consequence for anything reading this: an FP8 number from `gfx942` and one from
# `sm90` were not produced in the same format, and a kernel written against one
# encoding does not run against the other. Unconfirmed on hardware, like the rest
# of this table.
_AMD_FP8_FNUZ = frozenset({"gfx942"})

# Whether the hardware computes a format or fakes it. The distinction matters
# because torch will happily run an emulated format and report success: a Tesla
# T4 answers True to `torch.cuda.is_bf16_supported()` and then emulates BF16,
# which measured 3.4x slower than FP16 on the same model. Reporting "emulated"
# is the difference between an unexplained slowdown and an expected one.
#
# Thresholds, and what each one is:
#   fp16   sm60+  native half arithmetic (Pascal packed FP16 onwards)
#   bf16   sm80+  native; sm70-sm79 runs it emulated; below that, absent
#   int8   sm61+  native integer dot product (dp4a)
#   fp8    sm89+  Ada-class tensor cores
#   fp4    sm100+ Blackwell tensor cores
#
# These are hardware facts and say nothing about whether LM7 issues work in a
# format. Weight-only NVFP4 dequantizes to BF16 inside the kernel, so `fp4:
# native` on a Blackwell does not mean any FP4 matmul is executed -- see
# docs/nvidia-blackwell.md.
NATIVE = "native"
EMULATED = "emulated"
ABSENT = "absent"


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
                spec = TargetSpec(vendor, "gpu", architecture, ordinal=ordinal)
                generation = amd_generation(spec) if is_rocm else nvidia_generation(spec)
                if generation is not None:
                    capabilities["generation"] = generation
                precisions = precision_support(spec)
                if precisions:
                    capabilities["precision"] = precisions
                # Reported beside the precision map rather than inside it, because
                # it qualifies one entry rather than adding one: `fp8: native` on
                # gfx942 and on sm90 are different encodings. See `amd_fp8_format`.
                fp8_format = amd_fp8_format(spec)
                if fp8_format is not None:
                    capabilities["fp8_format"] = fp8_format
                devices.append(
                    DeviceInfo(
                        spec,
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


def compute_capability(target: TargetSpec) -> int | None:
    """The ``smXX`` number for a CUDA target, or None when it is not stated.

    None means "do not gate on architecture". An unqualified ``nvidia`` target
    has no architecture until it is resolved against real hardware, so refusing
    on a missing value would reject the common case.

    The number is compared as a plain integer everywhere, which orders correctly
    only because CUDA capabilities happen to sort that way as concatenated
    digits: 75 < 80 < 86 < 89 < 90 < 100 < 120. Blackwell is above Ada by this
    comparison, which is why `sm120` needed no special case anywhere in LM7.
    """
    architecture = target.architecture
    if not architecture or not architecture.startswith("sm"):
        return None
    try:
        return int(architecture.removeprefix("sm"))
    except ValueError:
        return None


def nvidia_generation(target: TargetSpec) -> str | None:
    """The marketing generation for a CUDA target, for example ``Blackwell``.

    Returns None for a non-NVIDIA target, an unqualified one, or a capability
    number newer than `_NVIDIA_GENERATIONS`.
    """
    if target.vendor != "nvidia":
        return None
    capability = compute_capability(target)
    if capability is None:
        return None
    for threshold, name in _NVIDIA_GENERATIONS:
        if capability >= threshold:
            return name
    return None


def gcn_architecture(target: TargetSpec) -> str | None:
    """The ``gfxNNN`` string for an AMD target, or None when it is not stated.

    The counterpart to `compute_capability`, and it stays a string on purpose:
    `gfx90a` does not parse as an integer, and `gfx` numbers do not order by
    capability anyway -- see `_AMD_GENERATIONS`. Callers match rather than
    compare.
    """
    architecture = target.architecture
    if target.vendor != "amd" or not architecture or not architecture.startswith("gfx"):
        return None
    return architecture


def amd_generation(target: TargetSpec) -> str | None:
    """The architecture generation for an AMD target, for example ``CDNA 3``.

    Returns None for a non-AMD target, an unqualified one, or a `gfx` string
    outside `_AMD_GENERATIONS`. AMD publishes no single marketing name spanning
    both product lines, so this reports the ISA family -- CDNA for the Instinct
    parts, RDNA for the Radeon ones -- which is the name their own ISA documents
    use.
    """
    architecture = gcn_architecture(target)
    if architecture is None:
        return None
    return _AMD_GENERATIONS.get(architecture)


def amd_fp8_format(target: TargetSpec) -> str | None:
    """Which FP8 encoding this AMD architecture implements, or None if it has no FP8.

    See `_AMD_FP8_FNUZ`: "fp8" names two incompatible encodings, and reporting
    only that the format is native would equate a `gfx942` measurement with an
    `sm90` one.
    """
    architecture = gcn_architecture(target)
    if architecture is None or architecture not in _AMD_FP8:
        return None
    return "fnuz" if architecture in _AMD_FP8_FNUZ else "ocp"


def precision_support(target: TargetSpec) -> dict[str, str]:
    """Which numeric formats this target computes natively, fakes, or lacks.

    NVIDIA and AMD are characterized, because they are the vendors whose
    architecture string LM7 already knows. Everything else returns an empty
    mapping rather than a guess: reporting "native" for a CPU whose AVX-512 BF16
    support was never probed would be exactly the kind of unmeasured claim this
    reports against. An unqualified `nvidia` or `amd` target also returns empty,
    since its architecture is unknown until it resolves against real hardware.

    Both vendors answer with the same six keys, so a row from one card compares
    against a row from another. What that comparison does *not* carry is the FP8
    encoding, which differs between them -- read `amd_fp8_format` alongside this.

    The NVIDIA half is confirmed on three generations of real silicon. The AMD
    half is read from ISA documentation and has never been run.
    """
    if target.vendor == "amd":
        return _amd_precision_support(target)
    capability = compute_capability(target) if target.vendor == "nvidia" else None
    if capability is None:
        return {}
    return {
        "fp32": NATIVE,
        "fp16": NATIVE if capability >= 60 else ABSENT,
        "bf16": NATIVE if capability >= 80 else (EMULATED if capability >= 70 else ABSENT),
        "int8": NATIVE if capability >= 61 else ABSENT,
        "fp8": NATIVE if capability >= 89 else ABSENT,
        "fp4": NATIVE if capability >= 100 else ABSENT,
    }


def _amd_precision_support(target: TargetSpec) -> dict[str, str]:
    """The AMD half of `precision_support`, keyed on membership rather than order.

    Nothing here is ever `emulated`. That state exists because a Tesla T4 answers
    True to `torch.cuda.is_bf16_supported()` and then fakes it; no equivalent
    case is known on any `gfx` part in `_AMD_GENERATIONS`, and inventing one
    would be the guess this function refuses to make. A part outside that table
    returns an empty mapping for the same reason.
    """
    architecture = gcn_architecture(target)
    if architecture is None or architecture not in _AMD_GENERATIONS:
        return {}
    return {
        "fp32": NATIVE,
        "fp16": NATIVE,
        "bf16": NATIVE if architecture in _AMD_BF16 else ABSENT,
        "int8": NATIVE if architecture in _AMD_INT8 else ABSENT,
        "fp8": NATIVE if architecture in _AMD_FP8 else ABSENT,
        # No shipping part in `_AMD_GENERATIONS` computes FP4. CDNA 4 adds FP6
        # and FP4 on paper; `gfx950` stays absent here until that is read from
        # something better than a product announcement.
        "fp4": ABSENT,
    }


def cuda_build_targets(target: TargetSpec) -> dict[str, Any] | None:
    """Which CUDA architectures the installed PyTorch was *built* for.

    `precision_support` describes the silicon. This describes the wheel, and the
    two disagree more often than is comfortable: a format the hardware computes
    natively is still unreachable if the PyTorch build shipped no kernels for
    this architecture, in which case CUDA JITs from PTX at first use -- slower to
    start, and only for features PTX can express.

    From compute capability 9.0 onward NVIDIA splits the target in two. `sm_90`
    is the portable, forward-compatible one; `sm_90a` adds the
    architecture-specific instructions (Hopper's `wgmma` and TMA among them) and
    is *not* forward compatible, so a kernel needing them must be compiled for
    the `a` variant explicitly. A build carrying only `sm_90` runs correctly on
    an H100 and cannot reach those instructions at all. Reporting the arch list
    is the difference between "my GPU supports this" and "my install can use it".

    ROCm answers the same question through the same API -- `get_arch_list()`
    returns `['gfx900', 'gfx906', 'gfx942', ...]` there -- and the question is the
    one that matters most on AMD, where wheels routinely ship kernels for a
    narrower set of parts than the runtime supports. The function keeps its
    CUDA-era name, and the `cuda_build` key with it, because that name is already
    in the `lm7 targets --json` output that other things read.

    The `a`-variant half is NVIDIA-only: ROCm expresses the equivalent as feature
    suffixes on the target string (`gfx942:sramecc+:xnack-`) rather than as a
    separate architecture, and LM7 strips those in `detect_targets`. AMD reports
    `architecture_specific: None` rather than False, which is the difference
    between "no" and "this vendor does not answer that question".

    Returns None for anything that is not a resolved NVIDIA or AMD target, or
    when torch reports no arch list (a CPU-only build).
    """
    if target.vendor == "amd":
        native = gcn_architecture(target)
    elif target.vendor == "nvidia":
        capability = compute_capability(target)
        native = f"sm_{capability}" if capability is not None else None
    else:
        return None
    try:
        architectures = list(torch.cuda.get_arch_list())
    except Exception:  # noqa: BLE001 - a torch without CUDA answers nothing useful
        return None
    if not architectures:
        return None
    return {
        "arch_list": architectures,
        # Whether this exact architecture has compiled kernels in the wheel. False
        # means it runs by JIT-ing PTX from an older target, which is legal and
        # slower to warm up. On ROCm there is no PTX equivalent: a missing `gfx`
        # target is a hard failure at load ("no kernel image is available"), not a
        # slow start, so False is a stronger signal there than here.
        "native_kernels": native in architectures if native else None,
        # The `a` variant, which is what architecture-specific instructions need.
        "architecture_specific": (
            (f"{native}a" in architectures) if native and target.vendor == "nvidia" else None
        ),
    }


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
CPU_TOPOLOGY_PATH = Path("/sys/devices/system/cpu")

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

# AArch64 prints no "model name" at all -- the kernel publishes the numeric
# "CPU implementer" and "CPU part" identifiers instead. Without these two tables
# a host that says "AMD EPYC 7B13" on x86 says "aarch64" on Arm, which names no
# chip and cannot be looked up. The implementers are the ones whose parts LM7
# might meet; the parts are Arm's own server cores, which is where Graviton,
# Azure Cobalt, Grace and Ampere Altra all land, because those vendors ship
# Neoverse cores rather than their own designs.
_ARM_IMPLEMENTERS = {
    "0x41": "Arm",
    "0x42": "Broadcom",
    "0x43": "Cavium",
    "0x48": "HiSilicon",
    "0x4e": "NVIDIA",
    "0x50": "Ampere",
    "0x51": "Qualcomm",
    "0x53": "Samsung",
    "0x56": "Marvell",
    "0x61": "Apple",
    "0xc0": "Ampere",
}

_ARM_PARTS = {
    "0xd0c": "Neoverse N1",
    "0xd40": "Neoverse V1",
    "0xd49": "Neoverse N2",
    "0xd4f": "Neoverse V2",
    "0xd84": "Neoverse V3",
    "0xd8e": "Neoverse N3",
}


def _arm_model_name(fields: dict[str, str]) -> str | None:
    """Name an AArch64 CPU from the identifiers the kernel does print.

    Degrades in two steps rather than guessing: a known Arm core gets its
    marketing name, a known implementer with an unrecognised part gets the
    vendor and the raw part number -- still something to look up -- and an
    implementer this does not know returns None, leaving the caller's existing
    fallback in place.
    """
    implementer = fields.get("CPU implementer", "").strip().lower()
    part = fields.get("CPU part", "").strip().lower()
    if not implementer or not part:
        return None
    vendor = _ARM_IMPLEMENTERS.get(implementer)
    if vendor is None:
        return None
    # Only implementer 0x41 is Arm itself, so only there does a part number mean
    # a Neoverse core; another vendor's 0xd49 would be its own design.
    core = _ARM_PARTS.get(part) if implementer == "0x41" else None
    return f"{vendor} {core}" if core else f"{vendor} {part}"


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
        # /proc/cpuinfo answers this on x86 and never on AArch64, which prints no
        # topology fields at all. sysfs carries the same (socket, core) pairs on
        # both, so it is the fallback rather than the source: x86 keeps reading
        # the file it always read.
        "physical_cores": info.get("physical_cores") or read_physical_cores(CPU_TOPOLOGY_PATH),
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


def read_physical_cores(root: Path) -> int | None:
    """Count physical cores from sysfs topology, for hosts ``/proc/cpuinfo`` fails.

    AArch64 prints no ``physical id`` or ``core id``, so the ``/proc/cpuinfo``
    count is always ``None`` there — but the kernel does publish the same
    information under ``/sys/devices/system/cpu/cpu*/topology/``, on every
    architecture. Counts distinct (package, core) pairs for the same reason
    :func:`parse_cpu_info` does: it folds SMT siblings together and stays right
    on a multi-socket host.

    Takes the root as an argument so it can be pointed at a captured tree in
    tests. Returns ``None`` rather than guessing when sysfs is absent, which is
    every non-Linux host, or unreadable.
    """
    cores: set[tuple[str, str]] = set()
    try:
        entries = sorted(root.glob("cpu[0-9]*"))
    except OSError:
        return None
    for entry in entries:
        topology = entry / "topology"
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            # An offline CPU has no topology directory. Skip it rather than
            # abandoning the count -- the online ones still answer the question.
            continue
        # Package IDs are not dense and not zero-based: a GCP Axion reports
        # every core in package 148. Only their distinctness is meaningful.
        if package and core:
            cores.add((package, core))
    return len(cores) or None


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
        model_name = model_name or fields.get("model name") or _arm_model_name(fields)
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
