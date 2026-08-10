from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import TargetNotFoundError


@dataclass(frozen=True)
class TargetSpec:
    vendor: str
    kind: str
    architecture: str | None = None
    model: str | None = None
    ordinal: int | None = None
    remote: bool = False

    def __str__(self) -> str:
        if self.vendor == "cpu":
            return "cpu" + (f":{self.architecture}" if self.architecture else "")
        if self.vendor == "qualcomm" and self.model:
            return f"qualcomm:{self.model}"
        if self.kind == "npu":
            # The kind has to survive the round trip: `intel` on its own already
            # means the Intel GPU, so an NPU spec cannot print as its vendor.
            return f"{self.vendor}:npu"
        qualifier = self.model or self.architecture
        return f"{self.vendor}:{qualifier}" if qualifier else self.vendor


@dataclass(frozen=True)
class DeviceInfo:
    target: TargetSpec
    name: str
    total_memory_bytes: int | None = None
    capabilities: Mapping[str, Any] = field(default_factory=dict)


_VENDORS = {
    "cpu",
    "nvidia",
    "amd",
    "intel",
    "apple",
    "tpu",
    "tenstorrent",
    "aws",
    "qualcomm",
    "arm",
}

# Tenstorrent qualifiers split into an architecture generation and a board
# model, so `tenstorrent:blackhole` pins the silicon and `tenstorrent:p150`
# pins the card.
_TENSTORRENT_ARCHITECTURES = frozenset({"wormhole", "blackhole"})

# Arm GPU qualifiers split the same way: a shader-core generation, or a product
# name. Arm ships exactly two product prefixes, so an unrecognized qualifier is
# a typo rather than a part LM7 has not heard of yet.
_ARM_ARCHITECTURES = ("valhall", "bifrost", "midgard")
_ARM_PRODUCT_PREFIXES = ("mali-", "immortalis-")


def parse_target(value: str | TargetSpec) -> TargetSpec:
    if isinstance(value, TargetSpec):
        return value
    if not isinstance(value, str) or not value.strip():
        raise TargetNotFoundError(f"Invalid target {value!r}; expected a non-empty string.")
    value = value.lower().strip()
    if value == "auto":
        return TargetSpec("auto", "auto")
    parts = value.split(":")
    if len(parts) > 2 or parts[0] not in _VENDORS or any(not p for p in parts):
        raise TargetNotFoundError(f"Invalid target {value!r}.")
    vendor = parts[0]
    qualifier = parts[1] if len(parts) == 2 else None
    if vendor == "cpu":
        return TargetSpec("cpu", "cpu", architecture=qualifier)
    if vendor == "intel" and qualifier == "gpu":
        return TargetSpec("intel", "gpu")
    if vendor == "intel" and qualifier == "npu":
        # Core Ultra "AI Boost". Its kind is neither "gpu" nor "accelerator":
        # PyTorch has no NPU device, so the silicon is reached only through the
        # OpenVINO NPU plugin, and backends gate on that kind to say so.
        return TargetSpec("intel", "npu")
    if vendor == "apple" and qualifier == "metal":
        return TargetSpec("apple", "gpu", architecture="metal")
    if vendor == "aws" and qualifier == "trainium":
        return TargetSpec("aws", "accelerator", model="trainium", remote=True)
    if vendor == "qualcomm" and qualifier == "sm8750":
        return TargetSpec(
            "qualcomm",
            "npu",
            architecture="v79",
            model="sm8750",
            remote=True,
        )
    if vendor == "arm":
        # An Arm GPU is never a torch device -- Mali is reached through Vulkan
        # and SPIR-V -- so this always describes deployment hardware the
        # compiler host does not own, exactly like qualcomm:sm8750.
        if qualifier is None:
            return TargetSpec("arm", "gpu", remote=True)
        if qualifier.startswith(_ARM_ARCHITECTURES):
            return TargetSpec("arm", "gpu", architecture=qualifier, remote=True)
        if qualifier.startswith(_ARM_PRODUCT_PREFIXES):
            return TargetSpec("arm", "gpu", model=qualifier, remote=True)
        raise TargetNotFoundError(
            f"Unsupported Arm GPU target {value!r}; expected a shader-core generation "
            f"({', '.join(_ARM_ARCHITECTURES)}) or a product name such as 'arm:mali-g715'."
        )
    if vendor == "tpu":
        return TargetSpec("tpu", "accelerator", model=qualifier)
    if vendor == "tenstorrent":
        if qualifier in _TENSTORRENT_ARCHITECTURES:
            return TargetSpec("tenstorrent", "accelerator", architecture=qualifier)
        return TargetSpec("tenstorrent", "accelerator", model=qualifier)
    if vendor in {"nvidia", "amd"}:
        architecture_prefixes = ("sm", "gfx")
        if qualifier and qualifier.startswith(architecture_prefixes):
            return TargetSpec(vendor, "gpu", architecture=qualifier)
        return TargetSpec(vendor, "gpu", model=qualifier)
    if vendor == "qualcomm":
        raise TargetNotFoundError(
            f"Unsupported Qualcomm target {value!r}; expected 'qualcomm:sm8750'."
        )
    if qualifier is not None:
        raise TargetNotFoundError(f"Unsupported target qualifier in {value!r}.")
    return TargetSpec(vendor, "gpu")
