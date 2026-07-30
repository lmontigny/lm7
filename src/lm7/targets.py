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


_VENDORS = {"cpu", "nvidia", "amd", "intel", "apple", "tpu", "tenstorrent", "aws"}

# Tenstorrent qualifiers split into an architecture generation and a board
# model, so `tenstorrent:blackhole` pins the silicon and `tenstorrent:p150`
# pins the card.
_TENSTORRENT_ARCHITECTURES = frozenset({"wormhole", "blackhole"})


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
    if qualifier is not None:
        raise TargetNotFoundError(f"Unsupported target qualifier in {value!r}.")
    return TargetSpec(vendor, "gpu")
