from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

_BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class ModelShape:
    """The handful of config fields that decide how big a KV cache is."""

    layers: int
    kv_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config: Any) -> ModelShape:
        """Read the shape out of a Hugging Face config.

        ``num_key_value_heads`` is the one that matters and the one that is
        often absent: a model without grouped-query attention simply does not
        set it, and falling back to ``num_attention_heads`` is what makes the
        estimate right for MHA instead of silently dividing by one.
        """
        layers = getattr(config, "num_hidden_layers", None)
        attention_heads = getattr(config, "num_attention_heads", None)
        hidden_size = getattr(config, "hidden_size", None)
        if layers is None or attention_heads is None:
            raise ValueError(
                "The model config does not declare num_hidden_layers and "
                "num_attention_heads, so its KV cache cannot be sized."
            )
        kv_heads = getattr(config, "num_key_value_heads", None) or attention_heads
        head_dim = getattr(config, "head_dim", None)
        if not head_dim:
            if not hidden_size:
                raise ValueError(
                    "The model config declares neither head_dim nor hidden_size, "
                    "so its KV cache cannot be sized."
                )
            head_dim = hidden_size // attention_heads
        return cls(int(layers), int(kv_heads), int(head_dim))


@dataclass(frozen=True)
class MemoryBudget:
    """What a serving configuration will cost, against what the device has.

    ``fits`` is only as good as ``weight_bytes``: when the weights are not known
    ahead of the load it is ``None``, and this reports the KV cost alone rather
    than pretending to a verdict it cannot reach.
    """

    kv_bytes_per_token: int
    kv_bytes: int
    weight_bytes: int | None
    total_bytes: int | None
    device_bytes: int | None
    fits: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        parts = [f"KV cache {_gib(self.kv_bytes)}"]
        if self.weight_bytes is not None:
            parts.append(f"weights {_gib(self.weight_bytes)}")
        if self.device_bytes is not None:
            parts.append(f"device {_gib(self.device_bytes)}")
        return ", ".join(parts)


def _gib(value: int) -> str:
    return f"{value / _BYTES_PER_GIB:.2f} GiB"


def kv_bytes_per_token(shape: ModelShape, dtype: torch.dtype) -> int:
    """Key and value, per layer, per token.

    The leading 2 is the key and the value, not a fudge factor.
    """
    return 2 * shape.layers * shape.kv_heads * shape.head_dim * dtype.itemsize


def plan_memory(
    shape: ModelShape,
    *,
    dtype: torch.dtype,
    max_model_len: int,
    max_num_seqs: int,
    device_bytes: int | None,
    weight_bytes: int | None = None,
    kv_cache_fraction: float | None = None,
) -> MemoryBudget:
    """Cost a serving configuration before anything is allocated.

    This is the cheapest thing in the serving layer and the one that saves the
    most time: a max_model_len that cannot fit fails forty seconds into a load
    otherwise, with a message about the allocator rather than about the request.
    """
    per_token = kv_bytes_per_token(shape, dtype)
    kv_bytes = per_token * max_model_len * max_num_seqs
    total = None if weight_bytes is None else weight_bytes + kv_bytes
    available = device_bytes
    if available is not None and kv_cache_fraction is not None:
        available = int(available * kv_cache_fraction)
    fits = None if (total is None or available is None) else total <= available
    return MemoryBudget(
        kv_bytes_per_token=per_token,
        kv_bytes=kv_bytes,
        weight_bytes=weight_bytes,
        total_bytes=total,
        device_bytes=available,
        fits=fits,
    )
