from __future__ import annotations

import pytest
import torch

from lm7.serving.budget import ModelShape, kv_bytes_per_token, plan_memory


class _Config:
    def __init__(self, **fields: object) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


def test_shape_reads_grouped_query_attention() -> None:
    shape = ModelShape.from_config(
        _Config(
            num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8, hidden_size=4096
        )
    )
    assert shape == ModelShape(layers=32, kv_heads=8, head_dim=128)


def test_shape_falls_back_to_attention_heads_without_gqa() -> None:
    """A multi-head model does not set num_key_value_heads at all.

    Defaulting it to anything but num_attention_heads understates the cache by
    the grouping factor, which is the difference between fitting and not.
    """
    shape = ModelShape.from_config(
        _Config(num_hidden_layers=12, num_attention_heads=12, hidden_size=768)
    )
    assert shape.kv_heads == 12
    assert shape.head_dim == 64


def test_shape_prefers_explicit_head_dim() -> None:
    shape = ModelShape.from_config(
        _Config(num_hidden_layers=4, num_attention_heads=8, hidden_size=512, head_dim=96)
    )
    assert shape.head_dim == 96


def test_shape_rejects_a_config_it_cannot_size() -> None:
    with pytest.raises(ValueError, match="num_hidden_layers"):
        ModelShape.from_config(_Config(hidden_size=768))


def test_kv_bytes_per_token_counts_key_and_value() -> None:
    shape = ModelShape(layers=2, kv_heads=4, head_dim=64)
    # 2 (K and V) * 2 layers * 4 heads * 64 dim * 2 bytes
    assert kv_bytes_per_token(shape, torch.float16) == 2048
    assert kv_bytes_per_token(shape, torch.float32) == 4096


def test_plan_memory_scales_with_sequences_and_length() -> None:
    shape = ModelShape(layers=2, kv_heads=4, head_dim=64)
    single = plan_memory(
        shape, dtype=torch.float16, max_model_len=1024, max_num_seqs=1, device_bytes=None
    )
    batched = plan_memory(
        shape, dtype=torch.float16, max_model_len=1024, max_num_seqs=8, device_bytes=None
    )
    assert batched.kv_bytes == single.kv_bytes * 8


def test_plan_memory_reports_no_verdict_without_weights() -> None:
    """Refusing to answer is the point: a verdict without the weights is a guess."""
    budget = plan_memory(
        ModelShape(layers=2, kv_heads=4, head_dim=64),
        dtype=torch.float16,
        max_model_len=128,
        max_num_seqs=1,
        device_bytes=1024**3,
    )
    assert budget.fits is None
    assert budget.total_bytes is None


def test_plan_memory_detects_a_configuration_that_cannot_fit() -> None:
    shape = ModelShape(layers=32, kv_heads=8, head_dim=128)
    budget = plan_memory(
        shape,
        dtype=torch.bfloat16,
        max_model_len=131072,
        max_num_seqs=64,
        device_bytes=80 * 1024**3,
        weight_bytes=16 * 1024**3,
    )
    assert budget.fits is False
    assert "KV cache" in budget.describe()


def test_kv_cache_fraction_shrinks_the_budget() -> None:
    shape = ModelShape(layers=8, kv_heads=8, head_dim=64)
    kwargs = {
        "dtype": torch.float16,
        "max_model_len": 4096,
        "max_num_seqs": 4,
        "device_bytes": 8 * 1024**3,
        "weight_bytes": 6 * 1024**3,
    }
    assert plan_memory(shape, **kwargs).fits is True  # type: ignore[arg-type]
    assert plan_memory(shape, kv_cache_fraction=0.7, **kwargs).fits is False  # type: ignore[arg-type]
