from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def cache_dir() -> Path:
    configured = os.environ.get("LM7_CACHE_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "lm7"


def _value_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            tuple(value.shape),
            str(value.dtype),
            tuple(value.stride()),
            str(value.device),
        )
    if isinstance(value, dict):
        return ("dict", tuple(sorted((str(k), _value_signature(v)) for k, v in value.items())))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_value_signature(v) for v in value))
    return (type(value).__module__, type(value).__qualname__)


def input_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    return (_value_signature(args), _value_signature(kwargs))


@dataclass
class MemoryCache:
    _values: dict[tuple[Any, ...], Any]

    def __init__(self) -> None:
        self._values = {}

    def get(self, key: tuple[Any, ...]) -> Any | None:
        return self._values.get(key)

    def put(self, key: tuple[Any, ...], value: Any) -> None:
        self._values[key] = value

    def clear(self) -> None:
        self._values.clear()


memory_cache = MemoryCache()


def clear_cache() -> None:
    memory_cache.clear()
    path = cache_dir()
    if path.exists():
        shutil.rmtree(path)
