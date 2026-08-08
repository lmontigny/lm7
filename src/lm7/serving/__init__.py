from __future__ import annotations

from .base import (
    Capabilities,
    RuntimeInfo,
    ServeRequest,
    ServerHandle,
    ServingRuntime,
    unmet_capabilities,
)
from .budget import MemoryBudget, ModelShape, kv_bytes_per_token, plan_memory
from .planner import RuntimeCandidate, ServePlan, plan_serving
from .registry import RuntimeRegistry
from .runtimes.eager import EagerServingRuntime
from .runtimes.vllm import VLLMServingRuntime

registry = RuntimeRegistry()
registry.register(EagerServingRuntime())
registry.register(VLLMServingRuntime())

__all__ = [
    "Capabilities",
    "EagerServingRuntime",
    "MemoryBudget",
    "ModelShape",
    "RuntimeCandidate",
    "RuntimeInfo",
    "RuntimeRegistry",
    "ServePlan",
    "ServeRequest",
    "ServerHandle",
    "ServingRuntime",
    "VLLMServingRuntime",
    "kv_bytes_per_token",
    "plan_memory",
    "plan_serving",
    "registry",
    "unmet_capabilities",
]
