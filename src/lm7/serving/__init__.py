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

# Named `runtime_registry` rather than `registry` so it does not shadow the
# `lm7.serving.registry` submodule. `lm7.backends` gets away with that shadowing
# because the assignment follows the import, but it leaves the name resolving to
# the module for any reader -- and for mypy.
runtime_registry = RuntimeRegistry()
runtime_registry.register(EagerServingRuntime())
runtime_registry.register(VLLMServingRuntime())

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
    "runtime_registry",
    "unmet_capabilities",
]
