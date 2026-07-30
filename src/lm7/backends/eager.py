from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..detection import torch_device
from .base import Artifact, BackendInfo, CompileRequest, Support


class EagerBackend:
    name = "eager"

    def probe(self) -> BackendInfo:
        return BackendInfo(
            self.name, torch.__version__, True, "PyTorch eager execution is available."
        )

    def supports(self, request: CompileRequest) -> Support:
        if request.target.kind == "npu":
            # An NPU is not a PyTorch device, so "eager on intel:npu" would be
            # eager on the CPU under another name. LM7 still falls back here
            # when an NPU compile fails, but that path warns; planning does not.
            return Support(
                False,
                f"PyTorch has no NPU device; {request.target} is reached through "
                "backend='openvino'.",
            )
        return Support(True, "Eager supports every detected local PyTorch device.", priority=0)

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        if request.transfers == "automatic":
            request.model.to(torch_device(request.target))
        return Artifact(self.name, request.target, request.model, metadata={"compiled": False})

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        assert artifact.callable is not None
        return artifact.callable
