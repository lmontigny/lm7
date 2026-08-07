from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..detection import torch_device
from ..errors import CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support


def cudagraphs_requested(compile_mode: str | None, options: Mapping[str, Any]) -> bool:
    """Whether this Inductor configuration asks TorchInductor for CUDA Graphs.

    The preset decides it and the name does not say so: `reduce-overhead` and
    `max-autotune` both set `triton.cudagraphs`, while `default` and
    `max-autotune-no-cudagraphs` leave it alone. Only one of those four names
    mentions CUDA Graphs at all, which is why this is worth reporting rather than
    leaving a reader to infer it.

    An explicit `triton.cudagraphs` in `options` wins, because that is what
    torch.compile does with it.
    """
    if "triton.cudagraphs" in options:
        return bool(options["triton.cudagraphs"])
    try:
        from torch._inductor import list_mode_options

        return bool(list_mode_options(compile_mode).get("triton.cudagraphs", False))
    except Exception:  # noqa: BLE001 - a private API that moved costs the label only
        return False


def cudagraph_skips() -> int:
    """How many times Inductor has declined to capture a CUDA Graph, process-wide.

    Requesting CUDA Graphs and getting them are different things: Inductor skips
    capture for mutated inputs, dynamic shapes, CPU scalars and several other
    reasons, bumping this counter and logging why. Comparing it either side of a
    compile is what turns "asked for" into "got".
    """
    try:
        from torch._dynamo.utils import counters

        return int(counters["inductor"]["cudagraph_skips"])
    except Exception:  # noqa: BLE001
        return 0


class InductorBackend:
    name = "inductor"

    def probe(self) -> BackendInfo:
        available = callable(getattr(torch, "compile", None))
        reason = (
            "torch.compile is available."
            if available
            else "This PyTorch build has no torch.compile."
        )
        return BackendInfo(self.name, torch.__version__, available, reason)

    def supports(self, request: CompileRequest) -> Support:
        probe = self.probe()
        if not probe.available:
            return Support(False, probe.reason)
        if request.target.vendor not in {"cpu", "nvidia", "amd", "intel", "apple"}:
            return Support(False, f"Inductor does not support target {request.target} in LM7 v0.1.")
        if request.target.kind == "npu":
            # torch.compile needs a torch device to lower to, and there is no
            # NPU one. Claiming this target would silently compile for the CPU.
            return Support(
                False,
                "PyTorch has no NPU device for TorchInductor to lower to; "
                f"{request.target} is reached through backend='openvino'.",
            )
        return Support(
            True, f"torch.compile supports {request.target.kind} execution.", priority=100
        )

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        try:
            device = torch_device(request.target)
            if request.transfers == "automatic":
                request.model.to(device)
            options = dict(request.options)
            compile_mode = options.pop("compile_mode", None)
            dynamic = options.pop("dynamic", None)
            fullgraph = options.pop("fullgraph", False)
            warmup = bool(options.pop("warmup", True))
            if compile_mode is not None and options:
                names = ", ".join(sorted(options))
                raise CompilationError(
                    "Inductor compile_mode presets cannot be combined with backend "
                    f"options ({names}); choose a preset or individual options. "
                    "The dynamic and fullgraph controls may be used with either."
                )
            compiled = torch.compile(
                request.model,
                backend="inductor",
                mode=compile_mode,
                dynamic=dynamic,
                fullgraph=fullgraph,
                options=options or None,
            )
            # torch.compile is lazy: the first call is part of compilation and must
            # remain inside this error boundary so configured fallback can work.
            #
            # `warmup=False` gives that up, and exists for models where compiling by
            # executing is not free. A graph that writes into a KV cache advances it
            # once per execution, so a warmup call consumes cache slots the caller
            # never asked for -- at a long enough prompt, past the end of the buffer
            # and into a device-side assert. Such a caller compiles on its own first
            # call instead, which means a compilation failure surfaces there rather
            # than as a CompilationError here, and `fallback` cannot act on it.
            # See src/lm7/generation.py.
            requested = cudagraphs_requested(compile_mode, options)
            skipped: int | None = None
            if warmup:
                warmup_args = _map_tensors(example_args, lambda tensor: tensor.to(device))
                warmup_kwargs = _map_tensors(example_kwargs, lambda tensor: tensor.to(device))
                skips_before = cudagraph_skips()
                with torch.inference_mode():
                    compiled(*warmup_args, **warmup_kwargs)
                skipped = cudagraph_skips() - skips_before
            return Artifact(
                self.name,
                request.target,
                compiled,
                metadata={
                    "compiled": True,
                    "compile_mode": compile_mode,
                    "warmup": warmup,
                    # Three separate facts, because a preset can ask for CUDA Graphs
                    # and Inductor can still decline: `cudagraphs` is what the
                    # configuration requested, `cudagraph_skips` is how many times
                    # capture was refused during this compile, and `cudagraphs_active`
                    # is the conjunction. The last two are None without a warmup,
                    # because nothing has run yet and neither answer is known.
                    "cudagraphs": requested,
                    "cudagraph_skips": skipped,
                    "cudagraphs_active": (requested and skipped == 0) if warmup else None,
                },
            )
        except Exception as exc:
            raise CompilationError(
                f"Compilation stage failed for target {request.target} with backend inductor: {exc}. "
                "Try backend='eager' or fallback='warn'."
            ) from exc

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        assert artifact.callable is not None
        return artifact.callable


def _map_tensors(value: Any, fn: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_tensors(item, fn) for key, item in value.items()}
    return value
