from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..errors import CompilationError
from .base import Artifact, BackendInfo, CompileRequest, Support

# Priority sits below `eager` deliberately -- see `supports()`. Kept as a named
# constant so the reason string and the value cannot drift apart.
TVM_PRIORITY = 0

_INSTALL_HINT = 'install LM7 with ".[tvm]"'


class TVMBackend:
    """Optional Apache TVM backend, compiling through the Relax IR.

    LM7 does **not** use `torch.compile(backend="tvm")`. PyTorch ships a `tvm`
    dynamo backend, but it imports `tvm.relay`, `tvm.contrib.graph_executor`,
    `tvm.auto_scheduler`, and `tvm.meta_schedule`, all of which TVM removed in
    the Relax migration -- and its ImportError handler then reports "Please
    install apache-tvm" even when TVM is installed. It cannot work with a
    current TVM.

    TVM's own `relax_dynamo()` backend is the supported replacement, but it
    converts through `from_fx`, whose operator table rejects `embedding` -- so
    no causal LM compiles. This backend therefore captures with `torch.export`
    and converts with `from_exported_program`, which has the wider table, then
    builds and runs on the Relax VM. That is what makes real models work.

    Compilation is in-process and per input signature, so this is a JIT backend
    in LM7's sense: nothing it produces outlives the process.

    See ``docs/tvm.md`` for the measured performance, which is the reason this
    backend is never selected automatically.
    """

    name = "tvm"

    def probe(self) -> BackendInfo:
        try:
            installed = importlib.util.find_spec("tvm") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            return BackendInfo(
                self.name, None, False, f"Apache TVM is not installed; {_INSTALL_HINT}."
            )
        version = _tvm_version()
        try:
            importlib.import_module("tvm.relax")
            importlib.import_module("tvm.relax.frontend.torch")
        except ImportError as exc:
            return BackendInfo(
                self.name,
                version,
                False,
                f"TVM is installed but its Relax PyTorch frontend is unavailable: {exc}. "
                "LM7 needs a Relax-era TVM (0.20 or newer).",
            )
        return BackendInfo(
            self.name, version, True, "Apache TVM can compile through Relax for LLVM CPU targets."
        )

    def supports(self, request: CompileRequest) -> Support:
        probe = self.probe()
        if not probe.available:
            return Support(False, probe.reason)
        if request.target.vendor != "cpu":
            return Support(
                False,
                "LM7 wires up TVM for CPU (LLVM) targets only; its CUDA codegen needs a "
                "discoverable CUDA toolkit that LM7 does not configure.",
            )
        # Reported as supported so `backend="tvm"` works, but at priority 0 --
        # tied with `eager` and below every real CPU backend -- because TVM's
        # untuned Relax codegen measured ~170x slower than eager on this
        # project's own MLP benchmark. `backend="auto"` resolves ties by name,
        # so `eager` wins and TVM is never selected implicitly.
        return Support(
            True,
            "TVM compiles through Relax; select it explicitly, as its untuned codegen is "
            "far slower than Inductor. See docs/tvm.md.",
            priority=TVM_PRIORITY,
        )

    def compile(
        self,
        request: CompileRequest,
        example_args: tuple[Any, ...],
        example_kwargs: Mapping[str, Any],
    ) -> Artifact:
        if example_kwargs:
            raise CompilationError(
                "The tvm backend captures positional inputs only; got keyword inputs "
                f"{', '.join(sorted(example_kwargs))}. Call the model positionally, for "
                "example model(input_ids, attention_mask)."
            )
        try:
            tvm = importlib.import_module("tvm")
            relax = importlib.import_module("tvm.relax")
            frontend = importlib.import_module("tvm.relax.frontend.torch")
        except ImportError as exc:  # pragma: no cover - probe already guards this
            raise CompilationError(f"Apache TVM could not be imported: {exc}.") from exc

        options = dict(request.options)
        target_string = options.pop("target", "llvm")
        try:
            with torch.no_grad():
                exported = torch.export.export(request.model, tuple(example_args))
            # keep_params_as_input=False bakes the weights into the module, so
            # the built VM takes only the call's real arguments.
            module = frontend.from_exported_program(exported, keep_params_as_input=False)
            target = tvm.target.Target(target_string)
            with target:
                executable = relax.build(module, target=target)
            device = tvm.cpu(0)
            machine = relax.VirtualMachine(executable, device)
            runner = _TVMRunner(machine, device, tvm)
            # Run once inside the compile boundary so a codegen failure surfaces
            # here, where LM7's fallback can still catch it, rather than on the
            # caller's first real call.
            runner(*example_args)
        except Exception as exc:
            raise CompilationError(
                f"Compilation stage failed for target {request.target} with backend tvm: "
                f"{exc}. Verify the TVM installation or use fallback='warn'."
            ) from exc
        return Artifact(
            self.name,
            request.target,
            runner,
            metadata={
                "compiled": True,
                "tvm_version": _tvm_version(),
                "tvm_target": target_string,
                "frontend": "relax.from_exported_program",
            },
        )

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        assert artifact.callable is not None
        return artifact.callable


class _TVMRunner:
    """Adapts the Relax VM to a torch-tensor-in, torch-tensor-out callable."""

    def __init__(self, machine: Any, device: Any, tvm: Any) -> None:
        self._machine = machine
        self._device = device
        self._tvm = tvm

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise CompilationError(
                "A TVM-compiled callable takes positional tensors only; got "
                f"{', '.join(sorted(kwargs))}."
            )
        inputs = [
            self._tvm.runtime.tensor(
                tensor.detach().cpu().contiguous().numpy(), device=self._device
            )
            for tensor in args
        ]
        outputs = self._machine["main"](*inputs)
        return _to_torch(outputs)


def _to_torch(value: Any) -> Any:
    """Convert a Relax VM return value back into torch tensors.

    The VM returns its own array type, or a nested container of them for a
    multi-output graph.
    """
    if hasattr(value, "numpy"):
        return torch.from_numpy(value.numpy())
    if isinstance(value, (list, tuple)) or type(value).__name__ == "Array":
        converted = tuple(_to_torch(item) for item in value)
        return converted[0] if len(converted) == 1 else converted
    return value


def _tvm_version() -> str | None:
    try:
        return importlib.metadata.version("apache-tvm")
    except importlib.metadata.PackageNotFoundError:
        try:
            return getattr(importlib.import_module("tvm"), "__version__", None)
        except ImportError:
            return None
