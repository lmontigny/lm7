from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from ..errors import ArtifactLoadError, CompilationError
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
            tvm, relax, frontend = _import_tvm()
        except ImportError as exc:  # pragma: no cover - probe already guards this
            raise CompilationError(f"Apache TVM could not be imported: {exc}.") from exc

        options = dict(request.options)
        try:
            with torch.no_grad():
                exported = torch.export.export(request.model, tuple(example_args))
            executable, target_string = _lower_and_build(exported, options, tvm, relax, frontend)
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

    def compile_exported(
        self,
        exported_program: torch.export.ExportedProgram,
        library_path: Path,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Path:
        """Build a standalone Relax VM library and save it to ``library_path``.

        Unlike ``compile()``, the saved library reloads through
        ``load_library()`` with only the TVM *runtime* -- ``tvm.runtime`` plus
        ``relax.VirtualMachine`` -- so the process loading it later never needs
        ``torch.export`` or the Relax PyTorch frontend that built it. Weights
        are baked in the same way as the JIT path
        (``keep_params_as_input=False``).

        The library embeds the exporting host's target triple (arm64 vs
        x86-64, and any ``mcpu`` given in ``options``), so it only reloads on
        a matching architecture -- see docs/tvm.md.
        """
        probe = self.probe()
        if not probe.available:
            raise CompilationError(probe.reason)
        args, kwargs = getattr(exported_program, "example_inputs", None) or ((), {})
        if kwargs:
            raise CompilationError(
                "The tvm backend captures positional inputs only; got keyword inputs "
                f"{', '.join(sorted(kwargs))}. Export the model with positional args, for "
                "example args=(input_ids, attention_mask)."
            )
        try:
            tvm, relax, frontend = _import_tvm()
        except ImportError as exc:  # pragma: no cover - probe already guards this
            raise CompilationError(f"Apache TVM could not be imported: {exc}.") from exc

        resolved_options = dict(options or {})
        try:
            executable, _ = _lower_and_build(
                exported_program, resolved_options, tvm, relax, frontend
            )
            library_path.parent.mkdir(parents=True, exist_ok=True)
            executable.export_library(str(library_path))
            if args:
                # Round-trips through the saved library rather than the
                # in-memory `executable`, so a save/reload mismatch -- not just
                # a codegen failure -- surfaces here.
                _TVMRunner(
                    relax.VirtualMachine(tvm.runtime.load_module(str(library_path)), tvm.cpu(0)),
                    tvm.cpu(0),
                    tvm,
                )(*args)
        except Exception as exc:
            library_path.unlink(missing_ok=True)
            raise CompilationError(
                f"TVM AOT export failed for {library_path}: {exc}. Verify the TVM "
                "installation or use fallback='warn'."
            ) from exc
        return library_path

    def load_library(self, library_path: Path) -> Callable[..., Any]:
        """Reload a library written by ``compile_exported()``.

        Uses only the TVM runtime -- no ``torch.export`` or Relax PyTorch
        frontend import required, unlike ``load()`` for the JIT path.
        """
        probe = self.probe()
        if not probe.available:
            raise ArtifactLoadError(probe.reason)
        try:
            tvm = importlib.import_module("tvm")
            relax = importlib.import_module("tvm.relax")
        except ImportError as exc:  # pragma: no cover - probe already guards this
            raise ArtifactLoadError(f"Apache TVM could not be imported: {exc}.") from exc
        try:
            loaded = tvm.runtime.load_module(str(library_path))
            device = tvm.cpu(0)
            machine = relax.VirtualMachine(loaded, device)
        except Exception as exc:
            raise ArtifactLoadError(
                f"Artifact load stage failed for {library_path}: {exc}. TVM libraries embed "
                "the exporting host's CPU architecture and do not reload on a different one."
            ) from exc
        return _TVMRunner(machine, device, tvm)

    def load(self, artifact: Artifact) -> Callable[..., Any]:
        assert artifact.callable is not None
        return artifact.callable


def _import_tvm() -> tuple[Any, Any, Any]:
    tvm = importlib.import_module("tvm")
    relax = importlib.import_module("tvm.relax")
    frontend = importlib.import_module("tvm.relax.frontend.torch")
    return tvm, relax, frontend


def _lower_and_build(
    exported_program: torch.export.ExportedProgram,
    options: Mapping[str, Any],
    tvm: Any,
    relax: Any,
    frontend: Any,
) -> tuple[Any, Any]:
    """Lower an ExportedProgram to Relax and build it for ``options["target"]``.

    Shared by the JIT and AOT export paths so they cannot drift on how the
    target option is read or how weights are baked in. Returns the built
    executable and the target value actually used (for artifact metadata).
    """
    resolved_options = dict(options)
    target_string = resolved_options.pop("target", "llvm")
    # keep_params_as_input=False bakes the weights into the module, so the
    # built VM/library takes only the call's real arguments.
    module = frontend.from_exported_program(exported_program, keep_params_as_input=False)
    target = tvm.target.Target(target_string)
    with target:
        executable = relax.build(module, target=target)
    return executable, target_string


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
        # DLPack rather than NumPy: it is zero-copy, and it keeps this adapter
        # working on a base LM7 install, which does not depend on NumPy.
        inputs = [self._tvm.runtime.from_dlpack(tensor.detach().contiguous()) for tensor in args]
        outputs = self._machine["main"](*inputs)
        return _to_torch(outputs)


def _to_torch(value: Any) -> Any:
    """Convert a Relax VM return value back into torch tensors.

    The VM returns its own array type, or a nested container of them for a
    multi-output graph. DLPack keeps the conversion zero-copy and NumPy-free.
    """
    if hasattr(value, "__dlpack__"):
        return torch.from_dlpack(value)
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
