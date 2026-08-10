"""``--backend trtllm``: get out of the way and let TensorRT-LLM serve.

The same shape as ``--backend vllm``, for the same reason. LM7's own server is a
single-stream reference: one static KV cache, one request at a time. TensorRT-LLM
owns attention kernels, a paged KV cache, a batch scheduler and an in-flight
batching decode loop -- and none of that can be expressed as the compiled
callable ``lm7.compile`` returns, because scheduling is a property of a *server*
holding many in-flight requests, not of a callable one caller invokes.

So nothing here proxies, wraps or re-implements an OpenAI schema. LM7 translates
its target and its flags into ``trtllm-serve``'s own argv, refuses the hardware
TensorRT-LLM has no kernels for, and hands over the process. What answers the
port afterwards is TensorRT-LLM, unmodified, and none of LM7's guarantees apply.
The honest framing is that this is a launcher, and it is documented as one in
docs/serving.md.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..detection import compute_capability, resolve_target
from ..errors import UnsupportedModelError
from ..targets import TargetSpec
from .engine import ServeConfig, resolve_model_source

# TensorRT-LLM builds kernels for Ampere and newer. Below that it does not fall
# back to something slower, it fails during engine construction, so the refusal
# belongs here where it can name the card instead of arriving as a CUDA error
# several minutes into a load.
_MINIMUM_CAPABILITY = 80

# Where this repo's own instructions put the environment (docs/tensorrt-llm.md).
# Checked because TensorRT-LLM *must* live in its own venv -- it pins a torch,
# a transformers and a tensorrt that conflict with every other environment here
# -- so the common case is TensorRT-LLM present on the machine and absent from
# the interpreter running `lm7`.
_TRTLLM_VENVS = ("~/.venv-trtllm/bin/trtllm-serve", "./.venv-trtllm/bin/trtllm-serve")

_NOT_INSTALLED = (
    "TensorRT-LLM is not installed, or trtllm-serve is not on PATH. LM7 does not depend on "
    "it: it pins torch, transformers and tensorrt to versions that conflict with LM7's other "
    "environments, so it needs a venv of its own -- see docs/tensorrt-llm.md for the install, "
    "then put that venv's bin on PATH. Or drop --backend trtllm to use LM7's own "
    "single-stream server."
)


def trtllm_executable() -> str | None:
    """The ``trtllm-serve`` command LM7 would hand over to, or None.

    Three places, in the order that matches how :func:`serve_with_trtllm`
    launches: LM7's own interpreter, then PATH, then the venv this repo's docs
    tell you to build. Importable is deliberately not the only test -- the
    handover is a subprocess, so what matters is whether a command can be run,
    and TensorRT-LLM normally lives in an environment LM7 cannot import at all.
    """
    if importlib.util.find_spec("tensorrt_llm") is not None:
        return sys.executable
    found = shutil.which("trtllm-serve")
    if found:
        return found
    for candidate in _TRTLLM_VENVS:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return None


def trtllm_available() -> bool:
    return trtllm_executable() is not None


def trtllm_supports(target: TargetSpec) -> None:
    """Refuse a target TensorRT-LLM cannot serve, naming why. Otherwise return.

    Two separate refusals because they have different fixes: another vendor's
    card is the wrong flag, and a pre-Ampere NVIDIA card is the wrong machine.
    """
    if target.vendor != "nvidia":
        raise UnsupportedModelError(
            f"TensorRT-LLM serves NVIDIA GPUs only; target {target} is {target.vendor}. "
            "Drop --backend trtllm to serve this target with LM7's own server."
        )
    capability = compute_capability(target)
    # None means the target is an unqualified `nvidia` with no architecture
    # resolved yet, which is the common case and not something to refuse on.
    if capability is not None and capability < _MINIMUM_CAPABILITY:
        raise UnsupportedModelError(
            f"TensorRT-LLM needs NVIDIA Ampere (sm80) or newer; {target.architecture} is below "
            "that. Drop --backend trtllm to serve this card with LM7's own server."
        )


def trtllm_argv(config: ServeConfig) -> list[str]:
    """The ``trtllm-serve`` command line for ``config``.

    A pure function of the config, with no ``tensorrt_llm`` import anywhere in
    it, so the translation stays unit-testable on a machine where TensorRT-LLM is
    not installed -- which is every machine in CI -- and so ``--dry-run`` prints
    something a reader can copy into a shell.

    ``trtllm-serve`` spells its flags with underscores, and LM7 does not rename
    them: the point of printing this argv is that it is the real command.
    """
    target = resolve_target(config.target)
    trtllm_supports(target)
    if config.quantize != "none":
        raise UnsupportedModelError(
            f"--quantize {config.quantize} is not passed through to TensorRT-LLM. LM7's "
            "--quantize quantizes weights in *its own* decode loop, and nothing of LM7 is in "
            "the request path here. TensorRT-LLM quantizes at engine build time from a "
            "checkpoint NVIDIA ModelOpt has already quantized -- serve one of those instead."
        )
    argv = [
        "trtllm-serve",
        # `trtllm-serve` is a command *group* on 1.2.x, not a command: the model
        # goes under `serve`, beside `disaggregated` and the MPI worker entry
        # points. Older releases took the model directly, so this word is a
        # version dependency and not decoration.
        "serve",
        # A local directory hands over as cleanly as a Hub id: it takes either in
        # this positional slot, exactly as `vllm serve` does.
        resolve_model_source(config.model),
        "--host",
        config.host,
        "--port",
        str(config.port),
        # TensorRT-LLM's name for the KV cache length. LM7's --max-model-len
        # means the same thing, so the flag passes through rather than being
        # reinterpreted.
        "--max_seq_len",
        str(config.max_model_len),
    ]
    # No --backend: that flag is *TensorRT-LLM's* runtime selector (`pytorch`,
    # `tensorrt` or `_autodeploy`), not LM7's, and the two would collide
    # confusingly in one command line. Leaving it off means TensorRT-LLM picks
    # its own default, which on 1.2.x is `pytorch` -- what an unmodified
    # `trtllm-serve` would do. See docs/serving.md.
    #
    # Last, so a passthrough wins over anything LM7 translated above, exactly as
    # --vllm-arg does for the other launcher. Without it a caller who needs one
    # flag LM7 does not model cannot use --backend trtllm at all -- and on a
    # desktop card the flag they need is --free_gpu_memory_fraction, since
    # TensorRT-LLM sizes its paged cache from free memory and takes nearly the
    # whole GPU by default.
    argv += list(config.trtllm_args)
    return argv


def trtllm_environment(config: ServeConfig) -> dict[str, str]:
    """The environment TensorRT-LLM is launched with, and what LM7 changes in it.

    Which is nothing at all today, and the function exists anyway so that
    ``--dry-run`` answers the question for both launchers the same way. Device
    selection is left alone for the reason given in :func:`serve_with_trtllm`.
    """
    return dict(os.environ)


def serve_with_trtllm(config: ServeConfig) -> int:
    """Replace this process's work with ``trtllm-serve``. Returns its exit code.

    A subprocess rather than an in-process import, and here that is load-bearing
    rather than merely tidy. TensorRT-LLM spawns MPI workers that **re-execute
    the parent's command line**; under ``python -m lm7`` those workers re-run the
    CLI, hit argparse and ``MPI_ABORT`` the job *after* the engine has finished
    building. ``trtllm-serve`` is its own entry point and its workers re-execute
    *it*, so handing over the process is what makes the MPI re-exec harmless.
    An earlier revision of this work drove the Python API in-process and had to
    be launched under ``trtllm-llmapi-launch`` to survive; see
    docs/tensorrt-llm.md.

    Device selection is left alone, for the same reason as the vLLM launcher: a
    target string cannot express a device ordinal, so there is nothing here that
    could set ``CUDA_VISIBLE_DEVICES`` more accurately than the caller already
    has, and setting it from a detected ordinal would hide every other GPU from a
    tensor-parallel run.
    """
    executable = trtllm_executable()
    if executable is None:
        raise UnsupportedModelError(_NOT_INSTALLED)
    argv = trtllm_argv(config)
    if executable == sys.executable:
        # Importable here, but the console script may not be on PATH -- normal
        # for an unactivated venv. The module is the same entry point, so it
        # keeps the `serve` subcommand that argv[1] already carries.
        argv = [sys.executable, "-m", "tensorrt_llm.commands.serve", *argv[1:]]
    else:
        argv = [executable, *argv[1:]]
    return subprocess.call(argv, env=trtllm_environment(config))


__all__ = [
    "serve_with_trtllm",
    "trtllm_argv",
    "trtllm_available",
    "trtllm_environment",
    "trtllm_executable",
    "trtllm_supports",
]
