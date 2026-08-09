"""``--backend vllm``: get out of the way and let vLLM serve.

LM7's own server is a single-stream reference. It has no continuous batching, no
paged KV cache and no prefix caching, and adding them would mean writing a
serving engine -- which is the same mistake as writing a compiler. So when a
caller asks for vLLM, nothing here proxies, wraps, or re-implements an OpenAI
schema: LM7 translates its target and its flags into vLLM's own argv and hands
over the process. What answers the port afterwards is vLLM, unmodified.

That means every vLLM feature works, and none of LM7's guarantees apply --
``lm7`` is not in the request path at all. The honest framing is that this is a
launcher, and it is documented as one in docs/serving.md.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..detection import resolve_target
from ..errors import UnsupportedModelError
from ..targets import TargetSpec
from .engine import ServeConfig, resolve_model_source

# Where vLLM has a device backend and LM7 has a target, and they mean the same
# hardware. Everything outside this map is refused rather than launched: vLLM
# silently falls back to whatever platform plugin it can load, so a
# `--target apple` that started a CPU server would answer requests while
# reporting nothing about having ignored the target.
_VLLM_PLATFORMS = {
    "nvidia": "cuda",
    "amd": "rocm",
    "cpu": "cpu",
    "tpu": "tpu",
    # Apple Silicon through the vllm-metal platform plugin, which is a plugin
    # rather than a fork -- `vllm serve` is the same command. Validated on an
    # M-series Mac; see docs/serving.md.
    "apple": "metal",
}

# Where vllm-metal's installer puts its environment. Checked because that
# installer deliberately builds an isolated venv -- vLLM pins a specific
# PyTorch, which is the same reason LM7 does not depend on vLLM -- so the
# common case is vLLM present on the machine and absent from LM7's interpreter.
_VLLM_METAL_VENV = "~/.venv-vllm-metal/bin/vllm"

_NOT_INSTALLED = (
    "vLLM is not installed, or is not on PATH. LM7 does not depend on it: vLLM pins a "
    "specific PyTorch, and pinning one here would decide the torch version for everyone "
    "who installs LM7. Install it yourself -- on Apple Silicon that is the vllm-metal "
    "plugin (see docs/serving.md), elsewhere 'uv pip install vllm' -- or drop "
    "--backend vllm to use LM7's own single-stream server."
)


def vllm_executable() -> str | None:
    """The ``vllm`` command LM7 would hand over to, or None.

    Three places, in the order that matches how ``serve_with_vllm`` launches:
    LM7's own interpreter, then PATH, then vllm-metal's default venv. Importable
    is deliberately *not* the only test -- the handover is a subprocess, so what
    matters is whether a command can be run, and on Apple Silicon vLLM normally
    lives in its own environment where LM7 cannot import it at all.
    """
    if importlib.util.find_spec("vllm") is not None:
        return sys.executable
    found = shutil.which("vllm")
    if found:
        return found
    candidate = Path(_VLLM_METAL_VENV).expanduser()
    return str(candidate) if candidate.exists() else None


def vllm_available() -> bool:
    return vllm_executable() is not None


def vllm_platform(target: TargetSpec) -> str:
    """vLLM's name for ``target``'s hardware, or a refusal explaining why not."""
    platform = _VLLM_PLATFORMS.get(target.vendor)
    if platform is None:
        supported = ", ".join(sorted(_VLLM_PLATFORMS))
        raise UnsupportedModelError(
            f"vLLM has no backend for target {target}; it serves {supported}. "
            "Drop --backend vllm to serve this target with LM7's own server."
        )
    return platform


def vllm_argv(config: ServeConfig) -> list[str]:
    """The ``vllm serve`` command line for ``config``.

    A pure function of the config, with no vLLM import anywhere in it, so the
    translation stays unit-testable on a machine where vLLM is not installed, and
    so the ``--dry-run`` output is something a user can copy into a shell.
    """
    target = resolve_target(config.target)
    vllm_platform(target)
    argv = [
        "vllm",
        "serve",
        # vLLM takes a local directory in the same positional slot as a Hub id,
        # so a local model hands over as cleanly as a Hub one.
        resolve_model_source(config.model),
        "--host",
        config.host,
        "--port",
        str(config.port),
        # vLLM's own name for the KV cache length. LM7's --max-model-len means the
        # same thing, so the flag passes through rather than being reinterpreted.
        "--max-model-len",
        str(config.max_model_len),
    ]
    if config.dtype != "auto":
        argv += ["--dtype", config.dtype]
    # No `--device`: vLLM's platform comes from which build is installed, and a
    # specific card comes from the environment -- see `serve_with_vllm`.
    return argv


def serve_with_vllm(config: ServeConfig) -> int:
    """Replace this process's work with ``vllm serve``. Returns vLLM's exit code.

    A subprocess rather than an in-process import, because vLLM's engine wants to
    own signal handling, worker processes and the CUDA context -- and because a
    launcher that has already handed over should not be holding a loaded model in
    memory behind it.

    The device selection is left alone. LM7 targets carry a device *ordinal* only
    when detection supplied one, and a target string cannot express one at all,
    so there is nothing here that could set ``CUDA_VISIBLE_DEVICES`` more
    accurately than the caller already has -- and setting it from a detected
    ordinal would silently hide every other GPU from a tensor-parallel run.
    ``VLLM_HOST_IP`` is the one exception; see :func:`vllm_environment`.
    """
    executable = vllm_executable()
    if executable is None:
        raise UnsupportedModelError(_NOT_INSTALLED)
    argv = vllm_argv(config)
    if executable == sys.executable:
        # Importable here, but the console script may not be on PATH -- normal
        # for a `pip install --target` layout or an unactivated venv.
        argv = [sys.executable, "-m", "vllm.entrypoints.cli.main", *argv[1:]]
    else:
        argv = [executable, *argv[1:]]
    return subprocess.call(argv, env=vllm_environment(config))


def vllm_environment(config: ServeConfig) -> dict[str, str]:
    """The environment vLLM is launched with, and the single thing LM7 changes.

    vLLM initializes a ``gloo`` process group even for a single-worker server,
    and picks the host's LAN address for it. On a Mac that **hangs**: startup
    stops after "PyTorch device set to: mps" with no error, no timeout and no
    further output, and the only clue is
    ``distributed_init_method=tcp://192.168.x.x:...`` in the log. Setting
    ``VLLM_HOST_IP`` to loopback fixes it -- measured here as a hang of over ten
    minutes becoming a 130-second startup.

    So LM7 sets it, but only when the server is being bound to loopback anyway
    (where a LAN address is certainly wrong) and only when the caller has not set
    it themselves. A multi-node vLLM deployment needs the real address, and that
    caller has already said so.
    """
    environment = dict(os.environ)
    if "VLLM_HOST_IP" in environment:
        return environment
    if config.host in ("127.0.0.1", "localhost", "::1"):
        environment["VLLM_HOST_IP"] = "127.0.0.1"
    return environment


__all__ = [
    "serve_with_vllm",
    "vllm_argv",
    "vllm_available",
    "vllm_environment",
    "vllm_executable",
    "vllm_platform",
]
