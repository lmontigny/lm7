"""What ``lm7 model serve`` does once argparse has finished.

Kept out of ``lm7/cli.py`` so that building the parser -- which happens for
``lm7 doctor`` and every other subcommand -- never imports FastAPI, Uvicorn or
Transformers. The parser lives with its siblings; only the handler is here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from ..detection import resolve_target
from ..errors import UnsupportedModelError
from .engine import ServeConfig, resolve_model_source

# The backends that are a *handover* rather than a server. LM7 translates the
# config into someone else's argv, hands over the process, and is not in the
# request path afterwards; what differs between them is only which module does
# the translating. Keeping them in one tuple is what makes `--dry-run`, the
# chat page and the refusals below identical for every launcher instead of
# per-backend special cases -- see docs/serving.md.
LAUNCHER_BACKENDS = ("vllm", "trtllm")

# argv, environment, executable: the three functions a launcher module provides.
Launcher = tuple[
    Callable[[ServeConfig], list[str]],
    Callable[[ServeConfig], dict[str, str]],
    Callable[[], str | None],
]


def _launcher(backend: str) -> Launcher:
    """The three functions that define a handover backend.

    Imported on demand rather than at module scope so that a ``--dry-run`` for
    one launcher never imports the other's module, and so that neither is
    imported at all for LM7's own server.
    """
    if backend == "vllm":
        from .vllm import vllm_argv, vllm_environment, vllm_executable

        return vllm_argv, vllm_environment, vllm_executable
    if backend == "trtllm":
        from .trtllm import trtllm_argv, trtllm_environment, trtllm_executable

        return trtllm_argv, trtllm_environment, trtllm_executable
    raise UnsupportedModelError(
        f"{backend!r} is not a launcher backend; expected one of {', '.join(LAUNCHER_BACKENDS)}."
    )


def serve_plan(config: ServeConfig) -> dict[str, Any]:
    """What this invocation would do, without loading a model or binding a port.

    Backs ``--dry-run``, which exists because loading a model is the expensive
    part of finding out that a target was misspelled. For a launcher backend it
    also answers the two questions that decide whether the handover will work at
    all: which executable LM7 found, and what it will change in the environment.
    """
    target = resolve_target(config.target)
    plan: dict[str, Any] = {
        "model": resolve_model_source(config.model),
        "target": str(target),
        "backend": config.backend,
        "max_model_len": config.max_model_len,
        "host": config.host,
        "port": config.port,
    }
    if config.backend in LAUNCHER_BACKENDS:
        argv_for, environment_for, executable_for = _launcher(config.backend)

        executable = executable_for()
        plan["runtime"] = config.backend
        plan["runtime_installed"] = executable is not None
        # Named because "not installed" is usually "installed in a different
        # environment" -- vllm-metal and TensorRT-LLM both need their own venv.
        plan["runtime_executable"] = executable
        plan["argv"] = argv_for(config)
        plan["ui_port"] = config.ui_port
        overrides = {
            name: value
            for name, value in environment_for(config).items()
            if os.environ.get(name) != value
        }
        plan["environment"] = overrides
    else:
        plan["runtime"] = "lm7"
        plan["dtype"] = config.dtype
        plan["compile_mode"] = config.compile_mode
        # Reported because it is the setting that decides whether a second,
        # unseen prompt length costs milliseconds or a fresh Inductor compile,
        # and --dry-run is where that is cheap to notice.
        plan["compile_prefill"] = config.compile_prefill
        plan["quantize"] = config.quantize
        plan["cors_origins"] = list(config.cors_origins)
        # Whether, not which: --dry-run output ends up in terminals and issues.
        plan["api_key"] = config.api_key is not None
        plan["endpoints"] = [
            "/health",
            "/metrics",
            "/v1/models",
            "/v1/chat/completions",
            "/v1/completions",
        ]
    return plan


def serve_model(config: ServeConfig, *, dry_run: bool = False, as_json: bool = False) -> int:
    """Run the server, or describe what running it would do.

    Blocks until interrupted, so there is no result object to print and no
    ``--json`` output beyond ``--dry-run``'s.
    """
    plan = serve_plan(config)
    if dry_run:
        print(json.dumps(plan, indent=2) if as_json else _format_plan(plan))
        return 0

    if config.backend in LAUNCHER_BACKENDS:
        # Nothing of LM7 is in the request path past this line -- see
        # serve/vllm.py and serve/trtllm.py.
        _refuse_foreign_passthrough(config)
        if config.ui_port is not None:
            # A static page on its own port, so LM7 hands out one HTML file and
            # the browser then talks to the launched server directly. Both vLLM
            # and trtllm-serve answer `access-control-allow-origin: *` by
            # default, so this needs no flag -- but a server started with
            # narrowed origins would have to include this one.
            from .ui import serve_page

            api = f"http://{config.host}:{config.port}"
            serve_page(config.ui_port, api, host=config.host)
            print(f"lm7: chat page on http://{config.host}:{config.ui_port} (talking to {api})")
        print(f"lm7: handing {plan['model']} to {config.backend} on {config.host}:{config.port}")
        return _serve_with_launcher(config)

    if config.ui_port is not None:
        raise UnsupportedModelError(
            f"--ui-port is for {' and '.join('--backend ' + name for name in LAUNCHER_BACKENDS)}, "
            "which own their port and serve no browser page. This server serves the chat page "
            f"itself at http://{config.host}:{config.port}/."
        )
    _refuse_foreign_passthrough(config)

    _require_serve_extra()
    from .engine import LM7ServeEngine
    from .server import run_server

    print(f"lm7: loading {plan['model']} for {plan['target']}...")
    engine = LM7ServeEngine.load(config)
    print(
        f"lm7: serving {engine.model_id} on http://{config.host}:{config.port} "
        f"({engine.target}, backend={engine.backend}, "
        f"max_model_len={engine.max_model_len}, "
        f"kv cache {engine.kv_cache_bytes / 1e6:.0f} MB)"
    )
    # Said out loud because it is the first thing that looks like a bug: the
    # graphs compile on their first call, so request one is slower than the rest
    # by however long Inductor takes. Which graphs those are is the whole of
    # --compile-prefill, and naming them is what makes a later stall legible --
    # with prefill compiled, every new prompt length pays the compile again.
    if config.compile_prefill:
        print(
            "lm7: the first request compiles the prefill and decode graphs and will be slower, "
            "and each new prompt length compiles the prefill again."
        )
    else:
        print(
            "lm7: the first request compiles the decode graph and will be slower. "
            "The prompt pass stays eager; --compile-prefill compiles it too."
        )
    run_server(config, engine)
    return 0


# Each launcher's verbatim passthrough, and the backend it belongs to. The two
# CLIs share no spelling, so handing vLLM's flags to trtllm-serve (or the
# reverse) could only ever produce an argv that does not parse.
_PASSTHROUGH = {"vllm": ("vllm_args", "--vllm-arg"), "trtllm": ("trtllm_args", "--trtllm-arg")}


def _refuse_foreign_passthrough(config: ServeConfig) -> None:
    """Refuse a passthrough aimed at a backend other than the one selected.

    Refused rather than ignored, like every other argument this command cannot
    honour: quietly dropping engine flags would start a server that is not the
    one that was asked for -- and the flags most likely to be passed this way are
    the ones that decide how much of the GPU it takes.
    """
    for backend, (field, flag) in _PASSTHROUGH.items():
        if config.backend == backend or not getattr(config, field):
            continue
        target = f"'{backend} serve'" if backend == "vllm" else "'trtllm-serve'"
        raise UnsupportedModelError(
            f"{flag} is passed through to {target} verbatim, and this is "
            f"--backend {config.backend}. Add --backend {backend} to hand the port over, "
            f"or drop {flag}."
        )


def _serve_with_launcher(config: ServeConfig) -> int:
    """Hand the process to the launcher named by ``config.backend``.

    Separate from :func:`_launcher` because launching is the one thing that is
    genuinely not shared: each of these replaces this process's work and returns
    the other server's exit code, so there is nothing left here to generalize.
    """
    if config.backend == "vllm":
        from .vllm import serve_with_vllm

        return serve_with_vllm(config)
    if config.backend == "trtllm":
        from .trtllm import serve_with_trtllm

        return serve_with_trtllm(config)
    # Not reachable through `serve_model`, which checks LAUNCHER_BACKENDS first.
    # Explicit anyway so that adding a name to that tuple without adding it here
    # raises instead of silently launching the wrong server.
    raise UnsupportedModelError(
        f"{config.backend!r} is listed as a launcher backend but has no launcher."
    )


def _require_serve_extra() -> None:
    """Name every missing package at once, before anything expensive happens.

    Checked here rather than left to the import in ``serve_model`` so that a
    missing extra is one sentence naming all of it, not three consecutive
    ``ModuleNotFoundError``s discovered one reinstall at a time -- and so that it
    is discovered before a multi-gigabyte download rather than after.
    """
    import importlib.util

    def missing(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is None
        except (ImportError, ValueError):
            # `find_spec` raises rather than returning None when a package is
            # present but broken -- a half-removed install, or a parent whose
            # import fails. Unimportable is unimportable either way.
            return True

    absent = [name for name in ("fastapi", "uvicorn", "pydantic") if missing(name)]
    if absent:
        raise UnsupportedModelError(
            f"lm7 model serve needs {', '.join(absent)} for its HTTP surface. "
            'Install the extra with: pip install "lm7[serve,hf]".'
        )


def _format_plan(plan: dict[str, Any]) -> str:
    lines = [f"{'model':<16}{plan['model']}", f"{'target':<16}{plan['target']}"]
    lines.append(f"{'runtime':<16}{plan['runtime']}")
    lines.append(f"{'address':<16}http://{plan['host']}:{plan['port']}")
    lines.append(f"{'max_model_len':<16}{plan['max_model_len']}")
    if plan["runtime"] in LAUNCHER_BACKENDS:
        state = plan["runtime_executable"] or "NOT FOUND"
        lines.append(f"{plan['runtime']:<16}{state}")
        lines.append(f"{'command':<16}{' '.join(plan['argv'])}")
        for name, value in plan["environment"].items():
            lines.append(f"{'env':<16}{name}={value}")
        if plan["ui_port"] is not None:
            lines.append(f"{'chat page':<16}http://{plan['host']}:{plan['ui_port']}")
    else:
        lines.append(f"{'quantize':<16}{plan['quantize']}")
        lines.append(f"{'prefill':<16}{'compiled' if plan['compile_prefill'] else 'eager'}")
        lines.append(f"{'cors_origins':<16}{', '.join(plan['cors_origins']) or 'none'}")
        lines.append(f"{'api_key':<16}{'required' if plan['api_key'] else 'none'}")
        lines.append(f"{'endpoints':<16}{' '.join(plan['endpoints'])}")
    return "\n".join(lines)


__all__ = ["serve_model", "serve_plan"]
