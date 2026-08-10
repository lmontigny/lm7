"""What ``lm7 model serve`` does once argparse has finished.

Kept out of ``lm7/cli.py`` so that building the parser -- which happens for
``lm7 doctor`` and every other subcommand -- never imports FastAPI, Uvicorn or
Transformers. The parser lives with its siblings; only the handler is here.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..detection import resolve_target
from ..errors import UnsupportedModelError
from .engine import ServeConfig, resolve_model_source


def serve_plan(config: ServeConfig) -> dict[str, Any]:
    """What this invocation would do, without loading a model or binding a port.

    Backs ``--dry-run``, which exists because loading a model is the expensive
    part of finding out that a target was misspelled. For ``--backend vllm`` it
    also answers the two questions that decide whether the handover will work at
    all: which ``vllm`` LM7 found, and what it will change in the environment.
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
    if config.backend == "vllm":
        from .vllm import vllm_argv, vllm_environment, vllm_executable

        executable = vllm_executable()
        plan["runtime"] = "vllm"
        plan["vllm_installed"] = executable is not None
        # Named because "not installed" is usually "installed in a different
        # environment" -- vllm-metal builds its own venv on purpose.
        plan["vllm_executable"] = executable
        plan["argv"] = vllm_argv(config)
        plan["ui_port"] = config.ui_port
        overrides = {
            name: value
            for name, value in vllm_environment(config).items()
            if os.environ.get(name) != value
        }
        plan["environment"] = overrides
    else:
        plan["runtime"] = "lm7"
        plan["dtype"] = config.dtype
        plan["compile_mode"] = config.compile_mode
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

    if config.backend == "vllm":
        # Nothing of LM7 is in the request path past this line -- see serve/vllm.py.
        from .vllm import serve_with_vllm

        if config.ui_port is not None:
            # A static page on its own port, so LM7 hands out one HTML file and
            # the browser then talks to vLLM directly. vLLM answers
            # `access-control-allow-origin: *` by default, so this needs no flag
            # -- but a server started with a narrowed --allowed-origins would
            # have to include this one.
            from .ui import serve_page

            api = f"http://{config.host}:{config.port}"
            serve_page(config.ui_port, api, host=config.host)
            print(f"lm7: chat page on http://{config.host}:{config.ui_port} (talking to {api})")
        print(f"lm7: handing {plan['model']} to vLLM on {config.host}:{config.port}")
        return serve_with_vllm(config)

    if config.ui_port is not None:
        raise UnsupportedModelError(
            "--ui-port is for --backend vllm, which owns its port and serves no browser "
            f"page. This server serves the chat page itself at http://{config.host}:{config.port}/."
        )
    if config.vllm_args:
        # Refused rather than ignored, like every other argument this server
        # cannot honour: quietly dropping engine flags would start a server that
        # is not the one that was asked for.
        raise UnsupportedModelError(
            "--vllm-arg is passed through to 'vllm serve' and means nothing to LM7's own "
            "server. Add --backend vllm to hand the port over, or drop --vllm-arg."
        )

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
    # by however long Inductor takes.
    print("lm7: the first request compiles the prefill and decode graphs and will be slower.")
    run_server(config, engine)
    return 0


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
    if plan["runtime"] == "vllm":
        state = plan["vllm_executable"] or "NOT FOUND"
        lines.append(f"{'vllm':<16}{state}")
        lines.append(f"{'command':<16}{' '.join(plan['argv'])}")
        for name, value in plan["environment"].items():
            lines.append(f"{'env':<16}{name}={value}")
        if plan["ui_port"] is not None:
            lines.append(f"{'chat page':<16}http://{plan['host']}:{plan['ui_port']}")
    else:
        lines.append(f"{'quantize':<16}{plan['quantize']}")
        lines.append(f"{'cors_origins':<16}{', '.join(plan['cors_origins']) or 'none'}")
        lines.append(f"{'api_key':<16}{'required' if plan['api_key'] else 'none'}")
        lines.append(f"{'endpoints':<16}{' '.join(plan['endpoints'])}")
    return "\n".join(lines)


__all__ = ["serve_model", "serve_plan"]
