from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import json
import threading
from collections.abc import Mapping
from typing import Any

from ...backends.base import Support
from ...errors import UnsupportedModelError
from ..base import Capabilities, RuntimeInfo, ServeRequest, ServerHandle, unmet_capabilities

SUPPORTED_VENDORS = frozenset({"nvidia", "amd", "tpu", "cpu"})


def serve_argv(request: ServeRequest) -> list[str]:
    """Translate a ``ServeRequest`` into the arguments ``vllm serve`` takes.

    Returned as argv rather than a config object on purpose: vLLM's server is
    configured by an ``argparse.Namespace``, and its own ``make_arg_parser`` is
    the only thing that knows the current spelling of every flag. LM7 produces
    argv, vLLM parses and validates it, and a constraint LM7 got wrong is
    rejected by vLLM before a GPU is touched -- see ``build_namespace``.

    Pure and free of any vLLM import, so the mapping is unit-testable on a
    machine where vLLM cannot be installed at all.
    """
    # Imported here rather than at module scope: `lm7.huggingface` imports
    # `lm7.api`, which imports this package, so a top-level import would close
    # a cycle at interpreter start.
    from ...huggingface import _model_id

    argv = [
        "--model",
        _model_id(request.model),
        "--host",
        request.host,
        "--port",
        str(request.port),
        "--dtype",
        request.dtype,
        "--max-model-len",
        str(request.max_model_len),
        "--max-num-seqs",
        str(request.max_num_seqs),
        "--tensor-parallel-size",
        str(request.tensor_parallel_size),
    ]
    if request.max_batched_tokens is not None:
        argv += ["--max-num-batched-tokens", str(request.max_batched_tokens)]
    if request.kv_cache_fraction is not None:
        argv += ["--gpu-memory-utilization", str(request.kv_cache_fraction)]
    if request.prefix_caching:
        argv.append("--enable-prefix-caching")
    if request.lora_adapters:
        argv.append("--enable-lora")
        argv += ["--lora-modules", *request.lora_adapters]
    if request.speculative_model is not None:
        argv += ["--speculative-config", json.dumps({"model": request.speculative_model})]
    for key, value in request.extra.items():
        flag = "--" + str(key).replace("_", "-").lstrip("-")
        if value is True:
            argv.append(flag)
        elif value is not False and value is not None:
            argv += [flag, str(value)]
    return argv


def _startup_message(failure: BaseException) -> str:
    """Translate the one vLLM startup failure that reads as someone else's bug.

    vLLM V1 runs its ``EngineCore`` in a spawned subprocess, so the child
    re-imports the caller's ``__main__``. A script that calls ``lm7.serve()`` at
    module scope therefore tries to start the engine again while importing, and
    multiprocessing answers with a bootstrapping error naming ``freeze_support``
    -- which says nothing about serving, LM7, or vLLM.
    """
    text = str(failure)
    if "bootstrapping phase" in text or "freeze_support" in text:
        return (
            "vLLM starts its engine in a spawned subprocess, which re-imports the "
            "module that called lm7.serve(). Put the call under "
            'if __name__ == "__main__": so importing the module does not start a '
            "second engine. The lm7 CLI already does this."
        )
    return f"vLLM failed to start: {failure}"


def _flexible_argument_parser() -> Any:
    """vLLM's own argparse subclass, from wherever this vLLM keeps it.

    It lived in ``vllm.utils`` and moved to ``vllm.utils.argparse_utils`` by
    0.26. Both are tried because LM7 pins no vLLM version -- there is no `vllm`
    extra -- so it meets whatever the user installed.
    """
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        from vllm.utils import FlexibleArgumentParser
    return FlexibleArgumentParser()


def build_namespace(request: ServeRequest) -> Any:
    """Parse and validate the request through vLLM's own CLI parser.

    This is the step that makes ``--dry-run`` worth printing: what comes back is
    a configuration vLLM has already accepted, not a string LM7 hopes is right.
    """
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args

    parser = make_arg_parser(_flexible_argument_parser())
    args = parser.parse_args(serve_argv(request))
    validate_parsed_serve_args(args)
    return args


class VLLMServingRuntime:
    """vLLM, run in-process through its own OpenAI-compatible server.

    LM7 does not reimplement the OpenAI schema and does not supervise a child
    process: vLLM ships ``run_server`` as an importable coroutine, so the engine
    and its FastAPI app run inside this interpreter. vLLM V1 already puts its
    ``EngineCore`` in a subprocess, so the isolation that a supervisor would
    have provided is inherited rather than rebuilt.

    Implemented but **not validated on real hardware** -- no GPU was available
    when this landed. ``serve_argv`` is unit-tested; the launch path is not.
    """

    name = "vllm"

    def probe(self) -> RuntimeInfo:
        try:
            installed = importlib.util.find_spec("vllm") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        if not installed:
            return RuntimeInfo(
                self.name,
                None,
                False,
                "vLLM is not installed; install it separately with 'pip install vllm'.",
            )
        try:
            version = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            version = None
        return RuntimeInfo(self.name, version, True, "vLLM is importable.")

    def capabilities(self) -> Capabilities:
        return Capabilities(
            continuous_batching=True,
            paged_kv_cache=True,
            prefix_caching=True,
            chunked_prefill=True,
            speculative_decoding=True,
            lora=True,
            streaming=True,
            cancellation=True,
            metrics=True,
        )

    def supports(self, request: ServeRequest) -> Support:
        probe = self.probe()
        if not probe.available:
            return Support(False, probe.reason)
        if request.target.vendor not in SUPPORTED_VENDORS:
            return Support(
                False,
                f"vLLM does not serve {request.target.vendor} targets; "
                f"it supports {', '.join(sorted(SUPPORTED_VENDORS))}.",
            )
        missing = unmet_capabilities(request, self.capabilities())
        if missing:
            return Support(False, f"vLLM does not implement {', '.join(missing)}.")
        if request.compile_backend != "auto":
            # Refused rather than ignored, for the same reason the capability
            # flags are: vLLM is handed a checkpoint and compiles it internally,
            # so there is nothing here for an LM7 compile backend to act on, and
            # accepting the flag would imply otherwise.
            return Support(
                False,
                f"vLLM compiles internally and cannot be driven by LM7's "
                f"{request.compile_backend!r} backend; --compile-backend applies "
                "to the built-in runtime only.",
            )
        return Support(True, "vLLM serves this target with paged KV and continuous batching.", 90)

    def describe(self, request: ServeRequest) -> Mapping[str, Any]:
        """The argv LM7 would hand vLLM, validated by vLLM where it is installed.

        The ``validated`` flag is the honest part: on a machine without vLLM the
        argv is LM7's translation and nothing has checked it, and this says so
        rather than implying a confirmation that did not happen.
        """
        argv = serve_argv(request)
        described: dict[str, Any] = {"runtime": self.name, "argv": argv, "validated": False}
        if self.probe().available:
            try:
                build_namespace(request)
            except SystemExit as exc:  # argparse rejects unknown or malformed flags
                described["error"] = f"vLLM rejected these arguments (exit {exc.code})."
                return described
            except Exception as exc:  # noqa: BLE001 - reporting vLLM's refusal is the point
                described["error"] = f"vLLM rejected these arguments: {exc}"
                return described
            described["validated"] = True
        return described

    def launch(self, request: ServeRequest) -> ServerHandle:
        """Build vLLM's engine and app here, and serve them on a worker thread.

        Deliberately *not* ``run_server``: that is vLLM's CLI entry point and it
        installs a SIGTERM handler, which raises "signal only works in main
        thread of the main interpreter" anywhere but the main thread. A library
        cannot take the main thread from its caller, so LM7 composes the pieces
        underneath it -- the engine client, ``build_app``, ``init_app_state`` --
        and drives uvicorn itself. Those are the functions vLLM exposes for
        embedding; the CLI is only one caller of them.
        """
        probe = self.probe()
        if not probe.available:
            raise UnsupportedModelError(probe.reason)
        import uvicorn
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.openai.api_server import (
            build_app,
            build_async_engine_client_from_engine_args,
            init_app_state,
        )

        args = build_namespace(request)
        engine_args = AsyncEngineArgs.from_cli_args(args)
        loop = asyncio.new_event_loop()
        failure: list[BaseException] = []
        started = threading.Event()
        holder: dict[str, Any] = {}

        async def serve() -> None:
            async with build_async_engine_client_from_engine_args(engine_args) as engine_client:
                app = build_app(args)
                await init_app_state(engine_client, app.state, args)
                config = uvicorn.Config(
                    app, host=request.host, port=request.port, log_level="warning"
                )
                server = uvicorn.Server(config)
                holder["server"] = server
                started.set()
                await server.serve()

        def run() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(serve())
            except asyncio.CancelledError:
                pass
            except BaseException as exc:  # noqa: BLE001 - re-raised by wait_ready/stop
                failure.append(exc)
            finally:
                started.set()
                loop.close()

        thread = threading.Thread(target=run, name="lm7-serve-vllm", daemon=True)
        thread.start()
        # The engine loads weights and warms up before uvicorn binds, so this
        # waits on the app rather than on the thread merely being alive -- and
        # surfaces a startup failure here instead of leaving the caller to poll
        # a port that will never open.
        started.wait(timeout=1800)
        if failure:
            raise UnsupportedModelError(_startup_message(failure[0])) from failure[0]

        def stop() -> None:
            server = holder.get("server")
            if server is not None:
                server.should_exit = True
            thread.join(timeout=60)
            if failure:
                raise failure[0]

        return ServerHandle(
            runtime=self.name,
            base_url=f"http://{request.host}:{request.port}",
            target=request.target,
            config={"argv": serve_argv(request), "vllm_version": probe.version},
            _stop=stop,
        )
