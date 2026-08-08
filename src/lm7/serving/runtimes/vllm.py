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


def build_namespace(request: ServeRequest) -> Any:
    """Parse and validate the request through vLLM's own CLI parser.

    This is the step that makes ``--dry-run`` worth printing: what comes back is
    a configuration vLLM has already accepted, not a string LM7 hopes is right.
    """
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser())
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
        probe = self.probe()
        if not probe.available:
            raise UnsupportedModelError(probe.reason)
        from vllm.entrypoints.openai.api_server import run_server

        args = build_namespace(request)
        loop = asyncio.new_event_loop()
        failure: list[BaseException] = []

        def run() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_server(args))
            except asyncio.CancelledError:
                pass
            except BaseException as exc:  # noqa: BLE001 - re-raised by stop(), not swallowed
                failure.append(exc)
            finally:
                loop.close()

        thread = threading.Thread(target=run, name="lm7-serve-vllm", daemon=True)
        thread.start()

        def stop() -> None:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=30)
            if failure:
                raise failure[0]

        return ServerHandle(
            runtime=self.name,
            base_url=f"http://{request.host}:{request.port}",
            target=request.target,
            config={"argv": serve_argv(request), "vllm_version": probe.version},
            _stop=stop,
        )
