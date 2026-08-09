from __future__ import annotations

import asyncio
import importlib.util
import threading
import time
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any

import torch

from ...backends.base import Support
from ...detection import torch_device
from ...errors import UnsupportedModelError
from ..base import Capabilities, RuntimeInfo, ServeRequest, ServerHandle, unmet_capabilities
from ..budget import ModelShape, plan_memory

_MISSING_DEPENDENCIES = (
    "The reference runtime needs FastAPI and Uvicorn for its HTTP surface and "
    'Transformers to load a model; install LM7 with ".[serve,hf]".'
)


class BuiltinServingRuntime:
    """LM7's own single-stream server. A reference, not a serving system.

    Every capability that makes a serving engine fast is absent here on purpose:
    one request is served at a time, the KV cache is the static one
    ``compile_generation`` allocates, and a second caller waits behind a lock.
    It exists so that ``fallback="warn"`` means the same thing for ``serve`` as
    it does for ``compile``, so the HTTP contract is testable on a CPU runner
    where vLLM does not install, and because an all-``False`` capability row is
    the most honest description of what LM7 implements by itself.
    """

    name = "builtin"

    def probe(self) -> RuntimeInfo:
        missing = [
            module
            for module in ("fastapi", "uvicorn", "transformers")
            if importlib.util.find_spec(module) is None
        ]
        if missing:
            return RuntimeInfo(self.name, None, False, _MISSING_DEPENDENCIES)
        return RuntimeInfo(self.name, None, True, "Reference single-stream server is available.")

    def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, cancellation=True, metrics=True)

    def supports(self, request: ServeRequest) -> Support:
        probe = self.probe()
        if not probe.available:
            return Support(False, probe.reason)
        missing = unmet_capabilities(request, self.capabilities())
        if missing:
            return Support(
                False,
                f"The reference runtime does not implement {', '.join(missing)}. "
                "It serves one request at a time; use a real serving runtime.",
            )
        # Priority 0 so that any installed engine outranks it in automatic
        # planning, exactly as `eager` sits below `inductor` for compile.
        return Support(True, "Reference single-stream server; no batching, no paging.", priority=0)

    def describe(self, request: ServeRequest) -> Mapping[str, Any]:
        from ...huggingface import _model_id

        return {
            "runtime": self.name,
            "model": _model_id(request.model),
            "max_model_len": request.max_model_len,
            "max_num_seqs": request.max_num_seqs,
            # Requested, not resolved: `auto` becomes a concrete backend only
            # when the planner runs inside compile_generation, which needs the
            # loaded model. The handle and /metrics report what actually ran.
            "compile_backend_requested": request.compile_backend,
            "note": "One request at a time; no batching, no paged KV cache.",
        }

    def launch(self, request: ServeRequest) -> ServerHandle:
        probe = self.probe()
        if not probe.available:
            raise UnsupportedModelError(probe.reason)
        server = _ReferenceServer(request)
        return server.start()


class _Metrics:
    """The few numbers a single-stream server can honestly report."""

    def __init__(self) -> None:
        self.requests = 0
        self.prompt_tokens = 0
        self.generated_tokens = 0
        self.ttft_ms_total = 0.0
        self.decode_ms_total = 0.0

    def record(self, prompt_tokens: int, generated: int, ttft_ms: float, decode_ms: float) -> None:
        self.requests += 1
        self.prompt_tokens += prompt_tokens
        self.generated_tokens += generated
        self.ttft_ms_total += ttft_ms
        self.decode_ms_total += decode_ms

    def to_dict(self) -> dict[str, Any]:
        decoded = max(self.generated_tokens - self.requests, 0)
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "ttft_ms": self.ttft_ms_total / self.requests if self.requests else 0.0,
            "tpot_ms": self.decode_ms_total / decoded if decoded else 0.0,
        }


class _ReferenceServer:
    runtime_name = "builtin"

    def __init__(self, request: ServeRequest) -> None:
        from ...huggingface import _model_id

        self.request = request
        self.model_id = _model_id(request.model)
        self.target = request.target
        self.metrics = _Metrics()
        # One request at a time is the whole point, and the lock is what makes
        # that true rather than merely likely: the runner owns exactly one
        # static cache, so two concurrent generations would interleave writes
        # into the same buffers and corrupt both.
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        import transformers

        # Both of these reach modules that import `lm7.api`, which imports this
        # package; deferring them to call time keeps that cycle from closing.
        from ...generation import compile_generation
        from ...huggingface import _resolve_dtype

        dtype = _resolve_dtype(self.request.dtype, self.target)
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=dtype
            ).eval()
        except Exception as exc:
            raise UnsupportedModelError(
                f"Hugging Face load stage failed for {self.request.model}: {exc}."
            ) from exc
        self.budget = plan_memory(
            ModelShape.from_config(model.config),
            dtype=dtype,
            max_model_len=self.request.max_model_len,
            max_num_seqs=self.request.max_num_seqs,
            device_bytes=None,
            weight_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
            kv_cache_fraction=self.request.kv_cache_fraction,
        )
        self.runner = compile_generation(
            model,
            self.target,
            backend=self.request.compile_backend,
            # The decode graph compiles once, at a fixed shape, and is reused for
            # every token of every request -- which is the whole win. The prefill
            # graph compiles *per prompt length*, so compiling it in a server
            # means a fresh compile the first time each new prompt length
            # arrives. A benchmark with one prompt length never sees that; a
            # server sees it constantly, so the prompt pass stays eager.
            compile_prefill=False,
            max_batch_size=1,
            max_sequence_length=self.request.max_model_len,
        )
        self.device = torch_device(self.target)

    def _encode(self, prompt: str) -> torch.Tensor:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return torch.as_tensor(encoded["input_ids"]).to(self.device)

    def apply_chat_template(self, messages: Iterable[dict[str, Any]]) -> str:
        conversation = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
        ]
        template = getattr(self.tokenizer, "chat_template", None)
        if template:
            text = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            return str(text)
        return "\n".join(f"{m['role']}: {m['content']}" for m in conversation) + "\nassistant:"

    def check_capacity(self, prompt: str, max_tokens: int) -> torch.Tensor:
        """Tokenize, and refuse anything the static cache cannot hold.

        Called before the response type is chosen so that an oversized request
        fails as a 400 rather than as a stream that dies after its first chunk.
        """
        input_ids = self._encode(prompt)
        prompt_tokens = int(input_ids.shape[-1])
        if prompt_tokens + max_tokens > self.request.max_model_len:
            raise ValueError(
                f"The prompt is {prompt_tokens} tokens and {max_tokens} more were asked for, "
                f"which exceeds the {self.request.max_model_len}-token cache this server "
                "allocated. Restart it with a larger --max-model-len."
            )
        return input_ids

    async def generate(
        self, prompt: str, max_tokens: int, is_disconnected: Any
    ) -> AsyncIterator[tuple[str, bool]]:
        """Yield ``(delta, finished)`` per decoded token.

        Each step runs in a worker thread so the event loop stays free to notice
        a client that has gone away -- which is what makes the ``cancellation``
        capability a fact rather than a claim.
        """
        input_ids = self.check_capacity(prompt, max_tokens)
        prompt_tokens = int(input_ids.shape[-1])
        with self._lock:
            started = time.perf_counter()
            state = await asyncio.to_thread(self.runner.prefill, input_ids)
            ttft_ms = (time.perf_counter() - started) * 1000
            token = state.next_token
            token_ids = [int(token.item())]
            # `decode` is typed as returning str or list[str] depending on
            # whether it was handed one sequence or many; this hands it one.
            text = str(self.tokenizer.decode(token_ids, skip_special_tokens=True))
            yield text, False

            decode_started = time.perf_counter()
            eos = self.tokenizer.eos_token_id
            try:
                for _ in range(max_tokens - 1):
                    if await is_disconnected():
                        break
                    token, state = await asyncio.to_thread(self.runner.decode, token, state)
                    token_id = int(token.item())
                    if token_id == eos:
                        break
                    token_ids.append(token_id)
                    updated = str(self.tokenizer.decode(token_ids, skip_special_tokens=True))
                    delta, text = updated[len(text) :], updated
                    if delta:
                        yield delta, False
            finally:
                # In a `finally` because an abandoned stream is closed *at* the
                # yield: the generator is thrown a GeneratorExit and everything
                # after the loop is skipped. Recording outside it loses exactly
                # the requests a server most wants counted -- the cancelled ones.
                decode_ms = (time.perf_counter() - decode_started) * 1000
                self.metrics.record(prompt_tokens, len(token_ids), ttft_ms, decode_ms)
        yield "", True

    def build_app(self) -> Any:
        from ._openai_app import build_app

        return build_app(self)

    def start(self) -> ServerHandle:
        import uvicorn

        config = uvicorn.Config(
            self.build_app(),
            host=self.request.host,
            port=self.request.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, name="lm7-serve", daemon=True)
        thread.start()

        deadline = time.monotonic() + 60
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            raise UnsupportedModelError("The reference server did not start within 60 seconds.")

        def stop() -> None:
            server.should_exit = True
            thread.join(timeout=30)

        # Port 0 asks the OS to choose, so the bound port is only knowable here.
        port = self.request.port
        if port == 0 and server.servers:
            port = server.servers[0].sockets[0].getsockname()[1]
        return ServerHandle(
            runtime=self.runtime_name,
            base_url=f"http://{self.request.host}:{port}",
            target=self.target,
            config={
                "model": self.model_id,
                "max_model_len": self.request.max_model_len,
                "max_num_seqs": self.request.max_num_seqs,
                "memory": self.budget.to_dict(),
                **self.compilation(),
            },
            _stop=stop,
        )

    def compilation(self) -> dict[str, Any]:
        """What the compiler actually did, as opposed to what was asked for.

        ``compile_generation`` resolves ``auto`` through LM7's own planner, so
        the answer is only knowable after the model is loaded -- and it differs
        per target. Reporting the request instead of the result is how a doc
        ends up claiming a compile that never happened.
        """
        # `runner.backend` is the string that was *asked for*, so it still reads
        # "auto" after the planner has chosen. The decode graph's CompiledModule
        # records what was actually selected, but only once it has compiled --
        # which happens on the first decode, not at load.
        decode_graph = getattr(self.runner, "_decode_graph", None)
        selected = getattr(decode_graph, "selected_backend", None)
        return {
            "compile_backend_requested": self.request.compile_backend,
            "compile_backend": selected,
            "compiled_decode": selected is not None,
            # Left eager on purpose: the prefill graph compiles per prompt
            # length, which a server pays repeatedly. See _load.
            "compiled_prefill": False,
        }
