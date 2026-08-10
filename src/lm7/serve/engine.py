"""The bridge between async HTTP requests and a synchronous compiled decode loop.

LM7 writes no kernels and, here, no KV cache and no scheduler either: the whole
inference path is :func:`lm7.compile_generation`, which owns the two compiled
graphs and the one static cache. This module adds exactly three things that HTTP
needs and the runner does not provide -- a chat template, token selection, and
serialization.

That last one is the load-bearing part. ``compile_generation`` allocates a single
static KV cache and every ``decode`` step mutates it in place, so two concurrent
requests would interleave writes into the same buffers and corrupt both answers
without raising. A lock is what makes "one at a time" true rather than likely.
This is a local single-user server; it does not batch, and a second caller waits.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..detection import resolve_target
from ..errors import UnsupportedModelError
from ..targets import TargetSpec

# What a caller may pass to `/v1/chat/completions` as `stop`.
StopSequences = Sequence[str]

# Files `save_pretrained` always writes, and the cheapest proof that a directory
# holds a model rather than happening to exist.
_MODEL_CONFIG_NAME = "config.json"


def resolve_model_source(model: str) -> str:
    """The string to hand ``from_pretrained``, from what the user typed.

    ``hf://owner/model`` is the Hub. A path to a directory holding a model --
    what ``save_pretrained`` writes -- is served directly, which is what makes a
    local fine-tune, a pre-downloaded checkpoint, or an air-gapped box reachable
    without a Hub round trip. An existing directory always wins, and there is no
    ambiguity to arbitrate: a Hub id is only ever accepted with its ``hf://``
    prefix, so a bare string that is not a directory was never valid anyway.

    The resolved absolute path is also the served model id. The server holds
    exactly one model and echoes back whatever name a client sends, so the id is
    read by humans debugging, and a path says which checkpoint far better than
    its last path component would.
    """
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        resolved = candidate.resolve()
        if not (resolved / _MODEL_CONFIG_NAME).is_file():
            raise UnsupportedModelError(
                f"{resolved} is a directory but does not contain {_MODEL_CONFIG_NAME}, so it "
                "does not hold a model saved by save_pretrained. Point at the directory that "
                "does, or use a Hugging Face URI such as 'hf://owner/model'."
            )
        return str(resolved)
    if candidate.exists():
        raise UnsupportedModelError(
            f"{model!r} is a file, not a directory. A local model is the directory "
            f"save_pretrained wrote, containing {_MODEL_CONFIG_NAME} and the weights."
        )
    if not model.startswith("hf://") and _looks_like_a_path(model):
        raise UnsupportedModelError(
            f"No such directory {model!r}. A local model is a directory containing "
            f"{_MODEL_CONFIG_NAME}; a Hub model is 'hf://owner/model'."
        )
    # Imported here rather than at module scope: `huggingface` pulls in
    # Transformers and the whole compile stack, and this module is imported by
    # `--dry-run`, which must not load either.
    from ..huggingface import _model_id

    return _model_id(model)


def _looks_like_a_path(model: str) -> bool:
    """Whether a user who typed this meant a path, so the error can say so."""
    return model.startswith((".", "~", "/")) or "\\" in model


# Awaited between decode steps to notice a client that has gone away. Kept as a
# parameter rather than reaching for `starlette.Request` so the engine stays
# importable, and testable, without FastAPI installed.
DisconnectProbe = Callable[[], Awaitable[bool]]


@dataclass(frozen=True)
class ServeConfig:
    """Everything fixed at server start, which is nearly everything.

    A static cache is allocated once at the size named here and never grows, so
    ``max_model_len`` is a property of the *server*, not of a request: a prompt
    that does not fit is refused rather than served with a reallocated cache.

    ``cors_origins`` defaults to every origin because the server binds loopback
    and holds no credentials, and a browser UI on another port is the ordinary
    way to use it. Narrow it when the server is reachable from anywhere else.
    """

    model: str
    target: str = "auto"
    backend: str = "auto"
    dtype: str = "auto"
    max_model_len: int = 4096
    compile_mode: str | None = None
    compile_prefill: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    # Only meaningful with `--backend vllm`: LM7's own server already
    # serves the chat page at `/`, while vLLM owns its port and ships none.
    ui_port: int | None = None
    quantize: str = "none"
    cors_origins: tuple[str, ...] = ("*",)
    api_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "target": self.target,
            "backend": self.backend,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "compile_mode": self.compile_mode,
            "compile_prefill": self.compile_prefill,
            "host": self.host,
            "port": self.port,
            "ui_port": self.ui_port,
            "quantize": self.quantize,
            "cors_origins": list(self.cors_origins),
            # Whether one is set, never which one: this dict backs --dry-run and
            # --json, and a key printed to a terminal is a key in a scrollback.
            "api_key": self.api_key is not None,
        }


@dataclass(frozen=True)
class TokenDelta:
    """One step of a generation: the new text, and whether that was the last of it.

    On an ordinary step ``text`` is a *delta*, and it can be empty even though a
    token was produced -- a BPE piece that is half a UTF-8 character decodes to
    nothing until its partner arrives.

    On the final step (``finished=True``) ``text`` is instead the whole
    completion. That asymmetry is deliberate: a stop sequence is only recognized
    once it has been decoded, so the authoritative answer is not always the
    concatenation of what was streamed, and a non-streaming caller should get the
    authoritative one.
    """

    text: str
    token_id: int | None
    finished: bool = False
    finish_reason: str | None = None
    token_count: int = 0


@dataclass
class Metrics:
    """Counters a one-request-at-a-time server can report without inventing any.

    Notably absent: queue depth, batch size, cache utilization. There is one
    request in flight and the cache is fully allocated whether or not it is used.
    """

    requests: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    ttft_ms_total: float = 0.0
    decode_ms_total: float = 0.0
    decode_steps: int = 0

    def record(
        self, prompt_tokens: int, generated: int, ttft_ms: float, decode_ms: float, steps: int
    ) -> None:
        self.requests += 1
        self.prompt_tokens += prompt_tokens
        self.generated_tokens += generated
        self.ttft_ms_total += ttft_ms
        self.decode_ms_total += decode_ms
        self.decode_steps += steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            # Time to first token is the prefill pass; time per output token is
            # the decode loop divided by the steps that ran, so a request that
            # stopped at its prefill token contributes to the first and not the
            # second.
            "ttft_ms": self.ttft_ms_total / self.requests if self.requests else 0.0,
            "tpot_ms": self.decode_ms_total / self.decode_steps if self.decode_steps else 0.0,
        }


def select_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick the next token id from one position of logits.

    ``temperature=0`` is greedy, which is what ``GenerationRunner`` does on its
    own -- the runner's ``state.next_token`` is an argmax. Everything above zero
    samples, so this reads ``state.logits`` rather than ``state.next_token`` and
    the runner's own choice is discarded.

    Sampling runs on the CPU in float32 whatever the model's device and dtype.
    It is three tensor ops on a single row, so the transfer dominates either way,
    and ``torch.multinomial`` with a seeded generator is not supported on every
    accelerator LM7 targets. ``decode`` moves the token back itself.
    """
    row = logits[:, -1, :].detach().to("cpu", torch.float32)
    if temperature <= 0.0:
        return row.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(row / temperature, dim=-1)
    if top_p < 1.0:
        ordered, indices = torch.sort(probabilities, dim=-1, descending=True)
        cumulative = ordered.cumsum(dim=-1)
        # Keep everything strictly below the threshold, plus the token that
        # crosses it -- otherwise a top_p smaller than the largest single
        # probability keeps nothing and multinomial raises.
        drop = cumulative - ordered > top_p
        ordered = ordered.masked_fill(drop, 0.0)
        probabilities = torch.zeros_like(probabilities).scatter_(-1, indices, ordered)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def _normalize_stop(stop: str | list[str] | None) -> tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,) if stop else ()
    return tuple(entry for entry in stop if entry)


def _stop_index(text: str, stops: StopSequences) -> int | None:
    """Where the earliest stop sequence begins in ``text``, if any.

    Searched over the whole accumulated text rather than the latest delta,
    because a stop sequence is tokenized however the model felt like it and can
    straddle two tokens.
    """
    positions = [text.index(stop) for stop in stops if stop in text]
    return min(positions) if positions else None


class LM7ServeEngine:
    """One model, one compiled runner, one static cache, one request at a time.

    Construct with :meth:`load` for the real thing; the initializer takes an
    already-built runner and tokenizer so a test can hand it fakes without a
    download.
    """

    def __init__(
        self,
        runner: Any,
        tokenizer: Any,
        config: ServeConfig,
        *,
        model_id: str,
        target: TargetSpec | None = None,
        backend: str | None = None,
    ) -> None:
        self.runner = runner
        self.tokenizer = tokenizer
        self.config = config
        self.model_id = model_id
        self.target = str(target if target is not None else getattr(runner, "target", "cpu"))
        self._requested_backend = (
            backend if backend is not None else str(getattr(runner, "backend", "eager"))
        )
        self.max_model_len = config.max_model_len
        self.metrics = Metrics()
        # Created on first use rather than here. An `asyncio.Lock` binds to the
        # running loop the first time it is awaited, and the engine is built
        # before Uvicorn starts one.
        self._lock: asyncio.Lock | None = None

    # -- construction -----------------------------------------------------

    @classmethod
    def load(cls, config: ServeConfig) -> LM7ServeEngine:
        """Download, load, and compile the model named by ``config``.

        Everything expensive happens here, on purpose: the first HTTP request
        should measure the model and not the toolchain. ``compile_generation``
        compiles lazily on the first call of each graph, so a first request is
        still slower than the rest -- see docs/serving.md.
        """
        # Imported at call time. `huggingface` pulls in Transformers and the
        # whole compile stack, and `lm7.serve` is imported by the CLI parser
        # before anyone has asked to serve anything.
        from ..generation import compile_generation
        from ..huggingface import (
            _apply_quantization,
            _resolve_dtype,
            _validate_quantization,
            normalize_quantization,
        )

        model_id = resolve_model_source(config.model)
        target = resolve_target(config.target)
        # Gated before the download, not after: every quantization refusal here
        # is a property of the target, backend and dtype, so finding out costs
        # nothing and finding out late costs a multi-gigabyte checkpoint.
        quantization = normalize_quantization(config.quantize)
        # `quantization` is what makes `dtype="auto"` mean BF16 on NVIDIA rather
        # than the FP16 an unquantized model gets. Resolving without it served
        # INT8 weights under FP16 compute, whose logits are NaN -- so every token
        # was an argmax over NaN, and a chat model whose EOS is not token 0 ran to
        # its full budget and returned an empty string with `finish_reason:
        # "length"` and no error. Measured on an RTX 4070 SUPER (Ada `sm89`).
        dtype = _resolve_dtype(config.dtype, target, quantization)
        if quantization != "none" and not config.model.startswith("hf://"):
            # The per-model gate is keyed by Hub id, and a directory does not
            # carry one -- transformers 5.x's `save_pretrained` writes no
            # `_name_or_path`, so there is nothing to recover it from. Refusing
            # here rather than passing the path through means the message names
            # the real problem instead of printing a filesystem path where the
            # error text promises a model id. Passing None would be worse: that
            # skips the gate entirely, quantizing models nobody has checked.
            raise UnsupportedModelError(
                f"--quantize {quantization} is not available for a local directory. LM7 gates "
                "quantization per model and identifies a model by its Hugging Face id, which a "
                "directory does not carry. Serve the hf:// form of a validated model to "
                "quantize it, or drop --quantize to serve this directory unquantized."
            )
        _validate_quantization(quantization, target, config.backend, config.dtype, model_id)
        transformers = _load_transformers()
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
            model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval()
        except Exception as exc:
            raise UnsupportedModelError(
                f"Model load stage failed for {config.model}: {exc}."
            ) from exc
        # Before `compile_generation`, which uses the model exactly as given and
        # so compiles whatever it is handed -- a model quantized here decodes
        # quantized. `_apply_quantization` refuses a filter that matched nothing
        # rather than reporting a quantization that did not happen.
        _apply_quantization(model, target, quantization)
        runner = compile_generation(
            model,
            target,
            backend=config.backend,
            compile_mode=config.compile_mode,
            compile_prefill=config.compile_prefill,
            max_batch_size=1,
            max_sequence_length=config.max_model_len,
        )
        return cls(runner, tokenizer, config, model_id=model_id, target=target)

    # -- prompt handling --------------------------------------------------

    def apply_chat_template(self, messages: Iterable[Any]) -> str:
        """Turn an OpenAI ``messages`` array into the string the model expects.

        The tokenizer's own template is the only correct answer -- the special
        tokens that mark a turn boundary are the checkpoint's, not a convention
        -- so a base model without one gets a plainly-labelled transcript and no
        pretence that it is the same thing.
        """
        conversation = [
            {"role": _message_field(message, "role"), "content": _message_field(message, "content")}
            for message in messages
        ]
        if getattr(self.tokenizer, "chat_template", None):
            rendered = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            return str(rendered)
        transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in conversation)
        return f"{transcript}\nassistant:"

    def encode(self, prompt: str) -> torch.Tensor:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return torch.as_tensor(encoded["input_ids"])

    def resolve_budget(self, prompt: str, max_tokens: int | None) -> tuple[torch.Tensor, int]:
        """Tokenize, and settle how many tokens this request may generate.

        Called before the response type is chosen so that an impossible request
        is a 400 with a reason, rather than a 200 whose stream dies after one
        chunk.

        ``max_tokens=None`` means "whatever still fits", which is the only budget
        that stays correct as a conversation grows: a client resending its
        transcript has a prompt that gets longer every turn, so any constant it
        picked at the start is eventually larger than the space left. An explicit
        ask is still refused rather than quietly narrowed -- a caller that
        requested 512 tokens and silently received 40 has been misled.
        """
        input_ids = self.encode(prompt)
        prompt_tokens = int(input_ids.shape[-1])
        remaining = self.max_model_len - prompt_tokens
        if remaining < 1:
            raise ValueError(
                f"The prompt is {prompt_tokens} tokens, which leaves no room in the "
                f"{self.max_model_len}-token static cache this server allocated at startup. "
                "Send a shorter conversation, or restart the server with a larger "
                "--max-model-len."
            )
        if max_tokens is None:
            return input_ids, remaining
        if max_tokens > remaining:
            raise ValueError(
                f"The prompt is {prompt_tokens} tokens and {max_tokens} more were requested, "
                f"which exceeds the {self.max_model_len}-token static cache this server "
                f"allocated at startup. Ask for at most {remaining}, omit max_tokens to use "
                "whatever fits, send a shorter conversation, or restart the server with a "
                "larger --max-model-len."
            )
        return input_ids, max_tokens

    # -- generation -------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        is_disconnected: DisconnectProbe | None = None,
    ) -> AsyncIterator[TokenDelta]:
        """Yield one :class:`TokenDelta` per step, then a final finished delta.

        Every PyTorch call goes through ``asyncio.to_thread``. That is not a
        micro-optimization: prefill on a 1B model is hundreds of milliseconds
        during which the event loop would otherwise answer nothing, so ``/health``
        would time out and a disconnected client would go unnoticed until the
        generation finished. Only one thread is ever in the runner, because the
        lock is held for the whole loop.
        """
        input_ids, max_tokens = self.resolve_budget(prompt, max_tokens)
        prompt_tokens = int(input_ids.shape[-1])
        stops = _normalize_stop(stop)
        # How many characters to keep back from the stream. A stop sequence is
        # only visible once every one of its characters has been decoded, so
        # emitting text the instant it appears would send the first half of a
        # stop sequence to the client and then have nothing to unsend it with.
        # Holding back one character less than the longest stop is the smallest
        # buffer that cannot leak one.
        hold = max((len(entry) for entry in stops), default=1) - 1
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        token_ids: list[int] = []
        emitted = ""
        text = ""
        finish_reason = "length"

        async with self._acquire():
            # The cache is reset by `prefill` itself, but a request abandoned
            # mid-stream left the runner's mask and positions where it stopped,
            # so this starts from a known state rather than a plausible one.
            self.runner.reset()

            started = time.perf_counter()
            state = await asyncio.to_thread(self.runner.prefill, input_ids)
            ttft_ms = (time.perf_counter() - started) * 1000.0

            token = select_token(
                state.logits, temperature=temperature, top_p=top_p, generator=generator
            )
            first = int(token.item())
            steps = 0
            decode_started = time.perf_counter()
            try:
                if self._is_eos(first):
                    finish_reason = "stop"
                else:
                    token_ids.append(first)
                    text = self._detokenize(token_ids)
                    for _ in range(max_tokens - 1):
                        if (cut := _stop_index(text, stops)) is not None:
                            finish_reason, text = "stop", text[:cut]
                            break
                        safe = text[: len(text) - hold] if hold else text
                        if len(safe) > len(emitted):
                            delta, emitted = safe[len(emitted) :], safe
                            yield TokenDelta(delta, token_ids[-1])
                        if is_disconnected is not None and await is_disconnected():
                            finish_reason = "cancelled"
                            break
                        _, state = await asyncio.to_thread(self.runner.decode, token, state)
                        steps += 1
                        token = select_token(
                            state.logits, temperature=temperature, top_p=top_p, generator=generator
                        )
                        token_id = int(token.item())
                        if self._is_eos(token_id):
                            finish_reason = "stop"
                            break
                        token_ids.append(token_id)
                        text = self._detokenize(token_ids)
                    else:
                        # The token budget ran out rather than the model stopping,
                        # so the last token has not been checked for a stop yet.
                        if (cut := _stop_index(text, stops)) is not None:
                            finish_reason, text = "stop", text[:cut]
            finally:
                # In a `finally` because an abandoned stream is closed *at* a
                # yield: the generator is thrown GeneratorExit and nothing after
                # the loop runs. Recording outside it would lose exactly the
                # requests worth counting -- the cancelled ones.
                decode_ms = (time.perf_counter() - decode_started) * 1000.0
                self.metrics.record(prompt_tokens, len(token_ids), ttft_ms, decode_ms, steps)

        # Whatever was held back, plus whatever the last token added.
        if len(text) > len(emitted) and token_ids:
            yield TokenDelta(text[len(emitted) :], token_ids[-1])
        yield TokenDelta(
            text, None, finished=True, finish_reason=finish_reason, token_count=len(token_ids)
        )

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int | None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        is_disconnected: DisconnectProbe | None = None,
    ) -> tuple[str, str, int]:
        """Run :meth:`generate` to the end. Returns ``(text, finish_reason, tokens)``.

        Reads the final delta rather than concatenating the stream: see
        :class:`TokenDelta` for why those two can differ.
        """
        text, reason, tokens = "", "length", 0
        async for delta in self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop=stop,
            is_disconnected=is_disconnected,
        ):
            if delta.finished:
                text, reason, tokens = delta.text, delta.finish_reason or reason, delta.token_count
        return text, reason, tokens

    # -- introspection ----------------------------------------------------

    @property
    def backend(self) -> str:
        """What compiled the decode graph, once anything has.

        ``--backend auto`` is a request, not an answer, and reporting it back as
        the server's backend would be reporting the question. ``compile_generation``
        compiles on the first call of each graph, so until a request has run there
        genuinely is no answer yet and the requested value is the honest one.
        """
        graphs = getattr(self.runner, "cudagraphs", None)
        if isinstance(graphs, dict):
            selected = graphs.get("decode", {}).get("backend")
            if selected:
                return str(selected)
        return self._requested_backend

    @property
    def kv_cache_bytes(self) -> int:
        return int(getattr(self.runner, "cache_bytes", 0))

    @property
    def warm(self) -> bool:
        """Whether the compile cost has been paid.

        The graphs compile on their first call, so the first request of a
        server's life is slower than every one after it by however long the
        backend takes. A client that knows this can say so instead of looking
        hung.
        """
        return self.metrics.requests > 0

    def graph_stats(self) -> dict[str, int]:
        """What the runner's Dynamo counters say about compiling so far.

        ``steady_frames`` is the number this path exists to make checkable:
        anything above zero means a *token* triggered a compile, which is the
        failure the split into separate prefill and decode graphs prevents. It
        is surfaced over HTTP rather than left in ``runner.counters`` because a
        server is exactly where that regression would go unnoticed.

        ``prefill_lengths`` is the cost the split accepts in exchange: the prompt
        pass is compiled per prompt length, so a varied workload pays repeatedly.
        See docs/kv-cache-decode.md.
        """
        counters = getattr(self.runner, "counters", None)
        steady = counters.get("steady", {}) if isinstance(counters, dict) else {}
        lengths = getattr(self.runner, "compiled_prefill_lengths", ())
        return {
            "prefill_lengths": len(lengths),
            "steady_frames": int(steady.get("frames", 0) or 0),
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "target": self.target,
            "backend": self.backend,
            "kv_cache_bytes": self.kv_cache_bytes,
            "max_model_len": self.max_model_len,
            "warm": self.warm,
            **self.graph_stats(),
            **self.metrics.to_dict(),
        }

    # -- internals --------------------------------------------------------

    def _acquire(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _detokenize(self, token_ids: list[int]) -> str:
        """Decode the whole sequence every step, and diff against what was sent.

        Decoding token-by-token is wrong for every tokenizer LM7 serves: a BPE
        piece can be half a UTF-8 character or carry a leading space that only
        exists in context, so per-token decoding produces replacement characters
        and lost spaces. Re-decoding a few hundred ids is microseconds against a
        decode step measured in milliseconds.
        """
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))

    def _is_eos(self, token_id: int) -> bool:
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            return False
        if isinstance(eos, int):
            return token_id == eos
        return token_id in set(eos)


def _message_field(message: Any, name: str) -> str:
    value = getattr(message, name, None)
    if value is None and isinstance(message, dict):
        value = message.get(name)
    return str(value if value is not None else "")


def _load_transformers() -> Any:
    import importlib

    try:
        return importlib.import_module("transformers")
    except ImportError as exc:
        raise UnsupportedModelError(
            'Serving a Hugging Face model needs Transformers. Install it with: pip install "lm7[hf]".'
        ) from exc


__all__ = [
    "DisconnectProbe",
    "LM7ServeEngine",
    "Metrics",
    "ServeConfig",
    "TokenDelta",
    "select_token",
]
