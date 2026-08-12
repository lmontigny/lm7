"""Two compiled graphs for autoregressive decoding: prefill, then KV-cache decode.

`lm7.compile()` compiles one forward pass. That is enough to measure a model and
not enough to serve one, because generation is two different workloads sharing a
set of weights:

    prefill   prompt tokens          -> next-token logits + a filled KV cache
    decode    one token + that cache -> next-token logits + one more cache entry

Their shapes differ by three orders of magnitude and only one of them repeats, so
compiling them together means recompiling on every prompt length, or specializing
the step that runs a thousand times for a shape it sees once. This module keeps
them apart: one graph per phase, one static cache allocated once on the target
device, and a count of every Dynamo frame, graph break and recompilation that
happened while doing it.

    runner = lm7.compile_generation(model, target="nvidia", max_sequence_length=8192)
    state = runner.prefill(input_ids)
    token, state = runner.decode(state.next_token, state)

LM7 owns the compile boundary, the cache lifetime and the counting. It does not
own the cache implementation or the attention kernels: the cache is Transformers'
``StaticCache`` and the model is whatever causal LM was passed in. See
docs/kv-cache-decode.md.
"""

from __future__ import annotations

import importlib
import inspect
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .api import compile as compile_module
from .detection import inference_context, resolve_target, synchronize, torch_device
from .errors import BackendUnavailableError, UnsupportedModelError
from .targets import TargetSpec

# What a model has to accept before this path can drive it. `past_key_values` is
# the cache itself and `cache_position` is where in that cache the incoming
# tokens belong -- without the second, a decode step cannot say which slot it is
# writing and the cache has to work it out from its own length, which is exactly
# the Python-side state that makes a graph recompile.
_REQUIRED_FORWARD_ARGUMENTS = ("past_key_values", "cache_position")

# Only the last position's logits are ever read here, and computing the rest is
# not free: an 8192-token prompt through a 128k vocabulary materializes 2 GiB of
# logits per batch element that are then discarded. Models that accept this
# argument run the vocabulary projection on one position instead.
_LOGITS_TO_KEEP = "logits_to_keep"

_BACKENDS = frozenset({"auto", "eager", "inductor"})

# What `backend="auto"` is allowed to land on. The set above validates the
# *string* a caller passes; this one validates what "auto" turns into, and the
# two are not the same check. A decode graph mutates a KV cache in place, and a
# backend that compiles by executing the artifact it just built spends cache
# slots nobody asked for -- only the Inductor backend implements the
# `warmup: False` option that declines that call. So a target whose
# highest-priority backend is something else (openxla on `tpu`, the Tenstorrent
# backend, OpenVINO on `intel:npu`) is refused here rather than handed a
# stateful graph no one has ever run through it.
_PLANNABLE_BACKENDS = frozenset({"eager", "inductor"})


def _counter_value(group: str, key: str) -> int:
    try:
        from torch._dynamo.utils import counters

        return int(counters[group][key])
    except Exception:  # noqa: BLE001 - a private counter that moved costs the label only
        return 0


def _graph_breaks() -> int:
    try:
        from torch._dynamo.utils import counters

        return sum(int(value) for value in counters["graph_break"].values())
    except Exception:  # noqa: BLE001
        return 0


def _recompiles() -> int:
    """Compiles beyond the first, summed over every code object Dynamo has seen.

    Dynamo counts compiles per frame, so "how many times did something recompile"
    is total compiles minus the number of distinct frames -- a quantity whose
    *difference* across a phase is well defined even though its absolute value is
    process-wide. A decode loop reporting anything but zero here is specializing
    on something that changes per token.
    """
    try:
        from torch._dynamo.convert_frame import FRAME_COMPILE_COUNTER

        return sum(FRAME_COMPILE_COUNTER.values()) - len(FRAME_COMPILE_COUNTER)
    except Exception:  # noqa: BLE001
        return 0


@dataclass(frozen=True)
class GraphCounters:
    """Dynamo and Inductor counters, as an absolute snapshot or as a delta.

    Every field is process-wide, so a delta attributes work to a phase only while
    nothing else in the process is compiling. That holds for a single-threaded
    benchmark and is worth stating rather than assuming.
    """

    frames: int
    unique_graphs: int
    graph_breaks: int
    recompiles: int
    cudagraph_skips: int

    def __sub__(self, other: GraphCounters) -> GraphCounters:
        return GraphCounters(
            frames=self.frames - other.frames,
            unique_graphs=self.unique_graphs - other.unique_graphs,
            graph_breaks=self.graph_breaks - other.graph_breaks,
            recompiles=self.recompiles - other.recompiles,
            cudagraph_skips=self.cudagraph_skips - other.cudagraph_skips,
        )

    def __add__(self, other: GraphCounters) -> GraphCounters:
        return GraphCounters(
            frames=self.frames + other.frames,
            unique_graphs=self.unique_graphs + other.unique_graphs,
            graph_breaks=self.graph_breaks + other.graph_breaks,
            recompiles=self.recompiles + other.recompiles,
            cudagraph_skips=self.cudagraph_skips + other.cudagraph_skips,
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


ZERO_COUNTERS = GraphCounters(0, 0, 0, 0, 0)


def graph_counters() -> GraphCounters:
    """Snapshot Dynamo's process-wide compilation counters."""
    from .backends.inductor import cudagraph_skips

    return GraphCounters(
        frames=_counter_value("frames", "total"),
        unique_graphs=_counter_value("stats", "unique_graphs"),
        graph_breaks=_graph_breaks(),
        recompiles=_recompiles(),
        cudagraph_skips=cudagraph_skips(),
    )


@dataclass(frozen=True)
class GenerationState:
    """Where a sequence has got to, and the cache that got it there.

    ``past_key_values`` is a handle, not a value: the cache buffers are mutated in
    place by every step, so an older state does not describe an earlier cache.
    ``sequence_length`` is how many slots are filled, which is also the cache
    position the next token will be written to.
    """

    past_key_values: Any
    sequence_length: int
    next_token: torch.Tensor
    logits: torch.Tensor
    # None when the batch is unpadded, which is the common case and the one the
    # benchmarks measure. Otherwise a (batch, max_sequence_length) mask, mutated
    # in place per step exactly as the cache is.
    attention_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class GenerationResult:
    tokens: torch.Tensor
    state: GenerationState
    prefill_ms: float
    decode_ms: float
    decode_steps: int

    @property
    def ms_per_decoded_token(self) -> float:
        return self.decode_ms / self.decode_steps if self.decode_steps else 0.0


class _PrefillGraph(torch.nn.Module):
    """The prompt pass: many tokens in, one position of logits out.

    Deliberately a separate class from ``_DecodeGraph`` with an identical body.
    Dynamo caches compiled code per *code object*, so one shared wrapper would put
    both phases in one cache entry, report the decode compile as a recompilation
    of the prefill, and -- once a second prompt length shows up -- make the graph
    that runs a thousand times share an eviction limit with the one that runs
    once. Two classes make the split real and make the counters mean what they say.
    """

    def __init__(self, model: torch.nn.Module, logits_to_keep: bool) -> None:
        super().__init__()
        self.model = model
        self.logits_to_keep = logits_to_keep

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Any,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        extra = {_LOGITS_TO_KEEP: 1} if self.logits_to_keep else {}
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
            **extra,
        ).logits


class _DecodeGraph(torch.nn.Module):
    """The steady-state pass: one token in, one position of logits out.

    See ``_PrefillGraph`` for why this is a copy rather than the same class.
    """

    def __init__(self, model: torch.nn.Module, logits_to_keep: bool) -> None:
        super().__init__()
        self.model = model
        self.logits_to_keep = logits_to_keep

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Any,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        extra = {_LOGITS_TO_KEEP: 1} if self.logits_to_keep else {}
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
            **extra,
        ).logits


class GenerationRunner:
    """A compiled prefill graph, a compiled decode graph, and one static cache.

    Built by :func:`compile_generation`, which documents the arguments.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target: TargetSpec,
        *,
        backend: str,
        compile_mode: str | None,
        compile_prefill: bool,
        max_batch_size: int,
        max_sequence_length: int,
    ) -> None:
        self.model = model
        self.target = target
        self.backend = backend
        self.compile_mode = compile_mode
        self.compile_prefill = compile_prefill
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.device = torch_device(target)
        self.dtype = _model_dtype(model)

        named, catch_all = _forward_arguments(model)
        missing = [
            name for name in _REQUIRED_FORWARD_ARGUMENTS if name not in named and not catch_all
        ]
        if missing:
            raise UnsupportedModelError(
                f"compile_generation needs a model whose forward accepts {', '.join(missing)}. "
                "That is the Hugging Face causal-LM contract; a model that cannot be handed a "
                "KV cache has no decode step to compile."
            )
        # Deliberately stricter than the check above. The required arguments are
        # allowed to arrive through `**kwargs`, because that is how Transformers
        # routes `cache_position`; this one is an optimization, so it is sent only
        # when the signature names it and cannot be quietly swallowed.
        self._logits_to_keep = _LOGITS_TO_KEEP in named

        # The weights move here, not on the first call, and that is load-bearing
        # rather than tidy. LM7's backends move the model as part of compiling,
        # and compiling happens inside the first call -- which this runner makes
        # under `inference_mode`, where `Module.to` cannot swap a parameter that
        # is a tensor subclass:
        #
        #     RuntimeError: _apply(): Couldn't swap Linear.weight
        #
        # which is every TorchAO-quantized linear, so every FP8 model. Moving
        # first and telling the backend transfers are explicit means nothing tries
        # to move anything again from inside that context.
        model.to(self.device)

        # The cache is allocated here rather than on the first call because the
        # runner's whole promise is that the buffers exist before decoding starts
        # and do not move afterwards. That means the device has to be known now,
        # which is why the target resolves eagerly rather than lazily as it does
        # for `lm7.compile`.
        self.past_key_values = _allocate_static_cache(
            model,
            max_batch_size=max_batch_size,
            max_sequence_length=max_sequence_length,
            dtype=self.dtype,
            device=self.device,
        )

        # `warmup: False` is the option that makes a stateful graph compilable at
        # all. LM7's Inductor backend otherwise compiles by *calling* the artifact,
        # and each execution of these graphs advances the cache by the tokens it
        # was given -- so an unasked-for warmup burns cache slots, and a prompt of
        # 512 tokens against a 533-slot cache indexes past the end of the buffer
        # and dies in a device-side assert rather than anywhere readable. These
        # graphs compile on their first real call instead.
        options: dict[str, Any] = {"warmup": False}
        if compile_mode:
            options["compile_mode"] = compile_mode
        self._prefill_graph = compile_module(
            _PrefillGraph(model, self._logits_to_keep).eval(),
            target=target,
            backend=backend if compile_prefill else "eager",
            transfers="explicit",
            fallback="error",
            cache=False,
            options=options if compile_prefill else None,
        )
        self._decode_graph = compile_module(
            _DecodeGraph(model, self._logits_to_keep).eval(),
            target=target,
            backend=backend,
            transfers="explicit",
            fallback="error",
            cache=False,
            options=options,
        )

        # One reusable position tensor for the decode step. Building
        # `torch.tensor([n], device="cuda")` every token would put a host-to-device
        # copy on the critical path of a loop whose entire point is that nothing
        # but the kernels is on it; `fill_` enqueues instead.
        self._decode_position = torch.zeros(1, dtype=torch.long, device=self.device)

        self._attention_mask: torch.Tensor | None = None
        self.prefill_compile = ZERO_COUNTERS
        self.decode_compile = ZERO_COUNTERS
        self.steady = ZERO_COUNTERS
        self.compiled_prefill_lengths: list[int] = []
        self._decode_compiled = False

    # -- introspection ----------------------------------------------------

    @property
    def counters(self) -> dict[str, dict[str, int]]:
        """Compilation counters per phase, as deltas.

        ``prefill`` and ``decode`` cover the first call at each shape, which is
        where compilation happens, summed over every shape that needed one.
        ``steady`` accumulates every call after that, and is the number this path
        exists to make checkable: anything nonzero in ``steady["frames"]`` means a
        token triggered a compile.
        """
        return {
            "prefill": self.prefill_compile.to_dict(),
            "decode": self.decode_compile.to_dict(),
            "steady": self.steady.to_dict(),
        }

    @property
    def cudagraphs(self) -> dict[str, dict[str, Any]]:
        """What each graph asked Inductor for, and whether it got it.

        Reported from what LM7 observed rather than from the compile artifact,
        because these graphs are compiled without a backend warmup and there is
        nothing to observe until they have run. Skips are counted process-wide, so
        both phases' compiles and every steady call contribute — capture that is
        refused on a later replay still shows up here.
        """
        from .backends.inductor import cudagraphs_requested

        requested = cudagraphs_requested(self.compile_mode, {})
        skips = self.prefill_compile.cudagraph_skips + self.decode_compile.cudagraph_skips
        skips += self.steady.cudagraph_skips
        return {
            "prefill": {
                **_artifact_metadata(self._prefill_graph),
                "cudagraphs": requested and self.compile_prefill,
                "cudagraph_skips": skips,
                "cudagraphs_active": requested and self.compile_prefill and skips == 0,
            },
            "decode": {
                **_artifact_metadata(self._decode_graph),
                "cudagraphs": requested,
                "cudagraph_skips": skips,
                "cudagraphs_active": requested and skips == 0,
            },
        }

    @property
    def cache_bytes(self) -> int:
        """Bytes of KV cache allocated on the device, whatever the sequence uses."""
        return _cache_bytes(self.past_key_values)

    def __repr__(self) -> str:
        return (
            f"GenerationRunner(target={self.target}, backend={self.backend}, "
            f"compile_mode={self.compile_mode!r}, max_batch_size={self.max_batch_size}, "
            f"max_sequence_length={self.max_sequence_length})"
        )

    # -- the two phases ---------------------------------------------------

    def reset(self) -> None:
        """Zero the cache for a new sequence, reallocating and recompiling nothing."""
        self.past_key_values.reset()

    @property
    def cache_sequence_length(self) -> int:
        """How many slots the cache itself believes are filled.

        Worth having as a public number rather than an internal assertion, because
        it is the one that decides where the next token is written -- see
        ``decode``. If it ever disagrees with ``state.sequence_length``, the cache
        and the positions have desynchronized and every subsequent token is wrong.
        """
        return int(self.past_key_values.get_seq_length())

    def prefill(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> GenerationState:
        """Run the prompt through the model, filling the cache from position zero.

        ``attention_mask`` is the prompt's own ``(batch, prompt)`` mask, as a
        tokenizer produces it for a left-padded batch. Pass it here and nowhere
        else: the runner widens it to cache length and extends it by one slot per
        decoded token, which is what the decode step actually needs. See
        ``_cache_length_mask``.
        """
        input_ids = self._check_prompt(input_ids)
        length = int(input_ids.shape[-1])
        cache_position = torch.arange(length, device=self.device)
        self._attention_mask = self._cache_length_mask(attention_mask, input_ids.shape)
        # A prompt pass is compiled per length. That is the cost this split accepts
        # in exchange for a decode graph compiled once, and it is recorded rather
        # than hidden: a workload with many prompt lengths pays it repeatedly.
        compiling = length not in self.compiled_prefill_lengths
        self.reset()
        logits = self._call(
            self._prefill_graph,
            input_ids,
            cache_position,
            self._attention_mask,
            "prefill",
            compiling,
        )
        if compiling:
            self.compiled_prefill_lengths.append(length)
        return self._state(logits, length)

    def decode(
        self, token: torch.Tensor, state: GenerationState
    ) -> tuple[torch.Tensor, GenerationState]:
        """Advance one token, reusing the compiled decode graph and the cache."""
        if state.sequence_length >= self.max_sequence_length:
            raise ValueError(
                f"The cache holds {self.max_sequence_length} tokens and is full at "
                f"{state.sequence_length}. A static cache cannot grow; build the runner with a "
                "larger max_sequence_length."
            )
        token = token.to(self.device, non_blocking=True).reshape(self.max_batch_size, 1)
        # This tensor is the *attention* position -- it decides the causal mask and
        # the rotary embedding. It is not where the key and value land: Transformers'
        # static layer writes at its own `cumulative_length` and advances it by the
        # number of tokens it was given, so the two agree only as long as every
        # execution of the graph is one the caller asked for. That is what the
        # `warmup: False` compile option protects.
        self._decode_position.fill_(state.sequence_length)
        if self._attention_mask is not None:
            # The slot this token is about to occupy becomes attendable. Written in
            # place, so the mask keeps its shape and the graph keeps its guards.
            self._attention_mask[:, state.sequence_length] = 1
        logits = self._call(
            self._decode_graph,
            token,
            self._decode_position,
            self._attention_mask,
            "decode",
            not self._decode_compiled,
        )
        self._decode_compiled = True
        next_state = self._state(logits, state.sequence_length + 1)
        return next_state.next_token, next_state

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        attention_mask: torch.Tensor | None = None,
    ) -> GenerationResult:
        """Greedy generation over the two graphs, timed by phase.

        The first token comes out of prefill, so ``max_new_tokens=1`` never enters
        the decode graph.
        """
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")
        synchronize(self.target)
        started = time.perf_counter()
        state = self.prefill(input_ids, attention_mask)
        synchronize(self.target)
        prefill_ms = (time.perf_counter() - started) * 1000.0

        tokens = [state.next_token]
        steps = max_new_tokens - 1
        started = time.perf_counter()
        for _ in range(steps):
            token, state = self.decode(state.next_token, state)
            tokens.append(token)
        synchronize(self.target)
        decode_ms = (time.perf_counter() - started) * 1000.0
        return GenerationResult(
            tokens=torch.cat(tokens, dim=-1),
            state=state,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            decode_steps=steps,
        )

    # -- internals --------------------------------------------------------

    def _call(
        self,
        graph: Any,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor | None,
        phase: str,
        compiling: bool,
    ) -> torch.Tensor:
        """Run one graph once, and charge whatever it compiled to the right phase.

        Once, exactly. These graphs are built with ``warmup=False`` precisely so
        that a call is a call: each execution advances the KV cache by the tokens
        it was handed, so a backend that compiles by executing would consume slots
        the caller never asked for and desynchronize the positions from the cache.
        """
        before = graph_counters()
        with inference_context(self.target):
            logits = graph(input_ids, cache_position, self.past_key_values, attention_mask)
            # Cloned inside the inference context, and on purpose. Under
            # `compile_mode="reduce-overhead"` these logits live in a CUDA Graph's
            # static output buffer, which the next replay overwrites -- so a state
            # holding the raw tensor would silently start describing a later step.
            # One (batch, 1, vocab) copy per token is the price of the state
            # meaning what it says.
            logits = logits.clone()
        delta = graph_counters() - before
        if not compiling:
            self.steady = self.steady + delta
        elif phase == "prefill":
            self.prefill_compile = self.prefill_compile + delta
        else:
            self.decode_compile = self.decode_compile + delta
        return logits

    def _state(self, logits: torch.Tensor, sequence_length: int) -> GenerationState:
        return GenerationState(
            past_key_values=self.past_key_values,
            sequence_length=sequence_length,
            next_token=logits[:, -1].argmax(dim=-1, keepdim=True),
            logits=logits,
            attention_mask=self._attention_mask,
        )

    def _cache_length_mask(
        self, attention_mask: torch.Tensor | None, prompt_shape: torch.Size
    ) -> torch.Tensor | None:
        """Widen a prompt-length mask to a cache-length one, or return None.

        Transformers builds the decode step's mask against the *whole* static
        cache, so a mask that covers only the prompt describes the wrong thing the
        moment a token is decoded. It describes it fluently, too: measured on
        SmolLM2-135M with a left-padded batch of two, 24 greedy tokens, against
        ``model.generate`` as the reference —

            no mask                the padded row diverges immediately
            prompt-length mask     both rows diverge, into repeated newlines
            cache-length mask      both rows match exactly

        — and none of the three raises. A cache-length mask is also fixed shape,
        so extending it by a slot per token is a write rather than a new graph.

        None means "every position is real", which is correct for an unpadded
        batch and is what the benchmarks measure.
        """
        if attention_mask is None:
            return None
        if tuple(attention_mask.shape) != tuple(prompt_shape):
            raise ValueError(
                f"attention_mask must be the prompt's own {tuple(prompt_shape)} mask; got "
                f"{tuple(attention_mask.shape)}. The runner widens it to cache length itself."
            )
        mask = torch.zeros(
            self.max_batch_size,
            self.max_sequence_length,
            dtype=attention_mask.dtype,
            device=self.device,
        )
        mask[:, : prompt_shape[-1]] = attention_mask.to(self.device)
        return mask

    def _check_prompt(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be (batch, sequence); got shape {tuple(input_ids.shape)}."
            )
        batch, length = input_ids.shape
        if batch != self.max_batch_size:
            raise ValueError(
                f"This runner's cache is allocated for batch {self.max_batch_size} and was given "
                f"batch {batch}. A static cache is buffers, not a bound: a smaller batch is also a "
                "new graph shape, and recompiling the decode step is the one thing this path "
                "exists to avoid."
            )
        if length >= self.max_sequence_length:
            raise ValueError(
                f"The prompt is {length} tokens and the cache holds {self.max_sequence_length}, "
                "leaving no room to decode. Build the runner with a larger max_sequence_length."
            )
        return input_ids.to(self.device, non_blocking=True)


def compile_generation(
    model: torch.nn.Module,
    target: str | TargetSpec | None = None,
    *,
    backend: str = "auto",
    compile_mode: str | None = None,
    compile_prefill: bool = True,
    max_batch_size: int = 1,
    max_sequence_length: int = 2048,
) -> GenerationRunner:
    """Compile a causal LM into separate prefill and KV-cache decode graphs.

    ``model`` must accept ``past_key_values`` and ``cache_position`` -- the Hugging
    Face causal-LM contract -- and is used exactly as given, so a model quantized
    before this call decodes quantized.

    ``backend`` accepts ``"auto"``, ``"eager"`` and ``"inductor"``, and ``"auto"``
    is checked against what it actually selects rather than only against the
    string: a target whose highest-priority backend is neither of the two is
    refused here, because that backend has never been handed a graph that writes
    into a KV cache. See ``_PLANNABLE_BACKENDS``.

    ``compile_mode`` is Inductor's preset, and is how CUDA Graphs are requested:
    ``"reduce-overhead"`` asks for them, ``None`` does not. Requesting is not
    getting, and ``runner.cudagraphs`` reports whether capture was refused.

    ``compile_prefill=False`` leaves the prompt pass in eager and compiles only the
    decode step, which is the boundary Transformers' own compiled generation draws
    -- see docs/huggingface-generation.md. It is an option here rather than the
    default because a served workload with one prompt length does not pay the
    recompilation cost that motivates it.

    ``max_batch_size`` and ``max_sequence_length`` size the static cache, which is
    allocated on the target device immediately and never grows. Prompts must
    arrive at exactly ``max_batch_size``; see ``GenerationRunner.prefill``.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if backend not in _BACKENDS:
        choices = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"backend must be one of: {choices}; got {backend!r}.")
    if backend == "eager" and compile_mode is not None:
        raise ValueError(
            "compile_mode is an Inductor preset and has no meaning for backend='eager'."
        )
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be at least 1.")
    if max_sequence_length < 2:
        raise ValueError(
            "max_sequence_length must be at least 2: one slot for a prompt token and one for a "
            "decoded token, or there is no decode step to compile."
        )
    resolved = resolve_target(target if target is not None else "auto")
    if backend == "auto":
        planned = _planned_backend(resolved)
        if planned not in _PLANNABLE_BACKENDS:
            raise BackendUnavailableError(
                f"backend='auto' selects {planned!r} for target {resolved}, which this path "
                "does not support. A decode graph mutates a KV cache in place, and a backend "
                "that compiles by executing the artifact it just built spends cache slots the "
                "caller never asked for; only the Inductor backend implements the "
                "`warmup: False` option that declines that call. Pass backend='eager' to run "
                f"{resolved} uncompiled."
            )
    return GenerationRunner(
        model.eval(),
        resolved,
        backend=backend,
        compile_mode=compile_mode,
        compile_prefill=compile_prefill,
        max_batch_size=max_batch_size,
        max_sequence_length=max_sequence_length,
    )


def _planned_backend(target: TargetSpec) -> str:
    """What ``backend="auto"`` resolves to for ``target``.

    Asked here rather than left to the compile call, because ``CompiledModule``
    plans lazily: without this the answer arrives on the first token, by which
    point the weights have moved and the cache has been allocated. The request is
    model-less for the same reason ``lm7.explain`` builds one -- every backend's
    ``supports`` answers from the target alone.
    """
    from .backends import registry
    from .backends.base import CompileRequest
    from .planner import plan

    request = CompileRequest(torch.nn.Identity(), target, "lazy", "explicit", "error", {})
    _, planned = plan(request, "auto", registry)
    return planned.selected


def _forward_arguments(model: torch.nn.Module) -> tuple[frozenset[str], bool]:
    """The names ``forward`` declares, and whether it also takes ``**kwargs``.

    Both halves are needed. A Transformers causal LM names ``past_key_values`` but
    routes ``cache_position`` through ``**kwargs``, so a names-only check rejects
    every model this path is for.
    """
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return frozenset(), False
    catch_all = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    return frozenset(parameters), catch_all


def _model_dtype(model: torch.nn.Module) -> torch.dtype:
    dtype = getattr(model, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    return next((parameter.dtype for parameter in model.parameters()), torch.float32)


def _allocate_static_cache(
    model: torch.nn.Module,
    *,
    max_batch_size: int,
    max_sequence_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Any:
    """A Transformers ``StaticCache``, fully materialized on ``device``.

    Materialized rather than left lazy on purpose. A lazily initialized cache
    allocates its buffers inside the first traced call, which puts the allocation
    in the graph and makes the second sequence a different graph.
    """
    try:
        cache_utils = importlib.import_module("transformers.cache_utils")
    except ImportError as exc:
        raise UnsupportedModelError(
            "compile_generation uses Transformers' StaticCache for the KV cache. "
            'Install it with: pip install "lm7[hf]".'
        ) from exc

    config = getattr(model, "config", None)
    if config is None:
        raise UnsupportedModelError(
            "compile_generation needs model.config to size the KV cache; this module has none."
        )
    if hasattr(config, "get_text_config"):
        config = config.get_text_config(decoder=True)
    try:
        cache = cache_utils.StaticCache(config=config, max_cache_len=max_sequence_length)
        num_heads, head_dim = _head_shapes(model, config)
        cache.early_initialization(max_batch_size, num_heads, head_dim, dtype, device)
    except Exception as exc:
        raise UnsupportedModelError(
            f"Could not allocate a static KV cache for this model: {exc}."
        ) from exc
    return cache


def _head_shapes(model: torch.nn.Module, config: Any) -> tuple[Any, Any]:
    """The ``(num_heads, head_dim)`` the cache buffers need.

    Transformers already answers this for its own models and the answer is not a
    one-liner -- a hybrid architecture carries a different head count per layer
    type, so both values can be per-layer lists. Its helper is asked first, and
    the two-attribute form is only the fallback.
    """
    shape = getattr(model, "_get_static_cache_init_shape", None)
    if callable(shape):
        num_heads, head_dim = shape()
        return num_heads, head_dim
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    num_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    return num_heads, head_dim


def _cache_bytes(cache: Any) -> int:
    total = 0
    for layer in getattr(cache, "layers", ()):
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if isinstance(tensor, torch.Tensor):
                total += tensor.numel() * tensor.element_size()
    return total


def _artifact_metadata(compiled: Any) -> dict[str, Any]:
    artifact = getattr(compiled, "artifact", None)
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    metadata["backend"] = getattr(compiled, "selected_backend", None)
    return metadata


__all__ = [
    "GenerationResult",
    "GenerationRunner",
    "GenerationState",
    "GraphCounters",
    "compile_generation",
    "graph_counters",
]
