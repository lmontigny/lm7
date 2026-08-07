from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import lm7
from lm7.errors import UnsupportedModelError
from lm7.generation import GraphCounters, _cache_bytes, _forward_arguments, _head_shapes

VOCAB = 32


class FakeStaticCache:
    """The one behaviour of Transformers' ``StaticCache`` this path turns on.

    ``StaticLayer.update`` writes at the cache's own ``cumulative_length`` and
    advances it once per call — it does *not* write at the ``cache_position`` it
    was handed, which only steers the causal mask and the rotary embedding. So a
    graph executed one extra time consumes an extra slot while the caller's
    positions stay put, and every token after that is computed against a cache
    that has silently shifted. This fake reproduces exactly that and nothing else.
    """

    def __init__(self, batch_size: int, max_cache_len: int, device=None) -> None:
        keys = torch.zeros(batch_size, 1, max_cache_len, 2, device=device)
        self.layers = [SimpleNamespace(keys=keys, values=keys.clone())]
        self.max_cache_len = max_cache_len
        self.cumulative_length = 0

    def write(self, input_ids: torch.Tensor) -> None:
        start = self.cumulative_length
        for offset in range(input_ids.shape[-1]):
            self.layers[0].keys[:, 0, start + offset, 0] = input_ids[:, offset].float()
        self.cumulative_length = start + int(input_ids.shape[-1])

    def reset(self) -> None:
        for layer in self.layers:
            layer.keys.zero_()
            layer.values.zero_()
        self.cumulative_length = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.cumulative_length


class FakeCausalLM(torch.nn.Module):
    """A causal LM small enough to test with, honest about its cache contract.

    The next token depends on both what the cache holds and where the caller said
    it was, so a cache that has advanced further than the positions produces
    different tokens rather than merely a different tensor somewhere internal.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            hidden_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=4
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        del attention_mask, use_cache, kwargs
        past_key_values.write(input_ids)
        stored = past_key_values.layers[0].keys[:, 0, :, 0].sum(dim=-1).long()
        position = int(cache_position[-1])
        chosen = (stored + position) % VOCAB
        logits = torch.zeros(input_ids.shape[0], 1, VOCAB)
        logits[torch.arange(input_ids.shape[0]), 0, chosen] = 1.0
        return SimpleNamespace(logits=logits)


class NoCacheModel(torch.nn.Module):
    def forward(self, input_ids):  # no past_key_values, no cache_position, no **kwargs
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, VOCAB))


@pytest.fixture
def fake_cache(monkeypatch):
    """Give the runner a cache without needing Transformers installed."""

    def allocate(model, *, max_batch_size, max_sequence_length, dtype, device):
        del model, dtype
        return FakeStaticCache(max_batch_size, max_sequence_length, device=device)

    monkeypatch.setattr(lm7.generation, "_allocate_static_cache", allocate)


def build(monkeypatch=None, **kwargs):
    options = {
        "target": "cpu",
        "backend": "eager",
        "max_batch_size": 1,
        "max_sequence_length": 16,
        **kwargs,
    }
    return lm7.compile_generation(FakeCausalLM(), **options)


def reference_loop(prompt: torch.Tensor, steps: int, max_cache_len: int) -> list[int]:
    """The same generation with no compiler and no runner in the way."""
    model = FakeCausalLM().eval()
    cache = FakeStaticCache(prompt.shape[0], max_cache_len)
    with torch.inference_mode():
        logits = model(prompt, past_key_values=cache, cache_position=torch.arange(prompt.shape[-1]))
        token = logits.logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = [int(token)]
        position = int(prompt.shape[-1])
        for _ in range(steps):
            logits = model(token, past_key_values=cache, cache_position=torch.tensor([position]))
            token = logits.logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens.append(int(token))
            position += 1
    return tokens


# -- construction ---------------------------------------------------------


def test_rejects_a_model_that_cannot_be_handed_a_cache(fake_cache):
    with pytest.raises(UnsupportedModelError, match="past_key_values"):
        lm7.compile_generation(NoCacheModel(), target="cpu", backend="eager")


def test_rejects_a_non_module():
    with pytest.raises(TypeError, match="torch.nn.Module"):
        lm7.compile_generation(object(), target="cpu")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"backend": "tensorrt"}, "backend must be one of"),
        ({"backend": "eager", "compile_mode": "reduce-overhead"}, "Inductor preset"),
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"max_sequence_length": 1}, "max_sequence_length"),
    ),
)
def test_rejects_impossible_configurations(fake_cache, kwargs, message):
    with pytest.raises(ValueError, match=message):
        build(**kwargs)


def test_cache_is_allocated_before_the_first_call(fake_cache):
    runner = build()
    assert runner.cache_sequence_length == 0
    # 1 layer x batch 1 x 1 head x 16 slots x 2 dims x 4 bytes, keys and values.
    assert runner.cache_bytes == 2 * 16 * 2 * 4


# -- the two phases -------------------------------------------------------


def test_prefill_then_decode_matches_an_uncompiled_loop(fake_cache):
    runner = build()
    prompt = torch.tensor([[3, 1, 4]])
    expected = reference_loop(prompt, steps=4, max_cache_len=16)

    state = runner.prefill(prompt)
    tokens = [int(state.next_token)]
    for _ in range(4):
        token, state = runner.decode(state.next_token, state)
        tokens.append(int(token))
    assert tokens == expected
    assert state.sequence_length == 3 + 4
    assert runner.cache_sequence_length == state.sequence_length


def test_the_inductor_path_advances_the_cache_exactly_once_per_call(fake_cache, monkeypatch):
    """The regression that ``warmup=False`` exists for.

    LM7's Inductor backend otherwise compiles by *calling* the artifact, so the
    first call through a graph executes the model twice — which for a graph that
    writes into a KV cache consumes two slots for one token and desynchronizes
    every position after it. Faking ``torch.compile`` to the identity keeps that
    backend warmup and removes the several seconds of real compilation, which is
    what makes this checkable in the portable suite instead of only on a GPU.
    """
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    runner = build(backend="inductor")
    prompt = torch.tensor([[3, 1, 4]])
    expected = reference_loop(prompt, steps=4, max_cache_len=16)

    state = runner.prefill(prompt)
    assert runner.cache_sequence_length == 3, "the prompt was written more than once"
    tokens = [int(state.next_token)]
    for _ in range(4):
        token, state = runner.decode(state.next_token, state)
        tokens.append(int(token))
    assert tokens == expected
    assert runner.cache_sequence_length == state.sequence_length


def test_graphs_are_compiled_without_a_backend_warmup(fake_cache, monkeypatch):
    requests = []

    def record(model, **kwargs):
        requests.append(kwargs)
        return model

    monkeypatch.setattr(torch, "compile", record)
    runner = build(backend="inductor", compile_mode="reduce-overhead")
    runner.prefill(torch.tensor([[3, 1, 4]]))
    assert runner._prefill_graph.options == {"warmup": False, "compile_mode": "reduce-overhead"}
    # `warmup` is the backend's own control and must not reach torch.compile as an
    # Inductor config key, which would raise on an unknown option.
    assert requests
    for kwargs in requests:
        assert "warmup" not in (kwargs.get("options") or {})


def test_generate_returns_the_prefill_token_first(fake_cache):
    runner = build()
    prompt = torch.tensor([[3, 1, 4]])
    expected = reference_loop(prompt, steps=4, max_cache_len=16)

    result = runner.generate(prompt, max_new_tokens=5)
    assert result.tokens.shape == (1, 5)
    assert result.tokens[0].tolist() == expected
    assert result.decode_steps == 4
    assert result.prefill_ms >= 0.0
    assert result.ms_per_decoded_token == pytest.approx(result.decode_ms / 4)


def test_generate_of_one_token_never_decodes(fake_cache):
    runner = build()
    result = runner.generate(torch.tensor([[3, 1, 4]]), max_new_tokens=1)
    assert result.decode_steps == 0
    assert result.ms_per_decoded_token == 0.0


def test_a_second_sequence_reuses_the_cache_and_repeats_itself(fake_cache):
    runner = build()
    prompt = torch.tensor([[3, 1, 4]])
    first = runner.generate(prompt, max_new_tokens=5)
    second = runner.generate(prompt, max_new_tokens=5)
    assert torch.equal(first.tokens, second.tokens)


def test_reset_empties_the_cache(fake_cache):
    runner = build()
    runner.prefill(torch.tensor([[3, 1, 4]]))
    assert runner.cache_sequence_length == 3
    runner.reset()
    assert runner.cache_sequence_length == 0


def test_batch_is_fixed_by_the_allocated_cache(fake_cache):
    runner = build(max_batch_size=2)
    with pytest.raises(ValueError, match="allocated for batch 2"):
        runner.prefill(torch.tensor([[3, 1, 4]]))
    state = runner.prefill(torch.tensor([[3, 1, 4], [2, 7, 5]]))
    assert state.next_token.shape == (2, 1)


def test_prompt_must_be_two_dimensional(fake_cache):
    runner = build()
    with pytest.raises(ValueError, match=r"\(batch, sequence\)"):
        runner.prefill(torch.tensor([3, 1, 4]))


def test_prompt_must_leave_room_to_decode(fake_cache):
    runner = build(max_sequence_length=4)
    with pytest.raises(ValueError, match="cache holds 4"):
        runner.prefill(torch.zeros(1, 4, dtype=torch.long))


def test_decode_refuses_to_run_past_the_cache(fake_cache):
    runner = build(max_sequence_length=4)
    state = runner.prefill(torch.zeros(1, 3, dtype=torch.long))
    _, state = runner.decode(state.next_token, state)
    assert state.sequence_length == 4
    with pytest.raises(ValueError, match="full at 4"):
        runner.decode(state.next_token, state)


# -- counters -------------------------------------------------------------


def test_steady_state_compiles_nothing(fake_cache):
    runner = build()
    runner.generate(torch.tensor([[3, 1, 4]]), max_new_tokens=8)
    steady = runner.counters["steady"]
    assert steady["frames"] == 0
    assert steady["graph_breaks"] == 0
    assert steady["recompiles"] == 0


def test_prefill_is_compiled_once_per_prompt_length(fake_cache):
    runner = build()
    runner.prefill(torch.zeros(1, 3, dtype=torch.long))
    runner.prefill(torch.zeros(1, 3, dtype=torch.long))
    assert runner.compiled_prefill_lengths == [3]
    runner.prefill(torch.zeros(1, 5, dtype=torch.long))
    assert runner.compiled_prefill_lengths == [3, 5]


def test_counters_report_every_phase(fake_cache):
    runner = build()
    runner.generate(torch.tensor([[3, 1, 4]]), max_new_tokens=4)
    assert set(runner.counters) == {"prefill", "decode", "steady"}
    for phase in runner.counters.values():
        assert set(phase) == {
            "frames",
            "unique_graphs",
            "graph_breaks",
            "recompiles",
            "cudagraph_skips",
        }


def test_graph_counters_are_a_difference_and_a_sum():
    before = GraphCounters(1, 2, 3, 4, 5)
    after = GraphCounters(4, 4, 4, 4, 4)
    assert (after - before).to_dict() == {
        "frames": 3,
        "unique_graphs": 2,
        "graph_breaks": 1,
        "recompiles": 0,
        "cudagraph_skips": -1,
    }
    assert (before + before).frames == 2


def test_graph_counters_snapshot_is_non_negative():
    counters = lm7.graph_counters()
    assert counters.frames >= 0
    assert counters.recompiles >= 0


def test_cudagraph_report_names_the_backend_once_a_graph_has_run(fake_cache):
    runner = build()
    # Each graph compiles on its own first call, so the decode half is unknown
    # until a token has been decoded rather than merely prefilled.
    assert runner.cudagraphs["decode"]["backend"] is None
    state = runner.prefill(torch.tensor([[3, 1, 4]]))
    assert runner.cudagraphs["prefill"]["backend"] == "eager"
    runner.decode(state.next_token, state)
    assert runner.cudagraphs["decode"]["backend"] == "eager"


def test_cudagraphs_are_not_claimed_when_no_preset_asked_for_them(fake_cache):
    runner = build()
    assert runner.cudagraphs["decode"]["cudagraphs"] is False
    assert runner.cudagraphs["decode"]["cudagraphs_active"] is False


def test_repr_names_the_configuration(fake_cache):
    assert "max_sequence_length=16" in repr(build())


# -- helpers --------------------------------------------------------------


def test_forward_arguments_reports_names_and_catch_all():
    named, catch_all = _forward_arguments(FakeCausalLM())
    assert "past_key_values" in named
    assert catch_all is True
    named, catch_all = _forward_arguments(NoCacheModel())
    assert named == frozenset({"input_ids"})
    assert catch_all is False


def test_head_shapes_prefers_the_transformers_answer():
    model = FakeCausalLM()
    assert _head_shapes(model, model.config) == (1, 4)
    model._get_static_cache_init_shape = lambda: ([2, 4], 64)
    assert _head_shapes(model, model.config) == ([2, 4], 64)


def test_head_shapes_falls_back_to_the_config():
    config = SimpleNamespace(hidden_size=64, num_attention_heads=8)
    assert _head_shapes(torch.nn.Identity(), config) == (8, 8)


def test_cache_bytes_ignores_a_cache_with_no_layers():
    assert _cache_bytes(SimpleNamespace()) == 0
