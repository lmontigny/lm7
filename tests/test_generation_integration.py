from __future__ import annotations

import os

import pytest
import torch

import lm7
from lm7.detection import resolve_target

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT = "The capital of France is"
RUN_HF_TESTS = os.environ.get("LM7_RUN_HF_TESTS") == "1"

pytestmark = [
    pytest.mark.hf,
    pytest.mark.skipif(not RUN_HF_TESTS, reason="set LM7_RUN_HF_TESTS=1"),
]


def load(dtype: torch.dtype):
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).eval()
    return tokenizer, model


def reference_tokens(model, input_ids, max_new_tokens: int) -> list[int]:
    """What ``model.generate`` produces, which is the only bar that matters here.

    A decode path that is fast and wrong is worse than no decode path, and the
    ways this one can be wrong — a cache written at the position the caller meant
    versus the position the cache itself is at, a graph executed once more than
    the caller asked for — all produce fluent text rather than an error.
    """
    with torch.inference_mode():
        generated = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return generated[0, input_ids.shape[-1] :].tolist()


# float32 deliberately, on GPU as well as CPU. Greedy decoding is token-exact
# only when the arithmetic is: measured on this model in bfloat16 on an RTX 4070
# SUPER, Transformers' *own* dynamic and static caches produce different text
# from each other by the fifth token, so "matches model.generate" is not a well
# defined assertion there. It is in float32, where every arm agrees exactly —
# runner and `model.generate`, eager and Inductor and CUDA Graphs.
# `test_bfloat16_decode_tracks_eager_in_logits` covers the narrow format with the
# quantity that survives it. See docs/kv-cache-decode.md.
EXACT_DTYPE = torch.float32


@pytest.mark.parametrize("backend", ("eager", "inductor"))
def test_runner_reproduces_model_generate(backend):
    target = resolve_target("auto")
    tokenizer, model = load(EXACT_DTYPE)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    expected = reference_tokens(model, input_ids, 12)

    runner = lm7.compile_generation(model, target=target, backend=backend, max_sequence_length=128)
    result = runner.generate(input_ids, max_new_tokens=12)
    assert result.tokens[0].tolist() == expected
    assert runner.cache_sequence_length == result.state.sequence_length


def logits_along_a_fixed_path(model, input_ids, backend, dtype, forced):
    """Decode a *given* token sequence, collecting the logits at each step.

    Forced rather than greedy, and that is the whole point. If each runner
    follows its own argmax, the moment one token differs the two are decoding
    different sentences, and comparing their logits after that measures the
    sentences rather than the arithmetic. Feeding both the same tokens keeps every
    step's inputs identical, so a difference is a difference in kernels.
    """
    del dtype
    torch._dynamo.reset()
    runner = lm7.compile_generation(
        model, target="nvidia", backend=backend, max_sequence_length=128
    )
    state = runner.prefill(input_ids)
    collected = [state.logits.float().cpu()]
    for token in forced:
        _, state = runner.decode(torch.tensor([[token]], device=runner.device), state)
        collected.append(state.logits.float().cpu())
    return collected


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable")
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    (
        # The two bounds are four orders of magnitude apart, and that gap is the
        # measurement. On an RTX 4070 SUPER along this fixed path, compiled and
        # eager logits differ by 1.3e-06 of the logit scale in float32 and by
        # 2.1e-02 in bfloat16 -- noise against something an argmax can act on.
        # This is why the token tests above demand exact equality in float32 and
        # nothing demands it in bfloat16. See docs/kv-cache-decode.md.
        (torch.float32, 1e-5),
        (torch.bfloat16, 5e-2),
    ),
)
def test_compiled_decode_tracks_eager_in_logits(dtype, tolerance):
    tokenizer, model = load(dtype)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    forced = tokenizer(" Paris is the capital city of", return_tensors="pt").input_ids[0].tolist()

    reference = logits_along_a_fixed_path(model, input_ids, "eager", dtype, forced)
    compiled = logits_along_a_fixed_path(model, input_ids, "inductor", dtype, forced)
    for step, (want, got) in enumerate(zip(reference, compiled)):
        difference = (want - got).abs().max().item()
        scale = want.abs().max().item()
        assert difference <= tolerance * scale, (
            f"step {step}: {difference:.4f} against a scale of {scale:.3f}"
        )


@pytest.mark.parametrize("backend", ("eager", "inductor"))
def test_a_left_padded_batch_reproduces_model_generate(backend):
    """The padded row is where a decode path quietly goes wrong.

    With no mask it attends to the padding; with the prompt's own mask it attends
    to the whole cache including slots nothing has written. Both keep generating
    fluent text. Only the cache-length mask the runner builds matches.
    """
    target = resolve_target("auto")
    tokenizer, model = load(EXACT_DTYPE)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = tokenizer(
        [PROMPT, "In 1969 humans first walked on the"], return_tensors="pt", padding=True
    )
    prompt_tokens = batch["input_ids"].shape[-1]
    with torch.inference_mode():
        expected = model.generate(**batch, max_new_tokens=24, do_sample=False)

    runner = lm7.compile_generation(
        model, target=target, backend=backend, max_batch_size=2, max_sequence_length=128
    )
    result = runner.generate(
        batch["input_ids"], max_new_tokens=24, attention_mask=batch["attention_mask"]
    )
    assert result.tokens.tolist() == expected[:, prompt_tokens:].tolist()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable")
def test_decode_compiles_once_and_never_again():
    tokenizer, model = load(torch.bfloat16)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    runner = lm7.compile_generation(
        model, target="nvidia", backend="inductor", max_sequence_length=512
    )
    runner.generate(input_ids, max_new_tokens=100)

    assert runner.counters["decode"]["frames"] >= 1
    steady = runner.counters["steady"]
    assert steady["frames"] == 0, "a token triggered a compile"
    assert steady["recompiles"] == 0
    assert steady["graph_breaks"] == 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable")
def test_reduce_overhead_captures_the_decode_step():
    """Requesting CUDA Graphs and getting them are different things.

    The decode graph mutates KV-cache buffers in place, which is the pattern
    Inductor normally declines to capture — see benchmarks/cudagraphs.py. It works
    here because the cache is materialized on the device before tracing, which is
    what lets Transformers mark its buffers as static addresses.
    """
    tokenizer, model = load(torch.bfloat16)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    runner = lm7.compile_generation(
        model,
        target="nvidia",
        backend="inductor",
        compile_mode="reduce-overhead",
        max_sequence_length=512,
    )
    result = runner.generate(input_ids, max_new_tokens=32)

    decode = runner.cudagraphs["decode"]
    assert decode["cudagraphs"] is True
    assert decode["cudagraphs_active"] is True, f"capture was refused: {decode}"
    assert result.tokens.shape == (1, 32)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable")
def test_cuda_graph_replay_does_not_alias_the_returned_logits():
    """Each step's logits must survive the next replay.

    A captured graph writes its output into one static buffer, so a state holding
    the raw tensor starts describing a later token as soon as one is decoded.
    """
    tokenizer, model = load(torch.bfloat16)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    runner = lm7.compile_generation(
        model,
        target="nvidia",
        backend="inductor",
        compile_mode="reduce-overhead",
        max_sequence_length=512,
    )
    state = runner.prefill(input_ids)
    _, state = runner.decode(state.next_token, state)
    returned = state.logits
    snapshot = returned.clone()
    for _ in range(4):
        _, state = runner.decode(state.next_token, state)
    assert torch.equal(returned, snapshot), "a later replay overwrote an earlier step's logits"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable")
def test_cache_stays_on_the_gpu():
    tokenizer, model = load(torch.bfloat16)
    runner = lm7.compile_generation(
        model, target="nvidia", backend="inductor", max_sequence_length=256
    )
    layer = runner.past_key_values.layers[0]
    assert layer.keys.device.type == "cuda"
    assert runner.cache_bytes > 0
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    before = layer.keys.data_ptr()
    runner.generate(input_ids, max_new_tokens=8)
    assert layer.keys.data_ptr() == before, "the cache buffer was reallocated mid-run"
