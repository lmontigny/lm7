"""What a dynamic sequence dimension buys on the prompt and costs on every token.

A decode artifact's graph can take one token per call or a bounded range of them,
and the two are not ordered: `dynamic` prefills a whole prompt in a single call,
`single-token` compiles for one shape and decodes faster for it. Which wins is a
property of the workload -- how long the prompt is against how much is generated
from it -- so this measures both halves rather than picking one.

    python benchmarks/exported_decode.py --output artifacts/exported-decode.json

Every number here comes from an artifact written by `lm7 model export --decode`
and reloaded through `lm7.load_artifact`, not from a program held in memory, so
it is what a deployed artifact actually costs. Correctness is not measured here;
it is gated in tests/test_export_decode_integration.py, which requires token
equality with eager.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from lm7 import load_artifact
from lm7.huggingface import export_hf_model

DEFAULT_MODEL = "hf://HuggingFaceTB/SmolLM2-135M-Instruct"


def _prompt(length: int) -> torch.Tensor:
    """Deterministic ids inside any vocabulary this benchmark is pointed at."""
    return torch.tensor([[(index * 37) % 4000 + 5 for index in range(length)]])


def _time(call, repeats: int) -> float:
    """Milliseconds for the best of `repeats`, after one warm call."""
    call()
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        best = min(best, (time.perf_counter() - started) * 1000)
    return best


def measure(artifact: Any, shape: str, lengths: list[int], steps: int, repeats: int) -> dict:
    """Prefill cost at each length, and the per-token cost of decoding afterwards."""
    record: dict[str, Any] = {"shape": shape, "prefill_ms": {}, "decode_ms_per_token": None}
    with torch.inference_mode():
        for length in lengths:
            prompt = _prompt(length)
            positions = torch.arange(length, dtype=torch.long)
            if shape == "dynamic":

                def prefill(prompt=prompt, positions=positions):
                    artifact(input_ids=prompt, cache_position=positions)
            else:
                # The fixed graph has no way to take a prompt at once, so this is
                # the same work expressed the only way it can be: a call per
                # token. That is the cost being compared, not a handicap.
                def prefill(prompt=prompt, length=length):
                    for position in range(length):
                        artifact(
                            input_ids=prompt[:, position : position + 1],
                            cache_position=torch.tensor([position], dtype=torch.long),
                        )

            record["prefill_ms"][str(length)] = round(_time(prefill, repeats), 2)

        # Decode is one token per call for both shapes -- the only difference is
        # whether the graph underneath was compiled for that shape alone.
        base = max(lengths)

        def decode_run():
            for offset in range(steps):
                artifact(
                    input_ids=torch.tensor([[7]]),
                    cache_position=torch.tensor([base + offset], dtype=torch.long),
                )

        record["decode_ms_per_token"] = round(_time(decode_run, repeats) / steps, 3)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefill and decode cost of the two shapes.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="aot_inductor", choices=("export", "aot_inductor"))
    parser.add_argument("--target", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--prompt-length", type=int, nargs="+", default=[32, 128, 250])
    parser.add_argument("--decode-steps", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-cache-len", type=int, default=512)
    parser.add_argument("--build-dir", default="build/exported-decode")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    build = Path(arguments.build_dir)
    if build.exists():
        shutil.rmtree(build)

    record: dict[str, Any] = {
        "model": arguments.model,
        "backend": arguments.backend,
        "target": arguments.target,
        "dtype": arguments.dtype,
        "max_cache_len": arguments.max_cache_len,
        "prompt_lengths": arguments.prompt_length,
        "decode_steps": arguments.decode_steps,
        "torch": torch.__version__,
        "shapes": {},
    }

    for shape in ("dynamic", "single-token"):
        started = time.perf_counter()
        exported = export_hf_model(
            arguments.model,
            output=str(build / f"{shape}.lm7"),
            target=arguments.target,
            backend=arguments.backend,
            dtype=arguments.dtype,
            decode=True,
            decode_shape=shape,
            max_cache_len=arguments.max_cache_len,
        )
        export_ms = (time.perf_counter() - started) * 1000
        artifact = load_artifact(exported.output)
        measured = measure(
            artifact, shape, arguments.prompt_length, arguments.decode_steps, arguments.repeats
        )
        measured["export_ms"] = round(export_ms, 1)
        measured["artifact_mib"] = round(exported.artifact_bytes / 1024**2, 1)
        record["shapes"][shape] = measured

    dynamic = record["shapes"]["dynamic"]
    fixed = record["shapes"]["single-token"]
    record["prefill_speedup"] = {
        length: round(fixed["prefill_ms"][length] / dynamic["prefill_ms"][length], 1)
        for length in dynamic["prefill_ms"]
    }
    record["decode_cost"] = round(dynamic["decode_ms_per_token"] / fixed["decode_ms_per_token"], 2)
    # Where the prompt saving is repaid by the slower token. Below this many
    # generated tokens `dynamic` is ahead; above it, `single-token` is.
    record["break_even_tokens"] = {
        length: (
            round(
                (fixed["prefill_ms"][length] - dynamic["prefill_ms"][length])
                / (dynamic["decode_ms_per_token"] - fixed["decode_ms_per_token"])
            )
            if dynamic["decode_ms_per_token"] > fixed["decode_ms_per_token"]
            else None
        )
        for length in dynamic["prefill_ms"]
    }

    print(json.dumps(record, indent=2, sort_keys=True))
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    print("\nprompt   dynamic    single-token   speedup   break-even")
    for length in arguments.prompt_length:
        key = str(length)
        print(
            f"{length:6d}  {dynamic['prefill_ms'][key]:8.1f}   {fixed['prefill_ms'][key]:10.1f}   "
            f"{record['prefill_speedup'][key]:6.1f}x   {record['break_even_tokens'][key]} tokens"
        )
    print(
        f"\ndecode: dynamic {dynamic['decode_ms_per_token']:.2f} ms/token, "
        f"single-token {fixed['decode_ms_per_token']:.2f} ms/token "
        f"({record['decode_cost']}x)"
    )


if __name__ == "__main__":
    main()
