"""Measure what CUDA Graphs actually do across the four Inductor presets.

Only one of the four preset names mentions CUDA Graphs, and it is not the only
one that enables them:

    default                      does not request them
    reduce-overhead              requests them
    max-autotune                 requests them, and autotunes
    max-autotune-no-cudagraphs   autotunes, does not request them

Requesting is not getting. Inductor declines capture for mutated inputs, dynamic
shapes, CPU scalars and more, bumping a skip counter each time. This script
separates the two and checks the behaviours that matter for an inference loop:

- capture succeeds, or is refused with a reason
- a second call at the same shape reuses the graph instead of recompiling
- a changed input shape forces a recompile
- a stateful KV-cache generation step prevents capture
- repeated calls do not grow device memory

    python benchmarks/cudagraphs.py --output artifacts/cudagraphs.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch

import lm7
from lm7.backends.inductor import cudagraph_skips, cudagraphs_requested
from lm7.detection import resolve_target, synchronize

MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")


def _frames() -> int:
    """How many distinct graphs Dynamo has compiled, process-wide."""
    try:
        from torch._dynamo.utils import counters

        return int(counters["frames"]["total"])
    except Exception:  # noqa: BLE001
        return 0


class _MLP(torch.nn.Module):
    def __init__(self, width: int = 2048) -> None:
        super().__init__()
        self.up = torch.nn.Linear(width, width * 2, bias=False)
        self.down = torch.nn.Linear(width * 2, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.gelu(self.up(x)))


def _time(callable_model: Any, x: torch.Tensor, target: Any, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        with torch.no_grad():
            callable_model(x)
        synchronize(target)
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def run_mode(mode: str, target: Any, *, width: int, repeats: int) -> dict[str, Any]:
    torch._dynamo.reset()
    options = {"compile_mode": mode}
    model = _MLP(width).cuda().to(torch.bfloat16).eval()
    record: dict[str, Any] = {
        "compile_mode": mode,
        "cudagraphs_requested": cudagraphs_requested(mode, {}),
    }

    fixed = torch.randn(64, width, device="cuda", dtype=torch.bfloat16)
    compiled = lm7.compile(
        model,
        target=target,
        backend="inductor",
        transfers="automatic",
        fallback="error",
        cache=False,
        options=options,
    )

    # First call: compile plus, where requested, capture.
    skips_before = cudagraph_skips()
    torch.cuda.reset_peak_memory_stats()
    synchronize(target)
    started = time.perf_counter()
    with torch.no_grad():
        compiled(fixed)
    synchronize(target)
    record["first_call_ms"] = (time.perf_counter() - started) * 1000.0
    record["cudagraph_skips_on_capture"] = cudagraph_skips() - skips_before
    record["captured"] = (
        record["cudagraphs_requested"] and record["cudagraph_skips_on_capture"] == 0
    )

    # Second call at the same shape must not compile a new graph.
    frames_after_first = _frames()
    with torch.no_grad():
        compiled(fixed)
    synchronize(target)
    record["frames_second_call"] = _frames() - frames_after_first
    record["reuses_graph"] = record["frames_second_call"] == 0

    record["fixed_shape_ms"] = _time(compiled, fixed, target, repeats)
    record["peak_memory_compile_bytes"] = int(torch.cuda.max_memory_allocated())

    # Memory stability has to compare two rounds that are both past compilation.
    # An earlier revision compared the first round against the second and called
    # it stable, but the first includes the compile workspace, so it passed for a
    # reason that had nothing to do with replay leaking memory.
    torch.cuda.reset_peak_memory_stats()
    _time(compiled, fixed, target, repeats)
    steady_first = int(torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    _time(compiled, fixed, target, repeats)
    steady_second = int(torch.cuda.max_memory_allocated())
    record["peak_memory_steady_bytes"] = steady_first
    record["peak_memory_steady_repeat_bytes"] = steady_second
    record["memory_stable"] = steady_second <= steady_first

    # Changing shapes: each new batch size should compile a new graph the first
    # time it is seen. LM7 keys its own variant cache on the input signature, so
    # this measures Dynamo underneath rather than LM7's cache.
    changing = []
    frames_before_dynamic = _frames()
    skips_before_dynamic = cudagraph_skips()
    for batch in (16, 32, 128):
        x = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16)
        seen = _frames()
        with torch.no_grad():
            compiled(x)
        synchronize(target)
        changing.append({"batch": batch, "new_frames": _frames() - seen})
    record["changing_shapes"] = changing
    record["frames_for_three_new_shapes"] = _frames() - frames_before_dynamic
    record["cudagraph_skips_on_new_shapes"] = cudagraph_skips() - skips_before_dynamic
    record["dynamic_shapes_recompile"] = record["frames_for_three_new_shapes"] > 0

    del model, compiled
    torch.cuda.empty_cache()
    return record


def run_stateful(mode: str, target: Any, *, repeats: int) -> dict[str, Any]:
    """A module that mutates a buffer in place, which is what a KV cache does.

    CUDA Graphs replay into fixed addresses, so a graph that writes into a
    persistent buffer and reads it back on the next call is exactly the pattern
    capture has to refuse or handle specially. This is the cheap stand-in for a
    generation loop -- no tokenizer, no model download, same mutation shape.
    """

    class Stateful(torch.nn.Module):
        def __init__(self, width: int = 512, steps: int = 8) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(width, width, bias=False)
            self.register_buffer("cache", torch.zeros(steps, width, dtype=torch.bfloat16))
            self.position = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.proj(x)
            self.cache[self.position % self.cache.shape[0]] = out[0]
            self.position += 1
            return out + self.cache.sum(dim=0)

    torch._dynamo.reset()
    model = Stateful().cuda().to(torch.bfloat16).eval()
    x = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
    record: dict[str, Any] = {
        "compile_mode": mode,
        "cudagraphs_requested": cudagraphs_requested(mode, {}),
    }
    skips_before = cudagraph_skips()
    frames_before = _frames()
    try:
        compiled = lm7.compile(
            model,
            target=target,
            backend="inductor",
            transfers="automatic",
            fallback="error",
            cache=False,
            options={"compile_mode": mode},
        )
        for _ in range(repeats):
            with torch.no_grad():
                compiled(x)
        synchronize(target)
        record["works"] = True
        record["cudagraph_skips"] = cudagraph_skips() - skips_before
        # Frames matter as much as skips here: the Python counter this module
        # increments is state Dynamo guards on, so a graph recompiled on every
        # call would show up here even when nothing declined to capture.
        record["frames"] = _frames() - frames_before
        record["recompiles_per_call"] = record["frames"] > 2
        record["captured"] = record["cudagraphs_requested"] and record["cudagraph_skips"] == 0
    except Exception as error:  # noqa: BLE001 - a refusal is a result
        record.update(
            {"works": False, "error_type": type(error).__name__, "error": str(error)[:300]}
        )
    del model
    torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA Graph behaviour across Inductor presets.")
    parser.add_argument("--mode", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--target", default="nvidia")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA GPU.")
    target = resolve_target(arguments.target)

    fixed_results = []
    for mode in arguments.mode:
        record = run_mode(mode, target, width=arguments.width, repeats=arguments.repeats)
        fixed_results.append(record)
        print(
            f"{mode:<28} requested={record['cudagraphs_requested']!s:<5}"
            f" captured={record['captured']!s:<5}"
            f" reuse={record['reuses_graph']!s:<5}"
            f" newshape_frames={record['frames_for_three_new_shapes']:<2}"
            f" fixed={record['fixed_shape_ms']:7.4f}ms"
            f" mem_stable={record['memory_stable']}"
        )

    stateful_results = []
    print("\nstateful (in-place buffer mutation, the KV-cache pattern):")
    for mode in arguments.mode:
        record = run_stateful(mode, target, repeats=arguments.repeats)
        stateful_results.append(record)
        status = "ok" if record.get("works") else f"FAIL {record.get('error_type')}"
        print(
            f"{mode:<28} {status:<20}"
            f" requested={record['cudagraphs_requested']!s:<5}"
            + (
                f" skips={record['cudagraph_skips']} frames={record['frames']}"
                f" captured={record['captured']}"
                if record.get("works")
                else ""
            )
        )

    report = {
        "schema_version": 1,
        "environment": {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "capability": "sm{}{}".format(*torch.cuda.get_device_capability(0)),
            "host": platform.node(),
        },
        "fixed_and_changing_shapes": fixed_results,
        "stateful": stateful_results,
    }
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON: {output}")


if __name__ == "__main__":
    main()
