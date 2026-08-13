"""Compiled vs eager text-to-image diffusion: per-step latency, and whether it agrees.

The headline is ``ms_per_step``. Encode and VAE decode are paid once per image
whatever the step count, so they are reported separately rather than folded into
one end-to-end number that would hide the only cost that scales.

``break_even_steps`` is the diffusion counterpart of ``break_even_tokens`` in
`exported_decode.py`: compiling costs a first call and pays it back per step, so
the question "is compiling worth it" has a step count for an answer, not a yes.

Correctness is checked from **identical initial latents**, which is the only way
to compare two diffusion arms at all -- different noise gives different images
and says nothing. Two numbers come out of it, and the second is the important
one: `max_abs_difference` on the final image, and the same on the *first step's*
latents. Divergence in a denoise loop compounds, so a small step-1 difference
beside a large final one is this path's version of the failure
`docs/exported-decode.md` records for stateful artifacts -- right at first,
quietly wrong by the end.

    python benchmarks/diffusion.py --model sd-turbo --target nvidia \
        --output artifacts/diffusion-4070s.json

Defaults are sized for the ~5 minute budget on an RTX 4070 SUPER: SD-Turbo,
512x512, batch 1, 4 steps, 3 images per arm after a warmup image.
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
from lm7.detection import resolve_target
from lm7.image_hub import load_pipeline

HF_MODELS = {
    # SD-Turbo is the reference: distilled to 4 steps and trained without
    # guidance, so a whole image is cheap enough to repeat inside the budget.
    "sd-turbo": "hf://stabilityai/sd-turbo",
    "sd15": "hf://stable-diffusion-v1-5/stable-diffusion-v1-5",
    # Fits 12 GiB at float16 and not at float32; not measured on the 4070 SUPER.
    "sdxl": "hf://stabilityai/stable-diffusion-xl-base-1.0",
    # The CI-sized pipeline, for checking this script runs without a download
    # budget. Its numbers mean nothing.
    "tiny": "hf://hf-internal-testing/tiny-stable-diffusion-pipe",
}

DEFAULT_PROMPT = "a red bicycle leaning against a white wall"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _measure(runner: Any, args: argparse.Namespace, latents: torch.Tensor) -> dict[str, Any]:
    """One warmup image, then `repeats` measured ones from the same latents."""
    first = runner.generate(
        args.prompt, steps=args.steps, guidance_scale=args.guidance_scale, latents=latents
    )
    samples = [
        runner.generate(
            args.prompt, steps=args.steps, guidance_scale=args.guidance_scale, latents=latents
        )
        for _ in range(args.repeats)
    ]
    per_step = [result.ms_per_step for result in samples]
    totals = [result.total_ms for result in samples]
    return {
        # The warmup call, which for a compiled arm is where compilation happened.
        "first_call_ms": first.total_ms,
        "first_call_ms_per_step": first.ms_per_step,
        "ms_per_step_median": statistics.median(per_step),
        "ms_per_step_min": min(per_step),
        "ms_per_step_p95": _percentile(per_step, 0.95),
        "encode_ms_median": statistics.median(r.encode_ms for r in samples),
        "denoise_ms_median": statistics.median(r.denoise_ms for r in samples),
        "decode_ms_median": statistics.median(r.decode_ms for r in samples),
        "total_ms_median": statistics.median(totals),
        "peak_memory_bytes": _peak_memory(runner.target),
        "counters": first.counters,
        # Nonzero means a step recompiled, which invalidates the per-step number
        # above rather than merely making it worse.
        "recompiled_during_loop": bool(first.counters["steady"]["frames"]),
        "images": samples[-1].images,
    }


def _peak_memory(target: Any) -> int | None:
    if target.vendor not in {"nvidia", "amd"}:
        return None
    return int(torch.cuda.max_memory_allocated(target.ordinal or 0))


def _first_step_latents(runner: Any, args: argparse.Namespace, latents: torch.Tensor) -> Any:
    """Denoise exactly one step, so divergence can be read before it compounds."""
    embeddings = runner.encode_prompt(args.prompt, guidance_scale=args.guidance_scale)
    return runner.denoise(latents, embeddings, steps=1, guidance_scale=args.guidance_scale)


def _agreement(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    difference = (reference.float() - candidate.float()).abs()
    mse = float((difference**2).mean())
    # PSNR against a [0, 1] signal. Infinite when the arms are bit-identical,
    # which is a real outcome on CPU and worth reporting as such.
    psnr = float("inf") if mse == 0 else float(10 * torch.log10(torch.tensor(1.0 / mse)))
    return {
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "psnr_db": psnr,
    }


def _break_even_steps(eager: dict[str, Any], compiled: dict[str, Any]) -> float | None:
    """How many steps before compiling has paid for its first call.

    None when the compiled arm is not faster per step, which is the answer that
    matters most: there is no step count at which it pays.
    """
    saved = eager["ms_per_step_median"] - compiled["ms_per_step_median"]
    if saved <= 0:
        return None
    overhead = compiled["first_call_ms"] - eager["first_call_ms"]
    return max(0.0, overhead / saved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sd-turbo", choices=sorted(HF_MODELS))
    parser.add_argument("--target", default="auto")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--size", default="512x512")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument("--output", default=None, help="write JSON here")
    args = parser.parse_args()

    width, height = (int(part) for part in args.size.lower().split("x"))
    target = resolve_target(args.target)
    model_uri = HF_MODELS[args.model]

    pipeline, dtype_name = load_pipeline(model_uri, target=target, dtype=args.dtype)

    results = []
    images: dict[str, torch.Tensor] = {}
    step_one: dict[str, torch.Tensor] = {}
    for backend in ("eager", "inductor"):
        runner = lm7.compile_diffusion(
            pipeline,
            target,
            backend=backend,
            compile_mode=args.compile_mode if backend == "inductor" else None,
            height=height,
            width=width,
        )
        # Every arm starts from the same noise, so the images are comparable.
        latents = runner.initial_latents(seed=args.seed)
        step_one[backend] = _first_step_latents(runner, args, latents)
        measured = _measure(runner, args, latents)
        images[backend] = measured.pop("images")
        results.append({"backend": backend, **measured})

    eager, compiled = results[0], results[1]
    report = {
        "schema_version": 1,
        "workload": {
            "model": args.model,
            "model_uri": model_uri,
            "pipeline_class": type(pipeline).__name__,
            "scheduler": type(pipeline.scheduler).__name__,
            "target": str(target),
            "dtype": dtype_name,
            "prompt": args.prompt,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "width": width,
            "height": height,
            "batch_size": 2 if args.guidance_scale > 1.0 else 1,
            "repeats": args.repeats,
            "seed": args.seed,
            "compile_mode": args.compile_mode,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "results": results,
        "comparison": {
            "ms_per_step_speedup": (
                eager["ms_per_step_median"] / compiled["ms_per_step_median"]
                if compiled["ms_per_step_median"]
                else None
            ),
            "break_even_steps": _break_even_steps(eager, compiled),
            "final_image": _agreement(images["eager"], images["inductor"]),
            "first_step_latents": _agreement(step_one["eager"], step_one["inductor"]),
        },
    }

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
