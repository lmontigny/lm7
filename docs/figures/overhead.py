"""Generate the overhead figure in the README's benchmark section.

Three bars: a forward pass with `torch.compile` called directly, and the same
forward pass through `lm7.compile` twice -- once with inputs already on the
device, once with LM7's default per-call transfer. The question the figure has
to answer is whether the layer costs anything, so each bar carries the spread
across runs as well as its median: the bars are closer together than the runs
are to each other, and a chart showing only medians would claim a precision
this machine does not have.

Like `architecture.py`, this is written rather than drawn, and for the same
reason: it reads the measured JSON instead of being redrawn from a screenshot,
so it cannot drift from what was actually run. Regenerate it after new runs and
the figure is the new runs.

    for run in 1 2 3 4 5 6 7; do
      python benchmarks/gpu.py --target apple --model smollm2 --dtype float16 \\
        --backend torch-compile inductor-placed inductor --repeats 100 \\
        --output artifacts/overhead-smollm2-run$run.json
    done
    python docs/figures/overhead.py

writes `lm7-overhead.svg` and, when cairosvg is available, a 2x
`lm7-overhead.png` beside this script. The README uses the PNG, which renders
identically everywhere where an SVG depends on the viewer's fonts.

    pip install cairosvg   # needs libcairo; on macOS, brew install cairo
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

# --- palette ---------------------------------------------------------------
# One hue plus a de-emphasis gray, which is the emphasis form: the two LM7 bars
# are the subject, and the direct `torch.compile` call is the reference they are
# read against. The hue is slot 1 of the reference categorical palette, and the
# pair was validated against this surface rather than assumed: CVD separation
# dE 15.0 (protan) and normal-vision dE 17.3, both clear of their floors. Two
# results are deliberate rather than overlooked -- the gray is below the chroma
# floor, which is what makes it a de-emphasis gray instead of a ninth hue; and
# at 2.90:1 it sits just under the 3:1 contrast bar, which obliges visible
# labels rather than color alone. Every bar carries its name and its value, so
# that obligation is met. A darker gray would clear the contrast bar and fail
# the separation floor against the blue -- they pull opposite ways here.
PAGE = "#f7f9fc"
CARD = "#ffffff"
CARD_LINE = "#dde4ec"
INK = "#1a2532"
MUTED = "#68788a"
GRID = "#eef2f7"
REFERENCE = "#8d99a6"  # torch.compile, the thing LM7 is measured against
LM7 = "#2a78d6"

SANS = (
    "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

WIDTH, HEIGHT = 900, 258
PLOT = {"x": 268, "y": 92, "w": 470, "h": 94}
BAR_H = 26
X_MAX = 10.0  # milliseconds; the axis starts at zero, because a bar chart must

# The arms, in the order they are drawn, with the label each gets on the left.
#
# Two bars, not three. The `inductor` arm -- LM7's default, which copies the
# caller's inputs to the device on every call -- is deliberately not drawn here:
# it is a third bar about a setting, and the figure is answering a question about
# a layer. It costs about 0.21 ms more, and because that is the arm a reader gets
# without passing anything, the README says so in prose beside the figure and
# keeps all three in its table. Dropping it from the picture is only honest while
# that stays true.
ARMS = {
    "torch-compile": ("torch.compile", REFERENCE),
    "inductor-placed": ("lm7.compile", LM7),
}


def _t(
    x: float,
    y: float,
    text: str,
    size: float,
    fill: str,
    weight: str = "400",
    anchor: str = "start",
    family: str = SANS,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(text)}</text>'
    )


def collect(paths: list[Path]) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Each arm's per-run median latency, and the environment they share."""
    runs: dict[str, list[float]] = {arm: [] for arm in ARMS}
    context: dict[str, Any] = {}
    for path in sorted(paths):
        report = json.loads(path.read_text())
        found = {result["backend"]: result for result in report["results"]}
        if not all(arm in found for arm in ARMS):
            # A run from a different arm set. Skipped rather than part-counted,
            # so that every bar rests on exactly the same set of runs.
            continue
        for arm in ARMS:
            runs[arm].append(float(found[arm]["latency_median_ms"]))
        context = {**report["workload"], **found["torch-compile"]["environment"]}
    missing = [arm for arm, values in runs.items() if not values]
    if missing:
        raise SystemExit(f"no run carried all of {', '.join(ARMS)}; missing {', '.join(missing)}")
    return runs, context


def render(runs: dict[str, list[float]], context: dict[str, Any]) -> str:
    scale = PLOT["w"] / X_MAX
    row_h = PLOT["h"] / len(ARMS)
    medians = {arm: statistics.median(values) for arm, values in runs.items()}
    baseline = medians["torch-compile"]

    opening = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Bar chart of median '
        f"forward-pass latency: torch.compile called directly at {baseline:.2f} "
        f"milliseconds, and the same model through lm7.compile at "
        f"{medians['inductor-placed']:.2f} milliseconds, with the spread across runs "
        f'wider than the gap between them">'
    )
    surface = (
        f'<rect x="16" y="16" width="{WIDTH - 32}" height="{HEIGHT - 32}" rx="10" '
        f'fill="{CARD}" stroke="{CARD_LINE}"/>'
    )
    s: list[str] = [opening, f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{PAGE}"/>', surface]

    s.append(
        _t(
            44,
            56,
            f"Median forward pass · {context['model_id'].split('/')[-1]} · "
            f"batch {context['batch_size']} · {context['dtype']} · "
            f"{context.get('device_name', 'GPU')} · torch {context.get('torch', '?')}",
            13,
            MUTED,
        )
    )

    # Vertical gridlines behind the bars, solid hairlines one shade off the card.
    for tick in range(0, int(X_MAX) + 1, 2):
        x = PLOT["x"] + tick * scale
        s.append(
            f'<line x1="{x:.1f}" y1="{PLOT["y"] - 8}" x2="{x:.1f}" '
            f'y2="{PLOT["y"] + PLOT["h"]}" stroke="{GRID}" stroke-width="1"/>'
        )
        s.append(
            _t(x, PLOT["y"] + PLOT["h"] + 20, str(tick), 11, MUTED, anchor="middle", family=MONO)
        )
    s.append(
        _t(
            PLOT["x"] + PLOT["w"] / 2,
            PLOT["y"] + PLOT["h"] + 42,
            "milliseconds per forward pass (lower is better)",
            12,
            MUTED,
            anchor="middle",
        )
    )

    for index, (arm, (label, colour)) in enumerate(ARMS.items()):
        centre = PLOT["y"] + index * row_h + row_h / 2
        top = centre - BAR_H / 2
        width = medians[arm] * scale
        s.append(_t(PLOT["x"] - 16, centre + 4, label, 13, INK, "500", anchor="end", family=MONO))
        s.append(
            f'<rect x="{PLOT["x"]}" y="{top:.1f}" width="{width:.1f}" height="{BAR_H}" '
            f'rx="4" fill="{colour}"/>'
        )

        # The spread across runs. Drawn in ink rather than in the surface colour
        # because it reaches past the end of its bar more often than not, and a
        # white line is invisible out there -- which hid most of the range in the
        # first version of this figure.
        low, high = min(runs[arm]), max(runs[arm])
        x1, x2 = PLOT["x"] + low * scale, PLOT["x"] + high * scale
        s.append(
            f'<line x1="{x1:.1f}" y1="{centre:.1f}" x2="{x2:.1f}" y2="{centre:.1f}" '
            f'stroke="{INK}" stroke-width="2" stroke-opacity="0.65"/>'
        )
        for cap in (x1, x2):
            s.append(
                f'<line x1="{cap:.1f}" y1="{centre - 7:.1f}" x2="{cap:.1f}" '
                f'y2="{centre + 7:.1f}" stroke="{INK}" stroke-width="2" stroke-opacity="0.65"/>'
            )

        value = f"{medians[arm]:.2f} ms"
        if arm != "torch-compile":
            value += f"  ({medians[arm] / baseline:.2f}x)"
        s.append(_t(x2 + 14, centre + 4, value, 13, INK, "600", family=MONO))

    s.append("</svg>")
    return "\n".join(s)


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=here.parents[1] / "artifacts",
        help="directory of benchmarks/gpu.py reports",
    )
    parser.add_argument("--glob", default="overhead-smollm2-run*.json")
    arguments = parser.parse_args()

    runs, context = collect(list(arguments.artifacts.glob(arguments.glob)))
    # Printed because the figure itself no longer says any of it: the run count
    # and the medians live in the README prose beside the image, and that prose
    # has to be updated by hand when these move.
    print(f"{len(next(iter(runs.values())))} runs")
    for arm, values in runs.items():
        print(
            f"  {arm:<16} median {statistics.median(values):6.2f} ms   "
            f"range {min(values):.2f}-{max(values):.2f}"
        )
    svg = here / "lm7-overhead.svg"
    svg.write_text(render(runs, context), encoding="utf-8")
    print(f"wrote {svg.name} ({svg.stat().st_size:,} bytes)")
    try:
        import cairosvg
    except (ImportError, OSError):
        print("cairosvg unavailable; skipping the PNG the README actually uses")
        return
    png = here / "lm7-overhead.png"
    cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2)
    print(f"wrote {png.name} ({png.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
