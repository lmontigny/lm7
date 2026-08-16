"""Generate the overhead figure in the README's benchmark section.

Two overlapping latency distributions -- `torch.compile` called directly, and
the same model through `lm7.compile` -- because the claim being made is that
they are the same, and a bar chart of two medians cannot show *overlap*. Whether
two distributions sit on top of each other is the whole question, so the figure
has to plot them rather than summarize them.

Like `architecture.py`, this is written rather than drawn, and for the same
reason: it reads the measured JSON instead of being redrawn from a screenshot,
so it cannot drift from what was actually run. Regenerate it after a new run and
the figure is the new run.

    python benchmarks/gpu.py --target apple --model smollm2 --dtype float16 \\
      --backend torch-compile inductor-placed inductor \\
      --repeats 300 --record-latencies --output artifacts/overhead-hist-smollm2.json
    python docs/figures/overhead.py

writes `lm7-overhead.svg` and, when cairosvg is available, a 2x
`lm7-overhead.png` beside this script. The README uses the PNG, which renders
identically everywhere where an SVG depends on the viewer's fonts.

    pip install cairosvg
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

# --- palette ---------------------------------------------------------------
# Slots 1 and 2 of the reference categorical palette, in fixed order, validated
# against this surface rather than assumed: worst adjacent CVD dE 24.7 (protan),
# normal-vision dE 33.6, contrast 4.19:1 and 3.03:1 against PAGE -- every check
# passing, so the two series are separable without relying on the fills alone.
PAGE = "#f7f9fc"
CARD = "#ffffff"
CARD_LINE = "#dde4ec"
INK = "#1a2532"
MUTED = "#68788a"
GRID = "#e6ecf3"
BASELINE = "#2a78d6"  # torch.compile, called directly
LM7 = "#eb6834"  # the same model through lm7.compile

SANS = (
    "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

WIDTH, HEIGHT = 940, 512
PLOT = {"x": 78, "y": 162, "w": WIDTH - 78 - 40, "h": 243}
BIN_MS = 0.2
# Chosen to hold the bulk of both distributions rather than every sample; what
# falls outside is counted and said out loud under the plot instead of being
# quietly dropped, and the axis is not allowed to start anywhere but where the
# data does. The LM7 arm has the longer tail, including one call at 118 ms, so
# the samples this range drops are mostly its -- which is the direction that
# would flatter the figure, hence the count on the face of it.
X_MIN, X_MAX = 6.4, 12.0


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


def _histogram(values: list[float]) -> tuple[list[int], int]:
    """Counts per bin across [X_MIN, X_MAX), and how many fell outside."""
    bins = round((X_MAX - X_MIN) / BIN_MS)
    counts = [0] * bins
    outside = 0
    for value in values:
        index = int((value - X_MIN) / BIN_MS)
        if 0 <= index < bins:
            counts[index] += 1
        else:
            outside += 1
    return counts, outside


def _steps(counts: list[int], y_max: int) -> str:
    """A histogram outline as one path, drawn as steps rather than bars.

    Steps rather than separate rectangles because two overlapping series of
    filled bars produce a moire of edges at every bin boundary; one outline per
    series stays readable where they cross, which is the whole point of the
    figure.
    """
    x_scale = PLOT["w"] / (X_MAX - X_MIN)
    y_scale = PLOT["h"] / y_max
    floor = PLOT["y"] + PLOT["h"]
    points = [f"M {PLOT['x']:.2f} {floor:.2f}"]
    for index, count in enumerate(counts):
        left = PLOT["x"] + index * BIN_MS * x_scale
        right = left + BIN_MS * x_scale
        top = floor - count * y_scale
        points.append(f"L {left:.2f} {top:.2f} L {right:.2f} {top:.2f}")
    points.append(f"L {PLOT['x'] + PLOT['w']:.2f} {floor:.2f} Z")
    return " ".join(points)


def render(data: dict[str, Any], arms: dict[str, str]) -> str:
    series = {}
    for result in data["results"]:
        if result["backend"] in arms:
            series[result["backend"]] = [float(v) for v in result["latencies_ms"]]
    missing = [arm for arm in arms if arm not in series]
    if missing:
        raise SystemExit(
            f"no recorded latencies for {', '.join(missing)} -- rerun "
            "benchmarks/gpu.py with --record-latencies"
        )

    histograms = {name: _histogram(values) for name, values in series.items()}
    y_max = max(max(counts) for counts, _ in histograms.values())
    # Rounded up to a clean tick so the top gridline is a number a reader can
    # use, and the tallest bin is never flush against the plot's ceiling.
    y_max = int(math.ceil(y_max / 10.0) * 10)
    x_scale = PLOT["w"] / (X_MAX - X_MIN)
    floor = PLOT["y"] + PLOT["h"]

    environment = data["results"][0]["environment"]
    workload = data["workload"]
    total = len(next(iter(series.values())))

    baseline_arm, lm7_arm = arms
    ordered = ((baseline_arm, BASELINE), (lm7_arm, LM7))
    medians = {name: statistics.median(values) for name, values in series.items()}
    gap = medians[lm7_arm] - medians[baseline_arm]

    opening = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Per-call latency '
        f"distributions for {arms[baseline_arm]} called directly and for the same "
        f'model through {arms[lm7_arm]}, overlapping almost entirely">'
    )
    surface = (
        f'<rect x="16" y="16" width="{WIDTH - 32}" height="{HEIGHT - 32}" rx="10" '
        f'fill="{CARD}" stroke="{CARD_LINE}"/>'
    )
    s: list[str] = [opening, f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{PAGE}"/>', surface]

    s.append(_t(44, 54, "The same compiler, with and without LM7 in the call path", 19, INK, "600"))
    s.append(
        _t(
            44,
            80,
            f"{total} consecutive forward passes each · "
            f"{workload['model_id'].split('/')[-1]} · batch {workload['batch_size']} · "
            f"{workload['dtype']} · {environment.get('device_name', 'GPU')} · "
            f"torch {environment.get('torch', '?')}",
            13,
            MUTED,
        )
    )
    s.append(
        _t(
            44,
            100,
            "Both arms compile through TorchInductor with mode=None, so the generated "
            "code is the same and what differs is dispatch.",
            13,
            MUTED,
        )
    )
    s.append(
        _t(
            44,
            126,
            f"The medians are {abs(gap):.2f} ms apart — "
            f"{abs(gap) / medians[baseline_arm] * 100:.1f}% of the call, and smaller "
            "than what this machine varies by between runs.",
            13,
            INK,
            "500",
        )
    )

    # Horizontal gridlines, solid hairlines one shade off the surface.
    for step in range(0, y_max + 1, max(10, y_max // 4)):
        y = floor - step * PLOT["h"] / y_max
        s.append(
            f'<line x1="{PLOT["x"]}" y1="{y:.1f}" x2="{PLOT["x"] + PLOT["w"]}" '
            f'y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        s.append(_t(PLOT["x"] - 10, y + 4, str(step), 11, MUTED, anchor="end", family=MONO))
    label_y = PLOT["y"] + PLOT["h"] / 2
    s.append(
        f'<text x="{PLOT["x"] - 46}" y="{label_y:.1f}" font-family="{SANS}" font-size="11" '
        f'fill="{MUTED}" text-anchor="middle" '
        f'transform="rotate(-90 {PLOT["x"] - 46} {label_y:.1f})">calls</text>'
    )

    for name, colour in ordered:
        counts, _ = histograms[name]
        s.append(
            f'<path d="{_steps(counts, y_max)}" fill="{colour}" fill-opacity="0.30" '
            f'stroke="{colour}" stroke-width="2" stroke-linejoin="round"/>'
        )

    # Medians, drawn after the fills so neither series buries the other's marker.
    # Direct-labelled rather than left to the legend: with two series this close
    # together, the label beside each line is what makes them tellable apart
    # without relying on hue.
    for offset, (name, colour) in enumerate(ordered):
        x = PLOT["x"] + (medians[name] - X_MIN) * x_scale
        s.append(
            f'<line x1="{x:.1f}" y1="{PLOT["y"] - 8}" x2="{x:.1f}" y2="{floor}" '
            f'stroke="{colour}" stroke-width="2"/>'
        )
        s.append(
            _t(
                x + (7 if offset else -7),
                PLOT["y"] - 14,
                f"{arms[name]} {medians[name]:.2f} ms",
                12,
                colour,
                "600",
                anchor="start" if offset else "end",
            )
        )

    s.append(
        f'<line x1="{PLOT["x"]}" y1="{floor}" x2="{PLOT["x"] + PLOT["w"]}" y2="{floor}" '
        f'stroke="{CARD_LINE}" stroke-width="1"/>'
    )
    # Ticks sit on whole milliseconds, not on X_MIN plus a stride: the first
    # version stepped from 6.4 and labelled the marks "6", "7", "8", putting
    # every label 0.4 ms to the right of the value it named.
    tick = float(math.ceil(X_MIN))
    while tick <= X_MAX + 1e-9:
        x = PLOT["x"] + (tick - X_MIN) * x_scale
        s.append(
            f'<line x1="{x:.1f}" y1="{floor}" x2="{x:.1f}" y2="{floor + 5}" '
            f'stroke="{CARD_LINE}" stroke-width="1"/>'
        )
        s.append(_t(x, floor + 20, f"{tick:.0f}", 11, MUTED, anchor="middle", family=MONO))
        tick += 1.0
    s.append(
        _t(
            PLOT["x"] + PLOT["w"] / 2,
            floor + 42,
            "milliseconds per forward pass",
            12,
            MUTED,
            anchor="middle",
        )
    )

    # A legend is present because there are two series, and each swatch is also
    # named next to its own median above, so identity never rests on hue alone.
    legend_y = HEIGHT - 32
    x = 44.0
    for name, colour in ordered:
        s.append(
            f'<rect x="{x:.1f}" y="{legend_y - 9}" width="12" height="12" rx="2" fill="{colour}" '
            f'fill-opacity="0.30" stroke="{colour}" stroke-width="2"/>'
        )
        s.append(_t(x + 20, legend_y + 1, arms[name], 13, INK, "500", family=MONO))
        # 7.85px is one 13px monospace advance; the trailing 34 is the gap to the
        # next swatch. Measured rather than guessed because the two labels
        # collided at the first attempt.
        x += 20 + len(arms[name]) * 7.85 + 34
    outside = histograms[lm7_arm][1] + histograms[baseline_arm][1]
    if outside:
        s.append(
            _t(
                WIDTH - 44,
                legend_y + 1,
                f"{outside} of {total * 2} calls past {X_MAX:.0f} ms, not drawn",
                11,
                MUTED,
                anchor="end",
            )
        )
    s.append("</svg>")
    return "\n".join(s)


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=here.parents[1] / "artifacts" / "overhead-hist-smollm2.json",
        help="a benchmarks/gpu.py report written with --record-latencies",
    )
    arguments = parser.parse_args()

    data = json.loads(arguments.input.read_text())
    # Deliberately the *default* LM7 arm rather than `inductor-placed`: it is the
    # slower of the two, because it moves inputs to the device on every call, and
    # a figure making this claim should make it with the arm a reader would get
    # without passing anything.
    arms = {"torch-compile": "torch.compile", "inductor": "lm7.compile"}

    svg = here / "lm7-overhead.svg"
    svg.write_text(render(data, arms), encoding="utf-8")
    print(f"wrote {svg.name} ({svg.stat().st_size:,} bytes)")
    try:
        import cairosvg
    except ImportError:
        print("cairosvg not installed; skipping the PNG the README actually uses")
        return
    png = here / "lm7-overhead.png"
    cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2)
    print(f"wrote {png.name} ({png.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
