"""Generate the README's Blackwell quantization speedup figure.

This is the second figure drawn from the same measured aggregates, and it does
not replace the first. `blackwell_quantization.py` beside it draws the full
record -- throughput and storage, every arm, the fidelity verdict on each --
and that is the right figure for the note, where a reader has already committed
to the detail. This one is for the README, where a reader has not.

So: one panel, one question -- how much faster than BF16 each quantization mode
runs on this card. Speedup is the only measure plotted, and three things the
sibling figure carries are deliberately absent here, each for the same reason:
a figure that answers one question is read, and a figure that answers three is
decoded.

- **Storage reduction** is the sibling's second panel. It is a different
  question on a different scale, and a reader comparing two panels of bars ends
  up reading one against the other's axis.
- **NVFP4 weight-only** is one more bar there. It is a slower duplicate of the
  story the dynamic arm tells properly. (INT8 weight-only is in neither figure:
  #211 removed it from the sibling as a nightly regression, and at 0.005x it
  was a hairline against any axis that also holds 1.86x.)
- **The per-mode fidelity gate** is the sibling's color coding and its
  `rejected` labels. It is the reason two of these modes are not admitted, so
  the footnote here says two of the five fail it -- but *which* two, by how
  much, and against which stack are three more questions, and they belong
  beside the caveats they need.

All of it stays in the quantization doc's table and in the sibling figure,
which is where a number that is not the headline belongs.

The slow arm that remains is deliberate. A chart where every bar wins is a
chart that was drawn to win, and FP8 weight-only at 0.82x is the reason the
figure separates narrow storage from narrow arithmetic at all.

    python docs/figures/blackwell_quantization_speedup.py

writes `blackwell-quantization-speedup.svg` and, when cairosvg is available, a 2x
`blackwell-quantization-speedup.png` beside this script. The README and the docs use
the PNG, which renders identically everywhere where an SVG depends on the
viewer's fonts.

    pip install cairosvg   # needs libcairo; on macOS, brew install cairo
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

# --- palette ---------------------------------------------------------------
# One hue plus the de-emphasis gray, which is the emphasis form used by the
# other figures here: the quantized arms are the subject and BF16 is the
# reference they are read against.
#
# There is no second color for the fidelity gate, because the gate is not in
# this figure at all. It went through three forms before coming out: red bars
# for rejected (which made the verdict look like a property of the
# measurement), a lighter blue (impossible -- every tint light enough to
# separate from the reference gray at dE 15 is too light to clear 2:1 against
# this white surface), and a text column beside each bar (which turned a
# five-bar chart into a table). The figure now answers one question, and the
# fidelity results live in the quantization doc's table where they can carry
# their own caveats.
#
# Checked against this surface rather than assumed: blue/gray separate by dE
# 15.0 (protan), 18.9 (deutan) and 17.3 unsimulated, all clear of the CVD
# target of 8 and the normal-vision floor of 15. Blue clears 3:1 against white
# at 4.42. The gray is below the chroma floor and sits at 2.90, which is what
# makes it a de-emphasis gray rather than a second hue, and which obliges
# visible labels instead of color alone -- every bar carries its name and its
# value.
PAGE = "#f7f9fc"
CARD = "#ffffff"
CARD_LINE = "#dde4ec"
INK = "#1a2532"
MUTED = "#68788a"
GRID = "#eef2f7"
RULE = "#c7d2de"  # the BF16 baseline at 1.00x, one step up from the grid
REFERENCE = "#8d99a6"  # BF16, the thing every other arm is measured against
QUANTIZED = "#2a78d6"

SANS = (
    "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

WIDTH, HEIGHT = 780, 420
PLOT = {"x": 268, "y": 104, "w": 390}
X_MAX = 2.0  # speedup; the axis starts at zero, because these are bars
ROW_H, BAR_H = 44, 26
VALUE_GAP = 10

# Median milliseconds per arm, from the fresh-process runs recorded in
# notes/blackwell-quantization-figure.md. BF16 is first because it is the
# divisor.
BASELINE = 46.8431
ARMS = [
    ("BF16", 46.8431, REFERENCE),
    ("FP8 weight-only", 57.0100, QUANTIZED),
    ("FP8 dynamic", 35.8532, QUANTIZED),
    ("FP8 dynamic rowwise", 34.4706, QUANTIZED),
    ("NVFP4 dynamic", 25.2347, QUANTIZED),
]


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


def render() -> str:
    scale = PLOT["w"] / X_MAX
    speedups = [BASELINE / median for _, median, _ in ARMS]
    plot_bottom = PLOT["y"] + len(ARMS) * ROW_H

    label = (
        "Bar chart of speedup against a BF16 baseline for five quantization modes on "
        "Llama-3.1-8B. Dynamic NVFP4 is fastest at 1.86 times BF16, rowwise dynamic FP8 "
        "reaches 1.36 times and per-tensor dynamic FP8 1.31 times, while FP8 weight-only "
        "is slower than BF16 at 0.82 times."
    )
    s: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{escape(label)}">'
        ),
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{PAGE}"/>',
        (
            f'<rect x="16" y="16" width="{WIDTH - 32}" height="{HEIGHT - 32}" rx="10" '
            f'fill="{CARD}" stroke="{CARD_LINE}"/>'
        ),
        _t(42, 52, "Quantization speedup on Blackwell", 18, INK, "650"),
        _t(
            42,
            74,
            "Llama-3.1-8B · batch 8 × sequence 128 · RTX PRO 6000 Blackwell (sm120) · "
            "TorchInductor",
            12,
            MUTED,
        ),
    ]

    # Gridlines first, so every bar sits on top of them. 1.00x is the question
    # the chart asks -- faster than BF16 or not -- so it gets a stronger rule
    # than the rest of the grid and the only tick that is named.
    for tick in (0.0, 0.5, 1.0, 1.5, 2.0):
        x = PLOT["x"] + tick * scale
        stroke = RULE if tick == 1.0 else GRID
        s.append(
            f'<line x1="{x:.1f}" y1="{PLOT["y"]}" x2="{x:.1f}" y2="{plot_bottom}" '
            f'stroke="{stroke}"/>'
        )
        s.append(_t(x, plot_bottom + 20, f"{tick:.1f}×", 10, MUTED, anchor="middle", family=MONO))
    s.append(
        _t(PLOT["x"] + scale, PLOT["y"] - 10, "BF16 baseline", 10, MUTED, "500", anchor="middle")
    )

    for i, ((name, _, colour), speedup) in enumerate(zip(ARMS, speedups)):
        y = PLOT["y"] + i * ROW_H + (ROW_H - BAR_H) / 2
        mid = y + BAR_H / 2 + 4
        width = speedup * scale
        s.append(_t(PLOT["x"] - 18, mid, name, 12, INK, "500", "end"))
        s.append(
            f'<rect x="{PLOT["x"]}" y="{y:.1f}" width="{width:.1f}" height="{BAR_H}" '
            f'rx="4" fill="{colour}"/>'
        )
        s.append(
            _t(PLOT["x"] + width + VALUE_GAP, mid, f"{speedup:.2f}×", 12, INK, "600", family=MONO)
        )

    s.extend(
        [
            _t(
                42,
                plot_bottom + 56,
                "Median of 30+ synchronized forwards, one fresh process per arm. Speed only: "
                "two of these five",
                11,
                MUTED,
            ),
            _t(
                42,
                plot_bottom + 72,
                "modes fail the top-1 fidelity gate and are not admitted. See the "
                "quantization doc for which, and by how much.",
                11,
                MUTED,
            ),
            "</svg>",
        ]
    )
    return "\n".join(s)


def main() -> None:
    here = Path(__file__).parent
    svg = here / "blackwell-quantization-speedup.svg"
    # newline="\n" so a regeneration on Windows does not rewrite every line.
    svg.write_text(render(), encoding="utf-8", newline="\n")
    try:
        import cairosvg
    except (ImportError, OSError):
        return
    cairosvg.svg2png(
        url=str(svg), write_to=str(here / "blackwell-quantization-speedup.png"), scale=2
    )


if __name__ == "__main__":
    main()
