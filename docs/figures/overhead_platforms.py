"""Generate the combined Apple Metal and NVIDIA CUDA overhead figure."""

from __future__ import annotations

import statistics
from pathlib import Path
from xml.sax.saxutils import escape

PAGE, CARD, LINE = "#f7f9fc", "#ffffff", "#dde4ec"
INK, MUTED, GRID = "#1a2532", "#68788a", "#eef2f7"
REFERENCE, LM7 = "#8d99a6", "#2a78d6"
SANS = "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
WIDTH, HEIGHT = 900, 390
PLOT_X, PLOT_Y, PLOT_W, X_MAX = 290, 92, 430, 30.0
BAR_H = 24

# Published seven-run aggregates. Endpoints preserve each reported range;
# repeated medians avoid pretending the unavailable raw Mac samples can be
# reconstructed. The RTX values are also kept here so regeneration is hermetic.
PLATFORMS = {
    "Apple M3 Pro · Metal": {
        "torch.compile": [7.65, 7.88, 7.88, 7.88, 7.88, 7.88, 8.43],
        "lm7.compile": [7.62, 7.91, 7.91, 7.91, 7.91, 7.91, 9.22],
    },
    "RTX 4070 SUPER · CUDA 13.0": {
        "torch.compile": [18.491, 18.550, 18.722, 19.116, 19.596, 21.134, 25.438],
        "lm7.compile": [19.092, 19.236, 19.758, 19.916, 20.156, 20.556, 22.295],
    },
}
COLOURS = {"torch.compile": REFERENCE, "lm7.compile": LM7}


def text(
    x: float,
    y: float,
    value: str,
    size: float,
    fill: str,
    weight: str = "400",
    anchor: str = "start",
    family: str = SANS,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def render() -> str:
    scale = PLOT_W / X_MAX
    output = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Grouped bar chart comparing '
            f'direct torch.compile and lm7.compile latency on Apple M3 Pro and RTX 4070 SUPER">'
        ),
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{PAGE}"/>',
        (
            f'<rect x="16" y="16" width="{WIDTH - 32}" height="{HEIGHT - 32}" rx="10" '
            f'fill="{CARD}" stroke="{LINE}"/>'
        ),
        text(44, 50, "LM7 overhead across Metal and CUDA", 17, INK, "650"),
        text(44, 72, "SmolLM2-135M-Instruct · batch 1 · float16 · median of 7 runs", 12, MUTED),
    ]

    for tick in range(0, 31, 5):
        x = PLOT_X + tick * scale
        output.append(
            f'<line x1="{x:.1f}" y1="{PLOT_Y}" x2="{x:.1f}" y2="320" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        output.append(text(x, 340, str(tick), 11, MUTED, anchor="middle", family=MONO))

    y = 100
    for platform, arms in PLATFORMS.items():
        output.append(text(44, y + 5, platform, 14, INK, "600"))
        y += 30
        baseline = statistics.median(arms["torch.compile"])
        for arm, values in arms.items():
            median = statistics.median(values)
            output.append(text(274, y + 5, arm, 12, INK, "500", anchor="end", family=MONO))
            output.append(
                f'<rect x="{PLOT_X}" y="{y - BAR_H / 2:.1f}" width="{median * scale:.1f}" '
                f'height="{BAR_H}" rx="4" fill="{COLOURS[arm]}"/>'
            )
            low, high = min(values), max(values)
            x1, x2 = PLOT_X + low * scale, PLOT_X + high * scale
            output.append(
                f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" '
                f'stroke="{INK}" stroke-width="2" stroke-opacity="0.65"/>'
            )
            for cap in (x1, x2):
                output.append(
                    f'<line x1="{cap:.1f}" y1="{y - 7}" x2="{cap:.1f}" y2="{y + 7}" '
                    f'stroke="{INK}" stroke-width="2" stroke-opacity="0.65"/>'
                )
            value = f"{median:.2f} ms"
            if arm == "lm7.compile":
                value += f"  ({median / baseline:.2f}x)"
            output.append(text(x2 + 12, y + 5, value, 12, INK, "600", family=MONO))
            y += 38
        y += 30

    output.append(
        text(
            PLOT_X + PLOT_W / 2,
            368,
            "milliseconds per forward pass (lower is better)",
            12,
            MUTED,
            anchor="middle",
        )
    )
    output.append("</svg>")
    return "\n".join(output)


def main() -> None:
    here = Path(__file__).parent
    svg = here / "lm7-overhead-platforms.svg"
    svg.write_text(render(), encoding="utf-8")
    print(f"wrote {svg.name} ({svg.stat().st_size:,} bytes)")
    try:
        import cairosvg
    except (ImportError, OSError):
        print("cairosvg unavailable; skipping PNG")
        return
    png = here / "lm7-overhead-platforms.png"
    cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2)
    print(f"wrote {png.name} ({png.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
