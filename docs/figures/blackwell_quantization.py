"""Generate the README Blackwell quantization speedup figure."""

from pathlib import Path
from xml.sax.saxutils import escape

PAGE, CARD, LINE = "#f7f9fc", "#ffffff", "#dde4ec"
INK, MUTED, GRID = "#1a2532", "#68788a", "#eef2f7"
REFERENCE, QUANTIZED = "#8d99a6", "#2a78d6"
SANS = "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
WIDTH, HEIGHT = 900, 410
PLOT_X, PLOT_Y, PLOT_W, X_MAX = 290, 112, 440, 2.0
BAR_H = 26

# Published RTX PRO 6000 Blackwell aggregates. Speedup is throughput relative
# to the synchronized BF16 baseline at the same batch and sequence shape.
ROWS = (
    ("BF16", 1.00, "baseline"),
    ("FP8 weight-only", 0.82, "3/4 · rejected"),
    ("FP8 dynamic", 1.31, "4/4"),
    ("FP8 dynamic rowwise", 1.36, "4/4"),
    ("NVFP4 dynamic", 1.86, "2/4 · rejected"),
)


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
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Horizontal bar chart '
            f'comparing quantization throughput with BF16 on RTX PRO 6000 Blackwell">'
        ),
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{PAGE}"/>',
        (
            f'<rect x="16" y="16" width="{WIDTH - 32}" height="{HEIGHT - 32}" rx="10" '
            f'fill="{CARD}" stroke="{LINE}"/>'
        ),
        text(44, 50, "Quantization speedup on Blackwell", 17, INK, "650"),
        text(
            44,
            72,
            "Llama-3.1-8B · batch 8 × sequence 128 · RTX PRO 6000 Blackwell (sm120) · TorchInductor",
            12,
            MUTED,
        ),
        text(820, 98, "fidelity", 11, MUTED, "500", anchor="end"),
    ]

    for tick in (0.0, 0.5, 1.0, 1.5, 2.0):
        x = PLOT_X + tick * scale
        output.append(
            f'<line x1="{x:.1f}" y1="{PLOT_Y}" x2="{x:.1f}" y2="330" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        output.append(text(x, 350, f"{tick:.1f}×", 11, MUTED, anchor="middle", family=MONO))

    for index, (label, speedup, fidelity) in enumerate(ROWS):
        y = 130 + index * 45
        colour = REFERENCE if label == "BF16" else QUANTIZED
        output.append(text(PLOT_X - 18, y + 5, label, 12, INK, "500", anchor="end"))
        output.append(
            f'<rect x="{PLOT_X}" y="{y - BAR_H / 2:.1f}" width="{speedup * scale:.1f}" '
            f'height="{BAR_H}" rx="4" fill="{colour}"/>'
        )
        output.append(
            text(
                PLOT_X + speedup * scale + 12,
                y + 5,
                f"{speedup:.2f}×",
                12,
                INK,
                "600",
                family=MONO,
            )
        )
        output.append(text(820, y + 5, fidelity, 11, MUTED, "500", anchor="end"))

    output.append(
        text(
            44,
            384,
            "Median of 30+ synchronized forwards; higher is better. Fidelity is top-1 agreement with BF16.",
            11,
            MUTED,
        )
    )
    output.append("</svg>")
    return "\n".join(output)


def main() -> None:
    here = Path(__file__).parent
    svg = here / "blackwell-quantization.svg"
    svg.write_text(render(), encoding="utf-8", newline="\n")
    try:
        import cairosvg
    except (ImportError, OSError):
        return
    cairosvg.svg2png(url=str(svg), write_to=str(here / "blackwell-quantization.png"), scale=2)


if __name__ == "__main__":
    main()
