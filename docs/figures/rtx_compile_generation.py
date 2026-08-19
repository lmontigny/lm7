"""Generate the README RTX compiled-generation speedup figure."""

from pathlib import Path
from xml.sax.saxutils import escape

PAGE, CARD, LINE = "#f7f9fc", "#ffffff", "#dde4ec"
INK, MUTED, GRID = "#1a2532", "#68788a", "#eef2f7"
BATCH_COLOURS = {1: "#2a78d6", 4: "#35a585", 8: "#d38b28"}
SANS = "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
WIDTH, HEIGHT = 900, 520
PLOT_X, PLOT_Y, PLOT_W, PLOT_BOTTOM, X_MAX = 210, 118, 580, 418, 6.2
BAR_H = 12

# Published RTX 4070 SUPER end-to-end speedups. The boolean marks cells where
# the first sixteen greedy tokens differed from eager in BF16. None is the
# 8,192-token, batch-8 shape that did not finish on the 12 GiB card.
ROWS = (
    ("512", {1: (5.93, False), 4: (4.25, False), 8: (3.42, False)}),
    ("1,024", {1: (4.55, False), 4: (3.35, False), 8: (3.01, False)}),
    ("2,048", {1: (3.76, False), 4: (2.50, False), 8: (1.55, False)}),
    ("4,096", {1: (2.71, False), 4: (1.51, False), 8: (1.08, True)}),
    ("8,192", {1: (1.94, True), 4: (1.06, True), 8: None}),
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
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Grouped horizontal bar chart '
            f"showing end-to-end compiled generation speedup by prompt length and batch size on an "
            f'RTX 4070 SUPER">'
        ),
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{PAGE}"/>',
        (
            f'<rect x="16" y="16" width="{WIDTH - 32}" height="{HEIGHT - 32}" rx="10" '
            f'fill="{CARD}" stroke="{LINE}"/>'
        ),
        text(44, 50, "Compiled generation speedup on RTX 4070 SUPER", 17, INK, "650"),
        text(
            44,
            72,
            "Llama-3.2-1B · BF16 · static KV cache · 100 generated tokens · end-to-end",
            12,
            MUTED,
        ),
        text(44, 109, "prompt tokens", 11, MUTED, "500"),
    ]

    legend_x = 470
    for index, batch in enumerate(BATCH_COLOURS):
        x = legend_x + index * 112
        output.append(
            f'<rect x="{x}" y="91" width="16" height="10" rx="2" fill="{BATCH_COLOURS[batch]}"/>'
        )
        output.append(text(x + 23, 101, f"batch {batch}", 11, MUTED, "500"))

    for tick in range(7):
        x = PLOT_X + tick * scale
        output.append(
            f'<line x1="{x:.1f}" y1="{PLOT_Y}" x2="{x:.1f}" y2="{PLOT_BOTTOM}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        output.append(text(x, 440, f"{tick}×", 11, MUTED, anchor="middle", family=MONO))

    baseline_x = PLOT_X + scale
    output.append(
        f'<line x1="{baseline_x:.1f}" y1="{PLOT_Y}" x2="{baseline_x:.1f}" '
        f'y2="{PLOT_BOTTOM}" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="4 4"/>'
    )
    output.append(text(baseline_x + 6, 130, "eager parity", 10, MUTED, "500"))

    offsets = {1: -16, 4: 0, 8: 16}
    for row_index, (prompt, batches) in enumerate(ROWS):
        centre_y = 154 + row_index * 61
        output.append(
            text(PLOT_X - 20, centre_y + 4, prompt, 12, INK, "600", anchor="end", family=MONO)
        )
        for batch, offset in offsets.items():
            y = centre_y + offset
            result = batches[batch]
            if result is None:
                output.append(text(PLOT_X + 9, y + 4, "not completed", 10, MUTED, "500"))
                continue
            speedup, diverged = result
            width = speedup * scale
            colour = BATCH_COLOURS[batch]
            opacity = 0.45 if diverged else 1.0
            stroke = (
                f' stroke="{colour}" stroke-width="1.5" stroke-dasharray="3 2"' if diverged else ""
            )
            output.append(
                f'<rect x="{PLOT_X}" y="{y - BAR_H / 2:.1f}" width="{width:.1f}" '
                f'height="{BAR_H}" rx="3" fill="{colour}" fill-opacity="{opacity}"{stroke}/>'
            )
            suffix = "†" if diverged else ""
            output.append(
                text(
                    PLOT_X + width + 8,
                    y + 4,
                    f"{speedup:.2f}×{suffix}",
                    10,
                    INK,
                    "600",
                    family=MONO,
                )
            )

    output.append(
        text(
            PLOT_X + PLOT_W / 2,
            466,
            "speedup over eager generation, including prompt prefill (higher is better)",
            11,
            MUTED,
            anchor="middle",
        )
    )
    output.append(
        text(
            44,
            495,
            "† BF16 greedy tokens diverged from eager; outlined bars are performance observations only.",
            10,
            MUTED,
        )
    )
    output.append("</svg>")
    return "\n".join(output)


def main() -> None:
    here = Path(__file__).parent
    svg = here / "rtx-compiled-generation.svg"
    svg.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {svg.name} ({svg.stat().st_size:,} bytes)")
    try:
        import cairosvg
    except (ImportError, OSError):
        print("cairosvg unavailable; skipping PNG")
        return
    png = here / "rtx-compiled-generation.png"
    cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2)
    print(f"wrote {png.name} ({png.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
