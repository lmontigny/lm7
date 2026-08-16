"""Generate the Blackwell quantization benefit figure from measured aggregates."""

from pathlib import Path
from xml.sax.saxutils import escape

ROWS = [
    ("BF16", 46.8431, 16.0605, "accepted"),
    ("INT8 weight-only", 8997.056, 9.0853, "accepted"),
    ("FP8 weight-only", 57.0100, 10.4276, "rejected"),
    ("FP8 dynamic", 35.8532, 10.4234, "accepted"),
    ("FP8 dynamic rowwise", 34.4706, 10.4276, "accepted"),
    ("NVFP4 weight-only", 57.6753, 6.0277, "rejected"),
    ("NVFP4 dynamic", 25.2347, 6.0277, "rejected"),
]
BASE_LATENCY, BASE_STORAGE = ROWS[0][1], ROWS[0][2]
WIDTH, HEIGHT = 1000, 540
INK, MUTED, GRID, BLUE, GRAY, RED = "#1a2532", "#68788a", "#e8edf3", "#2a78d6", "#8d99a6", "#c44b4b"
SANS = "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def text(x, y, value, size=12, fill=INK, weight="400", anchor="start", family=SANS):
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def render():
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="1000" height="540" fill="#f7f9fc"/>',
        '<rect x="16" y="16" width="968" height="508" rx="10" fill="#fff" stroke="#dde4ec"/>',
        text(42, 50, "What quantization buys on Blackwell", 18, INK, "650"),
        text(
            42,
            72,
            "Llama-3.1-8B · BF16 baseline · batch 8 × sequence 128 · TorchInductor",
            12,
            MUTED,
        ),
        text(365, 105, "Throughput vs BF16", 13, INK, "600", "middle"),
        text(785, 105, "Model storage reduction", 13, INK, "600", "middle"),
    ]
    left_x, right_x, bar_w = 255, 680, 270
    for tick in (0, 0.5, 1.0, 1.5, 2.0):
        for x0 in (left_x, right_x):
            x = x0 + tick / 2 * bar_w
            out.append(f'<line x1="{x}" y1="120" x2="{x}" y2="455" stroke="{GRID}"/>')
            out.append(text(x, 474, f"{tick:.1f}×", 10, MUTED, anchor="middle", family=MONO))
    for i, (label, latency, storage, status) in enumerate(ROWS):
        y = 140 + i * 45
        throughput = BASE_LATENCY / latency
        reduction = BASE_STORAGE / storage
        colour = BLUE if status == "accepted" else RED
        if label == "BF16":
            colour = GRAY
        out.append(text(235, y + 5, label, 12, INK, "500", "end"))
        out.append(
            f'<rect x="{left_x}" y="{y - 12}" width="{throughput / 2 * bar_w:.1f}" height="24" rx="4" fill="{colour}"/>'
        )
        out.append(
            f'<rect x="{right_x}" y="{y - 12}" width="{reduction / 2 * bar_w:.1f}" height="24" rx="4" fill="{colour}"/>'
        )
        out.append(
            text(
                left_x + throughput / 2 * bar_w + 8,
                y + 5,
                f"{throughput:.2f}×",
                11,
                INK,
                "600",
                family=MONO,
            )
        )
        out.append(
            text(
                right_x + reduction / 2 * bar_w + 8,
                y + 5,
                f"{reduction:.2f}×",
                11,
                INK,
                "600",
                family=MONO,
            )
        )
        if status == "rejected":
            out.append(text(962, y + 5, "rejected", 10, RED, "600", "end"))
    out.extend(
        [
            text(
                42,
                505,
                "Blue: passes 4/4 top-1 fidelity gate · Red: rejected · INT8 throughput 0.005×",
                11,
                MUTED,
            ),
            "</svg>",
        ]
    )
    return "\n".join(out)


def main():
    here = Path(__file__).parent
    svg = here / "blackwell-quantization.svg"
    svg.write_text(render(), encoding="utf-8")
    try:
        import cairosvg
    except (ImportError, OSError):
        return
    cairosvg.svg2png(url=str(svg), write_to=str(here / "blackwell-quantization.png"), scale=2)


if __name__ == "__main__":
    main()
