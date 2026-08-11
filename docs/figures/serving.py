"""Generate the compact LM7 model-serving figure.

    python docs/figures/serving.py

writes `lm7-model-serve.svg` and, when cairosvg is installed, a 2x PNG beside
it. Keep the labels aligned with `docs/serving.md`.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

PAGE = "#f7f9fc"
CARD = "#ffffff"
CARD_LINE = "#dde4ec"
INK = "#1a2532"
MUTED = "#68788a"
RAIL_NUM = "#b9c4d0"
NAVY = "#152441"
NAVY_INK = "#ffffff"
ARROW = "#94a3b4"
TEAL = "#0f8f8f"
GREEN_BAND = ("#eef9f1", "#5aab74")
SANS = (
    "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)

WIDTH = 1500
HEIGHT = 700
RAIL_W = 150
X0 = RAIL_W
CONTENT_W = WIDTH - X0 - 28


def text(x, y, value, size, fill, weight="400", anchor="start", spacing=None):
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    return (
        f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>'
        f"{escape(value)}</text>"
    )


def rect(x, y, width, height, fill, stroke=None, radius=12, stroke_width=1):
    border = f' stroke="{stroke}" stroke-width="{stroke_width}"' if stroke else ""
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
        f'fill="{fill}"{border}/>'
    )


def arrow(x1, y1, x2, y2):
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{ARROW}" stroke-width="1.8" '
        'fill="none" marker-end="url(#arrow)"/>'
    )


def rail(parts, y, number, label, second=""):
    parts.append(text(28, y + 34, number, 34, RAIL_NUM, "800"))
    parts.append(text(28, y + 54, label, 12, MUTED, "700", spacing="0.08em"))
    if second:
        parts.append(text(28, y + 70, second, 12, MUTED, "700", spacing="0.08em"))


def pill(parts, x, y, width, label, fill="#e8eef7", ink=INK):
    parts.append(rect(x, y, width, 34, fill, radius=17))
    parts.append(text(x + width / 2, y + 22, label, 13, ink, "700", "middle"))


def render() -> str:
    y_clients = 24
    y_serve = 186
    y_execute = 440
    centre = X0 + CONTENT_W / 2

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
            'aria-label="LM7 model serve in three layers: clients, serving layer, '
            'and model execution">'
        ),
        (
            '<defs><marker id="arrow" viewBox="0 0 8 8" refX="6" refY="4" '
            'markerWidth="5" markerHeight="5" orient="auto">'
            f'<path d="M 0 1 L 6 4 L 0 7 z" fill="{ARROW}"/></marker></defs>'
        ),
        rect(0, 0, WIDTH, HEIGHT, PAGE, radius=0),
    ]

    rail(parts, y_clients, "01", "CLIENTS")
    parts.append(rect(X0, y_clients, CONTENT_W, 110, CARD, CARD_LINE, radius=14))
    parts.append(text(X0 + 30, y_clients + 42, "OpenAI-compatible clients", 23, INK, "800"))
    parts.append(text(X0 + 30, y_clients + 72, "requests in · responses out", 14, MUTED))
    client_x = X0 + 590
    for width, label in [
        (190, "Built-in chat"),
        (180, "OpenAI SDK"),
        (240, "curl · Open WebUI"),
    ]:
        pill(parts, client_x, y_clients + 38, width, label)
        client_x += width + 18
    parts.append(arrow(centre, y_clients + 110, centre, y_serve - 8))

    rail(parts, y_serve, "02", "SERVING", "LAYER")
    parts.append(rect(X0, y_serve, CONTENT_W, 196, NAVY, radius=14))
    parts.append(text(X0 + 34, y_serve + 48, "lm7 model serve", 30, NAVY_INK, "800"))
    parts.append(
        text(
            X0 + 34,
            y_serve + 80,
            "One command puts an OpenAI-compatible API in front of a PyTorch model.",
            15,
            "#b9c7dd",
        )
    )

    card_y = y_serve + 108
    card_width = 290
    card_gap = 18
    card_x = X0 + 34
    cards = [
        ("Load model", "hf:// or local directory"),
        ("Choose hardware", 'target="auto"'),
        ("Choose compiler", 'backend="auto"'),
    ]
    for title, subtitle in cards:
        parts.append(rect(card_x, card_y, card_width, 62, CARD, radius=10))
        parts.append(text(card_x + 18, card_y + 25, title, 15, INK, "800"))
        parts.append(text(card_x + 18, card_y + 47, subtitle, 12.5, MUTED))
        card_x += card_width + card_gap

    pill(
        parts,
        card_x + 8,
        card_y + 14,
        288,
        "one model · one request at a time",
        "#243858",
        NAVY_INK,
    )
    parts.append(arrow(centre, y_serve + 196, centre, y_execute - 8))

    rail(parts, y_execute, "03", "MODEL", "EXECUTION")
    parts.append(
        rect(
            X0 - 12,
            y_execute - 18,
            CONTENT_W + 24,
            176,
            GREEN_BAND[0],
            GREEN_BAND[1],
            radius=14,
        )
    )
    parts.append(rect(X0, y_execute + 10, CONTENT_W, 118, CARD, CARD_LINE, radius=12))
    parts.append(
        text(
            X0 + 30,
            y_execute + 46,
            "LM7 compiles generation for the selected target",
            20,
            INK,
            "800",
        )
    )
    pill(parts, X0 + 30, y_execute + 68, 120, "CPU", "#e7f5ea", "#3f8b59")
    pill(parts, X0 + 168, y_execute + 68, 170, "Apple MPS", "#e7f5ea", "#3f8b59")
    pill(parts, X0 + 356, y_execute + 68, 180, "NVIDIA GPU", "#e7f5ea", "#3f8b59")
    parts.append(text(X0 + 610, y_execute + 90, "Need production throughput?", 14, MUTED, "700"))
    pill(
        parts,
        X0 + 830,
        y_execute + 68,
        430,
        "hand off to vLLM or TensorRT-LLM",
        "#eee9fb",
        "#7257bc",
    )

    parts.append(
        text(
            centre,
            HEIGHT - 24,
            "The client stays the same while LM7 chooses how and where the model runs.",
            14,
            MUTED,
            anchor="middle",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    here = Path(__file__).parent
    svg = here / "lm7-model-serve.svg"
    svg.write_text(render(), encoding="utf-8")
    print(f"wrote {svg.name} ({svg.stat().st_size:,} bytes)")
    try:
        import cairosvg
    except ImportError:
        print("cairosvg not installed; skipping PNG")
        return
    png = here / "lm7-model-serve.png"
    cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2)
    print(f"wrote {png.name} ({png.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
