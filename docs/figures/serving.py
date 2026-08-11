"""Generate the LM7 model-serving architecture figure.

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
BLUE_BAND = ("#eff5fd", "#4a7fd0")
PURPLE_BAND = ("#f5f2fd", "#8b6fd4")
GREEN_BAND = ("#eef9f1", "#5aab74")
SANS = (
    "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)

WIDTH = 1500
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
    border = f' stroke="{stroke}" stroke-width="{stroke_width}"' if stroke is not None else ""
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
        f'fill="{fill}"{border}/>'
    )


def line_arrow(x1, y1, x2, y2, colour=ARROW, width=1.6):
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{colour}" '
        f'stroke-width="{width}" fill="none" marker-end="url(#arrow)"/>'
    )


def rail(parts, y, number, label, second=""):
    parts.append(text(28, y + 34, number, 34, RAIL_NUM, "800"))
    parts.append(text(28, y + 54, label, 12, MUTED, "700", spacing="0.08em"))
    if second:
        parts.append(text(28, y + 70, second, 12, MUTED, "700", spacing="0.08em"))


def pill(parts, x, y, width, label, fill="#e8eef7", ink=INK):
    parts.append(rect(x, y, width, 30, fill, radius=15))
    parts.append(text(x + width / 2, y + 20, label, 12.5, ink, "700", "middle"))


def card(parts, x, y, width, height, title, subtitle, accent=TEAL):
    parts.append(rect(x, y, width, height, CARD, CARD_LINE, radius=12))
    parts.append(rect(x, y, 5, height, accent, radius=3))
    parts.append(text(x + 22, y + 31, title, 16, INK, "800"))
    parts.append(text(x + 22, y + 55, subtitle, 12.5, MUTED))


def render() -> str:
    height = 1000
    y_clients = 22
    y_api = 178
    y_runtime = 354
    y_execute = 642
    y_response = 866

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
            f'viewBox="0 0 {WIDTH} {height}" role="img" '
            'aria-label="LM7 model serve architecture: clients, API, serve runtime, '
            'execution path, and streamed response">'
        ),
        (
            '<defs><marker id="arrow" viewBox="0 0 8 8" refX="6" refY="4" '
            'markerWidth="5" markerHeight="5" orient="auto">'
            f'<path d="M 0 1 L 6 4 L 0 7 z" fill="{ARROW}"/></marker></defs>'
        ),
        rect(0, 0, WIDTH, height, PAGE, radius=0),
    ]

    rail(parts, y_clients, "01", "CLIENTS")
    parts.append(rect(X0, y_clients, CONTENT_W, 112, CARD, CARD_LINE, radius=14))
    parts.append(text(X0 + 28, y_clients + 38, "OpenAI-compatible clients", 22, INK, "800"))
    parts.append(
        text(
            X0 + 28,
            y_clients + 68,
            "Use the built-in page or tools that already speak the OpenAI API.",
            13.5,
            MUTED,
        )
    )
    client_x = X0 + 600
    for width, label in [
        (178, "Built-in chat"),
        (164, "OpenAI SDK"),
        (222, "curl · Open WebUI"),
    ]:
        pill(parts, client_x, y_clients + 40, width, label)
        client_x += width + 18
    parts.append(line_arrow(X0 + CONTENT_W / 2, y_clients + 112, X0 + CONTENT_W / 2, y_api - 8))

    rail(parts, y_api, "02", "HTTP API")
    parts.append(rect(X0, y_api, CONTENT_W, 130, NAVY, radius=14))
    parts.append(text(X0 + 32, y_api + 45, "lm7 model serve", 27, NAVY_INK, "800"))
    parts.append(
        text(X0 + 32, y_api + 75, "one model · one target · local endpoint", 14, "#b9c7dd")
    )
    pill(parts, X0 + 520, y_api + 26, 230, "/v1/chat/completions", CARD, INK)
    pill(parts, X0 + 770, y_api + 26, 205, "/v1/completions", CARD, INK)
    pill(parts, X0 + 995, y_api + 26, 155, "/v1/models", CARD, INK)
    pill(parts, X0 + 520, y_api + 75, 180, "SSE streaming", "#243858", NAVY_INK)
    pill(parts, X0 + 720, y_api + 75, 190, "sampling + stops", "#243858", NAVY_INK)
    pill(parts, X0 + 930, y_api + 75, 220, "/health · /metrics · /docs", "#243858", NAVY_INK)
    parts.append(line_arrow(X0 + CONTENT_W / 2, y_api + 130, X0 + CONTENT_W / 2, y_runtime - 8))

    rail(parts, y_runtime - 10, "03", "SERVE", "RUNTIME")
    parts.append(
        rect(
            X0 - 12,
            y_runtime - 18,
            CONTENT_W + 24,
            238,
            BLUE_BAND[0],
            BLUE_BAND[1],
            radius=14,
        )
    )
    runtime_gap = 16
    runtime_width = (CONTENT_W - 3 * runtime_gap) / 4
    runtime_cards = [
        ("Load the model", "hf:// model or local directory", "#4a7fd0"),
        ("Prepare requests", "chat template · limits · sampling", "#4a7fd0"),
        ("Own generation state", "one static KV cache", TEAL),
        ("Report what runs", "target · backend · dtype · memory", "#4a7fd0"),
    ]
    for index, (title, subtitle, accent) in enumerate(runtime_cards):
        x = X0 + index * (runtime_width + runtime_gap)
        card(parts, x, y_runtime + 16, runtime_width, 100, title, subtitle, accent)
    parts.append(rect(X0, y_runtime + 136, CONTENT_W, 56, NAVY, radius=10))
    parts.append(
        text(
            X0 + 28,
            y_runtime + 170,
            "Safety boundary: one model · one request at a time · cancellation on disconnect",
            15,
            NAVY_INK,
            "700",
        )
    )
    parts.append(line_arrow(X0 + CONTENT_W / 2, y_runtime + 220, X0 + CONTENT_W / 2, y_execute - 8))

    rail(parts, y_execute - 10, "04", "EXECUTION")
    parts.append(
        rect(
            X0 - 12,
            y_execute - 18,
            CONTENT_W + 24,
            176,
            PURPLE_BAND[0],
            PURPLE_BAND[1],
            radius=14,
        )
    )
    left_width = 790
    card(
        parts,
        X0,
        y_execute + 10,
        left_width,
        118,
        "LM7 compiled path · default",
        "compile_generation → prefill + decode graphs → static KV cache",
        TEAL,
    )
    pill(parts, X0 + 28, y_execute + 78, 116, "CPU", "#e7f5f4", TEAL)
    pill(parts, X0 + 160, y_execute + 78, 152, "Apple MPS", "#e7f5f4", TEAL)
    pill(parts, X0 + 328, y_execute + 78, 162, "NVIDIA GPU", "#e7f5f4", TEAL)
    pill(parts, X0 + 506, y_execute + 78, 236, "target=auto · backend=auto", "#e7f5f4", TEAL)

    right_x = X0 + left_width + 18
    right_width = CONTENT_W - left_width - 18
    card(
        parts,
        right_x,
        y_execute + 10,
        right_width,
        118,
        "Throughput handoff",
        "vLLM or TensorRT-LLM owns the port",
        "#8b6fd4",
    )
    pill(
        parts, right_x + 26, y_execute + 78, 190, "batching · paged attention", "#eee9fb", "#7257bc"
    )
    pill(parts, right_x + 232, y_execute + 78, 178, "LM7 steps out", "#eee9fb", "#7257bc")
    parts.append(
        line_arrow(X0 + CONTENT_W / 2, y_execute + 158, X0 + CONTENT_W / 2, y_response - 8)
    )

    rail(parts, y_response - 10, "05", "RESPONSE")
    parts.append(
        rect(
            X0 - 12,
            y_response - 18,
            CONTENT_W + 24,
            108,
            GREEN_BAND[0],
            GREEN_BAND[1],
            radius=14,
        )
    )
    parts.append(rect(X0, y_response + 8, CONTENT_W, 56, CARD, CARD_LINE, radius=12))
    parts.append(text(X0 + 28, y_response + 43, "OpenAI-compatible response", 18, INK, "800"))
    pill(parts, X0 + 490, y_response + 21, 170, "buffered JSON", "#e7f5ea", "#3f8b59")
    pill(parts, X0 + 680, y_response + 21, 190, "SSE token stream", "#e7f5ea", "#3f8b59")
    pill(
        parts,
        X0 + 890,
        y_response + 21,
        260,
        "same client · different hardware",
        "#e7f5ea",
        "#3f8b59",
    )

    parts.append(
        text(
            X0 + CONTENT_W / 2,
            height - 18,
            "Built-in serving is for local validation. Use a delegated engine when concurrency and throughput matter.",
            13,
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
