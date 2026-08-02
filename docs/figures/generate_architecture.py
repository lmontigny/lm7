"""Generate the layered architecture figure used at the top of the README.

The figure is written rather than drawn so that it stays truthful: every
backend, target string, and runtime below comes from this file's tables, which
mirror `src/lm7/targets.py`, each backend's accepted vendors, and the
`COMPILED_*_NAME` payloads in `src/lm7/exporting.py`.

    python docs/figures/generate_architecture.py

writes four SVGs beside this script -- one per group (compute, accelerators)
per theme (light, dark). The README picks between the themes with a `<picture>`
element, so the figure follows GitHub's light and dark modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

# --- palette ---------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    name: str
    page: str
    band: str
    card: str
    card_line: str
    ink: str
    muted: str
    rail: str
    orchestrator: str
    orchestrator_ink: str
    accent: str
    rail_num: str
    band_backend: str
    line_backend: str
    band_runtime: str
    line_runtime: str
    band_hw: str
    line_hw: str


LIGHT = Theme(
    name="light",
    page="#ffffff",
    band="#f6f8fa",
    card="#ffffff",
    card_line="#d5dde5",
    ink="#16202b",
    muted="#63737f",
    rail="#8896a4",
    orchestrator="#1c2b45",
    orchestrator_ink="#ffffff",
    accent="#2563eb",
    rail_num="#c3ccd6",
    band_backend="#eef4fb",
    line_backend="#cddef2",
    band_runtime="#f3f0fb",
    line_runtime="#ddd5f3",
    band_hw="#eef8f1",
    line_hw="#cfe8d8",
)

DARK = Theme(
    name="dark",
    page="#0d1117",
    band="#161b22",
    card="#1b222b",
    card_line="#39414b",
    ink="#e6edf3",
    muted="#98a5b3",
    rail="#6f7f8f",
    orchestrator="#132033",
    orchestrator_ink="#e6edf3",
    accent="#6ea8fe",
    rail_num="#3b4652",
    band_backend="#131c28",
    line_backend="#26384d",
    band_runtime="#181628",
    line_runtime="#332c4d",
    band_hw="#101f18",
    line_hw="#23402f",
)

# Badge colours are shared: they encode execution mode, not vendor.
BADGES = {
    "JIT": ("#0f7b3d", "#dcfce7"),
    "AOT": ("#8a5200", "#fef3c7"),
    "EXPORT": ("#7c2d92", "#f3e8ff"),
}
BADGE_DARK = {
    "JIT": ("#7ee2a8", "#123122"),
    "AOT": ("#f0c674", "#33270d"),
    "EXPORT": ("#dda9f0", "#2c1636"),
}


@dataclass(frozen=True)
class Hardware:
    """One piece of silicon, the target string that reaches it, and its backends."""

    label: str
    detail: str
    target: str
    backends: list[tuple[str, str]]


@dataclass(frozen=True)
class Column:
    vendor: str
    tint: str
    logo: str
    hardware: list[Hardware]
    runtime: list[str]
    footnote: str = ""
    extra: list[tuple[str, str]] = field(default_factory=list)


# --- the actual content ----------------------------------------------------
#
# `eager` is omitted from every column: it is the fallback for all of them and
# saying so once below the figure is clearer than repeating it eight times.

COLUMNS = [
    Column(
        vendor="Intel",
        logo="intel",
        tint="#0068b5",
        hardware=[
            Hardware(
                "CPU",
                "x86-64",
                "cpu",
                [
                    ("inductor", "JIT"),
                    ("aot_inductor", "AOT"),
                    ("openvino", "AOT"),
                    ("onnxruntime", "AOT"),
                    ("tvm", "JIT"),
                ],
            ),
            Hardware("GPU", "Arc, XPU", "intel", [("inductor", "JIT"), ("iree_vulkan", "EXPORT")]),
            Hardware("NPU", "Core Ultra AI Boost", "intel:npu", [("openvino", "AOT")]),
        ],
        runtime=["C++ / OpenMP", "oneDNN", "SYCL (XPU)", "OpenVINO IR", "Vulkan / SPIR-V"],
    ),
    Column(
        vendor="AMD",
        logo="amd",
        tint="#c8102e",
        hardware=[
            Hardware(
                "CPU",
                "x86-64, EPYC / Ryzen",
                "cpu",
                [
                    ("inductor", "JIT"),
                    ("aot_inductor", "AOT"),
                    ("zentorch", "JIT"),
                    ("onnxruntime", "AOT"),
                    ("tvm", "JIT"),
                ],
            ),
            Hardware("GPU", "ROCm", "amd", [("inductor", "JIT"), ("iree_vulkan", "EXPORT")]),
        ],
        runtime=["C++ / OpenMP", "ZenDNN", "Triton / ROCm, HIP", "Vulkan / SPIR-V"],
    ),
    Column(
        vendor="NVIDIA",
        logo="nvidia",
        tint="#76b900",
        hardware=[
            Hardware(
                "GPU",
                "Ampere and newer",
                "nvidia",
                [
                    ("inductor", "JIT"),
                    ("aot_inductor", "AOT"),
                    ("tensorrt", "AOT"),
                    ("onnxruntime", "AOT"),
                    ("iree_vulkan", "EXPORT"),
                ],
            ),
        ],
        runtime=["Triton / CUDA", "cuBLAS, cuDNN", "TensorRT engine", "Vulkan / SPIR-V"],
    ),
    Column(
        vendor="Apple",
        logo="apple",
        tint="#6e6e73",
        hardware=[
            Hardware(
                "CPU",
                "Apple silicon, ARM64",
                "cpu",
                [
                    ("inductor", "JIT"),
                    ("aot_inductor", "AOT"),
                    ("onnxruntime", "AOT"),
                    ("tvm", "JIT"),
                ],
            ),
            Hardware("GPU", "Metal", "apple", [("inductor", "JIT"), ("aot_inductor", "AOT")]),
        ],
        runtime=["C++ / OpenMP", "Metal / MPS"],
    ),
    Column(
        vendor="Arm",
        logo="arm",
        tint="#0091bd",
        hardware=[
            Hardware(
                "CPU",
                "ARM64 servers, SBCs",
                "cpu",
                [
                    ("inductor", "JIT"),
                    ("aot_inductor", "AOT"),
                    ("onnxruntime", "AOT"),
                    ("tvm", "JIT"),
                    ("executorch", "EXPORT"),
                    ("litert", "EXPORT"),
                ],
            ),
        ],
        runtime=["C++ / OpenMP", "XNNPACK (NEON, KleidiAI)", "ExecuTorch runtime"],
        footnote="Same cpu target as x86-64",
    ),
    Column(
        vendor="Google",
        logo="google",
        tint="#4285f4",
        hardware=[
            Hardware(
                "TPU", "via PyTorch/XLA", "tpu", [("openxla", "JIT"), ("stablehlo", "EXPORT")]
            ),
        ],
        runtime=["PJRT plugin", "OpenXLA", "StableHLO"],
    ),
    Column(
        vendor="Tenstorrent",
        logo="",
        tint="#7c68ee",
        hardware=[
            Hardware(
                "Wormhole / Blackhole",
                "PCIe accelerator cards",
                "tenstorrent",
                [("tenstorrent", "JIT"), ("stablehlo", "EXPORT")],
            ),
        ],
        runtime=["tt-xla (PJRT)", "tt-mlir", "tt-metal"],
    ),
    Column(
        vendor="Qualcomm",
        logo="qualcomm",
        tint="#3253dc",
        hardware=[
            Hardware("Snapdragon HTP", "8 Elite, v79", "qualcomm:sm8750", [("qnn", "EXPORT")]),
            Hardware(
                "Snapdragon CPU",
                "ARM64 phones",
                "cpu",
                [("executorch", "EXPORT"), ("litert", "EXPORT")],
            ),
        ],
        runtime=["ExecuTorch runtime", "QNN SDK, Hexagon HTP", "XNNPACK, LiteRT"],
        footnote="Export only; runs off-host",
    ),
]

# Brand marks from Simple Icons (https://simpleicons.org). The path data is
# CC0; the marks themselves remain the trademarks of their respective owners.
# Each entry is the brand colour and a 24x24 viewBox path.
LOGOS: dict[str, tuple[str, str]] = {
    "pytorch": (
        "#EE4C2C",
        (
            "M12.005 0L4.952 7.053a9.865 9.865 0 000 14.022 9.866 9.866 0 0014.022 0c3.984-3.9 3."
            "986-10.205.085-14.023l-1.744 1.743c2.904 2.905 2.904 7.634 0 10.538s-7.634 2.904-10."
            "538 0-2.904-7.634 0-10.538l4.647-4.646.582-.665zm3.568 3.899a1.327 1.327 0 00-1.327 "
            "1.327 1.327 1.327 0 001.327 1.328A1.327 1.327 0 0016.9 5.226 1.327 1.327 0 0015.573 "
            "3.9z"
        ),
    ),
    "intel": (
        "#0071C5",
        (
            "M20.42 7.345v9.18h1.651v-9.18zM0 7.475v1.737h1.737V7.474zm9.78.352v6.053c0 .513.044."
            "945.13 1.292.087.34.235.618.44.828.203.21.475.359.803.451.334.093.754.136 1.255.136h"
            ".216v-1.533c-.24 0-.445-.012-.593-.037a.672.672 0 0 1-.39-.173.693.693 0 0 1-.173-.3"
            "77 4.002 4.002 0 0 1-.037-.606v-2.182h1.193v-1.416h-1.193V7.827zm-3.505 2.312c-.396 "
            "0-.76.08-1.082.241-.327.161-.6.384-.822.668l-.087.117v-.902H2.658v6.256h1.639v-3.214"
            "c.018-.588.16-1.02.433-1.299.29-.297.642-.445 1.044-.445.476 0 .841.149 1.082.433.23"
            "5.284.359.686.359 1.2v3.324h1.663V12.97c.006-.89-.229-1.595-.686-2.09-.458-.495-1.1-"
            ".742-1.917-.742zm10.065.006a3.252 3.252 0 0 0-2.306.946c-.29.29-.525.637-.692 1.033a"
            "3.145 3.145 0 0 0-.254 1.273c0 .452.08.878.241 1.274.161.395.39.742.674 1.032.284.29"
            ".637.526 1.045.693.408.173.86.26 1.342.26 1.397 0 2.262-.637 2.782-1.23l-1.187-.904c"
            "-.248.297-.841.699-1.583.699-.464 0-.847-.105-1.138-.321a1.588 1.588 0 0 1-.593-.872"
            "l-.019-.056h4.915v-.587c0-.451-.08-.872-.235-1.267a3.393 3.393 0 0 0-.661-1.033 3.01"
            "3 3.013 0 0 0-1.02-.692 3.345 3.345 0 0 0-1.311-.248zm-16.297.118v6.256h1.651v-6.256"
            "zm16.278 1.286c1.132 0 1.664.797 1.664 1.255l-3.32.006c0-.458.525-1.255 1.656-1.261z"
            "m7.073 3.814a.606.606 0 0 0-.606.606.606.606 0 0 0 .606.606.606.606 0 0 0 .606-.606."
            "606.606 0 0 0-.606-.606zm-.008.105a.5.5 0 0 1 .002 0 .5.5 0 0 1 .5.501.5.5 0 0 1-.5."
            "5.5.5 0 0 1-.5-.5.5.5 0 0 1 .498-.5zm-.233.155v.699h.13v-.285h.093l.173.285h.136l-.1"
            "8-.297a.191.191 0 0 0 .118-.056c.03-.03.05-.074.05-.136 0-.068-.02-.117-.063-.154-.0"
            "37-.038-.105-.056-.185-.056zm.13.099h.154c.019 0 .037.006.056.012a.064.064 0 0 1 .03"
            "7.031c.013.013.012.031.012.056a.124.124 0 0 1-.012.055.164.164 0 0 1-.037.031c-.019."
            "006-.037.013-.056.013h-.154Z"
        ),
    ),
    "amd": (
        "#ED1C24",
        (
            "M18.324 9.137l1.559 1.56h2.556v2.557L24 14.814V9.137zM2 9.52l-2 4.96h1.309l.37-.982H"
            "3.9l.408.982h1.338L3.432 9.52zm4.209 0v4.955h1.238v-3.092l1.338 1.562h.188l1.338-1.5"
            "56v3.091h1.238V9.52H10.47l-1.592 1.845L7.287 9.52zm6.283 0v4.96h2.057c1.979 0 2.88-1"
            ".046 2.88-2.472 0-1.36-.937-2.488-2.747-2.488zm1.237.91h.792c1.17 0 1.63.711 1.63 1."
            "57 0 .728-.372 1.572-1.616 1.572h-.806zm-10.985.273l.791 1.932H2.008zm17.137.307l-1."
            "604 1.603v2.25h2.246l1.604-1.607h-2.246z"
        ),
    ),
    "nvidia": (
        "#76B900",
        (
            "M8.948 8.798v-1.43a6.7 6.7 0 0 1 .424-.018c3.922-.124 6.493 3.374 6.493 3.374s-2.774"
            " 3.851-5.75 3.851c-.398 0-.787-.062-1.158-.185v-4.346c1.528.185 1.837.857 2.747 2.38"
            "5l2.04-1.714s-1.492-1.952-4-1.952a6.016 6.016 0 0 0-.796.035m0-4.735v2.138l.424-.027"
            "c5.45-.185 9.01 4.47 9.01 4.47s-4.08 4.964-8.33 4.964c-.37 0-.733-.035-1.095-.097v1."
            "325c.3.035.61.062.91.062 3.957 0 6.82-2.023 9.593-4.408.459.371 2.34 1.263 2.73 1.65"
            "2-2.633 2.208-8.772 3.984-12.253 3.984-.335 0-.653-.018-.971-.053v1.864H24V4.063zm0 "
            "10.326v1.131c-3.657-.654-4.673-4.46-4.673-4.46s1.758-1.944 4.673-2.262v1.237H8.94c-1"
            ".528-.186-2.73 1.245-2.73 1.245s.68 2.412 2.739 3.11M2.456 10.9s2.164-3.197 6.5-3.53"
            "3V6.201C4.153 6.59 0 10.653 0 10.653s2.35 6.802 8.948 7.42v-1.237c-4.84-.6-6.492-5.9"
            "36-6.492-5.936z"
        ),
    ),
    "apple": (
        "#000000",
        (
            "M12.152 6.896c-.948 0-2.415-1.078-3.96-1.04-2.04.027-3.91 1.183-4.961 3.014-2.117 3."
            "675-.546 9.103 1.519 12.09 1.013 1.454 2.208 3.09 3.792 3.039 1.52-.065 2.09-.987 3."
            "935-.987 1.831 0 2.35.987 3.96.948 1.637-.026 2.676-1.48 3.676-2.948 1.156-1.688 1.6"
            "36-3.325 1.662-3.415-.039-.013-3.182-1.221-3.22-4.857-.026-3.04 2.48-4.494 2.597-4.5"
            "59-1.429-2.09-3.623-2.324-4.39-2.376-2-.156-3.675 1.09-4.61 1.09zM15.53 3.83c.843-1."
            "012 1.4-2.427 1.245-3.83-1.207.052-2.662.805-3.532 1.818-.78.896-1.454 2.338-1.273 3"
            ".714 1.338.104 2.715-.688 3.559-1.701"
        ),
    ),
    "arm": (
        "#0091BD",
        (
            "M5.419 8.534h1.614v6.911H5.419v-.72c-.71.822-1.573.933-2.07.933C1.218 15.658 0 13.88"
            "2 0 11.985c0-2.253 1.542-3.633 3.37-3.633.507 0 1.4.132 2.049.984zm-3.765 3.491c0 1."
            "198.751 2.202 1.918 2.202 1.015 0 1.959-.74 1.959-2.181 0-1.512-.934-2.233-1.959-2.2"
            "33-1.167-.01-1.918.974-1.918 2.212zm7.297-3.49h1.613v.618a3 3 0 0 1 .67-.578c.314-.1"
            "83.619-.233.984-.233.396 0 .822.06 1.269.324l-.66 1.462a1.432 1.432 0 0 0-.822-.244c"
            "-.345 0-.69.05-1.005.376-.446.477-.446 1.136-.446 1.593v3.582H8.94zm5.56 0h1.614v.63"
            "9c.538-.66 1.177-.822 1.705-.822.72 0 1.4.345 1.786 1.015.579-.822 1.441-1.015 2.05-"
            "1.015.842 0 1.573.396 1.969 1.086.132.233.365.74.365 1.745v4.272h-1.614V11.65c0-.771"
            "-.08-1.086-.152-1.228-.101-.264-.345-.609-.923-.609-.396 0-.741.213-.954.508-.284.39"
            "5-.315.984-.315 1.572v3.562H18.43V11.65c0-.771-.081-1.086-.152-1.228-.102-.264-.345-"
            ".609-.924-.609-.396 0-.74.213-.954.508-.284.395-.314.984-.314 1.572v3.562h-1.573z"
        ),
    ),
    "google": (
        "#4285F4",
        (
            "M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 1.147-2.933 2.4-6.053 "
            "2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.3"
            "07-2.307C18.747 1.44 16.133 0 12.48 0 5.867 0 .307 5.387.307 12s5.56 12 12.173 12c3."
            "573 0 6.267-1.173 8.373-3.36 2.16-2.16 2.84-5.213 2.84-7.667 0-.76-.053-1.467-.173-2"
            ".053H12.48z"
        ),
    ),
    "qualcomm": (
        "#3253DC",
        (
            "M12 0C6.22933 0 1.5761 4.48645 1.5761 10.47394c0 6.00417 4.65323 10.47394 10.4239 10"
            ".47394.98402 0 1.93468-.13343 2.8353-.3836l1.13412 2.9187c.11675.31688.35025.51702.7"
            "672.51702h1.80125c.43364 0 .75052-.28353.55038-.83391l-1.46768-3.81932c2.88534-1.817"
            "93 4.80333-5.03683 4.80333-8.8895C22.4239 4.48644 17.77067 0 12 0m4.53648 16.5615l-1"
            ".31758-3.41904c-.11675-.28353-.35024-.55038-.85059-.55038h-1.71786c-.43363 0-.7672.2"
            "8353-.56706.83391l1.73454 4.48645c-.56706.1501-1.18416.21682-1.81793.21682-4.2196 0-"
            "7.22168-3.31897-7.22168-7.65532C4.77832 6.1376 7.7804 2.81862 12 2.81862s7.22168 3.3"
            "1898 7.22168 7.65532c0 2.5351-1.01737 4.70327-2.6852 6.08756"
        ),
    ),
}


# --- geometry --------------------------------------------------------------

COL_W = 192
COL_GAP = 11
PAD = 26
RAIL_W = 86
TOP = 24

FONT = (
    "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def _t(x, y, text, size, fill, weight="400", anchor="start", family=FONT, spacing=None):
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>'
        f"{escape(text)}</text>"
    )


def _rect(x, y, w, h, fill, stroke=None, r=7, width=1, dash=None):
    s = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"{s}{d}/>'


def _logo(key: str, x: float, y: float, size: float, theme: Theme) -> str:
    """Place a 24x24 brand mark. Apple's mark is black, so it is lightened on dark."""
    colour, path = LOGOS[key]
    if theme.name == "dark" and colour.upper() in {"#000000", "#0F0F0F"}:
        colour = "#e6edf3"
    scale = size / 24
    return (
        f'<g transform="translate({x} {y}) scale({scale:.4f})">'
        f'<path d="{path}" fill="{colour}"/></g>'
    )


def _circle_num(cx: float, cy: float, n: int, theme: Theme) -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="9" fill="{theme.orchestrator}"/>' + _t(
        cx, cy + 3.5, str(n), 9.5, theme.orchestrator_ink, "700", "middle", MONO
    )


def _badge(x, y, kind, theme):
    fg, bg = (BADGE_DARK if theme.name == "dark" else BADGES)[kind]
    w = 44 if kind == "EXPORT" else 27
    out = [
        _rect(x, y - 8, w, 12, bg, r=3),
        _t(x + w / 2, y + 1, kind, 7.5, fg, "700", "middle", MONO, "0.03em"),
    ]
    return "".join(out), w


def _arrow(x1, y1, x2, y2, colour, marker="arrow"):
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{colour}" stroke-width="1.4" '
        f'fill="none" marker-end="url(#{marker})"/>'
    )


def render(theme: Theme) -> str:
    columns = COLUMNS
    n = len(columns)
    grid_w = n * COL_W + (n - 1) * COL_GAP
    width = RAIL_W + grid_w + PAD * 2
    x0 = PAD + RAIL_W

    # measure the backend layer: tallest column decides the band height
    def hw_block_h(hw: Hardware) -> int:
        return 44 + len(hw.backends) * 15 + 8

    backend_h = (
        max(sum(hw_block_h(h) for h in c.hardware) + 10 * (len(c.hardware) - 1) for c in columns)
        + 22
    )
    runtime_h = max(len(c.runtime) for c in columns) * 15 + 26
    hw_h = 70

    y_model = TOP
    model_h = 60
    y_orch = y_model + model_h + 30
    orch_h = 88
    y_backend = y_orch + orch_h + 40
    y_runtime = y_backend + backend_h + 26
    y_hw = y_runtime + runtime_h + 26
    height = y_hw + hw_h + PAD + 30

    s: list[str] = []
    s.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="LM7 architecture: model, orchestrator, backends, lowering and runtime, hardware">'
    )
    s.append(
        f'<defs><marker id="arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" '
        f'markerHeight="5" orient="auto"><path d="M 0 1 L 6 4 L 0 7 z" fill="{theme.rail}"/>'
        f"</marker></defs>"
    )
    s.append(_rect(0, 0, width, height, theme.page, r=0))

    def rail(y, num, label):
        s.append(_t(PAD, y + 16, num, 19, theme.rail_num, "800", family=MONO))
        s.append(_t(PAD, y + 31, label, 7.5, theme.muted, "700", spacing="0.1em"))

    # 01 model
    rail(y_model, "01", "MODEL")
    s.append(_rect(x0, y_model, grid_w, model_h, theme.card, theme.card_line))
    s.append(_logo("pytorch", x0 + 18, y_model + 17, 26, theme))
    s.append(_t(x0 + 54, y_model + 26, "PyTorch / Hugging Face model", 15, theme.ink, "700"))
    s.append(
        _t(
            x0 + 54,
            y_model + 44,
            "nn.Module, or an lm7 hf:// model id, with representative inputs",
            10,
            theme.muted,
        )
    )
    s.append(_t(x0 + grid_w - 20, y_model + 26, "LM7", 15, theme.accent, "800", "end"))
    s.append(
        _t(
            x0 + grid_w - 20,
            y_model + 44,
            "PyTorch-first compiler orchestrator",
            9.5,
            theme.muted,
            anchor="end",
        )
    )
    s.append(_arrow(x0 + grid_w / 2, y_model + model_h, x0 + grid_w / 2, y_orch - 4, theme.rail))

    # 02 orchestrator
    rail(y_orch, "02", "ORCHESTRATOR")
    s.append(_rect(x0, y_orch, grid_w, orch_h, theme.orchestrator, r=8))
    s.append(_t(x0 + 20, y_orch + 30, "lm7", 20, theme.orchestrator_ink, "800"))
    s.append(
        _t(x0 + 74, y_orch + 30, "one model, one target string", 10, theme.orchestrator_ink, "500")
    )
    s.append(
        _t(
            x0 + 20,
            y_orch + 72,
            "target detection  ·  backend selection  ·  shape and compile cache  ·  "
            "safe eager fallback  ·  artifact manifest",
            8.5,
            theme.orchestrator_ink,
            "500",
        )
    )
    for i, (call, note) in enumerate(
        [("lm7.compile()", "run it here"), ("lm7.export()", "ship an artifact")]
    ):
        bx = x0 + grid_w - 286 + i * 145
        s.append(_rect(bx, y_orch + 10, 134, 40, theme.card, theme.card_line, r=6))
        s.append(_t(bx + 67, y_orch + 27, call, 11, theme.ink, "700", "middle", MONO))
        s.append(_t(bx + 67, y_orch + 41, note, 8.5, theme.muted, anchor="middle"))

    # bands
    s.append(
        _rect(
            x0 - 10,
            y_backend - 16,
            grid_w + 20,
            backend_h + 18,
            theme.band_backend,
            theme.line_backend,
            r=10,
        )
    )
    s.append(
        _rect(
            x0 - 10,
            y_runtime - 14,
            grid_w + 20,
            runtime_h + 16,
            theme.band_runtime,
            theme.line_runtime,
            r=10,
        )
    )
    s.append(_rect(x0 - 10, y_hw - 14, grid_w + 20, hw_h + 16, theme.band_hw, theme.line_hw, r=10))

    rail(y_backend - 8, "03", "BACKENDS")
    rail(y_runtime - 6, "04", "LOWERING")
    rail(y_hw - 6, "05", "HARDWARE")

    for i, col in enumerate(columns):
        cx = x0 + i * (COL_W + COL_GAP)
        s.append(
            _arrow(cx + COL_W / 2, y_orch + orch_h, cx + COL_W / 2, y_backend - 18, theme.rail)
        )

        # 03 backends, one card per piece of silicon
        y = y_backend
        for hw in col.hardware:
            h = hw_block_h(hw)
            s.append(_rect(cx, y, COL_W, h, theme.card, theme.card_line))
            s.append(_t(cx + 10, y + 16, hw.label, 10.5, theme.ink, "700"))
            s.append(_t(cx + 10, y + 27, hw.detail, 7.5, theme.muted))
            s.append(_t(cx + 10, y + 39, f"target={hw.target}", 8.5, col.tint, "700", family=MONO))
            by = y + 54
            for name, kind in hw.backends:
                s.append(_t(cx + 10, by, name, 9.5, theme.ink, family=MONO))
                badge, _ = _badge(cx + COL_W - 52, by, kind, theme)
                s.append(badge)
                by += 15
            y += h + 10
        s.append(
            _arrow(
                cx + COL_W / 2, y_backend + backend_h - 8, cx + COL_W / 2, y_runtime - 6, theme.rail
            )
        )

        # 04 lowering and runtime
        s.append(_rect(cx, y_runtime, COL_W, runtime_h, theme.card, theme.card_line))
        ry = y_runtime + 19
        for line in col.runtime:
            s.append(_t(cx + 10, ry, line, 9.5, theme.ink, family=MONO))
            ry += 15
        s.append(
            _arrow(cx + COL_W / 2, y_runtime + runtime_h, cx + COL_W / 2, y_hw - 6, theme.rail)
        )

        # 05 hardware
        s.append(_rect(cx, y_hw, COL_W, hw_h, theme.card, theme.card_line))
        s.append(_rect(cx, y_hw, 4, hw_h, col.tint, r=2))
        s.append(_circle_num(cx + 20, y_hw + 20, i + 1, theme))
        if col.logo:
            s.append(_logo(col.logo, cx + 38, y_hw + 10, 20, theme))
            s.append(_t(cx + 64, y_hw + 25, col.vendor, 12.5, theme.ink, "700"))
        else:
            s.append(_t(cx + 38, y_hw + 25, col.vendor, 12.5, theme.ink, "700"))
        parts = ", ".join(h.label for h in col.hardware)
        s.append(_t(cx + 14, y_hw + 45, parts, 8.5, theme.muted))
        if col.footnote:
            s.append(_t(cx + 14, y_hw + 58, col.footnote, 7.5, col.tint, "700"))

    s.append(
        _t(
            x0 + grid_w / 2,
            height - 22,
            "JIT compiles in-process  ·  AOT also writes a reloadable .lm7  ·  "
            "EXPORT cannot run in-process",
            8.5,
            theme.muted,
            anchor="middle",
        )
    )
    s.append(
        _t(
            x0 + grid_w / 2,
            height - 10,
            "eager is the fallback on every target and is omitted above",
            8.5,
            theme.muted,
            anchor="middle",
        )
    )
    s.append("</svg>")
    return "\n".join(s)


def main() -> None:
    here = Path(__file__).parent
    for theme in (LIGHT, DARK):
        path = here / f"lm7-architecture-{theme.name}.svg"
        path.write_text(render(theme), encoding="utf-8")
        print(f"wrote {path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
