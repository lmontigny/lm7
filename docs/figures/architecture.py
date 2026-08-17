"""Generate the layered architecture figure at the top of the README.

The figure is written rather than drawn so it stays truthful: the backends and
badges below mirror `src/lm7/backends/*`, the vendor set in `src/lm7/targets.py`,
and `EXPORT_BACKENDS` in `src/lm7/exporting.py`. Correcting the figure is an edit
to one table here, not a redraw.

    python docs/figures/architecture.py

writes `lm7-architecture.svg` and, when cairosvg is available, a 2x
`lm7-architecture.png` beside this script. The README uses the PNG: it renders
identically everywhere, where an SVG depends on the fonts the viewer happens to
have and can reflow.

    pip install cairosvg
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

# --- palette ---------------------------------------------------------------

PAGE = "#f7f9fc"
CARD = "#ffffff"
CARD_LINE = "#dde4ec"
INK = "#1a2532"
MUTED = "#68788a"
RAIL_NUM = "#b9c4d0"
NAVY = "#152441"
NAVY_INK = "#ffffff"
ARROW = "#94a3b4"
JIT_LINE = "#0f8f8f"

BAND_BACKEND = ("#eff5fd", "#4a7fd0")
BAND_LOWER = ("#f5f2fd", "#8b6fd4")
BAND_HW = ("#eef9f1", "#5aab74")

# Badges encode when a backend runs, not which vendor it serves.
BADGE = {
    "JIT": ("#ffffff", "#12857f"),
    "AOT": ("#ffffff", "#7b8794"),
    "J+A": ("#ffffff", "#3b6fc4"),
    "JIT*": ("#ffffff", "#12857f"),
    "EXPORT": ("#ffffff", "#7b8794"),
}


@dataclass(frozen=True)
class Silicon:
    """One piece of silicon in a vendor column, and what compiles for it."""

    label: str
    detail: str
    target: str
    backends: list[tuple[str, str]]


@dataclass(frozen=True)
class Column:
    """One vendor: its silicon, what lowers beneath it, and what it runs on."""

    vendor: str
    tint: str
    logo: str
    silicon: list[Silicon]
    runtime: list[str]
    hardware: str
    hardware_detail: str
    extra: list[str] = field(default_factory=list)


# --- content ---------------------------------------------------------------
#
# Grouped by silicon rather than by vendor alone, because the target string is
# what selects it and a vendor has more than one. Intel's CPU is `cpu` like
# everyone else's; `intel` is the GPU and `intel:npu` the NPU.
#
# Backend names are the strings you pass to `backend=`, not prettified labels.
#
# JIT   compiles in this process
# AOT   writes a reloadable .lm7 artifact
# J+A   compiles in-process and also writes an artifact
# *     explicit opt-in; automatic selection never picks it

COLUMNS = [
    Column(
        vendor="Intel",
        tint="#0071c5",
        logo="intel",
        silicon=[
            Silicon(
                "CPU",
                "x86-64",
                "cpu",
                [
                    ("inductor", "JIT"),
                    ("aot_inductor", "AOT"),
                    ("openvino", "J+A"),
                    ("onnxruntime", "J+A"),
                    ("tvm", "JIT*"),
                ],
            ),
            Silicon("GPU", "Arc / XPU", "intel", [("inductor", "JIT"), ("iree_vulkan", "EXPORT")]),
            Silicon("NPU", "Core Ultra AI Boost", "intel:npu", [("openvino", "J+A")]),
        ],
        runtime=["CPU: C++ · oneDNN", "GPU: SYCL / XPU", "NPU: OpenVINO", "Vulkan / SPIR-V"],
        hardware="Intel CPU + GPU + NPU",
        hardware_detail="x86-64 · Arc · AI Boost",
    ),
    Column(
        vendor="NVIDIA",
        tint="#76b900",
        logo="nvidia",
        silicon=[
            Silicon(
                "GPU",
                "Ampere and newer",
                "nvidia",
                [
                    ("inductor", "JIT"),
                    ("aot_inductor", "AOT"),
                    ("tensorrt", "J+A"),
                    ("onnxruntime", "J+A"),
                    ("iree_vulkan", "EXPORT"),
                ],
            ),
        ],
        runtime=["Triton / CUDA", "cuBLAS · cuDNN", "TensorRT engine", "Vulkan / SPIR-V"],
        hardware="NVIDIA GPU",
        hardware_detail="CUDA · Ampere and newer",
    ),
    Column(
        vendor="AMD",
        tint="#ed1c24",
        logo="amd",
        silicon=[
            Silicon(
                "CPU",
                "x86-64 · EPYC / Ryzen",
                "cpu",
                [
                    ("inductor", "JIT"),
                    ("aot_inductor", "AOT"),
                    ("zentorch", "JIT*"),
                    ("onnxruntime", "J+A"),
                    ("tvm", "JIT*"),
                ],
            ),
            Silicon(
                "GPU",
                "ROCm",
                "amd",
                [("inductor", "JIT"), ("aot_inductor", "AOT"), ("iree_vulkan", "EXPORT")],
            ),
        ],
        runtime=["CPU: C++ · ZenDNN", "GPU: Triton · ROCm", "HIP · AMDGPU", "Vulkan / SPIR-V"],
        hardware="AMD CPU + GPU",
        hardware_detail="x86-64 · ROCm · Vulkan",
    ),
    Column(
        vendor="Arm",
        tint="#0091bd",
        logo="arm",
        silicon=[
            Silicon(
                "CPU",
                "ARM64 servers / SBCs",
                "cpu",
                [
                    ("inductor", "JIT"),
                    ("aot_inductor", "AOT"),
                    ("onnxruntime", "J+A"),
                    ("tvm", "JIT*"),
                    ("executorch", "EXPORT"),
                    ("litert", "EXPORT"),
                ],
            ),
        ],
        runtime=["C++ / OpenMP", "XNNPACK · NEON", "KleidiAI", "ExecuTorch runtime"],
        hardware="Arm CPU",
        hardware_detail="ARM64 · SBCs",
    ),
    Column(
        vendor="Apple",
        tint="#6e6e73",
        logo="apple",
        silicon=[
            Silicon(
                "CPU",
                "Apple silicon · ARM64",
                "cpu",
                [
                    ("inductor", "JIT"),
                    ("aot_inductor", "AOT"),
                    ("onnxruntime", "J+A"),
                    ("tvm", "JIT*"),
                ],
            ),
            Silicon(
                "GPU",
                "Metal",
                "apple",
                [("inductor", "JIT"), ("aot_inductor", "AOT")],
            ),
            Silicon(
                "SoC",
                "Mac / iOS · CPU/GPU/ANE",
                "apple",
                [("coreml", "EXPORT")],
            ),
        ],
        runtime=["CPU: C++ / OpenMP", "GPU: Metal / MPS", "iOS: Core ML runtime"],
        hardware="Apple Mac + iPhone",
        hardware_detail="CPU/GPU/ANE via Core ML",
    ),
    Column(
        vendor="Google",
        tint="#4285f4",
        logo="google",
        silicon=[
            Silicon(
                "TPU",
                "via PyTorch/XLA",
                "tpu",
                [("openxla", "JIT"), ("stablehlo", "EXPORT")],
            ),
        ],
        runtime=["PJRT plugin", "OpenXLA", "StableHLO"],
        hardware="Google TPU",
        hardware_detail="via PyTorch/XLA",
    ),
    Column(
        vendor="Tenstorrent",
        tint="#7c68ee",
        logo="",
        silicon=[
            Silicon(
                "Accelerator card",
                "Wormhole · Blackhole",
                "tenstorrent",
                [("tenstorrent", "JIT"), ("stablehlo", "EXPORT")],
            ),
        ],
        runtime=["tt-xla · PJRT", "tt-mlir", "tt-metal"],
        hardware="Tenstorrent",
        hardware_detail="Wormhole · Blackhole",
    ),
    Column(
        vendor="Qualcomm",
        tint="#3253dc",
        logo="qualcomm",
        silicon=[
            Silicon("Snapdragon HTP", "8 Elite · v79", "qualcomm:sm8750", [("qnn", "EXPORT")]),
            Silicon(
                "Snapdragon CPU",
                "ARM64 phones",
                "cpu",
                [("executorch", "EXPORT"), ("litert", "EXPORT")],
            ),
        ],
        runtime=["ExecuTorch runtime", "QNN SDK", "Hexagon HTP", "XNNPACK · LiteRT"],
        hardware="Snapdragon",
        hardware_detail="HTP NPU · ARM64 CPU",
    ),
]


# Brand marks use 24x24 viewBox paths. Simple Icons path data is CC0; the marks
# themselves remain the trademarks of their respective owners. A logo entry is
# either one coloured path or several coloured paths for multicolour marks.
LOGOS: dict[str, tuple[str, str] | list[tuple[str, str]]] = {
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
    "google": [
        (
            "#4285F4",
            (
                "M23.49 12.27c0-.79-.07-1.54-.19-2.27H12v4.51h6.47c-.29 "
                "1.48-1.14 2.73-2.4 3.58v2.98h3.86c2.26-2.08 3.56-5.15 "
                "3.56-8.8z"
            ),
        ),
        (
            "#34A853",
            (
                "M12 24c3.24 0 5.95-1.08 7.93-2.93l-3.86-2.98c-1.08.72-2.45 "
                "1.16-4.07 1.16-3.13 0-5.78-2.11-6.73-4.96H1.29v3.07C3.26 "
                "21.25 7.29 24 12 24z"
            ),
        ),
        (
            "#FBBC05",
            (
                "M5.27 14.24c-.25-.72-.39-1.49-.39-2.24s.14-1.52.39-2.24V6."
                "69H1.29C.47 8.31 0 10.1 0 12s.47 3.69 1.29 5.31l3.98-3.07z"
            ),
        ),
        (
            "#EA4335",
            (
                "M12 4.84c1.76 0 3.34.61 4.58 1.8L20 3.22C17.95 1.33 15.24 "
                "0 12 0 7.29 0 3.26 2.75 1.29 6.69l3.98 3.07C6.22 6.95 "
                "8.87 4.84 12 4.84z"
            ),
        ),
    ],
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

RAIL_W = 150
PAD = 22
COL_W = 186
COL_GAP = 9
TOP = 22

SANS = (
    "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)


def _t(x, y, text, size, fill, weight="400", anchor="start", spacing=None):
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    return (
        f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>'
        f"{escape(text)}</text>"
    )


def _rect(x, y, w, h, fill, stroke=None, r=10, width=1):
    s = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"{s}/>'


def _logo(key, x, y, size):
    paths = LOGOS[key]
    if isinstance(paths, tuple):
        paths = [paths]
    body = "".join(f'<path d="{path}" fill="{colour}"/>' for colour, path in paths)
    return f'<g transform="translate({x} {y}) scale({size / 24:.4f})">{body}</g>'


def _pill(x, y, kind):
    """Right-aligned badge. Width follows the label so longer badges do not float."""
    fg, bg = BADGE[kind]
    w = 16 + len(kind) * 8.2
    return (
        _rect(x - w, y - 14, w, 20, bg, r=10)
        + _t(x - w / 2, y + 0.5, kind, 12, fg, "700", "middle", "0.02em")
    ), w


def _arrow(x1, y1, x2, y2, colour=ARROW):
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{colour}" stroke-width="1.6" '
        f'fill="none" marker-end="url(#a)"/>'
    )


def render() -> str:
    n = len(COLUMNS)
    grid_w = n * COL_W + (n - 1) * COL_GAP
    width = RAIL_W + grid_w + PAD
    x0 = RAIL_W

    def card_h(si: Silicon) -> int:
        return 78 + len(si.backends) * 34 + 20

    head_h = 46
    backend_band_h = (
        head_h
        + max(sum(card_h(si) for si in c.silicon) + 12 * (len(c.silicon) - 1) for c in COLUMNS)
        + 34
    )
    runtime_h = max(len(c.runtime) for c in COLUMNS) * 32 + 50
    hw_h = 142

    y_model = TOP
    model_h = 96
    y_orch = y_model + model_h + 26
    orch_h = 132
    y_bus = y_orch + orch_h
    y_backend = y_bus + 86
    y_runtime = y_backend + backend_band_h + 46
    y_hw = y_runtime + runtime_h + 46
    height = y_hw + hw_h + 34 + PAD

    opening = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="LM7 architecture in five '
        f'layers: model, orchestrator, backends, lowering and runtime, hardware">'
    )
    marker = (
        '<defs><marker id="a" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" '
        'markerHeight="5" orient="auto">'
        f'<path d="M 0 1 L 6 4 L 0 7 z" fill="{ARROW}"/></marker></defs>'
    )
    s = [opening, marker, _rect(0, 0, width, height, PAGE, r=0)]

    def rail(y, num, label, second=""):
        s.append(_t(28, y + 38, num, 42, RAIL_NUM, "800"))
        s.append(_t(28, y + 62, label, 14.5, MUTED, "700", spacing="0.08em"))
        if second:
            s.append(_t(28, y + 81, second, 14.5, MUTED, "700", spacing="0.08em"))

    # 01 model
    rail(y_model, "01", "MODEL")
    s.append(_rect(x0, y_model, grid_w, model_h, CARD, CARD_LINE, r=14))
    s.append(_logo("pytorch", x0 + 42, y_model + 26, 44))
    s.append(_t(x0 + 108, y_model + 44, "PyTorch / Hugging Face model", 26, INK, "800"))
    s.append(
        _t(x0 + 108, y_model + 70, "nn.Module or hf:// id, with representative inputs", 15, MUTED)
    )
    bx = x0 + grid_w - 560
    s.append(_rect(bx, y_model + 24, 72, 44, NAVY, r=10))
    s.append(_t(bx + 36, y_model + 53, "LM7", 21, NAVY_INK, "800", "middle"))
    s.append(
        f'<path d="M {bx + 22} {y_model + 68} L {bx + 34} {y_model + 78} L {bx + 40} '
        f'{y_model + 68} z" fill="{NAVY}"/>'
    )
    s.append(_t(bx + 88, y_model + 44, "LM7: PyTorch-first compiler orchestration", 19, INK, "800"))
    s.append(
        _t(bx + 88, y_model + 68, "one model · many compiler stacks · local hardware", 14.5, MUTED)
    )
    s.append(_arrow(x0 + grid_w / 2, y_model + model_h, x0 + grid_w / 2, y_orch - 4))

    # 02 orchestrator
    rail(y_orch, "02", "ORCHESTRATION")
    s.append(_rect(x0, y_orch, grid_w, orch_h, NAVY, r=14))
    s.append(_t(x0 + 40, y_orch + 54, "LM7", 34, NAVY_INK, "800"))
    s.append(_t(x0 + 40, y_orch + 84, "one model · one target string", 16, "#b9c7dd"))
    for i, (a, b) in enumerate(
        [
            ("target detection", "backend selection"),
            ("shape + compile cache", "safe eager fallback"),
        ]
    ):
        fx = x0 + 300 + i * 230
        s.append(_t(fx, y_orch + 48, a, 15.5, NAVY_INK, "700"))
        s.append(_t(fx, y_orch + 80, b, 15.5, NAVY_INK, "700"))
    for i, call in enumerate(["lm7.compile()", "lm7.export()"]):
        px = x0 + grid_w - 400 + i * 200
        s.append(_rect(px, y_orch + 42, 182, 50, CARD, r=10))
        s.append(_t(px + 91, y_orch + 74, call, 19, INK, "800", "middle"))

    # the bus: which call reaches the backends, and how
    jit_y, aot_y = y_bus + 26, y_bus + 56
    s.append(
        f'<path d="M {x0 + grid_w - 309} {y_orch + orch_h - 40} L {x0 + grid_w - 309} '
        f'{jit_y} L {x0 + 30} {jit_y}" stroke="{JIT_LINE}" stroke-width="1.8" fill="none"/>'
    )
    s.append(_t(x0 + 40, jit_y - 9, "JIT · in-process execution", 15, JIT_LINE, "700"))
    s.append(
        f'<path d="M {x0 + grid_w - 109} {y_orch + orch_h - 40} L {x0 + grid_w - 109} '
        f'{aot_y} L {x0 + 30} {aot_y}" stroke="{ARROW}" stroke-width="1.8" fill="none"/>'
    )
    s.append(_t(x0 + grid_w - 420, aot_y - 9, "AOT · artifact creation", 15, MUTED, "700"))
    for i in range(n):
        cx = x0 + i * (COL_W + COL_GAP) + COL_W / 2
        s.append(_arrow(cx, aot_y, cx, y_backend - 6))

    # 03 backends
    rail(y_backend - 12, "03", "BACKENDS")
    s.append(
        _rect(
            x0 - 12,
            y_backend - 18,
            grid_w + 24,
            backend_band_h,
            BAND_BACKEND[0],
            BAND_BACKEND[1],
            r=14,
        )
    )
    for i, col in enumerate(COLUMNS):
        cx = x0 + i * (COL_W + COL_GAP)
        # Name the column, or the cards below read as anonymous CPU/GPU/NPU boxes.
        if col.logo:
            s.append(_logo(col.logo, cx + 2, y_backend - 2, 17))
            s.append(_t(cx + 25, y_backend + 15, col.vendor, 19, col.tint, "800"))
        else:
            s.append(_t(cx + 2, y_backend + 15, col.vendor, 19, col.tint, "800"))
        s.append(_rect(cx, y_backend + 24, COL_W, 2, col.tint, r=0))

        y = y_backend + head_h
        for si in col.silicon:
            h = card_h(si)
            s.append(_rect(cx, y, COL_W, h, CARD, CARD_LINE, r=12))
            s.append(_t(cx + 14, y + 31, si.label, 20, INK, "800"))
            s.append(_t(cx + 14, y + 54, si.detail, 14, MUTED))
            s.append(_t(cx + 14, y + 77, f"target={si.target}", 14, col.tint, "700"))
            by = y + 108
            for name, kind in si.backends:
                s.append(_t(cx + 14, by, name, 16.5, INK))
                pill, _ = _pill(cx + COL_W - 14, by, kind)
                s.append(pill)
                by += 34
            y += h + 12

    # 04 lowering and runtime
    rail(y_runtime - 12, "04", "LOWERING", "+ RUNTIME")
    s.append(
        _rect(
            x0 - 12, y_runtime - 18, grid_w + 24, runtime_h + 24, BAND_LOWER[0], BAND_LOWER[1], r=14
        )
    )
    for i, col in enumerate(COLUMNS):
        cx = x0 + i * (COL_W + COL_GAP)
        s.append(
            _arrow(cx + COL_W / 2, y_backend + backend_band_h - 22, cx + COL_W / 2, y_runtime - 8)
        )
        s.append(_rect(cx, y_runtime, COL_W, runtime_h, CARD, CARD_LINE, r=12))
        ry = y_runtime + 40
        for line in col.runtime:
            s.append(_t(cx + COL_W / 2, ry, line, 16.5, INK, "700", "middle"))
            ry += 32

    # 05 hardware
    rail(y_hw - 12, "05", "HARDWARE")
    s.append(_rect(x0 - 12, y_hw - 18, grid_w + 24, hw_h + 24, BAND_HW[0], BAND_HW[1], r=14))
    for i, col in enumerate(COLUMNS):
        cx = x0 + i * (COL_W + COL_GAP)
        s.append(_arrow(cx + COL_W / 2, y_runtime + runtime_h, cx + COL_W / 2, y_hw - 8))
        s.append(_rect(cx, y_hw, COL_W, hw_h, CARD, CARD_LINE, r=12))
        if col.logo:
            s.append(_logo(col.logo, cx + COL_W / 2 - 20, y_hw + 20, 40))
        else:
            s.append(_t(cx + COL_W / 2, y_hw + 50, "tt", 30, "#7c68ee", "800", "middle"))
        s.append(_t(cx + COL_W / 2, y_hw + 96, col.hardware, 18, INK, "800", "middle"))
        s.append(_t(cx + COL_W / 2, y_hw + 120, col.hardware_detail, 14.5, MUTED, anchor="middle"))

    s.append(
        _t(
            x0 + grid_w / 2,
            height - 20,
            "J+A = compiles in-process and also writes an artifact.  "
            "* = explicit opt-in, never chosen automatically.  "
            "Backend priority and artifact formats are in the table below.",
            15,
            MUTED,
            anchor="middle",
        )
    )
    s.append("</svg>")
    return "\n".join(s)


def main() -> None:
    here = Path(__file__).parent
    svg = here / "lm7-architecture.svg"
    svg.write_text(render(), encoding="utf-8")
    print(f"wrote {svg.name} ({svg.stat().st_size:,} bytes)")
    try:
        import cairosvg
    except ImportError:
        print("cairosvg not installed; skipping the PNG the README actually uses")
        return
    png = here / "lm7-architecture.png"
    cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2)
    print(f"wrote {png.name} ({png.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
