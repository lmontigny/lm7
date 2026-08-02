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

# --- geometry --------------------------------------------------------------

COL_W = 188
COL_GAP = 11
PAD = 22
RAIL_W = 0
TOP = 22

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


GROUPS = {
    "compute": ("Intel", "AMD", "NVIDIA", "Apple", "Arm"),
    "accelerators": ("Google", "Tenstorrent", "Qualcomm"),
}


def render(theme: Theme, vendors: tuple[str, ...]) -> str:
    columns = [c for c in COLUMNS if c.vendor in vendors]
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
    hw_h = 62

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
        s.append(_t(x0, y - 6, f"{num}  {label}", 8, theme.muted, "700", spacing="0.1em"))

    # 01 model
    rail(y_model, "01", "MODEL")
    s.append(_rect(x0, y_model, grid_w, model_h, theme.card, theme.card_line))
    s.append(_t(x0 + 20, y_model + 26, "PyTorch / Hugging Face model", 15, theme.ink, "700"))
    s.append(
        _t(
            x0 + 20,
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
    s.append(_rect(x0 - 8, y_backend - 14, grid_w + 16, backend_h + 14, theme.band, r=9))
    s.append(_rect(x0 - 8, y_runtime - 12, grid_w + 16, runtime_h + 12, theme.band, r=9))
    s.append(_rect(x0 - 8, y_hw - 12, grid_w + 16, hw_h + 12, theme.band, r=9))

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
        s.append(_t(cx + 14, y_hw + 22, col.vendor, 13, theme.ink, "700"))
        parts = ", ".join(h.label for h in col.hardware)
        s.append(_t(cx + 14, y_hw + 38, parts, 9, theme.muted))
        if col.footnote:
            s.append(_t(cx + 14, y_hw + 52, col.footnote, 8, col.tint, "600"))

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
    for group, vendors in GROUPS.items():
        for theme in (LIGHT, DARK):
            path = here / f"lm7-{group}-{theme.name}.svg"
            path.write_text(render(theme, vendors), encoding="utf-8")
            print(f"wrote {path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
