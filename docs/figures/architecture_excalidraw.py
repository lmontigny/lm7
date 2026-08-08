"""Generate an editable Excalidraw scene for the LM7 architecture diagram.

Mirrors `architecture.py`'s five-layer structure (model, orchestrator,
backends, lowering+runtime, hardware) but as a real .excalidraw JSON scene
rather than a fixed SVG/PNG: open it at https://excalidraw.com (File > Open)
or the Excalidraw VS Code extension to drag boxes around, retint, or add or
remove backends by hand.

    python docs/figures/architecture_excalidraw.py

writes `lm7-architecture.excalidraw` beside this script.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

random.seed(7)


def rid() -> str:
    return "".join(random.choice("0123456789abcdef") for _ in range(16))


def seed() -> int:
    return random.randint(1, 2**31 - 1)


NOW = int(time.time() * 1000)


def rect(elements, x, y, w, h, bg="transparent", stroke="#1e1e1e", stroke_width=2, rounded=True):
    elements.append(
        {
            "id": rid(),
            "type": "rectangle",
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "angle": 0,
            "strokeColor": stroke,
            "backgroundColor": bg,
            "fillStyle": "solid",
            "strokeWidth": stroke_width,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "roundness": {"type": 3} if rounded else None,
            "seed": seed(),
            "version": 1,
            "versionNonce": seed(),
            "isDeleted": False,
            "boundElements": None,
            "updated": NOW,
            "link": None,
            "locked": False,
        }
    )


def text(elements, x, y, s, size=16, color="#1e1e1e", align="left"):
    lines = s.split("\n")
    w = max(len(line) for line in lines) * size * 0.56
    h = len(lines) * size * 1.25
    if align == "center":
        x = x - w / 2
    elements.append(
        {
            "id": rid(),
            "type": "text",
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "angle": 0,
            "strokeColor": color,
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 2,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "roundness": None,
            "seed": seed(),
            "version": 1,
            "versionNonce": seed(),
            "isDeleted": False,
            "boundElements": None,
            "updated": NOW,
            "link": None,
            "locked": False,
            "text": s,
            "fontSize": size,
            "fontFamily": 2,
            "textAlign": "left",
            "verticalAlign": "top",
            "containerId": None,
            "originalText": s,
            "lineHeight": 1.25,
            "baseline": size,
        }
    )


def arrow(elements, x1, y1, x2, y2, color="#94a3b4", width=2):
    elements.append(
        {
            "id": rid(),
            "type": "arrow",
            "x": x1,
            "y": y1,
            "width": abs(x2 - x1),
            "height": abs(y2 - y1),
            "angle": 0,
            "strokeColor": color,
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": width,
            "strokeStyle": "solid",
            "roughness": 1,
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "roundness": {"type": 2},
            "seed": seed(),
            "version": 1,
            "versionNonce": seed(),
            "isDeleted": False,
            "boundElements": None,
            "updated": NOW,
            "link": None,
            "locked": False,
            "points": [[0, 0], [x2 - x1, y2 - y1]],
            "lastCommittedPoint": None,
            "startBinding": None,
            "endBinding": None,
            "startArrowhead": None,
            "endArrowhead": "arrow",
            "elbowed": False,
        }
    )


# --- content, mirrored from architecture.py's COLUMNS -----------------------
#
# Simplified relative to the SVG generator: one line per backend list rather
# than per-backend JIT/AOT badges, since Excalidraw text is meant to be
# hand-edited afterward rather than regenerated pixel-perfect.

COLUMNS = [
    {
        "vendor": "Intel",
        "tint": "#0071c5",
        "backends": "CPU  target=cpu\ninductor, aot_inductor,\nopenvino, onnxruntime, (tvm)\n\n"
        "GPU  target=intel\ninductor, (iree_vulkan)\n\nNPU  target=intel:npu\nopenvino",
        "runtime": "CPU: C++ / oneDNN\nGPU: SYCL / XPU\nNPU: OpenVINO\nVulkan / SPIR-V",
        "hardware": "Intel CPU + GPU + NPU\nx86-64 - Arc - AI Boost",
    },
    {
        "vendor": "NVIDIA",
        "tint": "#5c8a00",
        "backends": "GPU  target=nvidia\ninductor, aot_inductor,\ntensorrt, onnxruntime,\n(iree_vulkan)",
        "runtime": "Triton / CUDA\ncuBLAS - cuDNN\nTensorRT engine\nVulkan / SPIR-V",
        "hardware": "NVIDIA GPU\nCUDA - Ampere and newer",
    },
    {
        "vendor": "AMD",
        "tint": "#ed1c24",
        "backends": "CPU  target=cpu\ninductor, aot_inductor,\n(zentorch), onnxruntime, (tvm)\n\n"
        "GPU  target=amd\ninductor, (iree_vulkan)",
        "runtime": "CPU: C++ / ZenDNN\nGPU: Triton / ROCm\nHIP / AMDGPU\nVulkan / SPIR-V",
        "hardware": "AMD CPU + GPU\nx86-64 - ROCm - Vulkan",
    },
    {
        "vendor": "Arm",
        "tint": "#0091bd",
        "backends": "CPU  target=cpu\ninductor, aot_inductor,\nonnxruntime, (tvm),\n(executorch), (litert)",
        "runtime": "C++ / OpenMP\nXNNPACK / NEON\nKleidiAI\nExecuTorch runtime",
        "hardware": "Arm CPU\nARM64 servers - SBCs - phones",
    },
    {
        "vendor": "Apple",
        "tint": "#6e6e73",
        "backends": "CPU  target=cpu\ninductor, aot_inductor,\nonnxruntime, (tvm)\n\n"
        "GPU  target=apple\ninductor, aot_inductor",
        "runtime": "CPU: C++ / OpenMP\nGPU: Metal / MPS",
        "hardware": "Apple CPU + GPU\nApple silicon - Metal",
    },
    {
        "vendor": "Google",
        "tint": "#4285f4",
        "backends": "TPU  target=tpu\nopenxla, (stablehlo)",
        "runtime": "PJRT plugin\nOpenXLA\nStableHLO",
        "hardware": "Google TPU\nvia PyTorch/XLA",
    },
    {
        "vendor": "Tenstorrent",
        "tint": "#7c68ee",
        "backends": "Card  target=tenstorrent\ntenstorrent, (stablehlo)",
        "runtime": "tt-xla / PJRT\ntt-mlir\ntt-metal",
        "hardware": "Tenstorrent\nWormhole - Blackhole",
    },
    {
        "vendor": "Qualcomm",
        "tint": "#3253dc",
        "backends": "HTP  target=qualcomm:sm8750\n(qnn)\n\nCPU  target=cpu\n(executorch), (litert)",
        "runtime": "ExecuTorch runtime\nQNN SDK\nHexagon HTP\nXNNPACK / LiteRT",
        "hardware": "Snapdragon\nHTP NPU - ARM64 CPU",
    },
]


def build() -> dict:
    elements: list[dict] = []

    margin = 40
    col_w = 220
    col_gap = 20
    n = len(COLUMNS)
    grid_w = n * col_w + (n - 1) * col_gap
    x0 = margin

    y_model, model_h = 40, 100
    y_orch, orch_h = y_model + model_h + 60, 150
    y_band_backend, backend_band_h, backend_card_h = y_orch + orch_h + 70, 300, 230
    y_band_runtime, runtime_band_h, runtime_card_h = y_band_backend + backend_band_h + 70, 170, 110
    y_band_hw, hw_band_h, hw_card_h = y_band_runtime + runtime_band_h + 70, 150, 95

    height = y_band_hw + hw_band_h + margin
    width = x0 + grid_w + margin

    for y, label in (
        (y_model - 26, "01  MODEL"),
        (y_orch - 26, "02  ORCHESTRATION"),
        (y_band_backend - 34, "03  BACKENDS"),
        (y_band_runtime - 34, "04  LOWERING + RUNTIME"),
        (y_band_hw - 34, "05  HARDWARE"),
    ):
        text(elements, x0, y, label, size=14, color="#68788a")

    # 01 model
    rect(elements, x0, y_model, grid_w, model_h, bg="#ffffff", stroke="#1e1e1e")
    text(elements, x0 + 30, y_model + 22, "PyTorch / Hugging Face model", size=22, color="#1a2532")
    text(
        elements,
        x0 + 30,
        y_model + 56,
        "nn.Module or hf:// id, with representative inputs",
        size=14,
        color="#68788a",
    )
    arrow(elements, x0 + grid_w / 2, y_model + model_h, x0 + grid_w / 2, y_orch - 4)

    # 02 orchestrator
    rect(elements, x0, y_orch, grid_w, orch_h, bg="#152441", stroke="#0f172a")
    text(elements, x0 + 30, y_orch + 24, "LM7", size=30, color="#ffffff")
    text(elements, x0 + 30, y_orch + 64, "one model - one target string", size=14, color="#b9c7dd")
    text(
        elements,
        x0 + 340,
        y_orch + 24,
        "target detection      shape + compile cache\nbackend selection      safe eager fallback",
        size=14,
        color="#ffffff",
    )
    call_w = 200
    for i, call in enumerate(["lm7.compile()", "lm7.export()"]):
        cx = x0 + grid_w - 2 * call_w - 30 + i * (call_w + 20)
        rect(elements, cx, y_orch + orch_h - 66, call_w, 46, bg="#ffffff", stroke="#dde4ec")
        text(
            elements,
            cx + call_w / 2,
            y_orch + orch_h - 54,
            call,
            size=16,
            color="#1a2532",
            align="center",
        )

    # 03 backends band
    rect(
        elements,
        x0 - 12,
        y_band_backend - 12,
        grid_w + 24,
        backend_band_h,
        bg="#eff5fd",
        stroke="#4a7fd0",
        stroke_width=1,
    )
    for i, col in enumerate(COLUMNS):
        cx = x0 + i * (col_w + col_gap)
        arrow(elements, cx + col_w / 2, y_orch + orch_h, cx + col_w / 2, y_band_backend + 30)
        text(elements, cx, y_band_backend + 6, col["vendor"], size=15, color=col["tint"])
        card_y = y_band_backend + 34
        rect(elements, cx, card_y, col_w, backend_card_h, bg="#ffffff", stroke="#dde4ec")
        text(elements, cx + 12, card_y + 12, col["backends"], size=11.5, color="#1a2532")

    # 04 lowering + runtime band
    rect(
        elements,
        x0 - 12,
        y_band_runtime - 12,
        grid_w + 24,
        runtime_band_h,
        bg="#f5f2fd",
        stroke="#8b6fd4",
        stroke_width=1,
    )
    for i, col in enumerate(COLUMNS):
        cx = x0 + i * (col_w + col_gap)
        arrow(
            elements,
            cx + col_w / 2,
            y_band_backend + backend_band_h - 12,
            cx + col_w / 2,
            y_band_runtime + 20,
        )
        card_y = y_band_runtime + 20
        rect(elements, cx, card_y, col_w, runtime_card_h, bg="#ffffff", stroke="#dde4ec")
        text(elements, cx + 12, card_y + 12, col["runtime"], size=12.5, color="#1a2532")

    # 05 hardware band
    rect(
        elements,
        x0 - 12,
        y_band_hw - 12,
        grid_w + 24,
        hw_band_h,
        bg="#eef9f1",
        stroke="#5aab74",
        stroke_width=1,
    )
    for i, col in enumerate(COLUMNS):
        cx = x0 + i * (col_w + col_gap)
        arrow(
            elements,
            cx + col_w / 2,
            y_band_runtime + runtime_band_h - 12,
            cx + col_w / 2,
            y_band_hw + 20,
        )
        card_y = y_band_hw + 20
        rect(elements, cx, card_y, col_w, hw_card_h, bg="#ffffff", stroke="#dde4ec")
        text(elements, cx + 12, card_y + 14, col["hardware"], size=13, color="#1a2532")

    text(
        elements,
        x0,
        height - 26,
        '( ) = export-only or explicit opt-in, never chosen automatically by backend="auto".',
        size=13,
        color="#68788a",
    )

    return {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {"gridSize": 20, "viewBackgroundColor": "#f7f9fc"},
        "files": {},
    }


def main() -> None:
    scene = build()
    out = Path(__file__).parent / "lm7-architecture.excalidraw"
    out.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    print(f"wrote {out.name} ({len(scene['elements'])} elements)")


if __name__ == "__main__":
    main()
