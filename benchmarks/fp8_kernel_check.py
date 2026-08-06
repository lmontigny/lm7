"""Check which GEMM Inductor actually emits for each FP8 quantization mode.

A plausible output does not prove the tensor cores multiplied in FP8, and a mode
name does not prove what granularity TorchAO resolved. Both matter here because
`fp8-dynamic` and `fp8-dynamic-rowwise` differ only in scale layout, and TorchAO
resolves an omitted `granularity` to per-tensor rather than raising -- so the two
modes are indistinguishable from the call site and distinguishable only in what
they emit and what shape their scales are.

This reads two independent sources of truth:

- **The quantized weight**, whose `scale` shape is the granularity. `(1, 1)` is
  per-tensor; `(N, 1)` is per-row.
- **Inductor's generated code**, which either calls `_scaled_mm` -- the scaled
  narrow-format GEMM -- or a plain `mm` with dequantization around it. Only the
  first computes in FP8.

    python benchmarks/fp8_kernel_check.py --output artifacts/fp8-kernels.json

The table in docs/quantization.md under "Verifying that the narrow kernel
actually ran" was originally produced by hand; this is the repeatable version.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import torch

from lm7.huggingface import (
    FP8,
    FP8_DYNAMIC,
    FP8_DYNAMIC_ROWWISE,
    NO_QUANTIZATION,
    _apply_quantization,
    fp8_scale_granularity,
)
from lm7.targets import parse_target

MODES = (NO_QUANTIZATION, FP8, FP8_DYNAMIC, FP8_DYNAMIC_ROWWISE)


class _MLP(torch.nn.Module):
    def __init__(self, K: int, N: int) -> None:
        super().__init__()
        self.down_proj = torch.nn.Linear(K, N, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(x)


class _Block(torch.nn.Module):
    """One linear behind a `.mlp.` path, which is what LM7's FP8 filter selects."""

    def __init__(self, K: int, N: int) -> None:
        super().__init__()
        self.mlp = _MLP(K, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _build(K: int, N: int) -> torch.nn.Module:
    model = torch.nn.Sequential()
    model.add_module("layer", _Block(K, N))
    return model.cuda().to(torch.bfloat16).eval()


def _emitted_code(model: torch.nn.Module, x: torch.Tensor) -> str:
    """Compile once with tracing on and return the generated Triton/wrapper code."""
    from torch._inductor import config

    with tempfile.TemporaryDirectory() as directory:
        torch.compiler.config.force_disable_caches = True
        torch._dynamo.reset()
        with config.patch(
            {"trace.enabled": True, "trace.debug_dir": directory, "trace.output_code": True}
        ):
            compiled = torch.compile(model, fullgraph=True)
            with torch.no_grad():
                compiled(x)
            torch.cuda.synchronize()
        return "\n".join(
            path.read_text(errors="replace")
            for path in sorted(Path(directory).rglob("output_code.py"))
        )


def _calls(code: str, symbol: str) -> int:
    """Count calls to `symbol`, ignoring the substring hits that make this lie.

    Two traps, in opposite directions. `mm` occurs inside `_scaled_mm`, `bmm` and
    `addmm`, so an unanchored search reports a plain BF16 matmul for every
    dynamic mode. But calls arrive qualified -- `extern_kernels.mm(`,
    `torch._scaled_mm(` -- so excluding a preceding `.` as well as a preceding
    word character finds nothing at all, and every mode then looks dequantized.

    Excluding word characters only is what distinguishes the two: the `mm` in
    `extern_kernels.mm(` follows a dot and counts, the one in `torch._scaled_mm(`
    follows an underscore and does not.
    """
    return len(re.findall(rf"(?<!\w){re.escape(symbol)}\s*\(", code))


def check_mode(mode: str, M: int, K: int, N: int) -> dict[str, Any]:
    target = parse_target("nvidia")
    torch.manual_seed(0)
    model = _build(K, N)
    record: dict[str, Any] = {
        "quantization": mode,
        "M": M,
        "K": K,
        "N": N,
        "requested_granularity": fp8_scale_granularity(mode),
    }

    if mode != NO_QUANTIZATION:
        try:
            _, converted = _apply_quantization(model, target, mode)
        except Exception as error:  # noqa: BLE001 - a refused mode is a result
            record.update({"works": False, "error": f"{type(error).__name__}: {error}"[:300]})
            return record
        record["converted_modules"] = converted

    weight = model.layer.mlp.down_proj.weight
    scale = getattr(weight, "scale", None)
    record["weight_type"] = type(weight).__name__
    record["packed_dtype"] = str(getattr(weight, "_data", weight).dtype)
    record["scale_shape"] = list(scale.shape) if scale is not None else None
    if scale is not None:
        # The shape is the granularity, whatever the mode was called.
        record["observed_granularity"] = (
            "per-tensor" if tuple(scale.shape) in {(1,), (1, 1)} else "per-row"
        )

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    try:
        code = _emitted_code(model, x)
    except Exception as error:  # noqa: BLE001 - so is a kernel that will not compile
        record.update({"works": False, "error": f"{type(error).__name__}: {error}"[:300]})
        return record

    scaled = _calls(code, "_scaled_mm")
    plain = _calls(code, "mm") + _calls(code, "addmm")
    record.update(
        {
            "works": True,
            "scaled_mm_calls": scaled,
            "plain_mm_calls": plain,
            # The claim the docs make: a dynamic mode computes in FP8, which means
            # a scaled GEMM and no plain one.
            "computes_in_fp8": scaled > 0 and plain == 0,
        }
    )
    print(
        f"  {mode:<22} scale={record['scale_shape']!s:<12}"
        f" {record.get('observed_granularity') or '-':<10}"
        f" _scaled_mm={scaled}  mm={plain}"
        f"  -> {'FP8 GEMM' if record['computes_in_fp8'] else 'dequantized to BF16'}"
    )
    del model
    torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--shape", type=int, nargs=3, default=(1024, 4096, 4096))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This check needs a CUDA GPU.")

    M, K, N = arguments.shape
    print(f"{torch.cuda.get_device_name(0)}, shape {M} x {K} x {N}")
    results = [check_mode(mode, M, K, N) for mode in arguments.mode]

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "device": torch.cuda.get_device_name(0),
                    "compute_capability": list(torch.cuda.get_device_capability(0)),
                    "torch": torch.__version__,
                    "shape": {"M": M, "K": K, "N": N},
                    "results": results,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
