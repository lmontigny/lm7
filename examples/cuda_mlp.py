from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

import lm7


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    ).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and validate an LM7 CUDA model.")
    parser.add_argument(
        "--target",
        default="nvidia",
        help="NVIDIA target selector, such as nvidia or nvidia:sm89.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Retain TorchInductor IR and generated code in this directory.",
    )
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this PyTorch environment.")

    torch.manual_seed(0)
    source = _model()
    reference = copy.deepcopy(source).cuda()
    example_input = torch.randn(8, 16)
    expected = reference(example_input.cuda())

    options = None
    if arguments.debug_dir is not None:
        debug_dir = arguments.debug_dir.resolve()
        debug_dir.mkdir(parents=True, exist_ok=True)
        torch.compiler.config.force_disable_caches = True
        options = {
            "trace.enabled": True,
            "trace.debug_dir": str(debug_dir),
            "trace.fx_graph": True,
            "trace.fx_graph_transformed": True,
            "trace.ir_pre_fusion": True,
            "trace.ir_post_fusion": True,
            "trace.output_code": True,
        }

    compiled = lm7.compile(
        source,
        target=arguments.target,
        backend="inductor",
        transfers="automatic",
        fallback="error",
        options=options,
    )
    actual = compiled(example_input)
    torch.testing.assert_close(actual, expected)

    assert compiled.target is not None
    print(f"GPU: {torch.cuda.get_device_name(compiled.target.ordinal or 0)}")
    print(f"Target: {compiled.target}")
    print(f"Backend: {compiled.selected_backend}")
    print(f"Output: shape={tuple(actual.shape)}, device={actual.device}")
    print("TorchInductor output matches eager CUDA.")

    if arguments.debug_dir is not None:
        files = sorted(path for path in debug_dir.rglob("*") if path.is_file())
        print("Compiler debug files:")
        for path in files:
            print(f"- {path.relative_to(debug_dir)}")


if __name__ == "__main__":
    main()
