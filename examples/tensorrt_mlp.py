from __future__ import annotations

import copy

import torch

import lm7
from lm7.errors import BackendUnavailableError


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    ).eval()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this PyTorch environment.")

    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source).cuda().half()
    example_input = torch.randn(8, 16, dtype=torch.float16)
    expected = reference(example_input.cuda())

    try:
        compiled = lm7.compile(
            source.half(),
            target="nvidia",
            backend="tensorrt",
            transfers="automatic",
            fallback="error",
        )
        actual = compiled(example_input)
    except BackendUnavailableError as error:
        print(f"TensorRT unavailable: {error}")
        raise SystemExit('Install it with: pip install -e ".[tensorrt]"') from None

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    assert compiled.target is not None
    print(f"GPU: {torch.cuda.get_device_name(compiled.target.ordinal or 0)}")
    print(f"Target: {compiled.target}")
    print(f"Backend: {compiled.selected_backend}")
    print(f"Output: shape={tuple(actual.shape)}, device={actual.device}")
    print("Torch-TensorRT output matches eager CUDA.")
    print(
        "TensorRT is opt-in, not the NVIDIA default: it wins some steady-state causal-LM "
        "workloads but has the largest first-call cost -- see docs/nvidia-tensorrt-evaluation.md."
    )


if __name__ == "__main__":
    main()
