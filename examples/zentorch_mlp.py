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
    torch.manual_seed(0)
    source = model()
    reference = copy.deepcopy(source)
    example_input = torch.randn(8, 16)
    expected = reference(example_input)

    try:
        compiled = lm7.compile(
            source,
            target="cpu",
            backend="zentorch",
            fallback="error",
        )
        actual = compiled(example_input)
    except BackendUnavailableError as error:
        print(f"zentorch unavailable: {error}")
        raise SystemExit(
            "zentorch publishes x86-64 Linux wheels only. Install it with: "
            'pip install -e ".[zentorch]" -- see docs/zentorch.md.'
        ) from None

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    print(f"Target: {compiled.target}")
    print(f"Backend: {compiled.selected_backend}")
    print(f"Output: shape={tuple(actual.shape)}, device={actual.device}")
    print("zentorch output matches eager CPU.")
    print(
        "zentorch is explicit-only, like tvm: it ranks below inductor, so "
        "backend='auto' never selects it -- see docs/zentorch.md."
    )


if __name__ == "__main__":
    main()
