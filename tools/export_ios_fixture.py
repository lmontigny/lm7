"""Export the tiny Core ML fixture used by the iOS device validation gate.

Produces the artifact and the host golden output described in
docs/ios-device-testing.md. Run from a venv that has ExecuTorch and
coremltools installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import lm7

OUT = Path("artifacts/ios")


def main() -> None:
    torch.manual_seed(0)

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 2),
    ).eval()
    example = torch.randn(3, 4)

    OUT.mkdir(parents=True, exist_ok=True)

    artifact = lm7.export(
        model,
        args=(example,),
        target="apple",
        backend="coreml",
        output=str(OUT / "coreml-mlp.lm7"),
        options={
            "compute_unit": "all",
            "compute_precision": "float16",
        },
    )

    expected = model(example).detach()
    torch.save({"input": example, "expected": expected}, OUT / "golden.pt")

    # Device-side fixture: plain JSON so the iOS app needs no torch reader.
    (OUT / "golden.json").write_text(
        json.dumps(
            {
                "input_shape": list(example.shape),
                "input": example.flatten().tolist(),
                "output_shape": list(expected.shape),
                "expected": expected.flatten().tolist(),
            },
            indent=2,
        )
    )

    reloaded = artifact(example)
    max_abs_diff = (reloaded - expected).abs().max().item()

    print(f"pte:          {Path(artifact.path) / 'compiled_model.pte'}")
    print(f"input:        float32{list(example.shape)}")
    print(f"output:       float32{list(expected.shape)}")
    print(f"max_abs_diff: {max_abs_diff:.3e}")


if __name__ == "__main__":
    main()
