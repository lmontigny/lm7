from __future__ import annotations

import tempfile
from pathlib import Path

import torch

import lm7
from lm7.detection import detect_targets
from lm7.errors import BackendUnavailableError


def model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    ).eval()


def main() -> None:
    gpu_vendors = {"nvidia", "amd", "intel"}
    detected = next(
        (device for device in detect_targets() if device.target.vendor in gpu_vendors), None
    )
    if detected is None:
        raise SystemExit(
            "No NVIDIA, AMD, or Intel GPU detected -- iree_vulkan targets those vendors "
            'only, not Apple Metal or TPU. Install with: pip install -e ".[iree-vulkan]".'
        )
    target = detected.target.vendor

    torch.manual_seed(0)
    source = model()
    example_input = torch.randn(8, 16)
    expected = source(example_input)

    # Export-only and explicit: automatic backend selection never chooses
    # iree_vulkan, and a build host can produce the artifact without a Vulkan
    # device -- only the deployment host needs the Vulkan 1.3 driver.
    with tempfile.TemporaryDirectory(prefix="lm7-iree-vulkan-") as temporary_directory:
        output = Path(temporary_directory) / "model.lm7"
        try:
            artifact = lm7.export(
                source,
                args=(example_input,),
                target=target,
                backend="iree_vulkan",
                output=output,
            )
            actual = artifact(example_input)
        except BackendUnavailableError as error:
            print(f"IREE Vulkan unavailable: {error}")
            raise SystemExit('Install it with: pip install -e ".[iree-vulkan]"') from None

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

        print(f"Artifact: {artifact.path}")
        print(f"Backend: {artifact.manifest.backend}")
        print(f"Output: shape={tuple(actual.shape)}")
        print("IREE Vulkan output matches eager CPU.")
        print(
            "Pass options={'vulkan_target': ..., 'opt_level': 'O3'} to tune for a specific "
            "GPU microarchitecture -- see docs/iree-vulkan.md."
        )


if __name__ == "__main__":
    main()
