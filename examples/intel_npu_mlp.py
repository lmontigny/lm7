from __future__ import annotations

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
    if not any(device.target.kind == "npu" for device in detect_targets()):
        raise SystemExit(
            "No Intel NPU detected. `target='auto'` never picks this target -- see "
            "docs/intel-npu.md -- and this example needs the driver plus "
            'pip install -e ".[openvino]".'
        )

    torch.manual_seed(0)
    source = model()
    example_input = torch.randn(8, 16)
    expected = source(example_input)

    # intel:npu has no PyTorch device behind it: OpenVINO's NPU plugin is the
    # only backend that plans, so naming the target is enough.
    try:
        compiled = lm7.compile(source, target="intel:npu", backend="auto", fallback="error")
        actual = compiled(example_input)
    except BackendUnavailableError as error:
        print(f"Intel NPU plugin unavailable: {error}")
        raise SystemExit('Install it with: pip install -e ".[openvino]"') from None

    # The NPU plugin computes in FP16 with no FP32 mode to pin, so this is
    # looser than the CPU OpenVINO tolerance -- see docs/intel-npu.md.
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    print(f"Target: {compiled.target}")
    print(f"Backend: {compiled.selected_backend}")
    print(f"Output: shape={tuple(actual.shape)}")
    print("OpenVINO NPU output matches eager CPU within FP16 tolerance.")


if __name__ == "__main__":
    main()
