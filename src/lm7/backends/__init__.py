from .aot_inductor import AOTInductorBackend
from .eager import EagerBackend
from .executorch import ExecuTorchBackend
from .inductor import InductorBackend
from .iree_vulkan import IREEVulkanBackend
from .onnxruntime import ONNXRuntimeBackend
from .openvino import OpenVINOBackend
from .openxla import OpenXLABackend
from .registry import BackendRegistry
from .stablehlo import StableHLOBackend
from .tensorrt import TensorRTBackend
from .tenstorrent import TenstorrentBackend

registry = BackendRegistry()
registry.register(EagerBackend())
registry.register(InductorBackend())
registry.register(AOTInductorBackend())
registry.register(TensorRTBackend())
registry.register(IREEVulkanBackend())
registry.register(ONNXRuntimeBackend())
registry.register(OpenVINOBackend())
registry.register(OpenXLABackend())
registry.register(StableHLOBackend())
registry.register(ExecuTorchBackend())
registry.register(TenstorrentBackend())

__all__ = ["registry"]
