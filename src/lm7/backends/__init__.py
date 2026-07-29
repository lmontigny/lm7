from .aot_inductor import AOTInductorBackend
from .eager import EagerBackend
from .inductor import InductorBackend
from .iree_vulkan import IREEVulkanBackend
from .openvino import OpenVINOBackend
from .openxla import OpenXLABackend
from .registry import BackendRegistry
from .stablehlo import StableHLOBackend
from .tensorrt import TensorRTBackend

registry = BackendRegistry()
registry.register(EagerBackend())
registry.register(InductorBackend())
registry.register(AOTInductorBackend())
registry.register(TensorRTBackend())
registry.register(IREEVulkanBackend())
registry.register(OpenVINOBackend())
registry.register(OpenXLABackend())
registry.register(StableHLOBackend())

__all__ = ["registry"]
