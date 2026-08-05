from .aot_inductor import AOTInductorBackend
from .coreml import ExecuTorchCoreMLBackend
from .eager import EagerBackend
from .executorch import ExecuTorchBackend
from .inductor import InductorBackend
from .iree_vulkan import IREEVulkanBackend
from .litert import LiteRTBackend
from .onnxruntime import ONNXRuntimeBackend
from .openvino import OpenVINOBackend
from .openxla import OpenXLABackend
from .qnn import ExecuTorchQNNBackend
from .registry import BackendRegistry
from .stablehlo import StableHLOBackend
from .tensorrt import TensorRTBackend
from .tenstorrent import TenstorrentBackend
from .tvm import TVMBackend
from .zentorch import ZenTorchBackend

registry = BackendRegistry()
registry.register(EagerBackend())
registry.register(InductorBackend())
registry.register(AOTInductorBackend())
registry.register(TensorRTBackend())
registry.register(IREEVulkanBackend())
registry.register(LiteRTBackend())
registry.register(ONNXRuntimeBackend())
registry.register(OpenVINOBackend())
registry.register(OpenXLABackend())
registry.register(StableHLOBackend())
registry.register(ExecuTorchBackend())
registry.register(ExecuTorchQNNBackend())
registry.register(ExecuTorchCoreMLBackend())
registry.register(TenstorrentBackend())
registry.register(TVMBackend())
registry.register(ZenTorchBackend())

__all__ = ["registry"]
