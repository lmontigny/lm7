from .aot_inductor import AOTInductorBackend
from .eager import EagerBackend
from .inductor import InductorBackend
from .openxla import OpenXLABackend
from .registry import BackendRegistry
from .tensorrt import TensorRTBackend

registry = BackendRegistry()
registry.register(EagerBackend())
registry.register(InductorBackend())
registry.register(AOTInductorBackend())
registry.register(TensorRTBackend())
registry.register(OpenXLABackend())

__all__ = ["registry"]
