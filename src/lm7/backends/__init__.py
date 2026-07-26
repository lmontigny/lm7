from .aot_inductor import AOTInductorBackend
from .eager import EagerBackend
from .inductor import InductorBackend
from .registry import BackendRegistry

registry = BackendRegistry()
registry.register(EagerBackend())
registry.register(InductorBackend())
registry.register(AOTInductorBackend())

__all__ = ["registry"]
