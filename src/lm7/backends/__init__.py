from .eager import EagerBackend
from .inductor import InductorBackend
from .registry import BackendRegistry

registry = BackendRegistry()
registry.register(EagerBackend())
registry.register(InductorBackend())

__all__ = ["registry"]
