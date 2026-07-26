import pytest

from lm7.backends.eager import EagerBackend
from lm7.backends.registry import BackendRegistry


def test_registration_and_duplicate():
    registry = BackendRegistry()
    registry.register(EagerBackend())
    assert registry.get("eager").name == "eager"
    with pytest.raises(ValueError):
        registry.register(EagerBackend())
