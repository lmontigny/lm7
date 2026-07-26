class LM7Error(RuntimeError):
    """Base exception for LM7 failures."""


class TargetNotFoundError(LM7Error):
    pass


class BackendUnavailableError(LM7Error):
    pass


class UnsupportedModelError(LM7Error):
    pass


class CompilationError(LM7Error):
    pass


class ArtifactLoadError(LM7Error):
    pass


class InputDeviceError(LM7Error):
    pass
