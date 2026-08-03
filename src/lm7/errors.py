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


class ConfigurationError(LM7Error):
    """An option LM7 was asked for is not one LM7 offers.

    Deliberately not a CompilationError: `fallback="warn"` exists to survive a
    backend that cannot compile a model, not to paper over a typo in the call
    site. Falling back would answer a misspelled option by silently dropping the
    compiler, so this escapes the fallback boundary and reaches the caller.
    """


class ArtifactLoadError(LM7Error):
    pass


class InputDeviceError(LM7Error):
    pass
