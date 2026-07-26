from .api import backends, clear_cache, compile, detect_targets, explain, version
from .targets import DeviceInfo, TargetSpec, parse_target

__all__ = [
    "DeviceInfo",
    "TargetSpec",
    "backends",
    "clear_cache",
    "compile",
    "detect_targets",
    "explain",
    "parse_target",
    "version",
]

__version__ = "0.1.0"
