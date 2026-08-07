"""Engine cache metadata -- an LM7 responsibility, not the runtime's.

A TensorRT-LLM engine is expensive to build and narrowly pinned: it is valid only
for the architecture, the runtime version, and the shape bounds it was built
with. Reusing one across any of those silently produces wrong numbers or a crash
deep inside the runtime, so the cache key is computed here and written beside the
engine as a manifest.

This is the same lesson as the AOTInductor architecture guard, which
`docs/aot-artifact-compatibility.md` records: an artifact that does not say what
it was built for will eventually be loaded somewhere it does not belong.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..cache import cache_dir
from ..targets import TargetSpec
from .base import RuntimeInfo, ServeConfig


@dataclass(frozen=True)
class EngineIdentity:
    """What an engine is pinned to. Any difference means a rebuild."""

    runtime: str
    model_id: str
    architecture: str | None
    config: dict[str, Any]
    pinned: dict[str, str | None]

    def key(self) -> str:
        payload = json.dumps(
            {
                "runtime": self.runtime,
                "model_id": self.model_id,
                "architecture": self.architecture,
                "config": self.config,
                "pinned": self.pinned,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def engine_identity(
    runtime: RuntimeInfo, target: TargetSpec, model_id: str, config: ServeConfig
) -> EngineIdentity:
    return EngineIdentity(
        runtime=runtime.name,
        model_id=model_id,
        architecture=target.architecture,
        config=config.as_dict(),
        pinned=dict(runtime.pinned),
    )


def engine_dir(identity: EngineIdentity, root: Path | None = None) -> Path:
    base = root if root is not None else cache_dir() / "engines"
    return base / f"{identity.runtime}-{identity.key()}"


MANIFEST_NAME = "lm7-engine.json"


def write_manifest(directory: Path, identity: EngineIdentity) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "key": identity.key(),
                "runtime": identity.runtime,
                "model_id": identity.model_id,
                "architecture": identity.architecture,
                "config": identity.config,
                "pinned": identity.pinned,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def read_manifest(directory: Path) -> dict[str, Any] | None:
    path = directory / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def reusable(directory: Path, identity: EngineIdentity) -> tuple[bool, str]:
    """Whether the engine already in `directory` was built for this identity.

    Returns a reason on refusal rather than a bare False: "rebuild" is a
    multi-minute cost and a user is owed the reason they are paying it. The
    directory name already contains the key, so a mismatch here means either a
    hand-edited cache or a manifest from a different LM7 version -- both worth
    saying out loud rather than silently rebuilding over.
    """
    manifest = read_manifest(directory)
    if manifest is None:
        return False, f"no {MANIFEST_NAME} in {directory}"
    if manifest.get("key") != identity.key():
        return False, (
            f"engine in {directory} was built for key {manifest.get('key')!r}, "
            f"this request is {identity.key()!r}"
        )
    differences = [
        name
        for name, expected in (
            ("architecture", identity.architecture),
            ("model_id", identity.model_id),
            ("runtime", identity.runtime),
        )
        if manifest.get(name) != expected
    ]
    if differences:
        return False, f"manifest disagrees on {', '.join(differences)}"
    return True, "manifest matches"
