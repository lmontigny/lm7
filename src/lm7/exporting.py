from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .cache import input_signature
from .detection import resolve_target
from .errors import ArtifactLoadError, UnsupportedModelError
from .targets import TargetSpec, parse_target

FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
PROGRAM_NAME = "exported_program.pt2"


@dataclass(frozen=True)
class ArtifactManifest:
    format_version: int
    lm7_version: str
    torch_version: str
    created_at: str
    target: Mapping[str, Any]
    model_graph_hash: str
    cache_key: str
    input_signature: Any
    program_file: str
    program_sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactManifest:
        required = {field.name for field in cls.__dataclass_fields__.values()}
        missing = required - value.keys()
        if missing:
            raise ArtifactLoadError(
                f"Artifact manifest is missing required fields: {', '.join(sorted(missing))}."
            )
        return cls(**{name: value[name] for name in required})


@dataclass(frozen=True)
class ExportArtifact:
    path: Path
    manifest: ArtifactManifest
    exported_program: torch.export.ExportedProgram

    def module(self) -> torch.nn.Module:
        return self.exported_program.module()


def export(
    model: torch.nn.Module | torch.export.ExportedProgram,
    *,
    args: tuple[Any, ...] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    target: str | TargetSpec = "auto",
    output: str | os.PathLike[str],
    dynamic_shapes: Any = None,
    strict: bool = False,
) -> ExportArtifact:
    """Capture and persist a versioned LM7 source artifact."""
    kwargs = dict(kwargs or {})
    if isinstance(model, torch.export.ExportedProgram):
        if args is not None or kwargs:
            raise ValueError("args and kwargs cannot be supplied with an ExportedProgram.")
        exported_program = model
        signature: Any = None
    elif isinstance(model, torch.nn.Module):
        if args is None:
            raise ValueError("args must be supplied when exporting an nn.Module.")
        try:
            exported_program = torch.export.export(
                model,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
            )
        except Exception as exc:
            raise UnsupportedModelError(
                f"Model export stage failed for target {target}: {exc}. "
                "Check that the model is export-compatible and provide representative inputs."
            ) from exc
        signature = input_signature(args, kwargs)
    else:
        raise TypeError("model must be an nn.Module or torch.export.ExportedProgram.")

    resolved_target = _artifact_target(target)
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"Artifact output {destination} already exists; choose a new path or remove it explicitly."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    graph_hash = _graph_hash(exported_program)
    cache_key = artifact_cache_key(graph_hash, signature, resolved_target)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=str(destination.parent)))
    try:
        program_path = staging / PROGRAM_NAME
        torch.export.save(exported_program, program_path)
        program_sha256 = _file_sha256(program_path)
        manifest = ArtifactManifest(
            format_version=FORMAT_VERSION,
            lm7_version=_lm7_version(),
            torch_version=torch.__version__,
            created_at=datetime.now(timezone.utc).isoformat(),
            target=asdict(resolved_target),
            model_graph_hash=graph_hash,
            cache_key=cache_key,
            input_signature=_json_value(signature),
            program_file=PROGRAM_NAME,
            program_sha256=program_sha256,
        )
        (staging / MANIFEST_NAME).write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ExportArtifact(destination, manifest, exported_program)


def load_artifact(path: str | os.PathLike[str]) -> ExportArtifact:
    """Load an LM7 source artifact after validating its metadata and payload."""
    artifact_path = Path(path).expanduser().resolve()
    manifest_path = artifact_path / MANIFEST_NAME
    if not artifact_path.is_dir() or not manifest_path.is_file():
        raise ArtifactLoadError(
            f"Artifact load stage failed for {artifact_path}: {MANIFEST_NAME} was not found."
        )
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = ArtifactManifest.from_dict(raw_manifest)
    except ArtifactLoadError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ArtifactLoadError(
            f"Artifact load stage failed for {artifact_path}: invalid manifest: {exc}."
        ) from exc
    if manifest.format_version != FORMAT_VERSION:
        raise ArtifactLoadError(
            f"Unsupported LM7 artifact format {manifest.format_version}; "
            f"this LM7 version supports format {FORMAT_VERSION}."
        )
    program_path = artifact_path / manifest.program_file
    if not program_path.is_file():
        raise ArtifactLoadError(
            f"Artifact load stage failed for {artifact_path}: {manifest.program_file} is missing."
        )
    if _file_sha256(program_path) != manifest.program_sha256:
        raise ArtifactLoadError(
            f"Artifact load stage failed for {artifact_path}: program checksum does not match "
            "the manifest. Re-export the model."
        )
    try:
        exported_program = torch.export.load(program_path)
    except Exception as exc:
        raise ArtifactLoadError(
            f"Artifact load stage failed for {artifact_path}: torch.export.load failed: {exc}."
        ) from exc
    return ExportArtifact(artifact_path, manifest, exported_program)


def artifact_cache_key(model_graph_hash: str, signature: Any, target: TargetSpec) -> str:
    payload = {
        "format_version": FORMAT_VERSION,
        "lm7_version": _lm7_version(),
        "torch_version": torch.__version__,
        "model_graph_hash": model_graph_hash,
        "input_signature": _json_value(signature),
        "target": asdict(target),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact_target(target: str | TargetSpec) -> TargetSpec:
    parsed = parse_target(target)
    return resolve_target(parsed) if parsed.vendor == "auto" else parsed


def _graph_hash(exported_program: torch.export.ExportedProgram) -> str:
    graph = str(exported_program.graph_module.graph)
    state_metadata = sorted(
        (name, tuple(value.shape), str(value.dtype))
        for name, value in exported_program.state_dict.items()
    )
    value = json.dumps({"graph": graph, "state": state_metadata}, sort_keys=True)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _lm7_version() -> str:
    from . import __version__

    return __version__
