from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .detection import detect_targets, resolve_target
from .errors import ArtifactLoadError
from .exporting import ArtifactManifest, ExportArtifact, load_artifact
from .targets import TargetSpec

BUNDLE_FORMAT_VERSION = 1
BUNDLE_MANIFEST_NAME = "manifest.json"
TARGETS_DIR_NAME = "targets"


@dataclass(frozen=True)
class BundleManifest:
    bundle_format_version: int
    created_at: str
    model_graph_hash: str
    entries: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ArtifactBundle:
    path: Path
    manifest: BundleManifest

    def available_targets(self) -> tuple[TargetSpec, ...]:
        return tuple(_target_from_dict(entry["target"]) for entry in self.manifest.entries)

    def load(
        self,
        target: str | TargetSpec = "auto",
        *,
        backend: str = "auto",
    ) -> ExportArtifact:
        local_targets = (
            tuple(device.target for device in detect_targets())
            if target == "auto"
            else (resolve_target(target),)
        )
        selected = None
        for local_target in local_targets:
            candidates = [
                entry
                for entry in self.manifest.entries
                if _target_matches(entry["target"], local_target)
                and (backend == "auto" or entry["backend"] == backend)
            ]
            if candidates:
                selected = max(candidates, key=lambda entry: _backend_priority(entry["backend"]))
                break
        if selected is None:
            requested = "any detected local target" if target == "auto" else str(local_targets[0])
            available = ", ".join(
                f"{_target_from_dict(entry['target'])}/{entry['backend']}"
                for entry in self.manifest.entries
            )
            raise ArtifactLoadError(
                f"Bundle has no artifact compatible with {requested} and backend {backend!r}. "
                f"Available: {available or 'none'}."
            )
        artifact_path = self.path / selected["path"]
        manifest_path = artifact_path / "manifest.json"
        if _file_sha256(manifest_path) != selected["manifest_sha256"]:
            raise ArtifactLoadError(
                f"Bundled artifact {selected['path']} manifest checksum does not match."
            )
        return load_artifact(artifact_path)


def create_bundle(
    artifacts: Iterable[str | os.PathLike[str]],
    *,
    output: str | os.PathLike[str],
) -> ArtifactBundle:
    """Package independently built target artifacts into one immutable directory."""
    sources = tuple(Path(path).expanduser().resolve() for path in artifacts)
    if not sources:
        raise ValueError("At least one artifact is required to create a bundle.")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"Bundle output {destination} already exists; choose a new path or remove it explicitly."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=str(destination.parent)))
    entries: list[Mapping[str, Any]] = []
    seen_keys: set[str] = set()
    model_graph_hash: str | None = None
    try:
        targets_dir = staging / TARGETS_DIR_NAME
        targets_dir.mkdir()
        for source in sources:
            artifact_manifest = _read_artifact_manifest(source)
            _validate_artifact_payloads(source, artifact_manifest)
            if model_graph_hash is None:
                model_graph_hash = artifact_manifest.model_graph_hash
            elif artifact_manifest.model_graph_hash != model_graph_hash:
                raise ValueError(
                    "All bundled artifacts must contain the same exported model graph."
                )
            key = _entry_key(artifact_manifest)
            if key in seen_keys:
                raise ValueError(
                    f"Bundle already contains target/backend entry {key!r}; entries must be unique."
                )
            seen_keys.add(key)
            relative_path = Path(TARGETS_DIR_NAME) / key
            shutil.copytree(source, staging / relative_path)
            copied_manifest = staging / relative_path / "manifest.json"
            entries.append(
                {
                    "key": key,
                    "target": artifact_manifest.target,
                    "backend": artifact_manifest.backend,
                    "path": relative_path.as_posix(),
                    "manifest_sha256": _file_sha256(copied_manifest),
                }
            )
        assert model_graph_hash is not None
        manifest = BundleManifest(
            bundle_format_version=BUNDLE_FORMAT_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_graph_hash=model_graph_hash,
            entries=tuple(entries),
        )
        (staging / BUNDLE_MANIFEST_NAME).write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ArtifactBundle(destination, manifest)


def load_bundle(path: str | os.PathLike[str]) -> ArtifactBundle:
    bundle_path = Path(path).expanduser().resolve()
    manifest_path = bundle_path / BUNDLE_MANIFEST_NAME
    if not bundle_path.is_dir() or not manifest_path.is_file():
        raise ArtifactLoadError(
            f"Bundle load stage failed for {bundle_path}: {BUNDLE_MANIFEST_NAME} was not found."
        )
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = BundleManifest(
            bundle_format_version=value["bundle_format_version"],
            created_at=value["created_at"],
            model_graph_hash=value["model_graph_hash"],
            entries=tuple(value["entries"]),
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ArtifactLoadError(f"Bundle manifest is invalid: {exc}.") from exc
    if manifest.bundle_format_version != BUNDLE_FORMAT_VERSION:
        raise ArtifactLoadError(
            f"Unsupported bundle format {manifest.bundle_format_version}; "
            f"this LM7 version supports {BUNDLE_FORMAT_VERSION}."
        )
    return ArtifactBundle(bundle_path, manifest)


def _read_artifact_manifest(path: Path) -> ArtifactManifest:
    manifest_path = path / "manifest.json"
    if not path.is_dir() or not manifest_path.is_file():
        raise ArtifactLoadError(f"{path} is not an LM7 artifact directory.")
    try:
        return ArtifactManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    except ArtifactLoadError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactLoadError(f"Artifact manifest at {manifest_path} is invalid: {exc}.") from exc


def _validate_artifact_payloads(path: Path, manifest: ArtifactManifest) -> None:
    program_path = path / manifest.program_file
    if not program_path.is_file() or _file_sha256(program_path) != manifest.program_sha256:
        raise ArtifactLoadError(f"Artifact {path} has an invalid exported program payload.")
    if manifest.compiled_file:
        compiled_path = path / manifest.compiled_file
        if (
            not manifest.compiled_sha256
            or not compiled_path.is_file()
            or _file_sha256(compiled_path) != manifest.compiled_sha256
        ):
            raise ArtifactLoadError(f"Artifact {path} has an invalid compiled payload.")
    # The weights sibling some backends carry beside the graph: OpenVINO's .bin,
    # and ONNX Runtime's sidecar once the weights outgrow protobuf's 2 GiB
    # ceiling. `load_artifact` checks it too, but only when the artifact is
    # finally loaded -- by which point a corrupt payload has already been copied
    # into a bundle and shipped. Catching it here fails on the machine that can
    # still re-export.
    if manifest.compiled_weights_file:
        weights_path = path / manifest.compiled_weights_file
        if (
            not manifest.compiled_weights_sha256
            or not weights_path.is_file()
            or _file_sha256(weights_path) != manifest.compiled_weights_sha256
        ):
            raise ArtifactLoadError(f"Artifact {path} has an invalid compiled weights payload.")


def _entry_key(manifest: ArtifactManifest) -> str:
    target = _target_from_dict(manifest.target)
    qualifier = target.architecture or target.model or "generic"
    raw = f"{target.vendor}-{qualifier}--{manifest.backend}"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw)


def _target_from_dict(value: Mapping[str, Any]) -> TargetSpec:
    return TargetSpec(
        vendor=value["vendor"],
        kind=value["kind"],
        architecture=value.get("architecture"),
        model=value.get("model"),
        ordinal=value.get("ordinal"),
        remote=value.get("remote", False),
    )


def _target_matches(value: Mapping[str, Any], local: TargetSpec) -> bool:
    candidate = _target_from_dict(value)
    if candidate.vendor != local.vendor or candidate.kind != local.kind:
        return False
    if candidate.architecture and candidate.architecture != local.architecture:
        return False
    return not (candidate.model and local.model and candidate.model != local.model)


def _backend_priority(backend: str) -> int:
    # A compiled payload beats a bare ExportedProgram. OpenVINO sits below
    # AOTInductor, and TensorRT between them, to match the backend registry's
    # ranking.
    return {"aot_inductor": 100, "tensorrt": 90, "openvino": 80, "export": 0}.get(backend, 50)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
