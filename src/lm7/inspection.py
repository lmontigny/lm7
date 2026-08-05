from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import ArtifactLoadError
from .exporting import FORMAT_VERSION, MANIFEST_NAME, ArtifactManifest
from .targets import TargetSpec

LOW_DELEGATION_RATIO = 0.5


@dataclass(frozen=True)
class PayloadInspection:
    """Checksum result for one file declared by an artifact manifest."""

    kind: str
    file: str
    status: str
    expected_sha256: str | None
    actual_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactInspection:
    """Deployment metadata and integrity results for an LM7 artifact."""

    path: Path
    format_version: int
    backend: str
    target: str
    payload: str | None
    device_bound: bool
    host_executable: bool
    deployment: str
    checksums: tuple[PayloadInspection, ...]
    runtime_requirements: Mapping[str, Any]
    delegated_calls: int | None = None
    total_calls: int | None = None
    delegation_ratio: float | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def checksum_status(self) -> str:
        return "valid" if all(check.status == "valid" for check in self.checksums) else "invalid"

    @property
    def valid(self) -> bool:
        return not self.errors and self.checksum_status == "valid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "format_version": self.format_version,
            "backend": self.backend,
            "target": self.target,
            "payload": self.payload,
            "device_bound": self.device_bound,
            "host_executable": self.host_executable,
            "deployment": self.deployment,
            "checksum_status": self.checksum_status,
            "checksums": [check.to_dict() for check in self.checksums],
            "runtime_requirements": dict(self.runtime_requirements),
            "delegated_calls": self.delegated_calls,
            "total_calls": self.total_calls,
            "delegation_ratio": self.delegation_ratio,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "valid": self.valid,
        }


def inspect_artifact(path: str | Path) -> ArtifactInspection:
    """Inspect an artifact without importing or initializing its runtime backend."""
    artifact_path = Path(path).expanduser().resolve()
    manifest = _read_manifest(artifact_path)
    requirements = dict(manifest.runtime_requirements or {})
    checks = _payload_checks(artifact_path, manifest)
    errors = (
        *(
            f"{check.file}: {check.status.replace('_', ' ')}"
            for check in checks
            if check.status != "valid"
        ),
        *_deployment_errors(manifest.backend, requirements),
    )
    delegated_calls = _optional_int(requirements.get("delegated_calls"))
    total_calls = _optional_int(requirements.get("total_calls"))
    ratio = (
        delegated_calls / total_calls
        if delegated_calls is not None and total_calls is not None and total_calls > 0
        else None
    )
    warnings: list[str] = []
    if ratio is not None and ratio < LOW_DELEGATION_RATIO:
        warnings.append(
            f"Low delegate coverage: {delegated_calls}/{total_calls} calls "
            f"({ratio:.0%}); less than the {LOW_DELEGATION_RATIO:.0%} inspection heuristic."
        )

    return ArtifactInspection(
        path=artifact_path,
        format_version=manifest.format_version,
        backend=manifest.backend,
        target=_target_name(manifest.target),
        payload=manifest.compiled_file or manifest.program_file,
        device_bound=bool(requirements.get("device_bound", False)),
        host_executable=_host_executable(manifest.backend, requirements),
        deployment=_deployment_summary(manifest.backend, requirements),
        checksums=checks,
        runtime_requirements=requirements,
        delegated_calls=delegated_calls,
        total_calls=total_calls,
        delegation_ratio=ratio,
        warnings=tuple(warnings),
        errors=errors,
    )


def _read_manifest(artifact_path: Path) -> ArtifactManifest:
    manifest_path = artifact_path / MANIFEST_NAME
    if not artifact_path.is_dir() or not manifest_path.is_file():
        raise ArtifactLoadError(
            f"Artifact inspection failed for {artifact_path}: {MANIFEST_NAME} was not found."
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("manifest root must be an object")
        manifest = ArtifactManifest.from_dict(raw)
        if not isinstance(manifest.target, Mapping):
            raise TypeError("target must be an object")
        if manifest.runtime_requirements is not None and not isinstance(
            manifest.runtime_requirements, Mapping
        ):
            raise TypeError("runtime_requirements must be an object or null")
    except ArtifactLoadError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ArtifactLoadError(
            f"Artifact inspection failed for {artifact_path}: invalid manifest: {exc}."
        ) from exc
    if manifest.format_version != FORMAT_VERSION:
        raise ArtifactLoadError(
            f"Unsupported LM7 artifact format {manifest.format_version}; "
            f"this LM7 version supports format {FORMAT_VERSION}."
        )
    return manifest


def _payload_checks(
    artifact_path: Path, manifest: ArtifactManifest
) -> tuple[PayloadInspection, ...]:
    declared = [
        ("source", manifest.program_file, manifest.program_sha256),
        ("compiled", manifest.compiled_file, manifest.compiled_sha256),
        ("compiled_weights", manifest.compiled_weights_file, manifest.compiled_weights_sha256),
    ]
    return tuple(
        _inspect_payload(artifact_path, kind, name, digest)
        for kind, name, digest in declared
        if kind == "source"
        or (kind == "compiled" and manifest.backend != "export")
        or name is not None
        or digest is not None
    )


def _inspect_payload(
    artifact_path: Path,
    kind: str,
    name: Any,
    expected_sha256: Any,
) -> PayloadInspection:
    if not isinstance(name, str) or not name:
        expected = expected_sha256 if isinstance(expected_sha256, str) else None
        return PayloadInspection(kind, "<undeclared>", "missing_metadata", expected, None)
    if not isinstance(expected_sha256, str) or not expected_sha256:
        return PayloadInspection(kind, name, "missing_checksum", None, None)
    payload = (artifact_path / name).resolve()
    if not payload.is_relative_to(artifact_path):
        return PayloadInspection(kind, name, "invalid_path", expected_sha256, None)
    if not payload.is_file():
        return PayloadInspection(kind, name, "missing", expected_sha256, None)
    actual = _file_sha256(payload)
    return PayloadInspection(
        kind,
        name,
        "valid" if actual == expected_sha256 else "mismatch",
        expected_sha256,
        actual,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_name(value: Mapping[str, Any]) -> str:
    try:
        return str(
            TargetSpec(
                vendor=str(value["vendor"]),
                kind=str(value["kind"]),
                architecture=_optional_str(value.get("architecture")),
                model=_optional_str(value.get("model")),
                ordinal=_optional_int(value.get("ordinal")),
                remote=bool(value.get("remote", False)),
            )
        )
    except KeyError as exc:
        raise ArtifactLoadError(
            f"Artifact target is missing required field {exc.args[0]!r}."
        ) from exc


def _host_executable(backend: str, requirements: Mapping[str, Any]) -> bool:
    # Device-bound TensorRT/AOTInductor payloads can still execute on a compatible
    # build host. QNN is different: LM7 deliberately exposes it as deployment-only.
    return backend != "qnn"


def _deployment_errors(backend: str, requirements: Mapping[str, Any]) -> tuple[str, ...]:
    if backend != "qnn":
        return ()
    required = ("soc_model", "htp_arch", "precision", "qnn_sdk")
    errors = [
        f"runtime_requirements.{name}: missing" for name in required if not requirements.get(name)
    ]
    libraries = requirements.get("runtime_libraries")
    if not isinstance(libraries, list) or not libraries:
        errors.append("runtime_requirements.runtime_libraries: missing")
    return tuple(errors)


def _deployment_summary(backend: str, requirements: Mapping[str, Any]) -> str:
    if backend == "qnn":
        soc = requirements.get("soc_model", "matching Qualcomm SoC")
        sdk = requirements.get("qnn_sdk")
        suffix = f" and QNN SDK {sdk}" if sdk else " and a matching QNN SDK"
        return f"requires Android {soc} HTP runtime{suffix}"
    if backend == "executorch":
        return "portable ExecuTorch/XNNPACK runtime"
    if backend == "tensorrt":
        architecture = requirements.get("compute_capability") or "matching NVIDIA GPU"
        return f"requires matching CUDA, TensorRT, and GPU architecture ({architecture})"
    if backend == "aot_inductor" and requirements.get("device_bound"):
        # Name the architecture the way the tensorrt branch does. "Matching
        # PyTorch" is deliberately not claimed: a package built by one minor
        # release loads under its neighbour, so the CUDA runtime is the part
        # that has to line up -- see docs/aot-artifact-compatibility.md.
        architecture = requirements.get("compute_capability") or "matching GPU"
        cuda = requirements.get("cuda")
        runtime = f"CUDA {cuda} PyTorch runtime" if cuda else "matching CUDA PyTorch runtime"
        return f"requires a matching GPU architecture ({architecture}) and a {runtime}"
    if backend == "export":
        return "portable PyTorch ExportedProgram"
    if requirements.get("device_bound"):
        return "requires a compatible target device and runtime"
    return f"requires the {backend} runtime"


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None
