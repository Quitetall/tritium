"""Immutable PTQ artifacts, atomic export, and strict native reload."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .. import _tritium
from .errors import TritiumError

_ARTIFACT_KIND_V1 = "qwen3.6-language-mtp-salt-v2-matrix-profiles"
_ARTIFACT_KIND_V2 = "qwen3.6-language-mtp-salt-v2-model-weights"
_ARTIFACT_KIND_V3 = "qwen3.6-language-mtp-salt-v2-hf-bundle"
_MANIFEST = "tritium.json"
_PRESERVED_FILE = "preserved.safetensors"
_PROFILE_FILES = {
    "compact-v1": "compact.tsalt2",
    "near-lossless-v1": "near-lossless.tsalt2",
}
_TOP_LEVEL_FIELDS_V1 = {
    "schema_version",
    "artifact_kind",
    "complete_model",
    "packing",
    "completion_id",
    "campaign_id",
    "admission_id",
    "selection_id",
    "source_model_id",
    "source_identity_status",
    "official_payload_authenticated",
    "profiles",
}
_TOP_LEVEL_FIELDS_V2 = _TOP_LEVEL_FIELDS_V1 | {"preserved"}
_TOP_LEVEL_FIELDS_V3 = _TOP_LEVEL_FIELDS_V2 | {"hf_assets", "source_revision"}
_PROFILE_FIELDS = {
    "file",
    "package_id",
    "serialized_bytes",
    "resident_bytes",
}
_PRESERVED_FIELDS = {
    "file",
    "package_id",
    "tensors",
    "payload_bytes",
    "serialized_bytes",
}
_HF_ASSET_FIELDS = {"file", "package_id", "bytes"}
_HF_ASSET_FILES = (
    "chat_template.jinja",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
_QWEN36_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


@dataclass(frozen=True)
class ArtifactRef:
    """One strictly reopened native SALT V2 profile package."""

    profile: str
    path: Path
    package_id: str
    packing: str
    serialized_bytes: int
    resident_bytes: int
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "path": str(self.path),
            "package_id": self.package_id,
            "packing": self.packing,
            "serialized_bytes": self.serialized_bytes,
            "resident_bytes": self.resident_bytes,
        }


@dataclass(frozen=True)
class PreservedRef:
    """One content-bound exact-BF16 safetensors companion."""

    path: Path
    package_id: str
    tensors: int
    payload_bytes: int
    serialized_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "package_id": self.package_id,
            "tensors": self.tensors,
            "payload_bytes": self.payload_bytes,
            "serialized_bytes": self.serialized_bytes,
        }


@dataclass(frozen=True)
class HfAssetRef:
    """One exact, bounded Hugging Face language sidecar."""

    file: str
    path: Path
    package_id: str
    bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "path": str(self.path),
            "package_id": self.package_id,
            "bytes": self.bytes,
        }


@dataclass(frozen=True)
class ExportReceipt:
    """Proof that a complete bundle directory was atomically published."""

    artifact_dir: Path
    manifest_path: Path
    admission_id: str
    compact_package_id: str
    near_lossless_package_id: str
    preserved_package_id: Optional[str] = None
    schema_version: int = 1


@dataclass(frozen=True)
class QuantizationResult:
    """Verified two-profile PTQ result for Qwen3.6 language and MTP matrices.

    ``complete_model`` is intentionally false until Hugging Face configuration,
    tokenizer assets, and the Qwen3.6 runtime adapter are present. Schema-v2
    adds exact preserved BF16 tensors; schema-v3 adds the pinned language-side
    Hugging Face asset catalog. SALT V2 packages retain measured physical ledgers.
    """

    artifact_dir: Path
    packing: str
    completion_id: str
    campaign_id: str
    admission_id: str
    selection_id: str
    source_model_id: str
    source_identity_status: str
    official_payload_authenticated: bool
    compact: ArtifactRef
    near_lossless: ArtifactRef
    preserved: Optional[PreservedRef] = None
    hf_assets: Tuple[HfAssetRef, ...] = ()
    source_revision: Optional[str] = None
    complete_model: bool = False
    schema_version: int = 1

    def artifact(self, profile: str) -> ArtifactRef:
        if profile == "compact-v1":
            return self.compact
        if profile == "near-lossless-v1":
            return self.near_lossless
        raise ValueError("profile must be 'compact-v1' or 'near-lossless-v1'")

    def export(self, output_dir: Union[os.PathLike[str], str]) -> ExportReceipt:
        return export(self, output_dir)

    def save_pretrained(self, output_dir: Union[os.PathLike[str], str]) -> None:
        missing = ["qwen3.6_runtime_adapter"]
        if not self.hf_assets:
            missing[:0] = ["config", "tokenizer_assets"]
        if self.preserved is None:
            missing.insert(0, "preserved_bf16_tensors")
        raise TritiumError(
            "this matrix-profile result is not a self-contained Hugging Face model",
            code="incomplete_artifact",
            stage="export",
            details={
                "missing": missing,
                "requested_output": str(output_dir),
            },
        )


def _pairs_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _read_manifest(directory: Path) -> Dict[str, Any]:
    manifest = directory / _MANIFEST
    metadata = manifest.lstat()
    if manifest.is_symlink() or not manifest.is_file() or metadata.st_size > 1024 * 1024:
        raise ValueError("tritium.json must be an ordinary manifest no larger than 1 MiB")
    with manifest.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_pairs_without_duplicates)
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
        raise ValueError("tritium.json must declare an integer schema_version")
    version = value["schema_version"]
    if version not in {1, 2, 3}:
        raise ValueError("unsupported Tritium bundle schema_version")
    fields = {
        1: _TOP_LEVEL_FIELDS_V1,
        2: _TOP_LEVEL_FIELDS_V2,
        3: _TOP_LEVEL_FIELDS_V3,
    }[version]
    if set(value) != fields:
        raise ValueError(f"tritium.json fields do not match bundle schema version {version}")
    expected_kind = {
        1: _ARTIFACT_KIND_V1,
        2: _ARTIFACT_KIND_V2,
        3: _ARTIFACT_KIND_V3,
    }[version]
    if value["artifact_kind"] != expected_kind or value["complete_model"] is not False:
        raise ValueError("unsupported Tritium artifact kind")
    if version == 3 and value["source_revision"] != _QWEN36_REVISION:
        raise ValueError("schema-v3 source_revision differs from pinned Qwen3.6 revision")
    if value["packing"] not in {"d2", "b3", "s34"}:
        raise ValueError("invalid Tritium package codec")
    if type(value["official_payload_authenticated"]) is not bool:
        raise ValueError("official_payload_authenticated must be boolean")
    for field in (
        "completion_id",
        "campaign_id",
        "admission_id",
        "selection_id",
        "source_model_id",
        "source_identity_status",
    ):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"{field} must be a non-empty string")
    profiles = value["profiles"]
    if not isinstance(profiles, dict) or set(profiles) != set(_PROFILE_FILES):
        raise ValueError("bundle must contain exactly both governed profiles")
    return value


def _load_hf_assets(directory: Path, value: Any) -> Tuple[HfAssetRef, ...]:
    if not isinstance(value, list) or len(value) != len(_HF_ASSET_FILES):
        raise ValueError("hf_assets must contain exact language asset catalog")
    assets = []
    for expected_file, item in zip(_HF_ASSET_FILES, value):
        if not isinstance(item, dict) or set(item) != _HF_ASSET_FIELDS:
            raise ValueError("HF asset fields do not match bundle schema version 3")
        if item["file"] != expected_file:
            raise ValueError("HF assets are missing, duplicated, or out of canonical order")
        package_id = item["package_id"]
        byte_count = item["bytes"]
        if not isinstance(package_id, str) or not package_id:
            raise ValueError(f"HF asset {expected_file} package_id must be non-empty")
        if type(byte_count) is not int or byte_count <= 0:
            raise ValueError(f"HF asset {expected_file} bytes must be positive")
        path = directory / expected_file
        actual_id, actual_bytes = _tritium.verify_hf_asset(
            str(path), package_id, byte_count
        )
        assets.append(
            HfAssetRef(
                file=expected_file,
                path=path,
                package_id=actual_id,
                bytes=actual_bytes,
            )
        )
    return tuple(assets)


def _load_preserved(directory: Path, value: Any) -> PreservedRef:
    if not isinstance(value, dict) or set(value) != _PRESERVED_FIELDS:
        raise ValueError("preserved fields do not match bundle schema version 2")
    if value["file"] != _PRESERVED_FILE:
        raise ValueError("preserved tensors use a non-canonical filename")
    package_id = value["package_id"]
    integers = (value["tensors"], value["payload_bytes"], value["serialized_bytes"])
    if not isinstance(package_id, str) or not package_id:
        raise ValueError("preserved package_id must be a non-empty string")
    if any(type(item) is not int or item <= 0 for item in integers):
        raise ValueError("preserved tensor and byte ledgers must be positive integers")
    path = directory / _PRESERVED_FILE
    actual = _tritium.verify_preserved_safetensors(path.as_posix(), package_id, *integers)
    return PreservedRef(
        path=path,
        package_id=actual[0],
        tensors=actual[1],
        payload_bytes=actual[2],
        serialized_bytes=actual[3],
    )


def _load_profile(directory: Path, packing: str, profile: str, value: Any) -> ArtifactRef:
    if not isinstance(value, dict) or set(value) != _PROFILE_FIELDS:
        raise ValueError(f"{profile} fields do not match bundle schema version 1")
    if value["file"] != _PROFILE_FILES[profile]:
        raise ValueError(f"{profile} uses a non-canonical package filename")
    package_id = value["package_id"]
    serialized = value["serialized_bytes"]
    resident = value["resident_bytes"]
    if not isinstance(package_id, str) or not package_id:
        raise ValueError(f"{profile} package_id must be a non-empty string")
    if type(serialized) is not int or type(resident) is not int:
        raise ValueError(f"{profile} physical bytes must be integers")
    if serialized <= 0 or resident <= 0:
        raise ValueError(f"{profile} physical bytes must be positive")
    path = directory / value["file"]
    actual_id, actual_packing, actual_serialized, actual_resident = (
        _tritium.verify_salt_v2_package(
            str(path), package_id, serialized, resident
        )
    )
    if actual_packing != packing:
        raise ValueError(f"{profile} codec differs from the bundle recipe")
    return ArtifactRef(
        profile=profile,
        path=path,
        package_id=actual_id,
        packing=actual_packing,
        serialized_bytes=actual_serialized,
        resident_bytes=actual_resident,
    )


def load(
    artifact: Union[os.PathLike[str], str], *, device: Optional[str] = None
) -> QuantizationResult:
    """Strictly reopen a two-profile Tritium PTQ bundle.

    Loading re-parses and hashes every package through the native seek-backed
    reader. It returns artifact evidence, not an inference model; device
    placement is therefore rejected until the Qwen3.6 runtime adapter lands.
    """

    if device is not None:
        raise TritiumError(
            "matrix-bundle load does not perform device placement",
            code="unsupported_device_load",
            stage="load",
            details={"device": device},
        )
    requested = Path(artifact)
    if requested.is_symlink():
        raise ValueError("artifact must be an ordinary Tritium bundle directory")
    directory = requested.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("artifact must be an ordinary Tritium bundle directory")
    value = _read_manifest(directory)
    compact = _load_profile(
        directory, value["packing"], "compact-v1", value["profiles"]["compact-v1"]
    )
    near = _load_profile(
        directory,
        value["packing"],
        "near-lossless-v1",
        value["profiles"]["near-lossless-v1"],
    )
    preserved = (
        _load_preserved(directory, value["preserved"])
        if value["schema_version"] >= 2
        else None
    )
    hf_assets = (
        _load_hf_assets(directory, value["hf_assets"])
        if value["schema_version"] >= 3
        else ()
    )
    return QuantizationResult(
        artifact_dir=directory,
        packing=value["packing"],
        completion_id=value["completion_id"],
        campaign_id=value["campaign_id"],
        admission_id=value["admission_id"],
        selection_id=value["selection_id"],
        source_model_id=value["source_model_id"],
        source_identity_status=value["source_identity_status"],
        official_payload_authenticated=value["official_payload_authenticated"],
        compact=compact,
        near_lossless=near,
        preserved=preserved,
        hf_assets=hf_assets,
        source_revision=value.get("source_revision"),
        schema_version=value["schema_version"],
    )


def _manifest_bytes(result: QuantizationResult) -> bytes:
    artifact_kind = {
        1: _ARTIFACT_KIND_V1,
        2: _ARTIFACT_KIND_V2,
        3: _ARTIFACT_KIND_V3,
    }.get(result.schema_version)
    if artifact_kind is None:
        raise ValueError("unsupported Tritium bundle schema_version")
    value = {
        "schema_version": result.schema_version,
        "artifact_kind": artifact_kind,
        "complete_model": False,
        "packing": result.packing,
        "completion_id": result.completion_id,
        "campaign_id": result.campaign_id,
        "admission_id": result.admission_id,
        "selection_id": result.selection_id,
        "source_model_id": result.source_model_id,
        "source_identity_status": result.source_identity_status,
        "official_payload_authenticated": result.official_payload_authenticated,
        "profiles": {
            "compact-v1": {
                "file": _PROFILE_FILES["compact-v1"],
                "package_id": result.compact.package_id,
                "serialized_bytes": result.compact.serialized_bytes,
                "resident_bytes": result.compact.resident_bytes,
            },
            "near-lossless-v1": {
                "file": _PROFILE_FILES["near-lossless-v1"],
                "package_id": result.near_lossless.package_id,
                "serialized_bytes": result.near_lossless.serialized_bytes,
                "resident_bytes": result.near_lossless.resident_bytes,
            },
        },
    }
    if result.preserved is not None:
        value["preserved"] = {
            "file": _PRESERVED_FILE,
            "package_id": result.preserved.package_id,
            "tensors": result.preserved.tensors,
            "payload_bytes": result.preserved.payload_bytes,
            "serialized_bytes": result.preserved.serialized_bytes,
        }
    if result.schema_version == 3:
        if len(result.hf_assets) != len(_HF_ASSET_FILES):
            raise ValueError("schema version 3 requires exact HF asset catalog")
        if result.source_revision != _QWEN36_REVISION:
            raise ValueError("schema version 3 requires pinned Qwen3.6 revision")
        value["source_revision"] = result.source_revision
        value["hf_assets"] = [
            {
                "file": asset.file,
                "package_id": asset.package_id,
                "bytes": asset.bytes,
            }
            for asset in result.hf_assets
        ]
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def export(
    result: QuantizationResult, output_dir: Union[os.PathLike[str], str]
) -> ExportReceipt:
    """Verify, stage, sync, and atomically copy one complete matrix bundle."""

    if not isinstance(result, QuantizationResult):
        raise TypeError("export requires a QuantizationResult")
    current = load(result.artifact_dir)
    if current != result:
        raise TritiumError(
            "QuantizationResult fields differ from its verified artifact",
            code="artifact_mismatch",
            stage="export",
        )
    target = Path(output_dir).absolute()
    parent = target.parent.resolve(strict=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output directory already exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=".tritium-stage-", dir=parent))
    published = False
    try:
        for artifact in (current.compact, current.near_lossless):
            destination = staging / _PROFILE_FILES[artifact.profile]
            shutil.copyfile(artifact.path, destination)
            with destination.open("rb") as stream:
                os.fsync(stream.fileno())
        if current.preserved is not None:
            destination = staging / _PRESERVED_FILE
            shutil.copyfile(current.preserved.path, destination)
            with destination.open("rb") as stream:
                os.fsync(stream.fileno())
        for asset in current.hf_assets:
            destination = staging / asset.file
            shutil.copyfile(asset.path, destination)
            with destination.open("rb") as stream:
                os.fsync(stream.fileno())
        manifest = staging / _MANIFEST
        with manifest.open("xb") as stream:
            stream.write(_manifest_bytes(current))
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        staged = load(staging)
        if current.preserved is None:
            preserved_changed = staged.preserved is not None
        else:
            preserved_changed = (
                staged.preserved is None
                or staged.preserved.package_id != current.preserved.package_id
                or staged.preserved.tensors != current.preserved.tensors
                or staged.preserved.payload_bytes != current.preserved.payload_bytes
                or staged.preserved.serialized_bytes != current.preserved.serialized_bytes
            )
        staged_assets = tuple(
            (asset.file, asset.package_id, asset.bytes) for asset in staged.hf_assets
        )
        current_assets = tuple(
            (asset.file, asset.package_id, asset.bytes) for asset in current.hf_assets
        )
        if (
            staged.completion_id != current.completion_id
            or staged.campaign_id != current.campaign_id
            or staged.admission_id != current.admission_id
            or staged.selection_id != current.selection_id
            or staged.compact.package_id != current.compact.package_id
            or staged.near_lossless.package_id != current.near_lossless.package_id
            or preserved_changed
            or staged_assets != current_assets
        ):
            raise TritiumError(
                "staged bundle identity differs from source artifact",
                code="artifact_mismatch",
                stage="export",
            )
        _tritium.publish_directory_noreplace(str(staging), str(target))
        published = True
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    reopened = load(target)
    return ExportReceipt(
        artifact_dir=reopened.artifact_dir,
        manifest_path=reopened.artifact_dir / _MANIFEST,
        admission_id=reopened.admission_id,
        compact_package_id=reopened.compact.package_id,
        near_lossless_package_id=reopened.near_lossless.package_id,
        preserved_package_id=(
            reopened.preserved.package_id if reopened.preserved is not None else None
        ),
        schema_version=reopened.schema_version,
    )


__all__ = [
    "ArtifactRef",
    "ExportReceipt",
    "HfAssetRef",
    "PreservedRef",
    "QuantizationResult",
    "export",
    "load",
]
