#!/usr/bin/env python3
"""Generate deterministic CycloneDX inventories for Tritium model/ONNX bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


KINDS = {"model-bundle", "onnx-bundle"}
ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
HEX_PATTERN = re.compile(r"[0-9a-f]{64}")
PACKAGE_ID_PATTERN = re.compile(r"trp1_[0-9a-f]{64}")
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024 * 1024
MAX_MEMBER_BYTES = 256 * 1024 * 1024 * 1024
MAX_MEMBERS = 200_000
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_IDENTITY_BYTES = 4096
CHUNK_BYTES = 1024 * 1024
PINNED_MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

MODEL_HF_FILES = (
    "chat_template.jinja",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
MODEL_FILES = set(MODEL_HF_FILES) | {
    "compact.tsalt2",
    "near-lossless.tsalt2",
    "preserved.safetensors",
    "tritium.json",
}
ONNX_FILES = {
    "language.onnx",
    "mtp.onnx",
    "tritium-onnx-manifest.json",
    "weights.bin",
}
MODEL_TOP_FIELDS = {
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
    "preserved",
    "hf_assets",
    "source_revision",
}
MODEL_PROFILE_FIELDS = {
    "file",
    "package_id",
    "serialized_bytes",
    "resident_bytes",
}
MODEL_PRESERVED_FIELDS = {
    "file",
    "package_id",
    "tensors",
    "payload_bytes",
    "serialized_bytes",
}
MODEL_HF_FIELDS = {"file", "package_id", "bytes"}
ONNX_TOP_FIELDS = {
    "schema",
    "language",
    "mtp",
    "weights",
    "identity",
    "conversion",
    "sequence_mode",
}
ONNX_IDENTITY_FIELDS = {
    "source_model_id",
    "tokenizer_id",
    "recipe_id",
    "package_id",
    "converted_coverage_id",
    "deferred_coverage_id",
}
ONNX_CONVERSION_FIELDS = {
    "mode",
    "completion_id",
    "campaign_id",
    "admission_id",
    "selection_id",
}


class BundleSbomError(ValueError):
    """Bundle archive or manifest cannot support an exact SBOM."""


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def _open_artifact(path: Path) -> tuple[Path, int, list[int]]:
    if (
        not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
    ):
        raise BundleSbomError("platform lacks safe no-follow directory traversal")
    artifact = Path(os.path.abspath(path))
    parts = artifact.parts
    if len(parts) < 2 or not artifact.is_absolute():
        raise BundleSbomError("bundle artifact path must be absolute")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        file_flags |= os.O_CLOEXEC
    directories: list[int] = []
    try:
        directories.append(os.open(parts[0], directory_flags))
        for part in parts[1:-1]:
            directories.append(os.open(part, directory_flags, dir_fd=directories[-1]))
        descriptor = os.open(parts[-1], file_flags, dir_fd=directories[-1])
        return artifact, descriptor, directories
    except OSError as error:
        for directory in reversed(directories):
            os.close(directory)
        raise BundleSbomError(
            f"cannot open bundle artifact without symlink traversal: {error}"
        ) from error


def _verify_path_chain(artifact: Path, descriptor: int, directories: list[int]) -> None:
    parts = artifact.parts
    try:
        for index, part in enumerate(parts[1:-1]):
            current = os.stat(part, dir_fd=directories[index], follow_symlinks=False)
            if not os.path.samestat(current, os.fstat(directories[index + 1])):
                raise BundleSbomError("bundle artifact parent path changed while generating SBOM")
        current = os.stat(parts[-1], dir_fd=directories[-1], follow_symlinks=False)
    except OSError as error:
        raise BundleSbomError(
            "bundle artifact parent path changed while generating SBOM"
        ) from error
    if not os.path.samestat(current, os.fstat(descriptor)):
        raise BundleSbomError("bundle artifact path changed while generating SBOM")


def _component_ref(name: str) -> str:
    return f"bundle-file:{hashlib.sha256(name.encode('utf-8')).hexdigest()}"


def _canonical_member(name: str) -> str:
    logical = PurePosixPath(name)
    if (
        not name
        or logical.is_absolute()
        or "\\" in name
        or "/" in name
        or logical.name != name
        or name in {".", ".."}
    ):
        raise BundleSbomError(f"bundle member {name!r} is not a flat canonical path")
    return name


def _read_member(
    stream: BinaryIO, size: int, name: str, digest_tool: str
) -> tuple[dict[str, Any], bytes | None]:
    if size < 0 or size > MAX_MEMBER_BYTES:
        raise BundleSbomError(f"bundle member {name!r} exceeds physical bounds")
    digest = hashlib.sha256()
    remaining = size
    retained = (
        bytearray()
        if name in {"tritium.json", "tritium-onnx-manifest.json"}
        else None
    )
    with tempfile.TemporaryFile() as diagnostics:
        try:
            process = subprocess.Popen(
                [digest_tool, "release", "digest-stream"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=diagnostics,
            )
        except OSError as error:
            raise BundleSbomError(f"cannot start stream digest tool: {error}") from error
        assert process.stdin is not None
        assert process.stdout is not None
        try:
            while remaining:
                chunk = stream.read(min(CHUNK_BYTES, remaining))
                if not chunk:
                    raise BundleSbomError(f"bundle member {name!r} is truncated")
                remaining -= len(chunk)
                digest.update(chunk)
                process.stdin.write(chunk)
                if retained is not None:
                    if len(retained) + len(chunk) > MAX_MANIFEST_BYTES:
                        raise BundleSbomError(
                            f"bundle manifest {name!r} exceeds metadata bounds"
                        )
                    retained.extend(chunk)
            process.stdin.close()
            payload = process.stdout.read(MAX_IDENTITY_BYTES + 1)
            process.stdout.close()
            status = process.wait()
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        if status != 0:
            diagnostics.seek(0)
            detail = diagnostics.read(4096).decode("utf-8", errors="replace").strip()
            raise BundleSbomError(
                f"stream digest tool failed for {name!r} with status {status}: {detail}"
            )
    if len(payload) > MAX_IDENTITY_BYTES:
        raise BundleSbomError(f"stream identity for {name!r} exceeds metadata bounds")
    try:
        identity = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleSbomError(f"stream identity for {name!r} is not JSON") from error
    if not isinstance(identity, dict) or set(identity) != {
        "schema",
        "bytes",
        "sha256",
        "blake3",
        "package_id",
    }:
        raise BundleSbomError(f"stream identity for {name!r} has wrong fields")
    if identity.get("schema") != "tritium.stream-identity.v1":
        raise BundleSbomError(f"stream identity for {name!r} has wrong schema")
    if identity.get("bytes") != size or identity.get("sha256") != digest.hexdigest():
        raise BundleSbomError(f"stream identity for {name!r} differs from tar bytes")
    if HEX_PATTERN.fullmatch(identity.get("blake3", "")) is None:
        raise BundleSbomError(f"stream identity for {name!r} has invalid BLAKE3")
    if PACKAGE_ID_PATTERN.fullmatch(identity.get("package_id", "")) is None:
        raise BundleSbomError(f"stream identity for {name!r} has invalid package ID")
    return identity, bytes(retained) if retained is not None else None


def _tar_number(field: bytes, label: str) -> int:
    if not field or field[0] & 0x80:
        raise BundleSbomError(f"tar {label} is not canonical octal")
    stripped = field.rstrip(b"\0 ").lstrip(b" ")
    if not stripped or any(byte not in b"01234567" for byte in stripped):
        raise BundleSbomError(f"tar {label} is not canonical octal")
    return int(stripped, 8)


def _tar_text(field: bytes, label: str) -> str:
    raw, separator, padding = field.partition(b"\0")
    if separator and any(padding):
        raise BundleSbomError(f"tar {label} has nonzero bytes after terminator")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BundleSbomError(f"tar {label} is not UTF-8") from error


def _tar_zero_number(field: bytes, label: str) -> None:
    if field.rstrip(b"\0 ") and _tar_number(field, label) != 0:
        raise BundleSbomError(f"tar {label} must be zero for a regular file")


def _tar_header(header: bytes) -> tuple[str, int]:
    if len(header) != 512:
        raise BundleSbomError("tar header is truncated")
    expected_checksum = _tar_number(header[148:156], "checksum")
    observed_checksum = sum(header[:148]) + 8 * ord(" ") + sum(header[156:])
    if expected_checksum != observed_checksum:
        raise BundleSbomError("tar header checksum differs")
    if header[257:263] != b"ustar\0" or header[263:265] != b"00":
        raise BundleSbomError("tar header is not canonical POSIX ustar")
    if header[345:500].rstrip(b"\0"):
        raise BundleSbomError("tar member prefix violates flat bundle layout")
    if any(header[500:512]):
        raise BundleSbomError("tar header padding is not zero")
    mode = _tar_number(header[100:108], "mode")
    if mode > 0o7777:
        raise BundleSbomError("tar mode exceeds POSIX permission bits")
    _tar_number(header[108:116], "uid")
    _tar_number(header[116:124], "gid")
    _tar_number(header[136:148], "mtime")
    typeflag = header[156:157]
    if typeflag not in {b"\0", b"0"}:
        name = _tar_text(header[:100], "member name")
        raise BundleSbomError(f"bundle member {name!r} is not a regular file")
    if any(header[157:257]):
        raise BundleSbomError("tar regular file link name must be empty")
    _tar_text(header[265:297], "owner name")
    _tar_text(header[297:329], "group name")
    try:
        _tar_zero_number(header[329:337], "device fields")
        _tar_zero_number(header[337:345], "device fields")
    except BundleSbomError as error:
        raise BundleSbomError("tar device fields must be zero for a regular file") from error
    name = _tar_text(header[:100], "member name")
    return _canonical_member(name), _tar_number(header[124:136], "member size")


def _read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = stream.read(size - len(output))
        if not chunk:
            raise BundleSbomError(f"{label} is truncated")
        output.extend(chunk)
    return bytes(output)


def _inventory_stream(
    stream: BinaryIO, digest_tool: str
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    records: list[dict[str, Any]] = []
    manifests: dict[str, bytes] = {}
    seen: set[str] = set()
    portable: set[str] = set()
    total = 0
    while True:
        header = _read_exact(stream, 512, "tar header")
        if not any(header):
            if any(_read_exact(stream, 512, "second tar end block")):
                raise BundleSbomError("tar archive has only one canonical end block")
            trailing_bytes = 1024
            while chunk := stream.read(CHUNK_BYTES):
                trailing_bytes += len(chunk)
                if any(chunk):
                    raise BundleSbomError("tar archive has nonzero data after its end blocks")
            if trailing_bytes % 512:
                raise BundleSbomError("tar archive padding is not block aligned")
            break
        name, member_size = _tar_header(header)
        folded = name.casefold()
        if name in seen or folded in portable:
            raise BundleSbomError(f"bundle contains duplicate portable member {name!r}")
        if len(records) == MAX_MEMBERS:
            raise BundleSbomError(f"bundle exceeds {MAX_MEMBERS} members")
        if member_size > MAX_MEMBER_BYTES:
            raise BundleSbomError(f"bundle member {name!r} exceeds physical bounds")
        total += member_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise BundleSbomError("bundle uncompressed bytes exceed release bounds")
        identity, retained = _read_member(stream, member_size, name, digest_tool)
        padding = (-member_size) % 512
        if padding and any(_read_exact(stream, padding, f"tar padding for {name!r}")):
            raise BundleSbomError(f"tar padding for {name!r} is not zero")
        if retained is not None:
            manifests[name] = retained
        records.append(
            {
                "name": name,
                "bytes": member_size,
                "sha256": identity["sha256"],
                "blake3": identity["blake3"],
                "package_id": identity["package_id"],
            }
        )
        seen.add(name)
        portable.add(folded)
    if not records:
        raise BundleSbomError("bundle archive is empty")
    records.sort(key=lambda record: record["name"])
    return records, manifests


def _inventory(
    path: Path, stream: BinaryIO, digest_tool: str
) -> tuple[list[dict[str, Any]], dict[str, bytes], str]:
    name = path.name
    if name.endswith(".tar"):
        records, manifests = _inventory_stream(stream, digest_tool)
        return records, manifests, "tar"
    if not name.endswith((".tar.zst", ".tzst")):
        raise BundleSbomError("bundle must use .tar, .tar.zst, or .tzst")
    with tempfile.TemporaryFile() as diagnostics:
        try:
            process = subprocess.Popen(
                ["zstd", "--decompress", "--stdout", "--quiet"],
                stdin=stream,
                stdout=subprocess.PIPE,
                stderr=diagnostics,
            )
        except OSError as error:
            raise BundleSbomError(f"cannot start zstd decoder: {error}") from error
        assert process.stdout is not None
        try:
            records, manifests = _inventory_stream(process.stdout, digest_tool)
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        finally:
            process.stdout.close()
        status = process.wait()
        if status != 0:
            diagnostics.seek(0)
            detail = diagnostics.read(4096).decode("utf-8", errors="replace").strip()
            raise BundleSbomError(f"zstd decoder failed with status {status}: {detail}")
    return records, manifests, "tar-zstd"


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BundleSbomError(f"{label} must be a JSON object")
    return value


def _manifest(payload: bytes, label: str) -> dict[str, Any]:
    try:
        return _object(json.loads(payload), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleSbomError(f"{label} must be valid UTF-8 JSON") from error


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise BundleSbomError(f"{label} must be a positive integer")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BundleSbomError(f"{label} must be a non-empty string")
    return value


def _records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["name"]: record for record in records}


def _package_identity(
    descriptor: dict[str, Any], record: dict[str, Any], label: str
) -> None:
    if descriptor.get("package_id") != record["package_id"]:
        raise BundleSbomError(f"{label} package identity differs from archive")


def _validate_model_manifest(
    value: dict[str, Any], records: dict[str, dict[str, Any]]
) -> None:
    if set(value) != MODEL_TOP_FIELDS:
        raise BundleSbomError("model bundle manifest fields are not canonical schema-v3")
    if (
        value.get("schema_version") != 3
        or value.get("artifact_kind") != "qwen3.6-language-mtp-salt-v2-hf-bundle"
        or value.get("complete_model") is not False
        or value.get("source_revision") != PINNED_MODEL_REVISION
    ):
        raise BundleSbomError("model bundle manifest is not canonical schema-v3 language/MTP")
    if value.get("packing") not in {"d2", "b3", "s34"}:
        raise BundleSbomError("model bundle packing is not canonical")
    for field in (
        "completion_id",
        "campaign_id",
        "admission_id",
        "selection_id",
        "source_model_id",
        "source_identity_status",
    ):
        _nonempty(value.get(field), f"model {field}")
    if value.get("official_payload_authenticated") is not True:
        raise BundleSbomError("model bundle official payload is not authenticated")
    preserved = _object(value.get("preserved"), "model preserved descriptor")
    if set(preserved) != MODEL_PRESERVED_FIELDS:
        raise BundleSbomError("model preserved descriptor fields are not canonical")
    if preserved.get("file") != "preserved.safetensors" or _positive_integer(
        preserved.get("serialized_bytes"), "preserved.serialized_bytes"
    ) != records["preserved.safetensors"]["bytes"]:
        raise BundleSbomError("model preserved byte ledger differs from archive")
    _positive_integer(preserved.get("tensors"), "preserved.tensors")
    payload_bytes = _positive_integer(
        preserved.get("payload_bytes"), "preserved.payload_bytes"
    )
    if payload_bytes >= preserved["serialized_bytes"]:
        raise BundleSbomError("model preserved payload ledger is not canonical")
    _package_identity(preserved, records["preserved.safetensors"], "model preserved")
    profiles = _object(value.get("profiles"), "model profiles")
    if set(profiles) != {"compact-v1", "near-lossless-v1"}:
        raise BundleSbomError("model profiles are not the exact governed pair")
    for profile, filename in (
        ("compact-v1", "compact.tsalt2"),
        ("near-lossless-v1", "near-lossless.tsalt2"),
    ):
        descriptor = _object(profiles.get(profile), f"model profile {profile}")
        if set(descriptor) != MODEL_PROFILE_FIELDS:
            raise BundleSbomError(f"model profile {profile} fields are not canonical")
        if descriptor.get("file") != filename or _positive_integer(
            descriptor.get("serialized_bytes"), f"{profile}.serialized_bytes"
        ) != records[filename]["bytes"]:
            raise BundleSbomError(f"model profile {profile} byte ledger differs from archive")
        _positive_integer(descriptor.get("resident_bytes"), f"{profile}.resident_bytes")
        _package_identity(descriptor, records[filename], f"model profile {profile}")
    assets = value.get("hf_assets")
    if not isinstance(assets, list) or len(assets) != len(MODEL_HF_FILES):
        raise BundleSbomError("model bundle HF asset catalog is incomplete")
    for ordinal, (raw, expected_file) in enumerate(zip(assets, MODEL_HF_FILES)):
        descriptor = _object(raw, f"hf_assets[{ordinal}]")
        if set(descriptor) != MODEL_HF_FIELDS:
            raise BundleSbomError("model bundle HF asset fields are not canonical")
        filename = descriptor.get("file")
        if filename != expected_file:
            raise BundleSbomError("model bundle HF asset filenames differ from canonical catalog")
        if (
            _positive_integer(descriptor.get("bytes"), f"hf_assets[{ordinal}].bytes")
            != records[filename]["bytes"]
        ):
            raise BundleSbomError(f"HF asset {filename!r} byte ledger differs from archive")
        _package_identity(descriptor, records[filename], f"HF asset {filename!r}")


def _validate_onnx_manifest(
    value: dict[str, Any], records: dict[str, dict[str, Any]]
) -> None:
    if set(value) != ONNX_TOP_FIELDS:
        raise BundleSbomError("ONNX bundle manifest fields are not canonical schema-v2")
    if (
        value.get("schema") != "tritium-qwen35-onnx-bundle-v2"
        or value.get("sequence_mode") != "dynamic-cache-v1"
    ):
        raise BundleSbomError("ONNX bundle manifest is not canonical schema-v2")
    for field, filename in (("language", "language.onnx"), ("mtp", "mtp.onnx")):
        descriptor = _object(value.get(field), f"ONNX {field} descriptor")
        if set(descriptor) != {"file", "blake3"} or descriptor.get("file") != filename:
            raise BundleSbomError(f"ONNX {field} filename is not canonical")
        if descriptor.get("blake3") != records[filename]["blake3"]:
            raise BundleSbomError(f"ONNX {field} BLAKE3 differs from archive")
    weights = _object(value.get("weights"), "ONNX weights descriptor")
    if set(weights) != {"file", "blake3", "bytes"}:
        raise BundleSbomError("ONNX weights descriptor fields are not canonical")
    if weights.get("file") != "weights.bin" or _positive_integer(
        weights.get("bytes"), "ONNX weights.bytes"
    ) != records["weights.bin"]["bytes"]:
        raise BundleSbomError("ONNX weights byte ledger differs from archive")
    if weights.get("blake3") != records["weights.bin"]["blake3"]:
        raise BundleSbomError("ONNX weights BLAKE3 differs from archive")
    identity = _object(value.get("identity"), "ONNX identity")
    conversion = _object(value.get("conversion"), "ONNX conversion")
    if set(identity) != ONNX_IDENTITY_FIELDS or set(conversion) != ONNX_CONVERSION_FIELDS:
        raise BundleSbomError("ONNX identity or conversion fields are not canonical")
    for field in ONNX_IDENTITY_FIELDS:
        _nonempty(identity.get(field), f"ONNX identity.{field}")
    for field in ONNX_CONVERSION_FIELDS:
        _nonempty(conversion.get(field), f"ONNX conversion.{field}")
    if conversion.get("mode") not in {"qat-hard", "ptq", "refined"}:
        raise BundleSbomError("ONNX conversion mode is not canonical")


def generate(
    artifact: Path,
    artifact_id: str,
    kind: str,
    source_revision: str,
    digest_tool: str,
) -> dict[str, Any]:
    if ID_PATTERN.fullmatch(artifact_id) is None:
        raise BundleSbomError("artifact id must use lowercase portable identifier syntax")
    if kind not in KINDS:
        raise BundleSbomError(f"artifact kind must be one of {sorted(KINDS)}")
    if REVISION_PATTERN.fullmatch(source_revision) is None:
        raise BundleSbomError("source revision must be a full lowercase Git object ID")
    if not digest_tool:
        raise BundleSbomError("stream digest tool must be specified")
    artifact, descriptor, directories = _open_artifact(artifact)
    try:
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise BundleSbomError("bundle artifact must be an ordinary file")
            artifact_bytes = opened.st_size
            if artifact_bytes <= 0 or artifact_bytes > MAX_ARCHIVE_BYTES:
                raise BundleSbomError("bundle archive bytes are outside release bounds")
            archive_sha256 = _sha256_stream(stream)
            stream.seek(0)
            records, manifests, archive_format = _inventory(artifact, stream, digest_tool)
            stream.seek(0)
            if _sha256_stream(stream) != archive_sha256:
                raise BundleSbomError("bundle artifact changed while generating SBOM")
            final = os.fstat(stream.fileno())
            signature = lambda value: (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
            if signature(final) != signature(opened):
                raise BundleSbomError("bundle artifact metadata changed while generating SBOM")
            _verify_path_chain(artifact, stream.fileno(), directories)
    finally:
        for directory in reversed(directories):
            os.close(directory)
    names = {record["name"] for record in records}
    expected = MODEL_FILES if kind == "model-bundle" else ONNX_FILES
    if names != expected:
        missing = expected - names
        unknown = names - expected
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown {', '.join(sorted(unknown))}")
        raise BundleSbomError(
            "bundle inventory differs from canonical layout: " + "; ".join(details)
        )
    indexed_records = _records(records)
    if kind == "model-bundle":
        _validate_model_manifest(
            _manifest(manifests["tritium.json"], "model manifest"), indexed_records
        )
    else:
        _validate_onnx_manifest(
            _manifest(manifests["tritium-onnx-manifest.json"], "ONNX manifest"),
            indexed_records,
        )
    components = [
        {
            "type": "file",
            "bom-ref": _component_ref(record["name"]),
            "name": record["name"],
            "hashes": [{"alg": "SHA-256", "content": record["sha256"]}],
            "properties": [
                {"name": "tritium:bundle:member-bytes", "value": str(record["bytes"])},
                {"name": "tritium:bundle:member-blake3", "value": record["blake3"]},
                {
                    "name": "tritium:bundle:member-package-id",
                    "value": record["package_id"],
                },
            ],
        }
        for record in records
    ]
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "machine-learning-model",
                "bom-ref": artifact_id,
                "name": artifact_id,
                "hashes": [{"alg": "SHA-256", "content": archive_sha256}],
                "properties": [
                    {"name": "tritium:artifact:file", "value": artifact.name},
                    {"name": "tritium:artifact:bytes", "value": str(artifact_bytes)},
                    {"name": "tritium:artifact:kind", "value": kind},
                    {"name": "tritium:bundle:archive-format", "value": archive_format},
                    {"name": "tritium:source:revision", "value": source_revision},
                ],
            },
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "tritium-generate-bundle-sbom",
                        "version": "1",
                    }
                ]
            },
        },
        "components": components,
        "dependencies": [
            {"ref": artifact_id, "dependsOn": [item["bom-ref"] for item in components]}
        ],
    }


def write_sbom(document: dict[str, Any], output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise BundleSbomError("SBOM output already exists")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir():
        raise BundleSbomError("SBOM output parent must be a directory")
    payload = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=parent)
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        published = True
        temporary.unlink()
        if os.name != "nt":
            directory = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        if published:
            output.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--kind", choices=sorted(KINDS), required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--digest-tool", default=os.environ.get("TRITIUM_BIN", "tritium")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        document = generate(
            args.artifact.absolute(),
            args.artifact_id,
            args.kind,
            args.source_revision,
            args.digest_tool,
        )
        write_sbom(document, args.output)
    except (OSError, BundleSbomError, subprocess.SubprocessError) as error:
        print(f"generate-bundle-sbom: FAIL: {error}", file=sys.stderr)
        return 1
    print(f"generate-bundle-sbom: OK: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
