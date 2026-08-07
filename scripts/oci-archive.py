#!/usr/bin/env python3
"""Strict OCI-layout tar inventory and image-attestation contract."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Callable
from urllib.parse import urlsplit


MANIFEST = "application/vnd.oci.image.manifest.v1+json"
ATTESTATION = "application/vnd.in-toto+json"
SPDX_PREDICATE = "https://spdx.dev/Document"
SLSA_PREDICATE = "https://slsa.dev/provenance/v1"
BUILDKIT_BUILD_TYPE = (
    "https://github.com/moby/buildkit/blob/master/"
    "docs/attestations/slsa-definitions.md"
)
MAX_METADATA_BYTES = 32 * 1024 * 1024
MAX_RETAINED_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 200_000
CHUNK_BYTES = 1024 * 1024
HEX = frozenset("0123456789abcdef")
SPDX_ID = re.compile(r"SPDXRef-[A-Za-z0-9.-]+")
SPDX_CREATOR = re.compile(r"(?:Person|Organization|Tool):\s*\S(?:.*\S)?")
RFC3339 = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})"
)
DIGEST_LENGTHS = {
    "sha1": 40,
    "sha224": 56,
    "sha256": 64,
    "sha384": 96,
    "sha512": 128,
}
IdentityReader = Callable[[BinaryIO, int, str], dict[str, Any]]


class OciArchiveError(ValueError):
    """OCI transport tar is unsafe, incomplete, or not image-bound."""


def _json(data: bytes | None, label: str) -> dict[str, Any]:
    if data is None:
        raise OciArchiveError(f"{label} metadata exceeds size limit")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OciArchiveError(f"{label} must contain UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise OciArchiveError(f"{label} must be a JSON object")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise OciArchiveError(f"{label} must be a SHA-256 descriptor")
    raw = value[7:]
    if len(raw) != 64 or any(character not in HEX for character in raw):
        raise OciArchiveError(f"{label} must be a SHA-256 descriptor")
    return raw


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OciArchiveError(f"{label} must be an object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OciArchiveError(f"{label} must be a non-empty string")
    return value


def _https_url(value: Any, label: str) -> str:
    raw = _string(value, label)
    try:
        parsed = urlsplit(raw)
    except ValueError as error:
        raise OciArchiveError(f"{label} must be an HTTPS identity URI") from error
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or any(character.isspace() for character in raw)
    ):
        raise OciArchiveError(f"{label} must be an HTTPS identity URI")
    return raw


def _timestamp(value: Any, label: str) -> datetime:
    raw = _string(value, label)
    if RFC3339.fullmatch(raw) is None:
        raise OciArchiveError(f"{label} is not RFC 3339")
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    except ValueError as error:
        raise OciArchiveError(f"{label} is not a valid timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OciArchiveError(f"{label} must include a timezone")
    return parsed


def _validate_spdx(predicate: dict[str, Any]) -> None:
    if predicate.get("SPDXID") != "SPDXRef-DOCUMENT":
        raise OciArchiveError("SPDX predicate lacks canonical document identity")
    if predicate.get("spdxVersion") not in {"SPDX-2.2", "SPDX-2.3"}:
        raise OciArchiveError("SPDX predicate version is unsupported")
    if predicate.get("dataLicense") != "CC0-1.0":
        raise OciArchiveError("SPDX predicate data license is invalid")
    _string(predicate.get("name"), "SPDX predicate name")
    _https_url(predicate.get("documentNamespace"), "SPDX document namespace")
    creation = _object(predicate.get("creationInfo"), "SPDX creationInfo")
    _timestamp(creation.get("created"), "SPDX creation timestamp")
    creators = creation.get("creators")
    if not isinstance(creators, list) or not creators:
        raise OciArchiveError("SPDX predicate creators must be a non-empty array")
    for creator in creators:
        creator = _string(creator, "SPDX predicate creator")
        if SPDX_CREATOR.fullmatch(creator) is None:
            raise OciArchiveError("SPDX predicate creator lacks required type")
    packages = predicate.get("packages")
    if not isinstance(packages, list) or not packages:
        raise OciArchiveError("SPDX predicate packages must be a non-empty array")
    package_ids: set[str] = set()
    for package in packages:
        package = _object(package, "SPDX package")
        package_id = _string(package.get("SPDXID"), "SPDX package SPDXID")
        if SPDX_ID.fullmatch(package_id) is None or package_id in package_ids:
            raise OciArchiveError("SPDX package identities must be unique SPDXRef values")
        _string(package.get("name"), "SPDX package name")
        location = _string(package.get("downloadLocation"), "SPDX package download location")
        if location not in {"NONE", "NOASSERTION"} and (
            re.fullmatch(r"[A-Za-z][A-Za-z0-9+.-]*:[^\s]+", location) is None
        ):
            raise OciArchiveError("SPDX package download location is invalid")
        files_analyzed = package.get("filesAnalyzed", True)
        if type(files_analyzed) is not bool:
            raise OciArchiveError("SPDX package filesAnalyzed must be boolean")
        verification = package.get("packageVerificationCode")
        if files_analyzed:
            verification = _object(verification, "SPDX package verification code")
            value = verification.get("packageVerificationCodeValue")
            if (
                not isinstance(value, str)
                or len(value) != 40
                or any(character not in HEX for character in value)
            ):
                raise OciArchiveError("SPDX package verification code is invalid")
        elif verification is not None:
            raise OciArchiveError(
                "SPDX package verification code must be absent when files are unanalyzed"
            )
        package_ids.add(package_id)


def _scan_transport_headers(stream: BinaryIO, archive_bytes: int) -> None:
    """Reject tar extension records hidden by Python's logical-member API."""

    position = 0
    zero_blocks = 0
    extensions = {b"x", b"g", b"L", b"K", b"S"}
    allowed = {b"\0", b"0", b"5"}
    while position + 512 <= archive_bytes:
        stream.seek(position)
        header = stream.read(512)
        if len(header) != 512:
            raise OciArchiveError("OCI tar header is truncated")
        if not any(header):
            zero_blocks += 1
            position += 512
            if zero_blocks == 2:
                stream.seek(0)
                return
            continue
        if zero_blocks:
            raise OciArchiveError("OCI tar has data after one end block")
        kind = header[156:157]
        if kind in extensions:
            raise OciArchiveError("OCI tar contains hidden PAX/GNU extension record")
        if kind not in allowed:
            raise OciArchiveError("OCI tar contains unsupported transport record")
        size_field = header[124:136]
        if size_field[0] & 0x80:
            raise OciArchiveError("OCI tar uses non-portable base-256 size")
        raw_size = size_field.rstrip(b"\0 ").lstrip(b" ") or b"0"
        try:
            size = int(raw_size, 8)
        except ValueError as error:
            raise OciArchiveError("OCI tar member size is not canonical octal") from error
        position += 512 + ((size + 511) // 512) * 512
        if position > archive_bytes:
            raise OciArchiveError("OCI tar member payload is truncated")
    raise OciArchiveError("OCI tar is missing trailing end blocks")


def _validate_slsa_v1(
    predicate: dict[str, Any], source_revision: str
) -> tuple[str, str]:
    definition = _object(predicate.get("buildDefinition"), "SLSA buildDefinition")
    if definition.get("buildType") != BUILDKIT_BUILD_TYPE:
        raise OciArchiveError("SLSA provenance build type is not BuildKit v1")
    external = _object(
        definition.get("externalParameters"), "SLSA externalParameters"
    )
    config_source = _object(external.get("configSource"), "SLSA configSource")
    _string(config_source.get("path"), "SLSA config source path")
    request = _object(external.get("request"), "SLSA build request")
    _string(request.get("frontend"), "SLSA build frontend")
    args = _object(request.get("args"), "SLSA build arguments")
    if args.get("build-arg:SOURCE_REVISION") != source_revision:
        raise OciArchiveError("SLSA provenance does not bind source revision")
    internal = _object(
        definition.get("internalParameters"), "SLSA internalParameters"
    )
    build_config = _object(internal.get("buildConfig"), "SLSA buildConfig")
    llb = build_config.get("llbDefinition")
    if not isinstance(llb, list) or not llb:
        raise OciArchiveError("SLSA max provenance lacks LLB build definition")
    llb_ids: set[str] = set()
    for operation in llb:
        operation = _object(operation, "SLSA LLB operation")
        operation_id = _string(operation.get("id"), "SLSA LLB operation identity")
        if operation_id in llb_ids:
            raise OciArchiveError("SLSA LLB operation identities must be unique")
        op = _object(operation.get("op"), "SLSA LLB operation payload")
        if not op:
            raise OciArchiveError("SLSA LLB operation payload must not be empty")
        llb_ids.add(operation_id)
    _string(internal.get("builderPlatform"), "SLSA builder platform")
    dependencies = definition.get("resolvedDependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise OciArchiveError("SLSA provenance lacks resolved dependencies")
    for dependency in dependencies:
        dependency = _object(dependency, "SLSA resolved dependency")
        _string(dependency.get("uri"), "SLSA dependency URI")
        digests = _object(dependency.get("digest"), "SLSA dependency digest")
        if not digests:
            raise OciArchiveError("SLSA dependency digest must not be empty")
        for algorithm, digest in digests.items():
            algorithm = _string(algorithm, "SLSA dependency digest algorithm")
            digest = _string(digest, "SLSA dependency digest value")
            length = DIGEST_LENGTHS.get(algorithm)
            if length is None or len(digest) != length or any(
                character not in HEX for character in digest
            ):
                raise OciArchiveError("SLSA dependency digest is invalid")
    details = _object(predicate.get("runDetails"), "SLSA runDetails")
    builder = _object(details.get("builder"), "SLSA builder")
    builder_id = _https_url(builder.get("id"), "SLSA builder identity")
    metadata = _object(details.get("metadata"), "SLSA run metadata")
    invocation_id = _string(metadata.get("invocationID"), "SLSA invocation identity")
    started = _timestamp(metadata.get("startedOn"), "SLSA build start timestamp")
    finished = _timestamp(metadata.get("finishedOn"), "SLSA build finish timestamp")
    if finished < started:
        raise OciArchiveError("SLSA build finish timestamp precedes start timestamp")
    return builder_id, invocation_id


def _safe_name(name: str) -> str:
    logical = PurePosixPath(name)
    if (
        not name
        or logical.is_absolute()
        or ".." in logical.parts
        or "\\" in name
        or logical.as_posix() != name
    ):
        raise OciArchiveError(f"OCI member {name!r} has unsafe topology")
    return name


def _python_identity(stream: BinaryIO, size: int, name: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = stream.read(min(CHUNK_BYTES, remaining))
        if not chunk:
            raise OciArchiveError(f"OCI member {name!r} is truncated")
        remaining -= len(chunk)
        digest.update(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def inspect(
    stream: BinaryIO,
    archive_bytes: int,
    release: str,
    source_revision: str,
    identify: IdentityReader | None = None,
) -> dict[str, Any]:
    """Inventory one plain OCI tar and validate closed descriptor/attestation graph."""

    _scan_transport_headers(stream, archive_bytes)
    identity_reader = identify or _python_identity
    records: list[dict[str, Any]] = []
    files: dict[str, bytes | None] = {}
    sizes: dict[str, int] = {}
    portable: set[str] = set()
    directories: set[str] = set()
    retained_bytes = 0
    try:
        archive = tarfile.open(fileobj=stream, mode="r|")
        with archive:
            for member in archive:
                name = _safe_name(member.name.rstrip("/") if member.isdir() else member.name)
                folded = name.casefold()
                if folded in portable:
                    raise OciArchiveError(f"OCI archive has duplicate portable member {name!r}")
                if len(records) + len(directories) == MAX_MEMBERS:
                    raise OciArchiveError("OCI archive member count exceeds release bounds")
                if member.isdir():
                    if name not in {"blobs", "blobs/sha256"}:
                        raise OciArchiveError(f"OCI archive has unknown directory {name!r}")
                    directories.add(name)
                    portable.add(folded)
                    continue
                if not member.isfile():
                    raise OciArchiveError("OCI archive contains link or non-file member")
                if name not in {"oci-layout", "index.json"} and not name.startswith(
                    "blobs/sha256/"
                ):
                    raise OciArchiveError(f"OCI archive has unknown file {name!r}")
                if name.startswith("blobs/sha256/"):
                    raw_name = name.removeprefix("blobs/sha256/")
                    if len(raw_name) != 64 or any(character not in HEX for character in raw_name):
                        raise OciArchiveError("OCI blob path is not canonical SHA-256")
                elif member.size > MAX_METADATA_BYTES:
                    raise OciArchiveError("OCI layout metadata exceeds size limit")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise OciArchiveError(f"OCI member {name!r} is unreadable")
                retained: bytes | None = None
                if member.size <= MAX_METADATA_BYTES:
                    payload = extracted.read(MAX_METADATA_BYTES + 1)
                    if len(payload) != member.size:
                        raise OciArchiveError(f"OCI member {name!r} is truncated")
                    identity = identity_reader(io.BytesIO(payload), member.size, name)
                    if name in {"oci-layout", "index.json"} or (
                        retained_bytes + member.size <= MAX_RETAINED_BYTES
                    ):
                        retained = payload
                        retained_bytes += member.size
                else:
                    identity = identity_reader(extracted, member.size, name)
                if identity.get("bytes") != member.size:
                    raise OciArchiveError(f"OCI member {name!r} byte identity differs")
                sha256 = identity.get("sha256")
                if (
                    not isinstance(sha256, str)
                    or len(sha256) != 64
                    or any(character not in HEX for character in sha256)
                ):
                    raise OciArchiveError(f"OCI member {name!r} SHA-256 identity is invalid")
                if name.startswith("blobs/sha256/") and name != f"blobs/sha256/{sha256}":
                    raise OciArchiveError("OCI blob path does not match blob digest")
                record = {"name": name, "bytes": member.size, "sha256": sha256}
                for field in ("blake3", "package_id"):
                    if field in identity:
                        record[field] = identity[field]
                records.append(record)
                files[name] = retained
                sizes[name] = member.size
                portable.add(folded)
            data_end = archive.offset
    except tarfile.TarError as error:
        raise OciArchiveError(f"OCI archive is not a plain tar: {error}") from error
    if not records:
        raise OciArchiveError("OCI archive is empty")
    if archive_bytes % 512:
        raise OciArchiveError("OCI tar size is not block aligned")
    trailing_bytes = archive_bytes - data_end
    if trailing_bytes < 1024:
        raise OciArchiveError("OCI tar is missing trailing end blocks")
    stream.seek(data_end)
    remaining = trailing_bytes
    while remaining:
        chunk = stream.read(min(CHUNK_BYTES, remaining))
        if not chunk:
            raise OciArchiveError("OCI tar trailing end blocks are truncated")
        remaining -= len(chunk)
        if any(chunk):
            raise OciArchiveError("OCI tar has nonzero trailing payload")
    if set(files) - {"oci-layout", "index.json"} != {
        name for name in files if name.startswith("blobs/sha256/")
    }:
        raise OciArchiveError("OCI archive file topology is not canonical")
    if set(files) & {"oci-layout", "index.json"} != {"oci-layout", "index.json"}:
        raise OciArchiveError("OCI archive lacks layout metadata")
    layout = _json(files["oci-layout"], "oci-layout")
    if layout != {"imageLayoutVersion": "1.0.0"}:
        raise OciArchiveError("OCI layout must be canonical version 1.0.0")
    index = _json(files["index.json"], "index.json")
    if index.get("schemaVersion") != 2 or not isinstance(index.get("manifests"), list):
        raise OciArchiveError("OCI index schema or manifests are invalid")

    referenced: set[str] = set()

    def descriptor_blob(descriptor: dict[str, Any], label: str) -> tuple[str, bytes | None]:
        raw = _digest(descriptor.get("digest"), f"{label}.digest")
        name = f"blobs/sha256/{raw}"
        if name not in files:
            raise OciArchiveError(f"{label} blob is absent or digest-mismatched")
        if type(descriptor.get("size")) is not int or descriptor["size"] != sizes[name]:
            raise OciArchiveError(f"{label} descriptor size differs")
        referenced.add(name)
        return name, files[name]

    def json_blob(descriptor: dict[str, Any], label: str) -> dict[str, Any]:
        _, payload = descriptor_blob(descriptor, label)
        return _json(payload, label)

    descriptors = index["manifests"]
    images = [
        item
        for item in descriptors
        if isinstance(item, dict)
        and item.get("mediaType") == MANIFEST
        and item.get("platform") == {"architecture": "amd64", "os": "linux"}
    ]
    if len(images) != 1:
        raise OciArchiveError("OCI index must contain exactly one linux/amd64 image")
    image_descriptor = images[0]
    image_digest = "sha256:" + _digest(image_descriptor.get("digest"), "image.digest")
    image = json_blob(image_descriptor, "image manifest")
    if image.get("schemaVersion") != 2 or not isinstance(image.get("config"), dict):
        raise OciArchiveError("image manifest schema or config is invalid")
    config = json_blob(image["config"], "image config")
    layers = image.get("layers")
    if not isinstance(layers, list):
        raise OciArchiveError("image layers must be an array")
    for ordinal, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise OciArchiveError("image layer descriptor must be an object")
        descriptor_blob(layer, f"image layer {ordinal}")
    runtime = config.get("config")
    if not isinstance(runtime, dict):
        raise OciArchiveError("image runtime config must be an object")
    labels = runtime.get("Labels")
    expected_labels = {
        "org.opencontainers.image.revision": source_revision,
        "org.opencontainers.image.version": release,
        "io.tritium.artifact.schema": "3",
        "io.tritium.startup-receipt.schema": "1",
    }
    if not isinstance(labels, dict) or any(
        labels.get(key) != value for key, value in expected_labels.items()
    ):
        raise OciArchiveError("image labels do not bind release contracts")
    if runtime.get("User") in {None, "", "0", "0:0"} or runtime.get(
        "Entrypoint"
    ) != ["/usr/local/bin/tritium-serve"]:
        raise OciArchiveError("image runtime identity is not hardened")

    predicates: set[str] = set()
    builder_id: str | None = None
    invocation_id: str | None = None
    attestation_count = 0
    for descriptor in descriptors:
        if descriptor is image_descriptor:
            continue
        if not isinstance(descriptor, dict):
            raise OciArchiveError("OCI index descriptor must be an object")
        annotations = descriptor.get("annotations")
        if (
            not isinstance(annotations, dict)
            or annotations.get("vnd.docker.reference.type") != "attestation-manifest"
            or annotations.get("vnd.docker.reference.digest") != image_digest
        ):
            raise OciArchiveError("OCI index contains non-image or unbound descriptor")
        attestation_count += 1
        attestation = json_blob(descriptor, "attestation manifest")
        if attestation.get("schemaVersion") != 2 or not isinstance(
            attestation.get("config"), dict
        ):
            raise OciArchiveError("attestation manifest schema or config is invalid")
        descriptor_blob(attestation["config"], "attestation config")
        attestation_layers = attestation.get("layers")
        if not isinstance(attestation_layers, list) or not attestation_layers:
            raise OciArchiveError("attestation manifest layers are absent")
        for layer in attestation_layers:
            if not isinstance(layer, dict) or layer.get("mediaType") != ATTESTATION:
                raise OciArchiveError("attestation layer media type is invalid")
            layer_annotations = layer.get("annotations")
            predicate = (
                layer_annotations.get("in-toto.io/predicate-type")
                if isinstance(layer_annotations, dict)
                else None
            )
            payload = json_blob(layer, "attestation layer")
            if (
                payload.get("_type") != "https://in-toto.io/Statement/v1"
                or payload.get("predicateType") != predicate
                or not isinstance(payload.get("predicate"), dict)
            ):
                raise OciArchiveError("attestation statement type or predicate differs")
            subjects = payload.get("subject")
            if not isinstance(subjects, list) or not any(
                isinstance(subject, dict)
                and isinstance(subject.get("digest"), dict)
                and subject["digest"].get("sha256") == image_digest[7:]
                for subject in subjects
            ):
                raise OciArchiveError("attestation subject does not bind image manifest")
            if not isinstance(predicate, str):
                raise OciArchiveError("attestation predicate type is absent")
            if predicate in predicates:
                raise OciArchiveError("OCI archive contains duplicate attestation predicate")
            if predicate == SPDX_PREDICATE:
                _validate_spdx(payload["predicate"])
            elif predicate == SLSA_PREDICATE:
                builder_id, invocation_id = _validate_slsa_v1(
                    payload["predicate"], source_revision
                )
            else:
                raise OciArchiveError(
                    "OCI archive lacks image-bound SBOM and provenance: "
                    "unsupported attestation predicate"
                )
            predicates.add(predicate)
    if attestation_count == 0:
        raise OciArchiveError("OCI index lacks attestation manifest")
    if predicates != {SPDX_PREDICATE, SLSA_PREDICATE}:
        raise OciArchiveError("OCI archive lacks image-bound SBOM and provenance")
    if builder_id is None or invocation_id is None:
        raise OciArchiveError("OCI archive lacks semantic SLSA provenance identity")
    blobs = {name for name in files if name.startswith("blobs/sha256/")}
    if blobs != referenced:
        unknown = blobs - referenced
        missing = referenced - blobs
        details = []
        if unknown:
            details.append("unreferenced " + ", ".join(sorted(unknown)))
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        raise OciArchiveError("OCI blob closure differs: " + "; ".join(details))
    records.sort(key=lambda record: record["name"])
    return {
        "records": records,
        "image_manifest_digest": image_digest,
        "predicates": sorted(predicates),
        "platform": "linux/amd64",
        "builder_id": builder_id,
        "invocation_id": invocation_id,
    }
