#!/usr/bin/env python3
"""Create a new release-evidence registry joined to official Qwen source identity.

The input registry and receipts are immutable. This command copies the verified
official-identity receipt into the registry evidence root, adds it as the exact
source-admission child, validates the complete candidate registry, and publishes
a new registry path without replacing existing files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import runpy
from typing import Any


SOURCE_ADMISSION = runpy.run_path(
    Path(__file__).with_name("verify-qwen36-source-admission-receipt.py")
)
validate_source_admission = SOURCE_ADMISSION["validate"]
SourceAdmissionError = SOURCE_ADMISSION["SourceAdmissionError"]
OFFICIAL_IDENTITY = runpy.run_path(
    Path(__file__).with_name("verify-qwen36-official-source-identity.py")
)
validate_official_identity = OFFICIAL_IDENTITY["validate_identity_receipt"]
OfficialIdentityError = OFFICIAL_IDENTITY["OfficialIdentityError"]
RELEASE_STATUS = runpy.run_path(Path(__file__).with_name("release-evidence-status.py"))
evaluate_registry = RELEASE_STATUS["evaluate"]
EvidenceError = RELEASE_STATUS["EvidenceError"]


class RegistrationError(ValueError):
    """The registration inputs differ, are unsafe, or already exist."""


MAX_INPUT_BYTES = 32 * 1024 * 1024


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def strict_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RegistrationError(f"{label} must be an ordinary file")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_INPUT_BYTES:
        raise RegistrationError(f"{label} exceeds the bounded JSON input size")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate field {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RegistrationError(f"{label} must contain strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RegistrationError(f"{label} must be a JSON object")
    return value


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new(path: Path, data: bytes) -> None:
    if path.is_symlink() or path.exists():
        raise RegistrationError(f"refusing to replace existing output: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as error:
        raise RegistrationError(f"cannot create new output: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise RegistrationError(f"cannot durably publish output: {path}") from error


def _validate_registration_inputs(
    base: dict[str, Any],
    candidate: dict[str, Any],
    candidate_bytes: bytes,
    admission_path: Path,
    identity_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if base.get("schema") != "tritium.release-evidence-registry.v1":
        raise RegistrationError("base registry schema differs")
    if base.get("release") != candidate.get("release") or base.get(
        "source_revision"
    ) != candidate.get("source_revision"):
        raise RegistrationError("registry release identity differs from candidate")
    if base.get("candidate_manifest_sha256") != sha256(candidate_bytes):
        raise RegistrationError("base registry does not bind the supplied candidate bytes")

    entries = base.get("receipts")
    if not isinstance(entries, list):
        raise RegistrationError("base registry receipts must be an array")
    admission_entries = [
        entry for entry in entries
        if isinstance(entry, dict) and entry.get("kind") == "source-admission"
    ]
    if len(admission_entries) != 1:
        raise RegistrationError("base registry must contain exactly one source-admission")
    entry = admission_entries[0]
    admission_id = entry.get("id")
    if not isinstance(admission_id, str):
        raise RegistrationError("source-admission registry ID is malformed")
    try:
        admission = validate_source_admission(
            admission_path,
            str(base["source_revision"]),
            str(base["release"]),
            admission_path,
        )
        identity = validate_official_identity(identity_path)
    except (SourceAdmissionError, OfficialIdentityError) as error:
        raise RegistrationError("source identity input receipt failed validation") from error

    if admission["receipt_id"] != admission_id:
        raise RegistrationError("source-admission entry ID differs from its receipt")
    if entry.get("sha256") != sha256(admission_path.read_bytes()):
        raise RegistrationError("source-admission registry digest differs from receipt bytes")
    if entry.get("path") is None:
        raise RegistrationError("source-admission registry path is missing")
    admission_artifact_id = entry.get("artifact_id")
    if not isinstance(admission_artifact_id, str):
        raise RegistrationError("source-admission candidate artifact ID is malformed")
    if identity["source_admission_receipt_id"] != admission_id:
        raise RegistrationError("official identity receipt names a different admission")
    source = admission["receipt"]
    if any(
        identity[field] != expected
        for field, expected in (
            ("repository", source["repository"]),
            ("revision", source["revision"]),
            ("source_model_id", source["source_model_id"]),
            ("manifest_content_id", source["manifest_content_id"]),
            ("source_proof_id", source["proof_id"]),
        )
    ):
        raise RegistrationError("official identity receipt differs from admission identity")
    if any(
        isinstance(item, dict) and item.get("kind") == "official-source-identity"
        for item in entries
    ):
        raise RegistrationError("base registry already contains official source identity")
    return entry, admission, identity


def register(
    base_registry: Path,
    candidate_path: Path,
    admission_path: Path,
    identity_path: Path,
    output_registry: Path,
) -> dict[str, Any]:
    base_registry = base_registry.resolve(strict=True)
    candidate_path = candidate_path.resolve(strict=True)
    admission_path = admission_path.resolve(strict=True)
    identity_path = identity_path.resolve(strict=True)
    output_registry = output_registry.absolute()
    registry_root = base_registry.parent
    if output_registry.parent.resolve(strict=True) != registry_root:
        raise RegistrationError("output registry must remain in the base evidence directory")
    if output_registry == base_registry:
        raise RegistrationError("output registry must not replace the base registry")

    base = strict_json(base_registry, "base registry")
    candidate = strict_json(candidate_path, "candidate manifest")
    candidate_bytes = candidate_path.read_bytes()
    entry, _admission, identity = _validate_registration_inputs(
        base, candidate, candidate_bytes, admission_path, identity_path
    )
    logical_admission = PurePosixPath(entry.get("path", ""))
    if (
        logical_admission.is_absolute()
        or ".." in logical_admission.parts
        or "\\" in str(logical_admission)
    ):
        raise RegistrationError("source-admission registry path is unsafe")
    try:
        registered_admission_path = (
            registry_root / Path(logical_admission)
        ).resolve(strict=True)
    except OSError as error:
        raise RegistrationError("registered source-admission path is absent") from error
    if registered_admission_path != admission_path:
        raise RegistrationError("admission receipt path differs from its registry entry")
    identity_bytes = identity_path.read_bytes()
    identity_name = identity["receipt_id"].removeprefix("sha256:") + ".json"
    identity_relative = PurePosixPath("official-source-identity") / identity_name
    identity_output = registry_root / Path(identity_relative)
    new_entry = {
        "id": identity["receipt_id"],
        "kind": "official-source-identity",
        "path": identity_relative.as_posix(),
        "sha256": sha256(identity_bytes),
        "artifact_id": entry["artifact_id"],
        "parents": [entry["id"]],
    }
    updated = dict(base)
    updated["receipts"] = [*base["receipts"], new_entry]

    created_identity = False
    created_registry = False
    try:
        identity_output.parent.mkdir(mode=0o755, exist_ok=True)
        if identity_output.parent.is_symlink():
            raise RegistrationError("identity output directory must not be a symlink")
        write_new(identity_output, identity_bytes)
        created_identity = True
        write_new(output_registry, canonical(updated) + b"\n")
        created_registry = True
        report = evaluate_registry(output_registry, candidate_path, candidate)
        qwen_gate = next(
            row for row in report["rows"] if row["id"] == "qwen-source-admission"
        )
        if qwen_gate["status"] != "PASS":
            raise RegistrationError("registered Qwen source identity gate is not PASS")
    except (OSError, EvidenceError) as error:
        if created_registry:
            output_registry.unlink(missing_ok=True)
        if created_identity:
            identity_output.unlink(missing_ok=True)
        raise RegistrationError(
            f"new identity registry failed full validation: {error}"
        ) from error
    except RegistrationError:
        if created_registry:
            output_registry.unlink(missing_ok=True)
        if created_identity:
            identity_output.unlink(missing_ok=True)
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-registry", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--admission-receipt", required=True, type=Path)
    parser.add_argument("--official-identity-receipt", required=True, type=Path)
    parser.add_argument("--output-registry", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = register(
            args.base_registry,
            args.candidate,
            args.admission_receipt,
            args.official_identity_receipt,
            args.output_registry,
        )
    except (OSError, RegistrationError) as error:
        parser.error(str(error))
    gate = next(row for row in report["rows"] if row["id"] == "qwen-source-admission")
    print(json.dumps({
        "result": "pass" if gate["status"] == "PASS" else gate["status"].lower(),
        "gate": gate["id"],
        "registry": str(args.output_registry),
        "missing_kinds": gate["missing_kinds"],
    }, sort_keys=True, separators=(",", ":")))
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
