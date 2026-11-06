#!/usr/bin/env python3
"""Reissue Qwen source receipts for a relocated, byte-identical proof.

This creates a new receipt pair; it never edits or replaces the source receipts.
The official file inventory and Hub-response digest are carried forward only
after their existing identity receipt validates and binds the exact old
admission receipt. The new admission receipt changes only ``proof_path``; the
new official-identity receipt changes only its admission parent and derived
receipt ID. This is a host-local relocation, not a fresh Hub or checkpoint
verification and not release-registry admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import stat
from typing import Any


ADMISSION = runpy.run_path(
    Path(__file__).with_name("verify-qwen36-source-admission-receipt.py")
)
validate_source_admission = ADMISSION["validate"]
SourceAdmissionError = ADMISSION["SourceAdmissionError"]
IDENTITY = runpy.run_path(
    Path(__file__).with_name("verify-qwen36-official-source-identity.py")
)
validate_identity = IDENTITY["validate_identity_receipt"]
OfficialIdentityError = IDENTITY["OfficialIdentityError"]

MAX_RECEIPT_BYTES = 8 * 1024 * 1024
MAX_PROOF_BYTES = 64 * 1024 * 1024
REBIND_SCHEMA = "tritium.qwen36-source-evidence-rebind.v1"


class RebindError(ValueError):
    """Receipt inputs differ, are unsafe, or cannot be durably reissued."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ordinary_file(path: Path, maximum: int, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RebindError(f"cannot inspect {label}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0 or metadata.st_size > maximum:
        raise RebindError(f"{label} must be a bounded ordinary file")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise RebindError(f"cannot read {label}") from error
    if len(data) != metadata.st_size:
        raise RebindError(f"{label} changed while it was read")
    return data


def strict_json(data: bytes, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RebindError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RebindError(f"{label} must be a JSON object")
    return value


def write_new(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as error:
        raise RebindError(f"refusing to replace output: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise RebindError(f"cannot durably write output: {path}") from error


def json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True).encode(
        "utf-8"
    ) + b"\n"


def rebind(
    admission_path: Path,
    identity_path: Path,
    proof_path: Path,
    output_dir: Path,
) -> dict[str, str]:
    admission_bytes = ordinary_file(admission_path, MAX_RECEIPT_BYTES, "source-admission receipt")
    identity_bytes = ordinary_file(identity_path, MAX_RECEIPT_BYTES, "official identity receipt")
    proof_bytes = ordinary_file(proof_path, MAX_PROOF_BYTES, "source proof")
    admission_value = strict_json(admission_bytes, "source-admission receipt")
    identity_value = strict_json(identity_bytes, "official identity receipt")

    try:
        old_admission = validate_source_admission(
            admission_path,
            ADMISSION["PINNED_REVISION"],
            "1.1.0-rc.2",
            admission_path,
        )
        old_identity = validate_identity(identity_path)
    except (SourceAdmissionError, OfficialIdentityError) as error:
        raise RebindError("input receipt failed its strict validator") from error

    admission = admission_value.get("receipt")
    if not isinstance(admission, dict):
        raise RebindError("source-admission receipt body is malformed")
    if (
        old_identity.get("source_admission_receipt_id") != old_admission["receipt_id"]
        or any(
            old_identity.get(identity_field) != admission.get(admission_field)
            for identity_field, admission_field in (
                ("repository", "repository"),
                ("revision", "revision"),
                ("source_model_id", "source_model_id"),
                ("manifest_content_id", "manifest_content_id"),
                ("source_proof_id", "proof_id"),
            )
        )
    ):
        raise RebindError("official identity is not bound to this admission")
    if (
        len(proof_bytes) != admission["proof_bytes"]
        or sha256(proof_bytes) != admission_value["proof_sha256"]
    ):
        raise RebindError("relocated proof differs from the source-admission receipt")

    output_dir = output_dir.absolute()
    parent = output_dir.parent.resolve(strict=True)
    if output_dir.name in {"", ".", ".."} or output_dir.is_symlink() or output_dir.exists():
        raise RebindError("output directory must be a new path")
    output_dir = parent / output_dir.name
    proof_destination = output_dir / "source-proof.tq36"

    new_admission = json.loads(json.dumps(admission_value))
    new_admission["receipt"]["proof_path"] = str(proof_destination)
    new_admission_id = "sha256:" + sha256(canonical(new_admission))

    new_identity = json.loads(json.dumps(identity_value))
    new_identity["source_admission_receipt_id"] = new_admission_id
    new_identity.pop("receipt_id", None)
    new_identity["receipt_id"] = "sha256:" + sha256(canonical(new_identity))

    rebind_value: dict[str, Any] = {
        "schema": REBIND_SCHEMA,
        "result": "pass",
        "source_admission_parent_receipt_id": old_admission["receipt_id"],
        "source_admission_receipt_id": new_admission_id,
        "official_identity_parent_receipt_id": old_identity["receipt_id"],
        "official_identity_receipt_id": new_identity["receipt_id"],
        "repository": admission["repository"],
        "revision": admission["revision"],
        "source_model_id": admission["source_model_id"],
        "proof_id": admission["proof_id"],
        "proof_bytes": len(proof_bytes),
        "proof_sha256": admission_value["proof_sha256"],
        "proof_path": str(proof_destination),
        "changed_fields": [
            "source_admission.receipt.proof_path",
            "official_identity.source_admission_receipt_id",
        ],
    }
    rebind_value["receipt_id"] = "sha256:" + sha256(canonical(rebind_value))

    try:
        output_dir.mkdir(mode=0o755)
    except OSError as error:
        raise RebindError("cannot create a new output directory") from error
    try:
        write_new(proof_destination, proof_bytes)
        admission_output = output_dir / "source-admission.json"
        identity_output = output_dir / "official-source-identity.json"
        write_new(admission_output, json_bytes(new_admission))
        write_new(identity_output, json_bytes(new_identity))
        try:
            verified_admission = validate_source_admission(
                admission_output,
                ADMISSION["PINNED_REVISION"],
                "1.1.0-rc.2",
                admission_output,
            )
            verified_identity = validate_identity(identity_output)
        except (SourceAdmissionError, OfficialIdentityError) as error:
            raise RebindError("reissued receipt failed its strict validator") from error
        if (
            verified_admission["receipt_id"] != new_admission_id
            or verified_identity["receipt_id"] != new_identity["receipt_id"]
            or verified_identity["source_admission_receipt_id"] != new_admission_id
        ):
            raise RebindError("reissued receipt identities differ")
        write_new(output_dir / "rebind.json", json_bytes(rebind_value))
        directory = os.open(output_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except (OSError, RebindError):
        # Leave any partial new directory in place for explicit inspection. No
        # existing input or output is replaced or removed.
        raise

    return {
        "source_admission_receipt_id": new_admission_id,
        "official_identity_receipt_id": new_identity["receipt_id"],
        "rebind_receipt_id": rebind_value["receipt_id"],
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-admission-receipt", required=True, type=Path)
    parser.add_argument("--official-identity-receipt", required=True, type=Path)
    parser.add_argument("--source-proof", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = rebind(
            args.source_admission_receipt,
            args.official_identity_receipt,
            args.source_proof,
            args.output_dir,
        )
    except (RebindError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
