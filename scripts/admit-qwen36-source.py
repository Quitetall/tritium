#!/usr/bin/env python3
"""Persist a canonical Qwen3.6 source-admission receipt.

This producer records source coverage only. It does not claim calibration,
fitting, packaging, runtime, or official payload authentication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA = "tritium.qwen36-source-admission.v1"
RESULT = "pass"
PINNED_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
RECEIPT_FIELDS = (
    "proof_id",
    "manifest_content_id",
    "source_model_id",
    "repository",
    "revision",
    "identity_status",
    "official_payload_authenticated",
    "proof_bytes",
    "payload_bytes",
    "work_dir",
    "proof_path",
    "total_tensors",
    "total_coefficients",
    "language_tensors",
    "language_coefficients",
    "mtp_tensors",
    "mtp_coefficients",
    "vision_tensors",
    "vision_coefficients",
    "additive_tensors",
    "additive_coefficients",
    "preserved_tensors",
    "preserved_coefficients",
    "excluded_vision_tensors",
    "excluded_vision_coefficients",
)


class AdmissionReceiptError(ValueError):
    """Source-admission evidence is malformed or cannot be published."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def receipt_value(receipt: Any) -> dict[str, Any]:
    """Copy only the frozen native receipt fields into plain JSON values."""

    value = {field: getattr(receipt, field) for field in RECEIPT_FIELDS}
    if set(value) != set(RECEIPT_FIELDS):
        raise AdmissionReceiptError("native source receipt fields differ")
    if value["identity_status"] == "":
        raise AdmissionReceiptError("source identity status is empty")
    if value["repository"] != "Qwen/Qwen3.6-27B":
        raise AdmissionReceiptError("source repository differs from pinned Qwen3.6")
    if value["revision"] != PINNED_REVISION:
        raise AdmissionReceiptError("source revision differs from pinned Qwen3.6")
    if value["official_payload_authenticated"] is not False:
        raise AdmissionReceiptError(
            "source admission cannot claim official payload authentication"
        )
    for field in RECEIPT_FIELDS:
        if field in {
            "proof_id",
            "manifest_content_id",
            "source_model_id",
            "identity_status",
            "repository",
            "revision",
            "work_dir",
            "proof_path",
        }:
            if not isinstance(value[field], str) or not value[field]:
                raise AdmissionReceiptError(f"{field} must be a non-empty string")
        elif field != "official_payload_authenticated":
            if type(value[field]) is not int or value[field] < 0:
                raise AdmissionReceiptError(f"{field} must be a nonnegative integer")
    return value


def build_record(receipt: Any) -> dict[str, Any]:
    value = receipt_value(receipt)
    proof = Path(value["proof_path"])
    if proof.is_symlink() or not proof.is_file():
        raise AdmissionReceiptError("source proof path is not an ordinary file")
    if proof.stat().st_size != value["proof_bytes"]:
        raise AdmissionReceiptError("source proof byte count differs")
    return {
        "schema": SCHEMA,
        "result": RESULT,
        "receipt": value,
        "proof_sha256": _sha256_file(proof),
    }


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.is_symlink() or path.exists():
        raise AdmissionReceiptError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cursor = path.parent
    while True:
        if cursor.is_symlink():
            raise AdmissionReceiptError(
                f"output parent traverses a symlink: {path}"
            )
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with open(descriptor, "wb", closefd=True) as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise AdmissionReceiptError(
                f"refusing to replace existing output: {path}"
            ) from error
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        from tritium.salt import admit_qwen36_source

        receipt = admit_qwen36_source(
            args.model_dir,
            revision=args.revision,
            work_dir=args.work_dir,
        )
        record = build_record(receipt)
        _write_new(args.output, record)
    except (OSError, ValueError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps({
        "schema": SCHEMA,
        "result": RESULT,
        "proof_id": record["receipt"]["proof_id"],
        "source_model_id": record["receipt"]["source_model_id"],
        "additive_tensors": record["receipt"]["additive_tensors"],
        "mtp_tensors": record["receipt"]["mtp_tensors"],
        "vision_tensors": record["receipt"]["vision_tensors"],
        "output": str(args.output),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
