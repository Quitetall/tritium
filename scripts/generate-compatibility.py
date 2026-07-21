#!/usr/bin/env python3
"""Validate and render Tritium's receipt-backed compatibility matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "tritium.compatibility.v1"
STATUSES = {"qualified", "pending", "unsupported"}
REQUIRED_DIMENSIONS = {
    "platform",
    "python-torch",
    "accelerator",
    "onnx-runtime",
    "web",
    "artifact-schema",
}
TOP_LEVEL_FIELDS = {"schema", "release", "dimensions"}
ROW_FIELDS = {"id", "target", "status", "receipt", "blocker", "diagnostic"}
RECEIPT_FIELDS = {"path", "sha256", "source_revision"}
RECEIPT_SCHEMA = "tritium.compatibility-receipt.v1"
MAX_RECEIPT_BYTES = 4 * 1024 * 1024


class MatrixError(ValueError):
    """The matrix is not canonical or claims unsupported evidence."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MatrixError(f"{label} must be an object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MatrixError(f"{label} must be a non-empty string")
    return value


def _exact_fields(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise MatrixError(f"{label} has unknown fields: {', '.join(sorted(unknown))}")


def _contained_path(root: Path, raw: Any, label: str) -> Path:
    text = _string(raw, label)
    logical = PurePosixPath(text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in text:
        raise MatrixError(f"{label} must be a contained POSIX path")
    candidate = root.joinpath(*logical.parts)
    cursor = root
    for part in logical.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise MatrixError(f"{label} must not traverse a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise MatrixError(f"{label} does not exist: {text}") from error
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise MatrixError(f"{label} escapes the matrix directory") from error
    if not resolved.is_file():
        raise MatrixError(f"{label} must name a regular non-symlink file")
    return resolved


def _validate_receipt(value: Any, root: Path, row_id: str, label: str) -> None:
    receipt = _object(value, label)
    _exact_fields(receipt, RECEIPT_FIELDS, label)
    digest = _string(receipt.get("sha256"), f"{label}.sha256")
    revision = _string(receipt.get("source_revision"), f"{label}.source_revision")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise MatrixError(f"{label}.sha256 must be 64 lowercase hexadecimal characters")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise MatrixError(
            f"{label}.source_revision must be a full lowercase Git object ID"
        )
    path = _contained_path(root, receipt.get("path"), f"{label}.path")
    if path.stat().st_size > MAX_RECEIPT_BYTES:
        raise MatrixError(f"{label}.path exceeds {MAX_RECEIPT_BYTES} bytes")
    hasher = hashlib.sha256()
    payload = bytearray()
    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024):
            hasher.update(chunk)
            payload.extend(chunk)
    actual = hasher.hexdigest()
    if actual != digest:
        raise MatrixError(f"{label}.sha256 does not match {receipt['path']}")
    try:
        evidence = _object(json.loads(payload), f"{label}.contents")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MatrixError(f"{label}.path must contain a JSON receipt") from error
    if evidence.get("schema") != RECEIPT_SCHEMA:
        raise MatrixError(f"{label}.contents.schema must equal {RECEIPT_SCHEMA!r}")
    if evidence.get("target_id") != row_id:
        raise MatrixError(f"{label}.contents.target_id does not bind row {row_id!r}")
    if evidence.get("source_revision") != revision:
        raise MatrixError(f"{label}.contents.source_revision does not match the matrix")
    if evidence.get("passed") is not True:
        raise MatrixError(f"{label}.contents.passed must be true")


def validate_matrix(document: Any, matrix_path: Path) -> dict[str, Any]:
    matrix = _object(document, "matrix")
    _exact_fields(matrix, TOP_LEVEL_FIELDS, "matrix")
    if matrix.get("schema") != SCHEMA:
        raise MatrixError(f"matrix.schema must equal {SCHEMA!r}")
    _string(matrix.get("release"), "matrix.release")
    dimensions = _object(matrix.get("dimensions"), "matrix.dimensions")
    missing = REQUIRED_DIMENSIONS - set(dimensions)
    unknown = set(dimensions) - REQUIRED_DIMENSIONS
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown {', '.join(sorted(unknown))}")
        raise MatrixError("matrix.dimensions: " + "; ".join(details))

    seen: set[str] = set()
    for dimension, raw_rows in dimensions.items():
        if not isinstance(raw_rows, list) or not raw_rows:
            raise MatrixError(f"matrix.dimensions.{dimension} must be a non-empty array")
        for ordinal, raw_row in enumerate(raw_rows):
            label = f"matrix.dimensions.{dimension}[{ordinal}]"
            row = _object(raw_row, label)
            _exact_fields(row, ROW_FIELDS, label)
            row_id = _string(row.get("id"), f"{label}.id")
            _string(row.get("target"), f"{label}.target")
            if row_id in seen:
                raise MatrixError(f"duplicate compatibility row id {row_id!r}")
            seen.add(row_id)
            status = row.get("status")
            if status not in STATUSES:
                raise MatrixError(f"{label}.status must be one of {sorted(STATUSES)}")
            present = {key for key in ("receipt", "blocker", "diagnostic") if key in row}
            required = {
                "qualified": {"receipt"},
                "pending": {"blocker"},
                "unsupported": {"diagnostic"},
            }[status]
            if present != required:
                raise MatrixError(
                    f"{label} with status {status!r} requires exactly {sorted(required)}"
                )
            if status == "qualified":
                _validate_receipt(
                    row["receipt"], matrix_path.parent, row_id, f"{label}.receipt"
                )
            elif status == "pending":
                _string(row["blocker"], f"{label}.blocker")
            else:
                diagnostic = _string(row["diagnostic"], f"{label}.diagnostic")
                if not diagnostic.startswith("TRITIUM_UNSUPPORTED_"):
                    raise MatrixError(
                        f"{label}.diagnostic must be a TRITIUM_UNSUPPORTED_* code"
                    )
    return matrix


def render_markdown(matrix: dict[str, Any], source: Path, output: Path | None = None) -> str:
    output = output or Path("docs/compatibility.md")
    lines = [
        "# Tritium compatibility matrix",
        "",
        "<!-- Generated by scripts/generate-compatibility.py; do not edit. -->",
        "",
        f"Release candidate: `{matrix['release']}`",
        "",
        f"Source manifest: `{source.as_posix()}`",
        "",
        "A target is supported only when its row is `qualified` and links to a",
        "digest-bound receipt. `pending` is not support. `unsupported` must fail",
        "with the listed stable diagnostic instead of silently falling back.",
        "",
    ]
    for dimension, rows in matrix["dimensions"].items():
        title = dimension.replace("-", " ").title()
        lines.extend([f"## {title}", "", "| Target | Status | Evidence / failure |", "|---|---|---|"])
        for row in rows:
            if row["status"] == "qualified":
                receipt = row["receipt"]
                receipt_path = source.parent / receipt["path"]
                receipt_link = Path(os.path.relpath(receipt_path, output.parent)).as_posix()
                evidence = (
                    f"[`{receipt['path']}`]({receipt_link}) "
                    f"SHA-256 `{receipt['sha256']}`; revision `{receipt['source_revision']}`"
                )
            elif row["status"] == "pending":
                evidence = row["blocker"]
            else:
                evidence = f"`{row['diagnostic']}`"
            lines.append(f"| {row['target']} | **{row['status']}** | {evidence} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("release/compatibility-v1.1.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/compatibility.md"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        matrix = validate_matrix(document, args.input)
        rendered = render_markdown(matrix, args.input, args.output)
    except (OSError, json.JSONDecodeError, MatrixError) as error:
        parser.error(str(error))
    if args.check:
        try:
            existing = args.output.read_text(encoding="utf-8")
        except OSError as error:
            parser.error(str(error))
        if existing != rendered:
            parser.error(f"{args.output} is stale; regenerate it")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
