#!/usr/bin/env python3
"""Aggregate exact-wheel ABI3 smoke records into one compatibility receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


SCHEMA = "tritium.wheel-smoke.v1"
RECEIPT_SCHEMA = "tritium.compatibility-receipt.v1"
TARGET_ID = "python-abi3-39-plus"
MAX_EVIDENCE_BYTES = 1024 * 1024
VERSIONS = {
    "linux-x86_64-cpu": range(9, 15),
    "windows-x86_64-cpu": range(9, 15),
    "macos-arm64-cpu": range(11, 15),
}
TARGET_CONTRACTS = {
    "linux-x86_64-cpu": ("linux", {"x86_64", "amd64"}, r"manylinux.*_x86_64"),
    "windows-x86_64-cpu": ("win32", {"x86_64", "amd64"}, r"win_amd64"),
    "macos-arm64-cpu": ("darwin", {"arm64", "aarch64"}, r"macosx_.*_universal2"),
}
FIELDS = {
    "schema",
    "cell_id",
    "target_id",
    "source_revision",
    "passed",
    "python_implementation",
    "python_version",
    "host_os",
    "host_arch",
    "wheel",
    "sha256",
    "bytes",
    "version",
    "platform_tag",
}


class AggregateError(ValueError):
    """The ABI3 evidence set is incomplete, inconsistent or malformed."""


def expected_cells() -> set[str]:
    return {
        f"{target}-cp3.{minor}"
        for target, minors in VERSIONS.items()
        for minor in minors
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise AggregateError(f"evidence must be a regular non-symlink file: {path.name}")
    if path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise AggregateError(f"evidence exceeds {MAX_EVIDENCE_BYTES} bytes: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AggregateError(f"invalid JSON evidence {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise AggregateError(f"evidence must contain an object: {path.name}")
    if set(value) != FIELDS:
        raise AggregateError(
            f"evidence fields are not canonical in {path.name}: "
            f"missing={sorted(FIELDS - set(value))}, extra={sorted(set(value) - FIELDS)}"
        )
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AggregateError(f"{label} must be a non-empty string")
    return value


def _validate_cell(value: dict[str, object], revision: str, path: Path) -> str:
    if value["schema"] != SCHEMA or value["passed"] is not True:
        raise AggregateError(f"{path.name} is not passed {SCHEMA} evidence")
    if value["source_revision"] != revision:
        raise AggregateError(f"{path.name} source revision mismatch")
    if value["python_implementation"] != "CPython":
        raise AggregateError(f"{path.name} did not execute CPython")
    target = _string(value["target_id"], f"{path.name}.target_id")
    if target not in VERSIONS:
        raise AggregateError(f"{path.name} has unsupported target {target!r}")
    expected_os, expected_arches, expected_platform = TARGET_CONTRACTS[target]
    host_os = _string(value["host_os"], f"{path.name}.host_os")
    host_arch = _string(value["host_arch"], f"{path.name}.host_arch").lower()
    platform_tag = _string(value["platform_tag"], f"{path.name}.platform_tag")
    if host_os != expected_os or host_arch not in expected_arches:
        raise AggregateError(f"{path.name} host does not match target {target!r}")
    if re.fullmatch(expected_platform, platform_tag) is None:
        raise AggregateError(f"{path.name} platform tag does not match target {target!r}")
    wheel = _string(value["wheel"], f"{path.name}.wheel")
    candidate_version = _string(value["version"], f"{path.name}.version")
    wheel_match = re.fullmatch(
        r"tritium_torch-(?P<version>[^-]+)-cp39-abi3-(?P<platform>[^-]+)\.whl", wheel
    )
    if wheel_match is None or wheel_match.group("platform") != platform_tag:
        raise AggregateError(f"{path.name} wheel filename does not bind its platform tag")
    if wheel_match.group("version") != candidate_version:
        raise AggregateError(f"{path.name} wheel filename does not bind its version")
    version = _string(value["python_version"], f"{path.name}.python_version")
    match = re.fullmatch(r"3\.(\d+)(?:\.\d+)?(?:[-+].*)?", version)
    if match is None:
        raise AggregateError(f"{path.name} has invalid CPython version {version!r}")
    minor = int(match.group(1))
    cell = f"{target}-cp3.{minor}"
    if value["cell_id"] != cell:
        raise AggregateError(f"{path.name} cell_id is not runtime-derived {cell!r}")
    if minor not in VERSIONS[target]:
        raise AggregateError(f"{path.name} is outside the admitted interpreter matrix")
    digest = _string(value["sha256"], f"{path.name}.sha256")
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise AggregateError(f"{path.name}.sha256 must be lowercase SHA-256")
    if not isinstance(value["bytes"], int) or isinstance(value["bytes"], bool) or value["bytes"] <= 0:
        raise AggregateError(f"{path.name}.bytes must be a positive integer")
    return cell


def aggregate(evidence_dir: Path, revision: str) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise AggregateError("source revision must be a full lowercase Git object ID")
    paths = sorted(evidence_dir.glob("*.json"))
    cells: dict[str, tuple[Path, dict[str, object]]] = {}
    wheels: dict[str, tuple[object, ...]] = {}
    candidate_versions: set[object] = set()
    for path in paths:
        value = _load(path)
        cell = _validate_cell(value, revision, path)
        if cell in cells:
            raise AggregateError(f"duplicate ABI3 evidence cell {cell!r}")
        target = str(value["target_id"])
        identity = tuple(
            value[key] for key in ("wheel", "sha256", "bytes", "version", "platform_tag")
        )
        if target in wheels and wheels[target] != identity:
            raise AggregateError(f"target {target!r} did not reuse one exact wheel")
        wheels[target] = identity
        candidate_versions.add(value["version"])
        cells[cell] = (path, value)
    if len(candidate_versions) != 1:
        raise AggregateError(
            f"ABI3 matrix must use one candidate version, found {sorted(map(str, candidate_versions))}"
        )
    expected = expected_cells()
    if set(cells) != expected:
        raise AggregateError(
            f"ABI3 matrix mismatch; missing={sorted(expected - set(cells))}, "
            f"extra={sorted(set(cells) - expected)}"
        )
    evidence = [
        {
            "cell_id": cell,
            "path": path.name,
            "sha256": _sha256(path),
            "python_version": value["python_version"],
            "wheel_sha256": value["sha256"],
        }
        for cell, (path, value) in sorted(cells.items())
    ]
    return {
        "schema": RECEIPT_SCHEMA,
        "target_id": TARGET_ID,
        "source_revision": revision,
        "passed": True,
        "matrix_schema": SCHEMA,
        "cells": evidence,
    }


def _atomic_write(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = aggregate(args.evidence_dir, args.source_revision)
        _atomic_write(args.output, receipt)
    except (OSError, AggregateError) as error:
        parser.error(str(error))
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
