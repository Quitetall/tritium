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
RECEIPT_SCHEMA = "tritium.abi3-matrix-qualification.v1"
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
RECEIPT_FIELDS = {
    "schema", "receipt_id", "target_id", "source_revision", "release",
    "run_id", "passed", "matrix_schema", "cells",
}
CELL_FIELDS = {
    "cell_id", "path", "sha256", "python_version", "target_id", "wheel",
    "wheel_sha256", "wheel_bytes",
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


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


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


def aggregate(
    evidence_dir: Path, revision: str, release: str, run_id: str,
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise AggregateError("source revision must be a full lowercase Git object ID")
    if re.fullmatch(r"1\.1\.0-rc\.(0|[1-9][0-9]*)", release) is None:
        raise AggregateError("release must be a canonical v1.1 candidate")
    if not run_id:
        raise AggregateError("run id must be non-empty")
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
            "target_id": value["target_id"],
            "wheel": value["wheel"],
            "wheel_sha256": value["sha256"],
            "wheel_bytes": value["bytes"],
        }
        for cell, (path, value) in sorted(cells.items())
    ]
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "target_id": TARGET_ID,
        "source_revision": revision,
        "release": release,
        "run_id": run_id,
        "passed": True,
        "matrix_schema": SCHEMA,
        "cells": evidence,
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(_canonical(receipt)).hexdigest()
    return receipt


def validate_receipt(
    path: Path, revision: str, release: str,
) -> dict[str, object]:
    value = _load_receipt(path)
    if value["schema"] != RECEIPT_SCHEMA or value["passed"] is not True:
        raise AggregateError("compatibility receipt did not pass")
    if value["source_revision"] != revision or value["release"] != release:
        raise AggregateError("compatibility receipt release identity mismatch")
    if value["target_id"] != TARGET_ID or value["matrix_schema"] != SCHEMA:
        raise AggregateError("compatibility receipt matrix identity mismatch")
    if not isinstance(value["run_id"], str) or not value["run_id"]:
        raise AggregateError("compatibility receipt run id is invalid")
    cells = value["cells"]
    if not isinstance(cells, list) or len(cells) != len(expected_cells()):
        raise AggregateError("compatibility receipt cell count mismatch")
    observed: set[str] = set()
    paths: set[str] = set()
    portable_paths: set[str] = set()
    target_wheels: dict[str, tuple[object, object, object]] = {}
    for ordinal, cell in enumerate(cells):
        if not isinstance(cell, dict) or set(cell) != CELL_FIELDS:
            raise AggregateError(f"compatibility receipt cell {ordinal} fields mismatch")
        cell_id = _string(cell["cell_id"], f"cells[{ordinal}].cell_id")
        logical_path = _string(cell["path"], f"cells[{ordinal}].path")
        portable_path = logical_path.casefold()
        if (
            Path(logical_path).name != logical_path
            or "\\" in logical_path
            or cell_id in observed
            or logical_path in paths
            or portable_path in portable_paths
        ):
            raise AggregateError("compatibility receipt has duplicate cell or path")
        observed.add(cell_id)
        paths.add(logical_path)
        portable_paths.add(portable_path)
        for field in ("sha256", "wheel_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(cell[field])) is None:
                raise AggregateError(f"compatibility receipt {field} is invalid")
        if type(cell["wheel_bytes"]) is not int or cell["wheel_bytes"] <= 0:
            raise AggregateError("compatibility receipt wheel byte count is invalid")
        target = _string(cell["target_id"], f"cells[{ordinal}].target_id")
        wheel = _string(cell["wheel"], f"cells[{ordinal}].wheel")
        _string(cell["python_version"], f"cells[{ordinal}].python_version")
        if target not in VERSIONS or not cell_id.startswith(f"{target}-cp3."):
            raise AggregateError("compatibility receipt target/cell identity mismatch")
        platform_pattern = TARGET_CONTRACTS[target][2]
        wheel_match = re.fullmatch(
            r"tritium_torch-(?P<version>[^-]+)-cp39-abi3-(?P<platform>[^-]+)\.whl",
            wheel,
        )
        if (
            wheel_match is None
            or wheel_match.group("version") != release.replace("1.1.0-rc.", "1.1.0rc")
            or re.fullmatch(platform_pattern, wheel_match.group("platform")) is None
        ):
            raise AggregateError("compatibility receipt wheel target/version mismatch")
        identity = (wheel, cell["wheel_sha256"], cell["wheel_bytes"])
        if target in target_wheels and target_wheels[target] != identity:
            raise AggregateError("compatibility receipt target reused different wheels")
        target_wheels[target] = identity
    if observed != expected_cells():
        raise AggregateError("compatibility receipt cell identities mismatch")
    if set(target_wheels) != set(VERSIONS):
        raise AggregateError("compatibility receipt target inventory mismatch")
    unsigned = dict(value)
    receipt_id = unsigned.pop("receipt_id")
    expected_id = "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()
    if receipt_id != expected_id:
        raise AggregateError("compatibility receipt identity mismatch")
    return value


def _load_receipt(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise AggregateError("compatibility receipt must be a bounded ordinary file")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AggregateError("compatibility receipt must contain UTF-8 JSON") from error
    if not isinstance(value, dict) or set(value) != RECEIPT_FIELDS:
        raise AggregateError("compatibility receipt fields do not match frozen schema")
    return value


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
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = aggregate(
            args.evidence_dir, args.source_revision, args.release, args.run_id
        )
        _atomic_write(args.output, receipt)
        validate_receipt(args.output, args.source_revision, args.release)
    except (OSError, AggregateError) as error:
        parser.error(str(error))
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
