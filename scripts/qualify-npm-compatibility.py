#!/usr/bin/env python3
"""Project a validated npm qualification into the compatibility schema."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import runpy
import tempfile
from typing import Any


NPM = runpy.run_path(Path(__file__).with_name("verify-npm-archive-receipt.py"))
validate_npm_receipt = NPM["validate_receipt"]
NpmReceiptError = NPM["NpmReceiptError"]

SCHEMA = "tritium.compatibility-receipt.v1"
TARGET_ID = "node-22"
RELEASE_PATTERN = re.compile(r"1\.1\.0-rc\.(0|[1-9][0-9]*)")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
FIELDS = {
    "schema",
    "target_id",
    "source_revision",
    "passed",
    "install_smoke",
    "host_os",
    "host_arch",
    "toolchain",
    "package",
    "archive",
    "archive_sha256",
    "archive_bytes",
    "source_free",
    "installed_offline",
    "strict_typescript",
    "wasm_build_id",
    "wasm_guest_digest",
    "upstream_receipt_id",
}


class CompatibilityError(ValueError):
    """The projected compatibility receipt is malformed or unbound."""


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CompatibilityError(f"{label} must be an object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompatibilityError(f"{label} must be a non-empty string")
    return value


def _hex(value: Any, label: str) -> str:
    text = _string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise CompatibilityError(f"{label} must be lowercase SHA-256")
    return text


def project(qualified: dict[str, Any]) -> dict[str, Any]:
    """Create compatibility evidence from one already-validated npm receipt."""

    machine = _object(qualified["machine"], "receipt.machine")
    toolchain = _object(qualified["toolchain"], "receipt.toolchain")
    artifact = _object(qualified["artifact"], "receipt.artifact")
    evidence = _object(qualified["evidence"], "receipt.evidence")
    result = {
        "schema": SCHEMA,
        "target_id": TARGET_ID,
        "source_revision": _string(qualified["source_revision"], "receipt.source_revision"),
        "passed": True,
        "install_smoke": True,
        "host_os": _string(machine["system"], "receipt.machine.system"),
        "host_arch": _string(machine["architecture"], "receipt.machine.architecture"),
        "toolchain": {
            "node": _string(toolchain["node"], "receipt.toolchain.node"),
            "npm": _string(toolchain["npm"], "receipt.toolchain.npm"),
        },
        "package": _string(artifact["package"], "receipt.artifact.package"),
        "archive": _string(artifact["name"], "receipt.artifact.name"),
        "archive_sha256": _hex(artifact["sha256"], "receipt.artifact.sha256"),
        "archive_bytes": artifact["bytes"],
        "source_free": evidence["source_free"],
        "installed_offline": evidence["installed_offline"],
        "strict_typescript": evidence["strict_typescript"],
        "wasm_build_id": _string(evidence["wasm_build_id"], "receipt.evidence.wasm_build_id"),
        "wasm_guest_digest": _hex(
            evidence["wasm_guest_digest"], "receipt.evidence.wasm_guest_digest"
        ),
        "upstream_receipt_id": _string(qualified["receipt_id"], "receipt.receipt_id"),
    }
    validate_project(result)
    return result


def validate_project(
    value: Any, revision: str | None = None, release: str | None = None
) -> dict[str, Any]:
    receipt = _object(value, "compatibility receipt")
    if set(receipt) != FIELDS:
        raise CompatibilityError(
            "compatibility receipt fields do not match the frozen schema: "
            f"missing={sorted(FIELDS - set(receipt))}, extra={sorted(set(receipt) - FIELDS)}"
        )
    if receipt["schema"] != SCHEMA or receipt["target_id"] != TARGET_ID:
        raise CompatibilityError("compatibility receipt target/schema mismatch")
    observed_revision = _string(receipt["source_revision"], "source_revision")
    if REVISION_PATTERN.fullmatch(observed_revision) is None:
        raise CompatibilityError("source_revision must be a full lowercase Git object ID")
    if revision is not None and observed_revision != revision:
        raise CompatibilityError("compatibility receipt source revision mismatch")
    if release is not None:
        package = _string(receipt["package"], "package")
        if package != f"@tritium-ai/web@{release}":
            raise CompatibilityError("compatibility receipt package/release mismatch")
        build = _string(receipt["wasm_build_id"], "wasm_build_id")
        if build != f"tritium-wasm@{release}+source-git:{observed_revision}":
            raise CompatibilityError("compatibility receipt WASM build identity mismatch")
    if receipt["passed"] is not True or receipt["install_smoke"] is not True:
        raise CompatibilityError("compatibility receipt must prove a passing install smoke")
    if any(
        receipt[field] is not True
        for field in ("source_free", "installed_offline", "strict_typescript")
    ):
        raise CompatibilityError("compatibility receipt does not prove a clean offline install")
    _string(receipt["host_os"], "host_os")
    _string(receipt["host_arch"], "host_arch")
    toolchain = _object(receipt["toolchain"], "toolchain")
    if set(toolchain) != {"node", "npm"}:
        raise CompatibilityError("toolchain fields do not match the frozen schema")
    _string(toolchain["node"], "toolchain.node")
    _string(toolchain["npm"], "toolchain.npm")
    archive = _string(receipt["archive"], "archive")
    if Path(archive).name != archive or "\\" in archive or not archive.endswith(".tgz"):
        raise CompatibilityError("archive name is unsafe")
    _hex(receipt["archive_sha256"], "archive_sha256")
    if type(receipt["archive_bytes"]) is not int or receipt["archive_bytes"] <= 0:
        raise CompatibilityError("archive_bytes must be a positive integer")
    _hex(receipt["wasm_guest_digest"], "wasm_guest_digest")
    upstream_id = _string(receipt["upstream_receipt_id"], "upstream_receipt_id")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", upstream_id) is None:
        raise CompatibilityError("upstream_receipt_id must be a SHA-256 identity")
    return receipt


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    if path.is_symlink():
        raise CompatibilityError("output must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npm-receipt", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if REVISION_PATTERN.fullmatch(args.source_revision) is None:
            raise CompatibilityError("source revision must be a full lowercase Git object ID")
        if RELEASE_PATTERN.fullmatch(args.release) is None:
            raise CompatibilityError("release must be a canonical v1.1 candidate")
        qualified = validate_npm_receipt(
            args.npm_receipt, args.archive, args.source_revision, args.release
        )
        projected = project(qualified)
        validate_project(projected, args.source_revision, args.release)
        _atomic_write(args.output, projected)
    except (OSError, NpmReceiptError, CompatibilityError) as error:
        parser.error(str(error))
    print(json.dumps(projected, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
