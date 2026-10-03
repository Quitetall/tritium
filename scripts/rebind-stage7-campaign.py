#!/usr/bin/env python3
"""Rebind a pre-evidence Stage-7 campaign plan to clean repository HEAD.

This tool changes the campaign source revision and run identity, and can build
the prerequisite receipt index from three supplied receipts. It does not copy,
invent, or qualify measurements. Existing output is never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
from typing import Any


CAMPAIGN_SCHEMA = "tritium.stage7-campaign.v1"
CAMPAIGN_FIELDS = {
    "schema", "release", "source_revision", "run_id", "model", "smoke_model",
    "smoke_provenance", "provenance", "thresholds", "recipe_count",
    "recipe_grid_id", "token_evidence_pack", "evidence",
}
MAX_JSON_BYTES = 32 * 1024 * 1024
HEX = frozenset("0123456789abcdef")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
FILE_FIELDS = {"path", "bytes", "sha256"}
RECEIPT_SCHEMAS = {
    "smoke": "tritium.stage7-smoke.v2",
    "native-kernels": "tritium.stage7-native-kernels.v1",
    "hestia-gate-c": "tritium.stage7-hestia-gate-c.v1",
}


class RebindError(ValueError):
    """Campaign cannot be safely rebound to current source HEAD."""


def _read_regular_file(path: Path, label: str, max_bytes: int) -> bytes:
    """Read one bounded regular file and reject path or content replacement."""
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)

    def open_parent() -> tuple[int, str]:
        absolute = Path(os.path.abspath(path))
        parts = absolute.parts
        if not parts or parts[0] != os.sep or len(parts) < 2:
            raise RebindError(f"{label} must be an ordinary file")
        parent_fd = os.open(os.sep, directory_flags)
        try:
            for component in parts[1:-1]:
                next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
                os.close(parent_fd)
                parent_fd = next_fd
            return parent_fd, parts[-1]
        except OSError:
            os.close(parent_fd)
            raise

    parent_fd: int | None = None
    try:
        parent_fd, filename = open_parent()
        descriptor = os.open(filename, file_flags, dir_fd=parent_fd)
    except OSError as error:
        if parent_fd is not None:
            os.close(parent_fd)
        raise RebindError(f"{label} must be an ordinary file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RebindError(f"{label} must be an ordinary file")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise RebindError(f"{label} exceeds size bounds")

        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise RebindError(f"{label} changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RebindError(f"{label} changed while reading")

        after = os.fstat(descriptor)
        current_parent_fd, current_filename = open_parent()
        try:
            current_path = os.stat(
                current_filename, dir_fd=current_parent_fd, follow_symlinks=False
            )
        finally:
            os.close(current_parent_fd)
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns
        )
        if (
            not stat.S_ISREG(current_path.st_mode)
            or identity(before) != identity(after)
            or identity(before) != identity(current_path)
        ):
            raise RebindError(f"{label} changed while reading")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise RebindError(f"{label} changed while reading")
        return payload
    except OSError as error:
        raise RebindError(f"{label} changed while reading") from error
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_regular_file(path, "campaign template", MAX_JSON_BYTES),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RebindError("campaign template must contain strict UTF-8 JSON") from error
    if not isinstance(value, dict) or set(value) != CAMPAIGN_FIELDS:
        raise RebindError("campaign template fields differ from frozen schema")
    if value["schema"] != CAMPAIGN_SCHEMA:
        raise RebindError("campaign template schema differs")
    return value


def _source_identity(source_root: Path) -> str:
    try:
        top = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RebindError("source repository identity probe failed") from error
    if Path(top).resolve() != source_root.resolve():
        raise RebindError("source root must be repository top level")
    if dirty:
        raise RebindError("source repository must be clean before campaign rebind")
    if len(revision) != 40 or any(character not in HEX for character in revision):
        raise RebindError("source HEAD is not a canonical Git revision")
    return revision


def _count(value: Any, needle: str) -> int:
    if isinstance(value, dict):
        return sum(_count(key, needle) + _count(item, needle) for key, item in value.items())
    if isinstance(value, list):
        return sum(_count(item, needle) for item in value)
    return int(value == needle)


def _open_record(root: Path, record: Any, label: str) -> bytes:
    if not isinstance(record, dict) or set(record) != FILE_FIELDS:
        raise RebindError(f"{label} file record fields differ")
    logical_text = record["path"]
    if not isinstance(logical_text, str) or not logical_text:
        raise RebindError(f"{label}.path must be a nonempty string")
    logical = PurePosixPath(logical_text)
    if (
        logical.is_absolute()
        or ".." in logical.parts
        or "\\" in logical_text
        or logical.as_posix() != logical_text
    ):
        raise RebindError(f"{label}.path must be a contained POSIX path")
    if (
        type(record["bytes"]) is not int
        or record["bytes"] <= 0
        or not isinstance(record["sha256"], str)
        or len(record["sha256"]) != 64
        or any(character not in HEX for character in record["sha256"])
    ):
        raise RebindError(f"{label} byte or digest record is invalid")
    path = root.joinpath(*logical.parts)
    cursor = root
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise RebindError(f"{label}.path traverses a symlink")
    payload = _read_regular_file(path, f"{label}.path", MAX_JSON_BYTES)
    if len(payload) != record["bytes"]:
        raise RebindError(f"{label}.bytes differs from file")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != record["sha256"]:
        raise RebindError(f"{label}.sha256 differs from file")
    return payload


def _receipt_record(root: Path, path: Path, kind: str) -> dict[str, Any]:
    """Create a contained campaign file record for one prerequisite receipt."""
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise RebindError(
            f"{kind} receipt must be inside the campaign evidence directory"
        ) from error
    logical = PurePosixPath(relative.as_posix())
    if (
        logical.is_absolute()
        or ".." in logical.parts
        or "\\" in logical.as_posix()
        or logical.as_posix() in ("", ".")
    ):
        raise RebindError(f"{kind} receipt path is not a contained POSIX path")
    cursor = root
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise RebindError(f"{kind} receipt path traverses a symlink")
    payload = _read_regular_file(candidate, f"{kind} receipt", MAX_JSON_BYTES)
    return {
        "kind": kind,
        "path": logical.as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validate_prerequisites(
    value: dict[str, Any], root: Path, target_revision: str
) -> None:
    _open_record(root, value["token_evidence_pack"], "campaign token evidence pack")
    evidence = value["evidence"]
    if not isinstance(evidence, list) or len(evidence) != 3:
        raise RebindError("campaign prerequisite evidence inventory is incomplete")
    expected = ("smoke", "native-kernels", "hestia-gate-c")
    for ordinal, kind in enumerate(expected):
        record = evidence[ordinal]
        if not isinstance(record, dict) or set(record) != FILE_FIELDS | {"kind"}:
            raise RebindError(f"evidence[{ordinal}] fields differ")
        if record["kind"] != kind:
            raise RebindError("campaign prerequisite evidence order differs")
        receipt_payload = _open_record(
            root,
            {field: record[field] for field in FILE_FIELDS},
            f"evidence[{ordinal}]",
        )
        try:
            receipt = json.loads(
                receipt_payload,
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"invalid JSON constant {token}")
                ),
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise RebindError(f"evidence[{ordinal}] must contain strict UTF-8 JSON") from error
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != RECEIPT_SCHEMAS[kind]
        ):
            raise RebindError(f"evidence[{ordinal}] receipt schema differs")
        if receipt.get("source_revision") != target_revision:
            raise RebindError(
                f"evidence[{ordinal}] source revision differs from target HEAD"
            )


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RebindError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cursor = path.parent
    while True:
        if cursor.is_symlink():
            raise RebindError("output parent traverses a symlink")
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RebindError(f"refusing to replace existing output: {path}") from error
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def rebind(
    template: Path,
    *,
    source_root: Path,
    run_id: str,
    output: Path,
    smoke_receipt: Path | None = None,
    native_kernels_receipt: Path | None = None,
    hestia_gate_c_receipt: Path | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    target_revision = _source_identity(source_root)
    if not RUN_ID.fullmatch(run_id):
        raise RebindError("run_id must be 1-128 ASCII alphanumeric, dot, underscore, or hyphen")
    value = _load(template)
    old_revision = value["source_revision"]
    if (
        not isinstance(old_revision, str)
        or len(old_revision) != 40
        or any(character not in HEX for character in old_revision)
    ):
        raise RebindError("campaign source_revision is not canonical")
    if old_revision == target_revision:
        raise RebindError("campaign is already bound to current HEAD")
    if _count(value, old_revision) != 1:
        raise RebindError("old source revision appears outside top-level campaign identity")
    supplied_receipts = (
        smoke_receipt,
        native_kernels_receipt,
        hestia_gate_c_receipt,
    )
    if any(path is not None for path in supplied_receipts):
        if not all(path is not None for path in supplied_receipts):
            raise RebindError("all three prerequisite receipt paths must be supplied together")
        evidence_root = template.resolve(strict=True).parent
        value["evidence"] = [
            _receipt_record(evidence_root, path, kind)
            for path, kind in zip(
                supplied_receipts,
                ("smoke", "native-kernels", "hestia-gate-c"),
                strict=True,
            )
            if path is not None
        ]
    _validate_prerequisites(
        value, template.resolve(strict=True).parent, target_revision
    )
    if run_id == value["run_id"]:
        raise RebindError("new run_id must differ from template run_id")
    rebound = dict(value)
    rebound["source_revision"] = target_revision
    rebound["run_id"] = run_id
    _write_new(output, rebound)
    return {
        "schema": CAMPAIGN_SCHEMA,
        "old_source_revision": old_revision,
        "source_revision": target_revision,
        "run_id": run_id,
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--smoke-receipt", type=Path,
        help="Stage-7 smoke receipt inside the template evidence directory",
    )
    parser.add_argument(
        "--native-kernels-receipt", type=Path,
        help="Stage-7 native-kernels receipt inside the template evidence directory",
    )
    parser.add_argument(
        "--hestia-gate-c-receipt", type=Path,
        help="Stage-7 HESTIA gate-C receipt inside the template evidence directory",
    )
    args = parser.parse_args()
    try:
        result = rebind(
            args.template,
            source_root=args.source_root,
            run_id=args.run_id,
            output=args.output,
            smoke_receipt=args.smoke_receipt,
            native_kernels_receipt=args.native_kernels_receipt,
            hestia_gate_c_receipt=args.hestia_gate_c_receipt,
        )
    except (OSError, RebindError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
