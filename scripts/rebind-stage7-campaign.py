#!/usr/bin/env python3
"""Rebind a pre-evidence Stage-7 campaign plan to clean repository HEAD.

This tool changes only the campaign source revision and run identity. It does
not copy, invent, or qualify measurements. Existing output is never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
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


class RebindError(ValueError):
    """Campaign cannot be safely rebound to current source HEAD."""


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
    if path.is_symlink() or not path.is_file():
        raise RebindError("campaign template must be an ordinary file")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_JSON_BYTES:
        raise RebindError("campaign template exceeds size bounds")
    try:
        value = json.loads(
            path.read_bytes(),
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


def _open_record(root: Path, record: Any, label: str) -> Path:
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
    if not path.is_file() or path.is_symlink():
        raise RebindError(f"{label}.path must name an ordinary file")
    if path.stat().st_size != record["bytes"]:
        raise RebindError(f"{label}.bytes differs from file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise RebindError(f"{label}.sha256 differs from file")
    return path


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
        receipt_path = _open_record(
            root,
            {field: record[field] for field in FILE_FIELDS},
            f"evidence[{ordinal}]",
        )
        try:
            receipt = json.loads(
                receipt_path.read_bytes(),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"invalid JSON constant {token}")
                ),
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise RebindError(f"evidence[{ordinal}] must contain strict UTF-8 JSON") from error
        if (
            not isinstance(receipt, dict)
            or receipt.get("source_revision") != target_revision
        ):
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
    args = parser.parse_args()
    try:
        result = rebind(
            args.template,
            source_root=args.source_root,
            run_id=args.run_id,
            output=args.output,
        )
    except (OSError, RebindError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
