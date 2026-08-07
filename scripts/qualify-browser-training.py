#!/usr/bin/env python3
"""Assemble physical browser lane fragments into admitted release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile
from typing import Any


VERIFIER = runpy.run_path(
    Path(__file__).with_name("verify-browser-training-receipt.py")
)
SCHEMA = VERIFIER["SCHEMA"]
ENGINES = VERIFIER["ENGINES"]
MANIFEST_DIGEST = VERIFIER["MANIFEST_DIGEST"]
VECTOR_DIGEST = VERIFIER["VECTOR_DIGEST"]
MAX_RECEIPT_BYTES = VERIFIER["MAX_RECEIPT_BYTES"]
validate_lane = VERIFIER["validate_lane"]
validate_receipt = VERIFIER["validate"]
canonical = VERIFIER["canonical"]
sha256 = VERIFIER["sha256"]


class QualificationError(ValueError):
    """Browser lane aggregation cannot produce candidate-bound evidence."""


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise QualificationError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def require_clean_revision(repo: Path, revision: str) -> None:
    if git_output(repo, "rev-parse", "HEAD") != revision:
        raise QualificationError("browser qualification source revision is not HEAD")
    status = git_output(repo, "status", "--short", "--untracked-files=all")
    if status:
        raise QualificationError("browser qualification requires clean exact source")


def ordinary(path: Path, label: str, maximum: int) -> Path:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size <= 0
        or path.stat().st_size > maximum
    ):
        raise QualificationError(f"{label} must be a bounded ordinary file")
    return path.resolve(strict=True)


def load_lane(
    path: Path,
    ordinal: int,
    source_revision: str,
    release: str,
    archive: Path,
) -> tuple[dict[str, Any], Path]:
    path = ordinary(path, f"{ENGINES[ordinal]} lane fragment", MAX_RECEIPT_BYTES)
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(
            "browser lane fragment must contain UTF-8 JSON"
        ) from error
    lane = validate_lane(
        value,
        ordinal,
        path.parent.resolve(strict=True),
        source_revision,
        release,
        archive,
    )
    trace = (path.parent / lane["trace"]["file"]).resolve(strict=True)
    return lane, trace


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def assemble(
    stage: Path,
    *,
    archive: Path,
    lane_paths: tuple[Path, Path, Path],
    source_revision: str,
    release: str,
    run_id: str,
) -> dict[str, Any]:
    archive = ordinary(archive, "candidate npm archive", 512 * 1024 * 1024)
    if not archive.name.endswith(".tgz"):
        raise QualificationError("candidate npm archive must end in .tgz")
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise QualificationError("source revision must be 40 lowercase hexadecimal")
    if not release or not run_id:
        raise QualificationError("release and run id must be non-empty")
    stage.mkdir()
    traces = stage / "traces"
    traces.mkdir()
    lanes = []
    for ordinal, path in enumerate(lane_paths):
        lane, source_trace = load_lane(
            path, ordinal, source_revision, release, archive
        )
        lane = json.loads(json.dumps(lane))
        destination = traces / f"{ENGINES[ordinal]}.trace.json"
        shutil.copyfile(source_trace, destination)
        lane["trace"]["file"] = destination.relative_to(stage).as_posix()
        lane["trace"]["bytes"] = destination.stat().st_size
        lane["trace"]["sha256"] = sha256(destination)
        lanes.append(lane)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "result": "pass",
        "release": release,
        "source_revision": source_revision,
        "run_id": run_id,
        "artifact": {
            "kind": "npm-archive",
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
        },
        "manifest_digest": MANIFEST_DIGEST,
        "vector_digest": VECTOR_DIGEST,
        "lanes": lanes,
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    receipt_path = stage / "receipt.json"
    receipt_path.write_bytes(canonical(receipt) + b"\n")
    validate_receipt(receipt_path, source_revision, release, archive)
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            fsync_file(path)
    fsync_directory(traces)
    fsync_directory(stage)
    return receipt


def qualify(
    output_dir: Path,
    *,
    repo: Path,
    archive: Path,
    lane_paths: tuple[Path, Path, Path],
    source_revision: str,
    release: str,
    run_id: str,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise QualificationError(f"output directory already exists: {output_dir}")
    repo = repo.resolve(strict=True)
    require_clean_revision(repo, source_revision)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    stage.rmdir()
    try:
        receipt = assemble(
            stage,
            archive=archive,
            lane_paths=lane_paths,
            source_revision=source_revision,
            release=release,
            run_id=run_id,
        )
        os.replace(stage, output_dir)
        fsync_directory(output_dir.parent)
        return receipt
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--chrome-lane", type=Path, required=True)
    parser.add_argument("--firefox-lane", type=Path, required=True)
    parser.add_argument("--safari-lane", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify(
        args.output_dir.absolute(),
        repo=args.repo,
        archive=args.artifact.absolute(),
        lane_paths=(
            args.chrome_lane.absolute(),
            args.firefox_lane.absolute(),
            args.safari_lane.absolute(),
        ),
        source_revision=args.source_revision,
        release=args.release,
        run_id=args.run_id,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
