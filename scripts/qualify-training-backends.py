#!/usr/bin/env python3
"""Assemble seven candidate backend bundles into release qualification."""

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
    Path(__file__).with_name("verify-training-backend-receipt.py")
)
SCHEMA = VERIFIER["SCHEMA"]
FAMILIES = VERIFIER["FAMILIES"]
MANIFEST_SHA256 = VERIFIER["MANIFEST_SHA256"]
VECTOR_SHA256 = VERIFIER["VECTOR_SHA256"]
canonical = VERIFIER["canonical"]
sha256 = VERIFIER["sha256"]
candidate_inventory = VERIFIER["candidate_inventory"]
validate_receipt = VERIFIER["validate"]
TrainingBackendReceiptError = VERIFIER["TrainingBackendReceiptError"]


class QualificationError(ValueError):
    """Candidate backend bundles are incomplete, stale, or misassigned."""


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise QualificationError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def require_clean_revision(repo: Path, revision: str) -> None:
    if git_output(repo, "rev-parse", "HEAD") != revision:
        raise QualificationError("backend qualification source revision is not HEAD")
    if git_output(repo, "status", "--short", "--untracked-files=no"):
        raise QualificationError("backend qualification requires clean tracked source")


def parse_bindings(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        family, separator, artifact_id = value.partition("=")
        if separator != "=" or family not in FAMILIES or not artifact_id:
            raise QualificationError("bundle binding must be FAMILY=ARTIFACT_ID")
        if family in result:
            raise QualificationError(f"duplicate bundle binding for {family}")
        result[family] = artifact_id
    if tuple(result) != FAMILIES:
        raise QualificationError("bundle bindings must follow all seven families in order")
    if len(set(result.values())) != len(FAMILIES):
        raise QualificationError("bundle artifact IDs must be unique")
    return result


def assemble(
    stage: Path, *, repo: Path, candidate: Path, artifact_ids: dict[str, str],
    source_revision: str, release: str, run_id: str,
) -> dict[str, Any]:
    if tuple(artifact_ids) != FAMILIES or len(set(artifact_ids.values())) != len(FAMILIES):
        raise QualificationError("backend artifact mapping is incomplete or duplicate")
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise QualificationError("source revision must be 40 lowercase hexadecimal")
    if not release or not run_id:
        raise QualificationError("release and run id must be non-empty")
    if candidate.is_symlink() or not candidate.is_file():
        raise QualificationError("candidate manifest must be ordinary")
    inventory = candidate_inventory(candidate)
    bundles = []
    for family in FAMILIES:
        artifact_id = artifact_ids[family]
        entry = inventory.get(artifact_id)
        if entry is None:
            raise QualificationError(f"{family} bundle is absent from candidate")
        identity, _, kind = entry
        if kind != "training-receipt-bundle":
            raise QualificationError(f"{family} artifact is not a training receipt bundle")
        bundles.append({
            "family": family,
            "artifact": {
                "id": identity[0], "kind": identity[1], "name": identity[2],
                "bytes": identity[3], "sha256": identity[4], "blake3": identity[5],
            },
        })
    receipt: dict[str, Any] = {
        "schema": SCHEMA, "result": "pass", "release": release,
        "source_revision": source_revision, "run_id": run_id,
        "candidate_manifest_sha256": sha256(candidate),
        "manifest_sha256": MANIFEST_SHA256, "vectors_sha256": VECTOR_SHA256,
        "bundles": bundles,
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    stage.mkdir()
    receipt_path = stage / "receipt.json"
    receipt_path.write_bytes(canonical(receipt) + b"\n")
    validate_receipt(receipt_path, source_revision, release, candidate, repo)
    return receipt


def qualify(
    output_dir: Path, *, repo: Path, candidate: Path,
    artifact_ids: dict[str, str], source_revision: str, release: str, run_id: str,
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
            stage, repo=repo, candidate=candidate.resolve(strict=True),
            artifact_ids=artifact_ids, source_revision=source_revision,
            release=release, run_id=run_id,
        )
        os.replace(stage, output_dir)
        return receipt
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--bundle", action="append", default=[], metavar="FAMILY=ARTIFACT_ID"
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify(
        args.output_dir.absolute(), repo=args.repo,
        candidate=args.candidate.absolute(), artifact_ids=parse_bindings(args.bundle),
        source_revision=args.source_revision, release=args.release, run_id=args.run_id,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
