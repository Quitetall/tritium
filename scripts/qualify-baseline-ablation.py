#!/usr/bin/env python3
"""Aggregate raw matched-byte baseline traces into release evidence."""

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
    Path(__file__).with_name("verify-estimator-refinement-receipt.py")
)
SCHEMA = VERIFIER["ABLATION_SCHEMA"]
TRACE_FIELDS = VERIFIER["ABLATION_TRACE_FIELDS"]
TRACE_SCHEMA = VERIFIER["ABLATION_TRACE_SCHEMA"]
canonical = VERIFIER["canonical"]
sha256 = VERIFIER["sha256"]
inventory = VERIFIER["inventory"]
object_ = VERIFIER["object_"]
derive_trace = VERIFIER["derive_ablation_trace"]
validate_receipt = VERIFIER["validate_ablation"]

MAX_TRACE_BYTES = 32 * 1024 * 1024


class QualificationError(ValueError):
    """Raw baseline execution cannot satisfy frozen admission."""


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise QualificationError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def require_clean_revision(repo: Path, revision: str) -> None:
    if git_output(repo, "rev-parse", "HEAD") != revision:
        raise QualificationError("ablation qualification source revision is not HEAD")
    if git_output(repo, "status", "--short", "--untracked-files=no"):
        raise QualificationError("ablation qualification requires clean tracked source")


def load_trace(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size <= 0
        or path.stat().st_size > MAX_TRACE_BYTES
    ):
        raise QualificationError("baseline trace must be a bounded ordinary file")
    try:
        trace = object_(json.loads(path.read_bytes()), TRACE_FIELDS, "baseline trace")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise QualificationError("baseline trace must contain frozen JSON") from error
    if trace["schema"] != TRACE_SCHEMA:
        raise QualificationError("baseline trace schema differs")
    return trace


def artifact_record(value: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": value[0], "kind": value[1], "name": value[2],
        "bytes": value[3], "sha256": value[4],
    }


def assemble(
    stage: Path,
    *,
    candidate: Path,
    trace_path: Path,
    model_artifact_id: str,
    source_revision: str,
    release: str,
    run_id: str,
) -> dict[str, Any]:
    load_trace(trace_path)
    trace_path = trace_path.resolve(strict=True)
    artifacts = inventory(candidate)
    model = artifacts.get(model_artifact_id)
    if model is None or model[1] != "model-bundle":
        raise QualificationError("ablation model is absent from candidate")
    skeleton = {"run_id": run_id, "model_artifact_id": model_artifact_id}
    derived = derive_trace(
        trace_path,
        receipt=skeleton,
        revision=source_revision,
        release=release,
    )
    stage.mkdir()
    retained = stage / "baseline-ablation-execution.json"
    shutil.copyfile(trace_path, retained)
    receipt: dict[str, Any] = {
        "schema": SCHEMA, "result": "pass", "release": release,
        "source_revision": source_revision, "run_id": run_id,
        "candidate_manifest_sha256": sha256(candidate),
        "anchor_artifact": artifact_record(model),
        "model_artifact_id": model_artifact_id,
        **derived,
        "trace": {
            "path": retained.name, "bytes": retained.stat().st_size,
            "sha256": sha256(retained),
        },
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    receipt_path = stage / "receipt.json"
    receipt_path.write_bytes(canonical(receipt) + b"\n")
    validate_receipt(receipt_path, source_revision, release, candidate)
    return receipt


def qualify(
    output_dir: Path,
    *,
    repo: Path,
    candidate: Path,
    trace_path: Path,
    model_artifact_id: str,
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
            candidate=candidate,
            trace_path=trace_path,
            model_artifact_id=model_artifact_id,
            source_revision=source_revision,
            release=release,
            run_id=run_id,
        )
        os.replace(stage, output_dir)
        return receipt
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-artifact-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify(
        args.output_dir.absolute(),
        repo=args.repo,
        candidate=args.candidate.absolute(),
        trace_path=args.trace.absolute(),
        model_artifact_id=args.model_artifact_id,
        source_revision=args.source_revision,
        release=args.release,
        run_id=args.run_id,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
