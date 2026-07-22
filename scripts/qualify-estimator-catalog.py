#!/usr/bin/env python3
"""Run installed candidate wheel estimator catalog and seal its evidence."""

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
SCHEMA = VERIFIER["ESTIMATOR_SCHEMA"]
TRACE_SCHEMA = VERIFIER["TRACE_SCHEMA"]
TRACE_FIELDS = VERIFIER["TRACE_FIELDS"]
TRACE_WHEEL_FIELDS = VERIFIER["TRACE_WHEEL_FIELDS"]
ENVIRONMENT_FIELDS = VERIFIER["ENVIRONMENT_FIELDS"]
ESTIMATOR_CASE_FIELDS = VERIFIER["ESTIMATOR_CASE_FIELDS"]
PLUGIN_FIELDS = VERIFIER["PLUGIN_FIELDS"]
ESTIMATORS = VERIFIER["ESTIMATORS"]
canonical = VERIFIER["canonical"]
sha256 = VERIFIER["sha256"]
inventory = VERIFIER["inventory"]
object_ = VERIFIER["object_"]
validate_receipt = VERIFIER["validate_estimators"]

MAX_TRACE_BYTES = 32 * 1024 * 1024


class QualificationError(ValueError):
    """Installed estimator execution is stale, incomplete, or unbound."""


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise QualificationError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def require_clean_revision(repo: Path, revision: str) -> None:
    if git_output(repo, "rev-parse", "HEAD") != revision:
        raise QualificationError("estimator qualification source revision is not HEAD")
    if git_output(repo, "status", "--short", "--untracked-files=no"):
        raise QualificationError("estimator qualification requires clean tracked source")


def ordinary(path: Path, label: str, *, max_bytes: int | None = None) -> Path:
    size = path.stat().st_size if path.exists() else 0
    if (
        path.is_symlink()
        or not path.is_file()
        or size <= 0
        or (max_bytes is not None and size > max_bytes)
    ):
        raise QualificationError(f"{label} must be a bounded ordinary file")
    return path.resolve(strict=True)


def executable(path: Path, label: str) -> Path:
    try:
        path = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise QualificationError(f"{label} does not exist") from error
    if not path.is_file() or path.stat().st_size <= 0 or not os.access(path, os.X_OK):
        raise QualificationError(f"{label} must be executable")
    return path


def anchor(candidate: Path, wheel: Path) -> dict[str, Any]:
    wheel = ordinary(wheel, "candidate wheel")
    matches = [
        value for value in inventory(candidate).values()
        if value[1] == "python-wheel" and value[2:] == (
            wheel.name, wheel.stat().st_size, sha256(wheel)
        )
    ]
    if len(matches) != 1:
        raise QualificationError("candidate must bind exactly one matching wheel")
    value = matches[0]
    return {
        "id": value[0], "kind": value[1], "name": value[2],
        "bytes": value[3], "sha256": value[4],
    }


def load_trace(
    trace_path: Path, *, wheel: Path, source_revision: str, release: str,
    run_id: str,
) -> dict[str, Any]:
    trace_path = ordinary(
        trace_path, "estimator execution trace", max_bytes=MAX_TRACE_BYTES
    )
    try:
        trace = object_(json.loads(trace_path.read_bytes()), TRACE_FIELDS, "trace")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise QualificationError("estimator trace must contain frozen UTF-8 JSON") from error
    wheel_record = object_(trace["wheel"], TRACE_WHEEL_FIELDS, "trace wheel")
    environment = object_(trace["environment"], ENVIRONMENT_FIELDS, "environment")
    if (
        trace["schema"] != TRACE_SCHEMA or trace["result"] != "pass"
        or trace["release"] != release or trace["source_revision"] != source_revision
        or trace["run_id"] != run_id
        or wheel_record != {
            "name": wheel.name, "bytes": wheel.stat().st_size,
            "sha256": sha256(wheel),
        }
        or environment["tritium"] != release.replace("-rc.", "rc")
        or environment["device"] != "cpu"
    ):
        raise QualificationError("estimator trace identity differs")
    cases = trace["estimators"]
    if not isinstance(cases, list) or len(cases) != len(ESTIMATORS):
        raise QualificationError("estimator trace inventory is incomplete")
    for ordinal, expected in enumerate(ESTIMATORS):
        case = object_(cases[ordinal], ESTIMATOR_CASE_FIELDS, f"estimators[{ordinal}]")
        if (case["name"], case["algorithm_id"], case["physical_planes"]) != expected:
            raise QualificationError("estimator trace order differs")
    object_(trace["external_plugin"], PLUGIN_FIELDS, "external plugin")
    return trace


def seal(
    stage: Path, *, candidate: Path, wheel: Path, trace_path: Path,
    source_revision: str, release: str, run_id: str,
) -> dict[str, Any]:
    candidate = ordinary(candidate, "candidate manifest")
    wheel = ordinary(wheel, "candidate wheel")
    trace = load_trace(
        trace_path, wheel=wheel, source_revision=source_revision,
        release=release, run_id=run_id,
    )
    stage.mkdir()
    retained = stage / "estimator-execution.json"
    shutil.copyfile(trace_path, retained)
    receipt: dict[str, Any] = {
        "schema": SCHEMA, "result": "pass", "release": release,
        "source_revision": source_revision, "run_id": run_id,
        "candidate_manifest_sha256": sha256(candidate),
        "anchor_artifact": anchor(candidate, wheel),
        "estimators": trace["estimators"],
        "external_plugin": trace["external_plugin"],
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
    output_dir: Path, *, repo: Path, candidate: Path, wheel: Path,
    python: Path, source_revision: str, release: str, run_id: str,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise QualificationError(f"output directory already exists: {output_dir}")
    repo = repo.resolve(strict=True)
    require_clean_revision(repo, source_revision)
    candidate = ordinary(candidate, "candidate manifest")
    wheel = ordinary(wheel, "candidate wheel")
    python = executable(python, "installed-wheel Python")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    worker_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.worker.", dir=output_dir.parent)
    )
    seal_stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.seal.", dir=output_dir.parent)
    )
    seal_stage.rmdir()
    trace = worker_dir / "worker.json"
    try:
        environment = {
            key: value for key, value in os.environ.items()
            if not any(token in key.upper() for token in ("TOKEN", "KEY", "SECRET", "PASSWORD"))
        }
        environment.update({
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "PIP_NO_INDEX": "1", "PYTHONHASHSEED": "0",
        })
        result = subprocess.run(
            [
                str(python), "-I", "-m", "tritium.torch.qualify_estimators",
                "--wheel", str(wheel.resolve(strict=True)),
                "--source-revision", source_revision, "--release", release,
                "--run-id", run_id, "--output", str(trace),
            ],
            env=environment, capture_output=True, check=False, timeout=600,
        )
        if result.returncode != 0:
            raise QualificationError("installed estimator worker failed")
        trace = ordinary(
            trace, "installed estimator trace", max_bytes=MAX_TRACE_BYTES
        )
        receipt = seal(
            seal_stage, candidate=candidate, wheel=wheel,
            trace_path=trace, source_revision=source_revision,
            release=release, run_id=run_id,
        )
        os.replace(seal_stage, output_dir)
        return receipt
    finally:
        shutil.rmtree(worker_dir, ignore_errors=True)
        shutil.rmtree(seal_stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify(
        args.output_dir.absolute(), repo=args.repo,
        candidate=args.candidate.absolute(), wheel=args.wheel.absolute(),
        python=args.python.absolute(), source_revision=args.source_revision,
        release=args.release, run_id=args.run_id,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
