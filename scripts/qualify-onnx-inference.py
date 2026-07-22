#!/usr/bin/env python3
"""Aggregate whole-Qwen ONNX execution traces into release evidence."""

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


VERIFIER = runpy.run_path(Path(__file__).with_name("verify-onnx-inference-receipt.py"))
SCHEMA = VERIFIER["SCHEMA"]
TRACE_SCHEMA = VERIFIER["TRACE_SCHEMA"]
TRACE_FIELDS = VERIFIER["TRACE_FIELDS"]
canonical = VERIFIER["canonical"]
sha256 = VERIFIER["sha256"]
inventory = VERIFIER["inventory"]
object_ = VERIFIER["object_"]
derive_trace = VERIFIER["derive_trace"]
validate_receipt = VERIFIER["validate"]

MAX_TRACE_BYTES = VERIFIER["MAX_TRACE_BYTES"]


class QualificationError(ValueError):
    """Whole-model ONNX execution cannot satisfy frozen admission."""


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise QualificationError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def require_clean_revision(repo: Path, revision: str) -> None:
    if git_output(repo, "rev-parse", "HEAD") != revision:
        raise QualificationError("ONNX qualification source revision is not HEAD")
    if git_output(repo, "status", "--short", "--untracked-files=no"):
        raise QualificationError("ONNX qualification requires clean tracked source")


def load_trace(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size <= 0
        or path.stat().st_size > MAX_TRACE_BYTES
    ):
        raise QualificationError("ONNX trace must be a bounded ordinary file")
    try:
        trace = object_(json.loads(path.read_bytes()), TRACE_FIELDS, "ONNX trace")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise QualificationError("ONNX trace must contain frozen JSON") from error
    if trace["schema"] != TRACE_SCHEMA:
        raise QualificationError("ONNX trace schema differs")
    return trace


def artifact_record(value: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": value[0],
        "kind": value[1],
        "name": value[2],
        "bytes": value[3],
        "sha256": value[4],
    }


def ordinary(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise QualificationError(f"{label} must be an ordinary nonempty file")
    return path.resolve(strict=True)


def directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise QualificationError(f"{label} must be an ordinary directory")
    return path.resolve(strict=True)


def executable(path: Path, label: str) -> Path:
    try:
        path = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise QualificationError(f"{label} does not exist") from error
    if not path.is_file() or path.stat().st_size <= 0 or not os.access(path, os.X_OK):
        raise QualificationError(f"{label} must be executable")
    return path


def run_installed_worker(
    trace: Path,
    *,
    candidate: Path,
    wheel: Path,
    python: Path,
    onnx_bundle: Path,
    native_bundle: Path,
    wheel_artifact_id: str,
    onnx_artifact_id: str,
    model_artifact_id: str,
    profile: str,
    conversion_mode: str,
    source_revision: str,
    release: str,
    run_id: str,
) -> None:
    candidate = ordinary(candidate, "candidate manifest")
    wheel = ordinary(wheel, "candidate wheel")
    python = executable(python, "installed-wheel Python")
    onnx_bundle = directory(onnx_bundle, "unpacked ONNX bundle")
    native_bundle = directory(native_bundle, "unpacked native bundle")
    artifacts = inventory(candidate)
    wheel_value = artifacts.get(wheel_artifact_id)
    onnx_value = artifacts.get(onnx_artifact_id)
    model_value = artifacts.get(model_artifact_id)
    if (
        wheel_value is None
        or wheel_value[1] != "python-wheel"
        or wheel_value[2:] != (wheel.name, wheel.stat().st_size, sha256(wheel))
    ):
        raise QualificationError("installed worker wheel differs from candidate")
    if onnx_value is None or onnx_value[1] != "onnx-bundle":
        raise QualificationError("installed worker ONNX artifact is absent")
    if model_value is None or model_value[1] != "model-bundle":
        raise QualificationError("installed worker native model is absent")

    trace.parent.mkdir(parents=True, exist_ok=True)
    wheel_record = trace.parent / "wheel-record.json"
    onnx_record = trace.parent / "onnx-record.json"
    model_record = trace.parent / "model-record.json"
    wheel_record.write_bytes(canonical(artifact_record(wheel_value)) + b"\n")
    onnx_record.write_bytes(canonical(artifact_record(onnx_value)) + b"\n")
    model_record.write_bytes(canonical(artifact_record(model_value)) + b"\n")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(
            token in key.upper() for token in ("TOKEN", "KEY", "SECRET", "PASSWORD")
        )
    }
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONHASHSEED": "0",
            "PATH": str(python.parent),
        }
    )
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-m",
            "tritium.torch.qualify_onnx",
            "--wheel",
            str(wheel),
            "--wheel-record",
            str(wheel_record),
            "--artifact-record",
            str(onnx_record),
            "--model-record",
            str(model_record),
            "--model-artifact-id",
            model_artifact_id,
            "--onnx-bundle",
            str(onnx_bundle),
            "--native-bundle",
            str(native_bundle),
            "--profile",
            profile,
            "--conversion-mode",
            conversion_mode,
            "--source-revision",
            source_revision,
            "--release",
            release,
            "--run-id",
            run_id,
            "--candidate-manifest-sha256",
            sha256(candidate),
            "--output",
            str(trace),
        ],
        cwd=trace.parent,
        env=environment,
        capture_output=True,
        check=False,
        timeout=21_600,
    )
    if result.returncode != 0:
        raise QualificationError("installed whole-model ONNX worker failed")
    load_trace(trace)


def assemble(
    stage: Path,
    *,
    candidate: Path,
    trace_path: Path,
    wheel_artifact_id: str,
    onnx_artifact_id: str,
    model_artifact_id: str,
    source_revision: str,
    release: str,
    run_id: str,
) -> dict[str, Any]:
    load_trace(trace_path)
    trace_path = trace_path.resolve(strict=True)
    artifacts = inventory(candidate)
    wheel = artifacts.get(wheel_artifact_id)
    onnx = artifacts.get(onnx_artifact_id)
    model = artifacts.get(model_artifact_id)
    if wheel is None or wheel[1] != "python-wheel":
        raise QualificationError("ONNX qualification wheel is absent from candidate")
    if onnx is None or onnx[1] != "onnx-bundle":
        raise QualificationError("ONNX qualification bundle is absent from candidate")
    if model is None or model[1] != "model-bundle":
        raise QualificationError("ONNX source model is absent from candidate")
    skeleton = {
        "run_id": run_id,
        "model_artifact_id": model_artifact_id,
        "wheel": artifact_record(wheel),
        "artifact": artifact_record(onnx),
    }
    candidate_sha256 = sha256(candidate)
    derived = derive_trace(
        trace_path,
        receipt=skeleton,
        revision=source_revision,
        release=release,
        candidate_sha256=candidate_sha256,
    )
    stage.mkdir()
    retained = stage / "onnx-execution.json"
    shutil.copyfile(trace_path, retained)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "result": "pass",
        "release": release,
        "source_revision": source_revision,
        "run_id": run_id,
        "candidate_manifest_sha256": candidate_sha256,
        "wheel": skeleton["wheel"],
        "artifact": skeleton["artifact"],
        "model_artifact_id": model_artifact_id,
        **derived,
        "trace": {
            "file": retained.name,
            "bytes": retained.stat().st_size,
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
    wheel_artifact_id: str,
    onnx_artifact_id: str,
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
            wheel_artifact_id=wheel_artifact_id,
            onnx_artifact_id=onnx_artifact_id,
            model_artifact_id=model_artifact_id,
            source_revision=source_revision,
            release=release,
            run_id=run_id,
        )
        os.replace(stage, output_dir)
        return receipt
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def qualify_installed(
    output_dir: Path,
    *,
    repo: Path,
    candidate: Path,
    wheel: Path,
    python: Path,
    onnx_bundle: Path,
    native_bundle: Path,
    wheel_artifact_id: str,
    onnx_artifact_id: str,
    model_artifact_id: str,
    profile: str,
    conversion_mode: str,
    source_revision: str,
    release: str,
    run_id: str,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise QualificationError(f"output directory already exists: {output_dir}")
    repo = repo.resolve(strict=True)
    require_clean_revision(repo, source_revision)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    worker_dir = Path(tempfile.mkdtemp(prefix="tritium-onnx-worker-"))
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    stage.rmdir()
    trace = worker_dir / "onnx-execution.json"
    try:
        run_installed_worker(
            trace,
            candidate=candidate,
            wheel=wheel,
            python=python,
            onnx_bundle=onnx_bundle,
            native_bundle=native_bundle,
            wheel_artifact_id=wheel_artifact_id,
            onnx_artifact_id=onnx_artifact_id,
            model_artifact_id=model_artifact_id,
            profile=profile,
            conversion_mode=conversion_mode,
            source_revision=source_revision,
            release=release,
            run_id=run_id,
        )
        receipt = assemble(
            stage,
            candidate=candidate,
            trace_path=trace,
            wheel_artifact_id=wheel_artifact_id,
            onnx_artifact_id=onnx_artifact_id,
            model_artifact_id=model_artifact_id,
            source_revision=source_revision,
            release=release,
            run_id=run_id,
        )
        os.replace(stage, output_dir)
        return receipt
    finally:
        shutil.rmtree(worker_dir, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--candidate", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trace", type=Path)
    mode.add_argument("--python", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--onnx-bundle", type=Path)
    parser.add_argument("--native-bundle", type=Path)
    parser.add_argument("--profile", choices=("compact-v1", "near-lossless-v1"))
    parser.add_argument("--conversion-mode", choices=("ptq", "refined"))
    parser.add_argument("--wheel-artifact-id", required=True)
    parser.add_argument("--onnx-artifact-id", required=True)
    parser.add_argument("--model-artifact-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    common = {
        "repo": args.repo,
        "candidate": args.candidate.absolute(),
        "wheel_artifact_id": args.wheel_artifact_id,
        "onnx_artifact_id": args.onnx_artifact_id,
        "model_artifact_id": args.model_artifact_id,
        "source_revision": args.source_revision,
        "release": args.release,
        "run_id": args.run_id,
    }
    if args.trace is not None:
        receipt = qualify(
            args.output_dir.absolute(),
            trace_path=args.trace.absolute(),
            **common,
        )
    else:
        missing = [
            name
            for name in (
                "wheel",
                "onnx_bundle",
                "native_bundle",
                "profile",
                "conversion_mode",
            )
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "installed mode requires --"
                + ", --".join(name.replace("_", "-") for name in missing)
            )
        receipt = qualify_installed(
            args.output_dir.absolute(),
            wheel=args.wheel.absolute(),
            python=args.python.absolute(),
            onnx_bundle=args.onnx_bundle.absolute(),
            native_bundle=args.native_bundle.absolute(),
            profile=args.profile,
            conversion_mode=args.conversion_mode,
            **common,
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
