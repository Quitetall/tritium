#!/usr/bin/env python3
"""Aggregate raw seven-target training traces into release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import statistics
import subprocess
import tempfile
from typing import Any


PERFORMANCE = runpy.run_path(
    Path(__file__).with_name("verify-training-performance-receipt.py")
)
SCHEMA = PERFORMANCE["SCHEMA"]
TRACE_SCHEMA = PERFORMANCE["TRACE_SCHEMA"]
TRACE_FIELDS = PERFORMANCE["TRACE_FIELDS"]
SAMPLE_FIELDS = PERFORMANCE["SAMPLE_FIELDS"]
FAMILIES = PERFORMANCE["FAMILIES"]
canonical = PERFORMANCE["canonical"]
sha256 = PERFORMANCE["sha256"]
inventory = PERFORMANCE["inventory"]
object_ = PERFORMANCE["object_"]
number = PERFORMANCE["number"]
integer = PERFORMANCE["integer"]
percentile95 = PERFORMANCE["percentile95"]
validate_receipt = PERFORMANCE["validate"]
TrainingPerformanceError = PERFORMANCE["TrainingPerformanceError"]

BACKENDS = runpy.run_path(
    Path(__file__).with_name("verify-training-backend-receipt.py")
)
validate_backends = BACKENDS["validate"]

WORKLOAD_ID = "training-manifest-v2-full-117"
MAX_TRACE_BYTES = 32 * 1024 * 1024


class QualificationError(ValueError):
    """Raw training performance traces cannot satisfy frozen admission."""


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise QualificationError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def require_clean_revision(repo: Path, revision: str) -> None:
    if git_output(repo, "rev-parse", "HEAD") != revision:
        raise QualificationError("performance qualification source revision is not HEAD")
    if git_output(repo, "status", "--short", "--untracked-files=no"):
        raise QualificationError("performance qualification requires clean tracked source")


def parse_traces(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        family, separator, raw_path = value.partition("=")
        if separator != "=" or family not in FAMILIES or not raw_path:
            raise QualificationError("trace binding must be FAMILY=PATH")
        if family in result:
            raise QualificationError(f"duplicate trace binding for {family}")
        result[family] = Path(raw_path).absolute()
    if tuple(result) != FAMILIES:
        raise QualificationError("trace bindings must follow all seven families in order")
    return result


def load_trace(path: Path, family: str) -> dict[str, Any]:
    if (
        path.is_symlink() or not path.is_file() or path.stat().st_size <= 0
        or path.stat().st_size > MAX_TRACE_BYTES
    ):
        raise QualificationError(f"{family} trace must be a bounded ordinary file")
    try:
        trace = object_(json.loads(path.read_bytes()), TRACE_FIELDS, f"{family} trace")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise QualificationError(f"{family} trace must contain canonical JSON") from error
    if trace["schema"] != TRACE_SCHEMA or trace["family"] != family:
        raise QualificationError(f"{family} trace identity differs")
    return trace


def aggregate_trace(trace: dict[str, Any]) -> dict[str, Any]:
    warmups = trace["warmups_ms"]
    samples = trace["samples"]
    if not isinstance(warmups, list) or len(warmups) < 10:
        raise QualificationError("performance trace requires at least ten warmups")
    for value in warmups:
        number(value, "warmup milliseconds", 1e-12)
    if not isinstance(samples, list) or len(samples) < 30:
        raise QualificationError("performance trace requires at least thirty samples")
    elapsed = []
    resident = []
    scratch = []
    transfers = 0
    synchronizations = 0
    energies = []
    for ordinal, raw in enumerate(samples):
        sample = object_(raw, SAMPLE_FIELDS, f"samples[{ordinal}]")
        elapsed.append(number(sample["elapsed_ms"], "sample elapsed", 1e-12))
        if integer(sample["cases"], "sample cases", 117) != 117:
            raise QualificationError("performance sample is not the full 117-case corpus")
        resident.append(integer(sample["peak_resident_bytes"], "resident bytes", 1))
        scratch.append(integer(sample["peak_scratch_bytes"], "scratch bytes", 0))
        transfers += integer(sample["host_transfers"], "host transfers", 0)
        synchronizations += integer(
            sample["global_synchronizations"], "global synchronizations", 0
        )
        if sample["native_execution"] is not True or sample["budget_pass"] is not True:
            raise QualificationError("performance sample is non-native or over budget")
        energy = sample["energy_joules"]
        if energy is not None:
            energies.append(number(energy, "sample energy", 1e-12))
    median = statistics.median(elapsed)
    return {
        "warmup_iterations": len(warmups), "sample_count": len(samples),
        "cases_per_sample": 117, "median_ms": median,
        "p95_ms": percentile95(elapsed),
        "cases_per_second": 117000.0 / median,
        "peak_resident_bytes": max(resident), "peak_scratch_bytes": max(scratch),
        "host_transfers": transfers, "global_synchronizations": synchronizations,
        "native_execution": True, "budget_pass": True,
        "energy_joules": sum(energies) if len(energies) == len(samples) else None,
    }


def assemble(
    stage: Path, *, repo: Path, candidate: Path, backend_receipt_path: Path,
    trace_paths: dict[str, Path], source_revision: str, release: str, run_id: str,
) -> dict[str, Any]:
    if tuple(trace_paths) != FAMILIES:
        raise QualificationError("all seven ordered traces are required")
    backend = validate_backends(
        backend_receipt_path, source_revision, release, candidate, repo
    )
    backend_artifacts = {
        bundle["family"]: bundle["artifact"]["id"] for bundle in backend["bundles"]
    }
    artifacts = inventory(candidate)
    stage.mkdir()
    traces_dir = stage / "traces"
    traces_dir.mkdir()
    measurements = []
    cpu_median = None
    budget_id = None
    for ordinal, family in enumerate(FAMILIES):
        source_trace = trace_paths[family].resolve(strict=True)
        trace = load_trace(source_trace, family)
        if trace["workload_id"] != WORKLOAD_ID:
            raise QualificationError("performance workload differs from frozen corpus")
        if budget_id is None:
            budget_id = trace["budget_id"]
        if trace["budget_id"] != budget_id:
            raise QualificationError("performance budget differs across targets")
        if trace["artifact_id"] != backend_artifacts[family]:
            raise QualificationError("performance trace binds a different backend artifact")
        candidate_artifact = artifacts.get(trace["artifact_id"])
        if candidate_artifact is None:
            raise QualificationError("performance artifact is absent from candidate")
        identity, bundle_path = candidate_artifact
        try:
            bundle = json.loads(bundle_path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise QualificationError("backend bundle is not UTF-8 JSON") from error
        if bundle.get("physical_device") != trace["physical_device"]:
            raise QualificationError("performance device differs from backend bundle")
        aggregate = aggregate_trace(trace)
        if cpu_median is None:
            cpu_median = aggregate["median_ms"]
        destination = traces_dir / f"{ordinal:02d}-{family}.json"
        shutil.copyfile(source_trace, destination)
        measurements.append({
            "family": family,
            "tier": "throughput" if family in {"cpu", "cuda", "rocm", "metal", "wgpu"}
            else "bounded-latency",
            "artifact": {
                "id": identity[0], "kind": identity[1], "name": identity[2],
                "bytes": identity[3], "sha256": identity[4], "blake3": identity[5],
            },
            "physical_device": trace["physical_device"], **aggregate,
            "cpu_relative_speed": cpu_median / aggregate["median_ms"],
            "trace": {
                "path": destination.relative_to(stage).as_posix(),
                "bytes": destination.stat().st_size, "sha256": sha256(destination),
            },
        })
    receipt: dict[str, Any] = {
        "schema": SCHEMA, "result": "pass", "release": release,
        "source_revision": source_revision, "run_id": run_id,
        "candidate_manifest_sha256": sha256(candidate),
        "backend_manifest_receipt_id": backend["receipt_id"],
        "workload_id": WORKLOAD_ID, "budget_id": budget_id,
        "measurements": measurements,
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    receipt_path = stage / "receipt.json"
    receipt_path.write_bytes(canonical(receipt) + b"\n")
    validate_receipt(receipt_path, source_revision, release, candidate)
    return receipt


def qualify(
    output_dir: Path, *, repo: Path, candidate: Path, backend_receipt_path: Path,
    trace_paths: dict[str, Path], source_revision: str, release: str, run_id: str,
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
            backend_receipt_path=backend_receipt_path.resolve(strict=True),
            trace_paths=trace_paths, source_revision=source_revision,
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
    parser.add_argument("--backend-receipt", type=Path, required=True)
    parser.add_argument("--trace", action="append", default=[], metavar="FAMILY=PATH")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify(
        args.output_dir.absolute(), repo=args.repo,
        candidate=args.candidate.absolute(),
        backend_receipt_path=args.backend_receipt.absolute(),
        trace_paths=parse_traces(args.trace), source_revision=args.source_revision,
        release=args.release, run_id=args.run_id,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
