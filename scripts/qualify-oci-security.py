#!/usr/bin/env python3
"""Qualify one exact OCI archive with offline vulnerability and secret scans."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "tritium.oci-security-qualification.v1"
HEX = frozenset("0123456789abcdef")
MAX_REPORT_BYTES = 128 * 1024 * 1024
MAX_LAYOUT_MEMBER_BYTES = 512 * 1024 * 1024
RC_PATTERN = re.compile(r"1\.1\.0-rc\.(0|[1-9][0-9]*)")
RUN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
VERSION_PATTERN = re.compile(r"v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?")
MIN_TRIVY_VERSION = (0, 69, 0)


class SecurityScanError(ValueError):
    """OCI security evidence is absent, stale, malformed, or non-green."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exact_hex(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(c not in HEX for c in value):
        raise SecurityScanError(f"{label} must be {length} lowercase hexadecimal characters")
    return value


def ordinary(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise SecurityScanError(f"{label} must be an ordinary file")
    return path.resolve(strict=True)


def parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise SecurityScanError(f"{label} must be UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SecurityScanError(f"{label} must be UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise SecurityScanError(f"{label} must be UTC timestamp")
    return parsed


def trivy_version(value: Any) -> str:
    if not isinstance(value, str):
        raise SecurityScanError("Trivy version is malformed")
    match = VERSION_PATTERN.fullmatch(value)
    if match is None or tuple(map(int, match.groups())) < MIN_TRIVY_VERSION:
        raise SecurityScanError("Trivy 0.69.0 or newer is required")
    return value


def run(command: list[str], *, timeout: float, output: Path | None = None) -> str:
    try:
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE if output is None else subprocess.DEVNULL,
            stderr=subprocess.PIPE, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SecurityScanError(f"scanner command failed: {error}") from error
    if result.returncode != 0:
        raise SecurityScanError(
            f"scanner command failed ({result.returncode}): {result.stderr.strip()[-2000:]}"
        )
    if output is not None:
        ordinary(output, "scanner report")
        if output.stat().st_size > MAX_REPORT_BYTES:
            raise SecurityScanError("scanner report exceeds byte limit")
        return output.read_text(encoding="utf-8")
    if len(result.stdout.encode()) > 1024 * 1024:
        raise SecurityScanError("scanner metadata exceeds byte limit")
    return result.stdout


def report_findings(report: Any, scanner: str) -> int:
    if not isinstance(report, dict) or report.get("SchemaVersion") != 2:
        raise SecurityScanError(f"{scanner} report schema differs")
    if report.get("ArtifactType") != "container_image":
        raise SecurityScanError(f"{scanner} report is not a container image scan")
    results = report.get("Results")
    if not isinstance(results, list):
        raise SecurityScanError(f"{scanner} report results are malformed")
    key = "Vulnerabilities" if scanner == "vulnerability" else "Secrets"
    count = 0
    for result in results:
        if not isinstance(result, dict):
            raise SecurityScanError(f"{scanner} report result is malformed")
        findings = result.get(key, [])
        if findings is None:
            findings = []
        if not isinstance(findings, list) or any(not isinstance(item, dict) for item in findings):
            raise SecurityScanError(f"{scanner} findings are malformed")
        count += len(findings)
    return count


def stage_oci_layout(archive: Path, destination: Path) -> None:
    """Extract a tarred OCI layout into a scanner-readable directory safely."""
    destination.mkdir()
    seen: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:*") as stream:
            for member in stream.getmembers():
                name = member.name
                logical = PurePosixPath(name)
                if (
                    not name
                    or logical.is_absolute()
                    or ".." in logical.parts
                    or "\\" in name
                    or name in seen
                ):
                    raise SecurityScanError("OCI archive contains an unsafe or duplicate member")
                seen.add(name)
                target = destination.joinpath(*logical.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isreg() or member.size < 0 or member.size > MAX_LAYOUT_MEMBER_BYTES:
                    raise SecurityScanError("OCI archive contains a non-regular or oversized member")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise SecurityScanError("OCI archive member has no readable payload")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    except (OSError, tarfile.TarError) as error:
        raise SecurityScanError(f"OCI archive extraction failed: {error}") from error
    if not (destination / "index.json").is_file() or not (destination / "oci-layout").is_file():
        raise SecurityScanError("OCI archive does not contain a complete layout")


def atomic_create(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise SecurityScanError("refusing to overwrite output")
    parent = path.parent.resolve(strict=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    if args.flavor not in {"cpu", "cuda"}:
        raise SecurityScanError("flavor must be cpu or cuda")
    if RC_PATTERN.fullmatch(args.release) is None or RUN_PATTERN.fullmatch(args.run_id) is None:
        raise SecurityScanError("release or run ID is malformed")
    revision = exact_hex(args.source_revision, 40, "source revision")
    if args.max_db_age_hours <= 0 or args.max_db_age_hours > 24 or args.timeout <= 0:
        raise SecurityScanError("scan limits must be positive")
    archive = ordinary(args.archive, "OCI archive")
    cache = args.cache_dir.resolve(strict=True)
    if args.cache_dir.is_symlink() or not cache.is_dir():
        raise SecurityScanError("Trivy cache must be an ordinary directory")
    executable_raw = shutil.which(args.trivy)
    if executable_raw is None:
        raise SecurityScanError("Trivy executable is unavailable")
    executable = Path(executable_raw).resolve(strict=True)
    if not executable.is_file():
        raise SecurityScanError("Trivy executable must resolve to a file")
    if (cache / "db").is_symlink():
        raise SecurityScanError("Trivy DB directory must not be a symlink")
    db = ordinary(cache / "db" / "trivy.db", "Trivy vulnerability database")
    db_metadata = ordinary(cache / "db" / "metadata.json", "Trivy DB metadata")
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    base = [str(executable), "--cache-dir", str(cache)]
    version = json.loads(run(base + ["version", "--format", "json"], timeout=60))
    if not isinstance(version, dict):
        raise SecurityScanError("Trivy version metadata is malformed")
    version_number = trivy_version(version.get("Version"))
    database = version.get("VulnerabilityDB")
    if not isinstance(database, dict):
        raise SecurityScanError("Trivy vulnerability DB metadata is absent")
    updated = parse_utc(database.get("UpdatedAt"), "Trivy DB UpdatedAt")
    downloaded = parse_utc(database.get("DownloadedAt"), "Trivy DB DownloadedAt")
    next_update = parse_utc(database.get("NextUpdate"), "Trivy DB NextUpdate")
    if (updated > started_at or downloaded < updated or downloaded > started_at
            or next_update < started_at
            or started_at - updated > timedelta(hours=args.max_db_age_hours)):
        raise SecurityScanError("Trivy vulnerability database is stale or future-dated")
    with tempfile.TemporaryDirectory(prefix="tritium-oci-scan-") as raw:
        temporary = Path(raw)
        layout = temporary / "layout"
        stage_oci_layout(archive, layout)
        vuln_output = temporary / "vulnerability.json"
        secret_output = temporary / "secret.json"
        common = [
            "image", "--input", str(layout), "--format", "json", "--offline-scan",
            "--skip-db-update", "--skip-java-db-update", "--skip-check-update",
        ]
        vuln_command = base + common + [
            "--scanners", "vuln", "--severity", "HIGH,CRITICAL", "--output",
            str(vuln_output),
        ]
        secret_command = base + common + [
            "--scanners", "secret", "--output", str(secret_output),
        ]
        vulnerability_report = json.loads(
            run(vuln_command, timeout=args.timeout, output=vuln_output)
        )
        secret_report = json.loads(run(secret_command, timeout=args.timeout, output=secret_output))
    vulnerabilities = report_findings(vulnerability_report, "vulnerability")
    leaked_secrets = report_findings(secret_report, "secret")
    if vulnerabilities or leaked_secrets:
        raise SecurityScanError(
            f"image has {vulnerabilities} HIGH/CRITICAL vulnerabilities and "
            f"{leaked_secrets} secret findings"
        )
    receipt = {
        "schema": SCHEMA, "release": args.release, "source_revision": revision,
        "run_id": args.run_id, "flavor": args.flavor,
        "started_at_utc": started_at.isoformat(timespec="seconds"),
        "duration_ms": (time.monotonic() - started) * 1000,
        "artifact": {"kind": "oci-image", "name": archive.name,
                     "bytes": archive.stat().st_size, "sha256": sha256(archive)},
        "scanner": {"name": "trivy", "version": version_number,
                    "executable_sha256": sha256(executable),
                    "commands": [
                        [*vuln_command[:5], "<layout>", *vuln_command[6:]],
                        [*secret_command[:5], "<layout>", *secret_command[6:]],
                    ]},
        "database": {"updated_at": database["UpdatedAt"],
                     "downloaded_at": database["DownloadedAt"],
                     "next_update": database["NextUpdate"],
                     "trivy_db_sha256": sha256(db),
                     "metadata_sha256": sha256(db_metadata),
                     "max_age_hours": args.max_db_age_hours},
        "findings": {"high_or_critical_vulnerabilities": vulnerabilities,
                     "secret_findings": leaked_secrets},
        "result": "pass",
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    validate_receipt(receipt, artifact_path=archive, revision=revision, release=args.release)
    return receipt


def validate_receipt(receipt: dict[str, Any], *, artifact_path: Path,
                     revision: str, release: str) -> dict[str, Any]:
    fields = {
        "schema", "receipt_id", "release", "source_revision", "run_id", "flavor",
        "started_at_utc", "duration_ms", "artifact", "scanner", "database", "findings",
        "result",
    }
    if not isinstance(receipt, dict) or set(receipt) != fields:
        raise SecurityScanError("security receipt fields differ")
    if receipt.get("schema") != SCHEMA or receipt.get("result") != "pass":
        raise SecurityScanError("security receipt schema or result differs")
    if receipt.get("release") != release or receipt.get("source_revision") != revision:
        raise SecurityScanError("security receipt release identity differs")
    if receipt.get("flavor") not in {"cpu", "cuda"} or not isinstance(
        receipt.get("run_id"), str
    ) or RUN_PATTERN.fullmatch(receipt["run_id"]) is None:
        raise SecurityScanError("security receipt run identity is malformed")
    started_at = parse_utc(receipt.get("started_at_utc"), "security receipt timestamp")
    duration = receipt.get("duration_ms")
    if type(duration) not in {int, float} or not math.isfinite(duration) or duration < 0:
        raise SecurityScanError("security receipt duration is malformed")
    artifact = receipt.get("artifact")
    if not isinstance(artifact, dict) or set(artifact) != {"kind", "name", "bytes", "sha256"}:
        raise SecurityScanError("security receipt artifact fields differ")
    artifact_path = ordinary(artifact_path, "candidate OCI image")
    actual = (artifact_path.name, artifact_path.stat().st_size, sha256(artifact_path))
    declared = (artifact.get("name"), artifact.get("bytes"), artifact.get("sha256"))
    if artifact.get("kind") != "oci-image" or actual != declared:
        raise SecurityScanError("security receipt does not bind candidate OCI bytes")
    scanner = receipt.get("scanner")
    if not isinstance(scanner, dict) or set(scanner) != {
        "name", "version", "executable_sha256", "commands"
    } or scanner.get("name") != "trivy":
        raise SecurityScanError("security receipt scanner identity differs")
    trivy_version(scanner.get("version"))
    exact_hex(scanner.get("executable_sha256"), 64, "scanner executable SHA-256")
    commands = scanner.get("commands")
    if not isinstance(commands, list) or len(commands) != 2 or any(
        not isinstance(command, list) for command in commands
    ):
        raise SecurityScanError("security receipt command contract differs")
    common = [
        "<executable>", "--cache-dir", "<cache>", "image", "--input",
        "<layout>", "--format", "json", "--offline-scan", "--skip-db-update",
        "--skip-java-db-update", "--skip-check-update",
    ]
    normalized = []
    for command in commands:
        if len(command) < 4 or not all(isinstance(item, str) and item for item in command):
            raise SecurityScanError("security receipt command contract differs")
        item = list(command)
        item[0] = "<executable>"
        item[2] = "<cache>"
        if item[5] != "<layout>":
            raise SecurityScanError("security receipt command layout placeholder differs")
        item[-1] = "<output>"
        normalized.append(item)
    expected = [
        common + ["--scanners", "vuln", "--severity", "HIGH,CRITICAL", "--output", "<output>"],
        common + ["--scanners", "secret", "--output", "<output>"],
    ]
    if normalized != expected:
        raise SecurityScanError("security receipt command contract differs")
    database = receipt.get("database")
    if not isinstance(database, dict) or set(database) != {
        "updated_at", "downloaded_at", "next_update", "trivy_db_sha256",
        "metadata_sha256", "max_age_hours",
    }:
        raise SecurityScanError("security receipt database fields differ")
    updated_at = parse_utc(database.get("updated_at"), "security receipt database updated_at")
    downloaded_at = parse_utc(
        database.get("downloaded_at"), "security receipt database downloaded_at"
    )
    exact_hex(database.get("trivy_db_sha256"), 64, "Trivy DB SHA-256")
    exact_hex(database.get("metadata_sha256"), 64, "Trivy metadata SHA-256")
    if (type(database.get("max_age_hours")) not in {int, float}
            or database["max_age_hours"] <= 0 or database["max_age_hours"] > 24):
        raise SecurityScanError("security receipt database age policy is malformed")
    next_update = parse_utc(database.get("next_update"), "security receipt database next_update")
    if (updated_at > started_at or downloaded_at < updated_at or downloaded_at > started_at
            or next_update < started_at
            or started_at - updated_at > timedelta(hours=database["max_age_hours"])):
        raise SecurityScanError("security receipt database is stale or future-dated")
    findings = receipt.get("findings")
    if findings != {"high_or_critical_vulnerabilities": 0, "secret_findings": 0}:
        raise SecurityScanError("security receipt contains blocking findings")
    supplied = receipt.get("receipt_id")
    if not isinstance(supplied, str) or not supplied.startswith("sha256:"):
        raise SecurityScanError("security receipt ID is malformed")
    exact_hex(supplied[7:], 64, "security receipt ID")
    unsigned = dict(receipt)
    del unsigned["receipt_id"]
    expected = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if not secrets.compare_digest(supplied, expected):
        raise SecurityScanError("security receipt content digest differs")
    return receipt


def load_receipt(path: Path, *, artifact_path: Path,
                 revision: str, release: str) -> dict[str, Any]:
    path = ordinary(path, "security receipt")
    if path.stat().st_size > 32 * 1024 * 1024:
        raise SecurityScanError("security receipt exceeds byte limit")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SecurityScanError("security receipt must contain UTF-8 JSON") from error
    return validate_receipt(value, artifact_path=artifact_path, revision=revision, release=release)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--flavor", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trivy", default="trivy")
    parser.add_argument("--max-db-age-hours", type=float, default=24)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    try:
        receipt = qualify(args)
        atomic_create(args.output, canonical(receipt))
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as error:
        print(f"qualify-oci-security: BLOCKED: {error}", file=sys.stderr)
        return 1
    print(f"qualify-oci-security: PASS: {receipt['receipt_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
