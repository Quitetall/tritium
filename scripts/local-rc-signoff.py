#!/usr/bin/env python3
"""Seal or verify the detached, evidence-bound Tritium local-RC sign-off."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SCHEMA = "tritium.local-rc-signoff.v1"
REPORT_SCHEMA = "tritium.release-gate-report.v1"
FIELDS = {
    "schema", "release", "source_revision", "candidate_manifest_sha256",
    "evidence_registry_sha256", "evidence_report_sha256", "signer_principal",
}
HEX = frozenset("0123456789abcdef")


class SignoffError(ValueError):
    """The sign-off is incomplete, stale, or cryptographically invalid."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ordinary(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise SignoffError(f"{label} must be an ordinary file")
    return path.resolve(strict=True)


def json_file(path: Path, label: str) -> dict[str, Any]:
    path = ordinary(path, label)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise SignoffError(f"{label} exceeds the metadata size limit")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SignoffError(f"{label} must contain UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise SignoffError(f"{label} must contain a JSON object")
    return value


def statement(report_path: Path, registry: Path, candidate: Path,
              principal: str) -> dict[str, str]:
    report = json_file(report_path, "evidence report")
    if report.get("schema") != REPORT_SCHEMA or report.get("ready") is not True:
        raise SignoffError("evidence report is not a complete passing report")
    if not principal or any(character.isspace() for character in principal):
        raise SignoffError("signer principal must be one non-empty token")
    release = report.get("release")
    if not isinstance(release, str) or not release:
        raise SignoffError("evidence report has an invalid release")
    registry_digest = sha256(ordinary(registry, "evidence registry"))
    if report.get("evidence_registry_sha256") != registry_digest:
        raise SignoffError("evidence report does not bind the exact registry")
    actual_candidate_digest = sha256(ordinary(candidate, "candidate manifest"))
    candidate_digest = report.get("candidate_manifest_sha256")
    revision = report.get("source_revision")
    if not isinstance(candidate_digest, str) or len(candidate_digest) != 64:
        raise SignoffError("evidence report has an invalid candidate digest")
    if not isinstance(revision, str) or len(revision) != 40:
        raise SignoffError("evidence report has an invalid source revision")
    if any(character not in HEX for character in candidate_digest + revision):
        raise SignoffError("evidence report identities must be lowercase hexadecimal")
    if candidate_digest != actual_candidate_digest:
        raise SignoffError("evidence report does not bind the exact candidate manifest")
    return {
        "schema": SCHEMA,
        "release": release,
        "source_revision": revision,
        "candidate_manifest_sha256": candidate_digest,
        "evidence_registry_sha256": registry_digest,
        "evidence_report_sha256": sha256(ordinary(report_path, "evidence report")),
        "signer_principal": principal,
    }


def atomic_create(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise SignoffError(f"refusing to overwrite {path}")
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


def seal(report: Path, registry: Path, candidate: Path, principal: str,
         key: Path, output: Path) -> None:
    value = statement(report, registry, candidate, principal)
    signature = Path(str(output) + ".sig")
    if signature.exists() or signature.is_symlink():
        raise SignoffError(f"refusing to overwrite {signature}")
    atomic_create(output, canonical(value))
    try:
        subprocess.run(
            ["ssh-keygen", "-Y", "sign", "-f", str(ordinary(key, "signing key")),
             "-n", "tritium-release", str(output)],
            check=True, timeout=30, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if not signature.is_file() or signature.is_symlink():
            raise SignoffError("ssh-keygen did not create an ordinary signature file")
    except (OSError, subprocess.SubprocessError, SignoffError) as error:
        output.unlink(missing_ok=True)
        signature.unlink(missing_ok=True)
        raise SignoffError(f"signing failed: {error}") from error


def verify(statement_path: Path, signature: Path, report: Path, registry: Path,
           candidate: Path, principal: str, allowed_signers: Path) -> dict[str, str]:
    actual = json_file(statement_path, "sign-off statement")
    if set(actual) != FIELDS or actual.get("schema") != SCHEMA:
        raise SignoffError("sign-off statement fields do not match the frozen schema")
    expected = statement(report, registry, candidate, principal)
    if actual != expected:
        raise SignoffError("sign-off statement does not match current evidence")
    try:
        result = subprocess.run(
            ["ssh-keygen", "-Y", "verify", "-f", str(ordinary(allowed_signers, "allowed signers")),
             "-I", principal, "-n", "tritium-release", "-s", str(ordinary(signature, "signature"))],
            input=canonical(actual), check=False, timeout=30,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SignoffError(f"signature verification failed: {error}") from error
    if result.returncode != 0:
        raise SignoffError("signature is not valid for the allowed release signer")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal_parser = subparsers.add_parser("seal")
    verify_parser = subparsers.add_parser("verify")
    for item in (seal_parser, verify_parser):
        item.add_argument("--report", type=Path, required=True)
        item.add_argument("--registry", type=Path, required=True)
        item.add_argument("--candidate", type=Path, required=True)
        item.add_argument("--principal", required=True)
    seal_parser.add_argument("--key", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    verify_parser.add_argument("--statement", type=Path, required=True)
    verify_parser.add_argument("--signature", type=Path, required=True)
    verify_parser.add_argument("--allowed-signers", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "seal":
            seal(args.report, args.registry, args.candidate, args.principal,
                 args.key, args.output)
            print(f"local-rc-signoff: SEALED: {args.output}")
        else:
            value = verify(args.statement, args.signature, args.report, args.registry,
                           args.candidate, args.principal, args.allowed_signers)
            print(f"local-rc-signoff: LOCAL_RC_READY: {value['release']}")
    except (OSError, SignoffError) as error:
        print(f"local-rc-signoff: BLOCKED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
