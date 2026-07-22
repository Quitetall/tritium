#!/usr/bin/env python3
"""Validate a complete pre-review registry and seal independent sign-off."""

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


REPRODUCTION = runpy.run_path(
    Path(__file__).with_name("verify-release-reproduction.py")
)
REVIEW_SCHEMA = REPRODUCTION["REVIEW_SCHEMA"]
REVIEW_ATTESTATION_SCHEMA = REPRODUCTION["REVIEW_ATTESTATION_SCHEMA"]
REVIEW_ATTESTATION_FIELDS = REPRODUCTION["REVIEW_ATTESTATION_FIELDS"]
REVIEWER_FIELDS = REPRODUCTION["REVIEWER_FIELDS"]
FINDING_FIELDS = REPRODUCTION["FINDING_FIELDS"]
REVIEW_SCOPES = REPRODUCTION["REVIEW_SCOPES"]
canonical = REPRODUCTION["canonical"]
sha256 = REPRODUCTION["sha256"]
candidate_artifacts = REPRODUCTION["candidate_artifacts"]
validate_receipt = REPRODUCTION["validate_independent_review"]
REVIEW_SIGNATURE_NAMESPACE = REPRODUCTION["REVIEW_SIGNATURE_NAMESPACE"]

STATUS = runpy.run_path(Path(__file__).with_name("release-evidence-status.py"))
REGISTRY_SCHEMA = STATUS["SCHEMA"]
TOP_FIELDS = STATUS["TOP_FIELDS"]
RECEIPT_FIELDS = STATUS["RECEIPT_FIELDS"]
KNOWN_KINDS = STATUS["KNOWN_KINDS"]
evaluate = STATUS["evaluate"]
review_scope_sha256 = STATUS["review_scope_sha256"]

MAX_INPUT_BYTES = 32 * 1024 * 1024
HEX = frozenset("0123456789abcdef")


class QualificationError(ValueError):
    """Independent review cannot sign off an incomplete or mismatched scope."""


def object_(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise QualificationError(f"{label} fields do not match frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationError(f"{label} must be non-empty")
    return value


def ordinary(path: Path, label: str) -> Path:
    if (
        path.is_symlink() or not path.is_file() or path.stat().st_size <= 0
        or path.stat().st_size > MAX_INPUT_BYTES
    ):
        raise QualificationError(f"{label} must be a bounded ordinary file")
    return path.resolve(strict=True)


def load(path: Path, fields: set[str], label: str) -> dict[str, Any]:
    path = ordinary(path, label)
    try:
        return object_(json.loads(path.read_bytes()), fields, label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"{label} must contain UTF-8 JSON") from error


def anchor_record(candidate: Path, anchor: Path) -> dict[str, Any]:
    anchor = ordinary(anchor, "anchor wheel")
    matches = [
        item for item in candidate_artifacts(candidate)
        if item[1] == "python-wheel" and item[2:] == (
            anchor.name, anchor.stat().st_size, sha256(anchor)
        )
    ]
    if len(matches) != 1:
        raise QualificationError("candidate must bind exactly one matching anchor wheel")
    item = matches[0]
    return {
        "id": item[0], "kind": item[1], "name": item[2],
        "bytes": item[3], "sha256": item[4],
    }


def validate_registry(
    registry_path: Path, candidate: Path, *, revision: str, release: str,
    digest_tool: str,
) -> tuple[dict[str, Any], list[str], str]:
    registry = load(registry_path, TOP_FIELDS, "pre-review registry")
    if (
        registry["schema"] != REGISTRY_SCHEMA
        or registry["release"] != release
        or registry["source_revision"] != revision
        or registry["candidate_manifest_sha256"] != sha256(candidate)
    ):
        raise QualificationError("pre-review registry identity differs from candidate")
    try:
        candidate_document = json.loads(candidate.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError("candidate manifest must contain UTF-8 JSON") from error
    entries = registry["receipts"]
    if not isinstance(entries, list):
        raise QualificationError("pre-review registry receipts must be an array")
    ids = []
    kinds = []
    for ordinal, raw in enumerate(entries):
        entry = object_(raw, RECEIPT_FIELDS, f"registry.receipts[{ordinal}]")
        ids.append(string(entry["id"], "registry receipt id"))
        kinds.append(string(entry["kind"], "registry receipt kind"))
    expected_kinds = KNOWN_KINDS - {"independent-review"}
    if (
        len(ids) != len(set(ids)) or len(kinds) != len(set(kinds))
        or set(kinds) != expected_kinds
    ):
        raise QualificationError("pre-review registry must contain every non-review kind once")
    report = evaluate(registry_path, candidate, candidate_document, digest_tool)
    for row in report["rows"]:
        if row["id"] == "reproduction-signoff":
            if (
                row["satisfied_kinds"] != ["second-machine"]
                or row["missing_kinds"] != ["independent-review"]
                or row["structural_kinds"]
            ):
                raise QualificationError("pre-review reproduction gate differs from policy")
        elif row["status"] != "PASS":
            raise QualificationError(f"pre-review gate {row['id']} is not PASS")
    return registry, ids, review_scope_sha256(registry)


def validate_attestation(
    attestation_path: Path, *, revision: str, release: str, candidate: Path,
    reviewed_ids: list[str], scope_sha256: str,
) -> dict[str, Any]:
    attestation = load(
        attestation_path, REVIEW_ATTESTATION_FIELDS, "review attestation"
    )
    if (
        attestation["schema"] != REVIEW_ATTESTATION_SCHEMA
        or attestation["release"] != release
        or attestation["source_revision"] != revision
        or attestation["candidate_manifest_sha256"] != sha256(candidate)
        or attestation["review_scope_sha256"] != scope_sha256
        or attestation["reviewed_receipt_ids"] != reviewed_ids
        or attestation["scopes"] != REVIEW_SCOPES
        or attestation["verdict"] != "pass"
    ):
        raise QualificationError("review attestation identity, scope, or verdict differs")
    string(attestation["run_id"], "review run_id")
    reviewer = object_(attestation["reviewer"], REVIEWER_FIELDS, "reviewer")
    if reviewer["independent"] is not True:
        raise QualificationError("release reviewer must be independent")
    for field in REVIEWER_FIELDS - {"independent"}:
        string(reviewer[field], f"reviewer.{field}")
    findings = object_(attestation["findings"], FINDING_FIELDS, "review findings")
    if any(type(findings[field]) is not int or findings[field] < 0 for field in FINDING_FIELDS):
        raise QualificationError("review finding counts must be non-negative integers")
    if (
        findings["total"] != findings["verified"] + findings["false_positive"]
        or findings["fixed"] != findings["verified"] or findings["open"] != 0
    ):
        raise QualificationError("review attestation contains unresolved findings")
    return attestation


def verify_signature(
    attestation_path: Path, signature_path: Path, policy_path: Path,
    principal: str,
) -> None:
    attestation_path = ordinary(attestation_path, "review attestation")
    signature_path = ordinary(signature_path, "review signature")
    policy_path = ordinary(policy_path, "reviewer signer policy")
    string(principal, "review signer principal")
    try:
        result = subprocess.run(
            [
                "ssh-keygen", "-Y", "verify", "-f", str(policy_path),
                "-I", principal, "-n", REVIEW_SIGNATURE_NAMESPACE,
                "-s", str(signature_path),
            ],
            input=attestation_path.read_bytes(), capture_output=True,
            check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise QualificationError("independent review signature verifier failed") from error
    if result.returncode != 0:
        raise QualificationError("independent review signature is not trusted")


def assemble(
    stage: Path, *, candidate: Path, anchor: Path, registry_path: Path,
    attestation_path: Path, signature_path: Path, policy_path: Path,
    signer_principal: str, source_revision: str, release: str, digest_tool: str,
) -> dict[str, Any]:
    candidate = ordinary(candidate, "candidate manifest")
    anchor = ordinary(anchor, "anchor wheel")
    registry_path = ordinary(registry_path, "pre-review registry")
    if len(source_revision) != 40 or any(character not in HEX for character in source_revision):
        raise QualificationError("source revision must be 40 lowercase hexadecimal")
    _, reviewed_ids, scope_sha256 = validate_registry(
        registry_path, candidate, revision=source_revision, release=release,
        digest_tool=digest_tool,
    )
    attestation = validate_attestation(
        attestation_path, revision=source_revision, release=release,
        candidate=candidate, reviewed_ids=reviewed_ids, scope_sha256=scope_sha256,
    )
    verify_signature(
        attestation_path, signature_path, policy_path, signer_principal
    )
    stage.mkdir()
    support = stage / "support"
    support.mkdir()
    retained = support / "review-attestation.json"
    shutil.copyfile(ordinary(attestation_path, "review attestation"), retained)
    retained_signature = support / "review-attestation.json.sig"
    retained_policy = support / "trusted-reviewers.allowed_signers"
    shutil.copyfile(ordinary(signature_path, "review signature"), retained_signature)
    shutil.copyfile(ordinary(policy_path, "reviewer signer policy"), retained_policy)
    receipt: dict[str, Any] = {
        **{field: attestation[field] for field in REVIEW_ATTESTATION_FIELDS - {"schema"}},
        "schema": REVIEW_SCHEMA, "result": "pass",
        "anchor_artifact": anchor_record(candidate, anchor),
        "review_attestation": {
            "path": retained.relative_to(stage).as_posix(),
            "bytes": retained.stat().st_size, "sha256": sha256(retained),
        },
        "review_signature": {
            "path": retained_signature.relative_to(stage).as_posix(),
            "bytes": retained_signature.stat().st_size,
            "sha256": sha256(retained_signature),
        },
        "signer_policy": {
            "path": retained_policy.relative_to(stage).as_posix(),
            "bytes": retained_policy.stat().st_size,
            "sha256": sha256(retained_policy),
        },
        "signer_principal": signer_principal,
        "signature_namespace": REVIEW_SIGNATURE_NAMESPACE,
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    receipt_path = stage / "receipt.json"
    receipt_path.write_bytes(canonical(receipt) + b"\n")
    validate_receipt(receipt_path, source_revision, release, candidate, anchor)
    return receipt


def qualify(
    output_dir: Path, *, candidate: Path, anchor: Path, registry_path: Path,
    attestation_path: Path, signature_path: Path, policy_path: Path,
    signer_principal: str, source_revision: str, release: str, digest_tool: str,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise QualificationError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    stage.rmdir()
    try:
        receipt = assemble(
            stage, candidate=candidate, anchor=anchor, registry_path=registry_path,
            attestation_path=attestation_path, signature_path=signature_path,
            policy_path=policy_path, signer_principal=signer_principal,
            source_revision=source_revision,
            release=release, digest_tool=digest_tool,
        )
        os.replace(stage, output_dir)
        return receipt
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--anchor-wheel", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument(
        "--signer-policy", type=Path,
        default=Path("release/trusted-reviewers-v1.1.allowed_signers"),
    )
    parser.add_argument("--signer-principal", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--digest-tool", default=os.environ.get("TRITIUM_BIN", "tritium"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify(
        args.output_dir.absolute(), candidate=args.candidate.absolute(),
        anchor=args.anchor_wheel.absolute(), registry_path=args.registry.absolute(),
        attestation_path=args.attestation.absolute(),
        signature_path=args.signature.absolute(),
        policy_path=args.signer_policy.absolute(),
        signer_principal=args.signer_principal,
        source_revision=args.source_revision, release=args.release,
        digest_tool=args.digest_tool,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
