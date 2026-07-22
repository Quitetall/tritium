#!/usr/bin/env python3
"""Seal model-zoo, documentation-claim, and governance release evidence."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import shutil
import subprocess
import tempfile
from typing import Any


VERIFIER = runpy.run_path(
    Path(__file__).with_name("verify-zoo-community-receipt.py")
)
ZOO_SCHEMA = VERIFIER["ZOO_SCHEMA"]
CLAIMS_SCHEMA = VERIFIER["CLAIMS_SCHEMA"]
GOVERNANCE_SCHEMA = VERIFIER["GOVERNANCE_SCHEMA"]
CLAIM_SOURCE_SCHEMA = VERIFIER["CLAIM_SOURCE_SCHEMA"]
EXPECTED_MODELS = VERIFIER["EXPECTED_MODELS"]
CLAIM_DOCUMENTS = VERIFIER["CLAIM_DOCUMENTS"]
GOVERNANCE_FILES = VERIFIER["GOVERNANCE_FILES"]
canonical = VERIFIER["canonical"]
sha256 = VERIFIER["sha256"]
validate_zoo = VERIFIER["validate_zoo"]
validate_claims = VERIFIER["validate_claims"]
validate_governance = VERIFIER["validate_governance"]
CLAIM_GENERATOR = runpy.run_path(
    Path(__file__).with_name("generate-release-claims.py")
)
check_claim_documents = CLAIM_GENERATOR["check"]

SOURCE_SCHEMA = "tritium.model-zoo-source.v1"
REVIEW_SCHEMA = "tritium.governance-review-attestation.v1"
REGISTRY_SCHEMA = "tritium.release-evidence-registry.v1"
GENERATOR_ID = VERIFIER["GENERATOR_ID"]
HEX = frozenset("0123456789abcdef")
MAX_INPUT_BYTES = 32 * 1024 * 1024
SOURCE_FIELDS = {"schema", "release", "source_revision", "models"}
SOURCE_MODEL_FIELDS = {
    "tier", "role", "model_id", "revision", "tokenizer_sha256", "license",
    "card", "artifact_ids", "evidence_receipt_ids",
}
REVIEW_FIELDS = {
    "schema", "release", "source_revision", "reviewed_at_utc", "reviewer",
    "reviewed_files", "repository_links_checked", "contacts_checked",
    "independent_from_maintainers", "unstaffed_channels_advertised", "result",
}
REVIEWER_FIELDS = {"id", "organization"}
REGISTRY_FIELDS = {
    "schema", "release", "source_revision", "candidate_manifest_sha256",
    "receipts",
}
REGISTRY_ENTRY_FIELDS = {"id", "kind", "path", "sha256", "artifact_id", "parents"}


class QualificationError(ValueError):
    """Zoo/community evidence cannot be sealed without exact release inputs."""


def object_(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise QualificationError(f"{label} fields do not match frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationError(f"{label} must be non-empty")
    return value


def hex_(value: Any, length: int, label: str) -> str:
    text = string(value, label)
    if len(text) != length or any(character not in HEX for character in text):
        raise QualificationError(
            f"{label} must be {length} lowercase hexadecimal characters"
        )
    return text


def digest(value: Any, label: str) -> str:
    text = string(value, label)
    if not text.startswith("sha256:"):
        raise QualificationError(f"{label} must be a canonical digest")
    hex_(text.removeprefix("sha256:"), 64, label)
    return text


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


def contained(root: Path, logical_value: Any, label: str) -> Path:
    text = string(logical_value, label)
    logical = PurePosixPath(text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in text:
        raise QualificationError(f"{label} path is unsafe")
    cursor = root.resolve(strict=True)
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise QualificationError(f"{label} traverses a symlink")
    return ordinary(cursor, label)


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise QualificationError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def require_clean_revision(repo: Path, revision: str) -> None:
    if git_output(repo, "rev-parse", "HEAD") != revision:
        raise QualificationError("zoo/community source revision is not HEAD")
    if git_output(repo, "status", "--short", "--untracked-files=no"):
        raise QualificationError("zoo/community qualification requires clean tracked source")


def file_record(path: Path, logical_path: str) -> dict[str, Any]:
    return {
        "path": logical_path,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def seal(value: dict[str, Any]) -> dict[str, Any]:
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical(value) + b"\n")


def candidate_artifacts(candidate: Path) -> dict[str, dict[str, Any]]:
    try:
        document = json.loads(candidate.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError("candidate manifest must contain UTF-8 JSON") from error
    artifacts = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(artifacts, list):
        raise QualificationError("candidate manifest artifacts must be an array")
    indexed: dict[str, dict[str, Any]] = {}
    for ordinal, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise QualificationError(f"candidate artifact {ordinal} must be an object")
        artifact_id = string(artifact.get("id"), f"candidate artifact {ordinal}.id")
        if artifact_id in indexed:
            raise QualificationError("candidate artifact IDs must be unique")
        indexed[artifact_id] = artifact
    return indexed


def anchor_record(
    candidate: Path, anchor: Path, artifacts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    anchor = ordinary(anchor, "anchor wheel")
    matches = []
    for artifact_id, artifact in artifacts.items():
        identity = artifact.get("identity")
        if (
            artifact.get("kind") == "python-wheel"
            and isinstance(identity, dict)
            and identity.get("bytes") == anchor.stat().st_size
            and identity.get("sha256") == sha256(anchor)
        ):
            matches.append(artifact_id)
    if len(matches) != 1:
        raise QualificationError("candidate must bind exactly one matching anchor wheel")
    return {
        "id": matches[0], "kind": "python-wheel", "name": anchor.name,
        "bytes": anchor.stat().st_size, "sha256": sha256(anchor),
    }


def registry_sources(
    registry_path: Path, candidate: Path, revision: str, release: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    registry = load(registry_path, REGISTRY_FIELDS, "source registry")
    if (
        registry["schema"] != REGISTRY_SCHEMA
        or registry["release"] != release
        or registry["source_revision"] != revision
        or registry["candidate_manifest_sha256"] != sha256(candidate)
    ):
        raise QualificationError("source registry identity differs from candidate")
    raw_entries = registry["receipts"]
    if not isinstance(raw_entries, list):
        raise QualificationError("source registry receipts must be an array")
    entries: dict[str, dict[str, Any]] = {}
    root = registry_path.parent.resolve(strict=True)
    for ordinal, raw in enumerate(raw_entries):
        entry = object_(raw, REGISTRY_ENTRY_FIELDS, f"registry.receipts[{ordinal}]")
        receipt_id = digest(entry["id"], f"registry.receipts[{ordinal}].id")
        if receipt_id in entries:
            raise QualificationError("source registry receipt IDs must be unique")
        receipt_path = contained(root, entry["path"], f"registry.receipts[{ordinal}].path")
        if sha256(receipt_path) != hex_(entry["sha256"], 64, "registry receipt sha256"):
            raise QualificationError("source registry receipt bytes drifted")
        try:
            receipt = json.loads(receipt_path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise QualificationError("source registry receipt must contain UTF-8 JSON") from error
        if (
            not isinstance(receipt, dict)
            or receipt.get("receipt_id") != receipt_id
            or receipt.get("result") != "pass"
            or receipt.get("release") != release
            or receipt.get("source_revision") != revision
        ):
            raise QualificationError("source registry receipt identity differs")
        entries[receipt_id] = entry
    return entries, registry


def validate_review(
    review_path: Path, revision: str, release: str
) -> dict[str, Any]:
    review = load(review_path, REVIEW_FIELDS, "governance review attestation")
    if (
        review["schema"] != REVIEW_SCHEMA
        or review["release"] != release
        or review["source_revision"] != revision
        or review["result"] != "pass"
    ):
        raise QualificationError("governance review identity or result differs")
    reviewer = object_(review["reviewer"], REVIEWER_FIELDS, "reviewer")
    string(reviewer["id"], "reviewer.id")
    string(reviewer["organization"], "reviewer.organization")
    try:
        datetime.fromisoformat(string(review["reviewed_at_utc"], "reviewed_at_utc").replace("Z", "+00:00"))
    except ValueError as error:
        raise QualificationError("reviewed_at_utc must be ISO-8601") from error
    if review["reviewed_files"] != list(GOVERNANCE_FILES):
        raise QualificationError("governance review file inventory differs from policy")
    for field in (
        "repository_links_checked", "contacts_checked", "independent_from_maintainers"
    ):
        if review[field] is not True:
            raise QualificationError(f"governance review {field} must pass")
    if review["unstaffed_channels_advertised"] is not False:
        raise QualificationError("governance review advertises an unstaffed channel")
    return review


def fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    if os.name != "nt":
        for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def assemble(
    stage: Path, *, repo: Path, candidate: Path, anchor: Path, registry_path: Path,
    models_path: Path, review_path: Path, source_revision: str, release: str,
    run_id: str,
) -> dict[str, dict[str, Any]]:
    repo = repo.resolve(strict=True)
    candidate = ordinary(candidate, "candidate manifest")
    anchor = ordinary(anchor, "anchor wheel")
    hex_(source_revision, 40, "source revision")
    string(release, "release")
    string(run_id, "run id")
    artifacts = candidate_artifacts(candidate)
    anchor_identity = anchor_record(candidate, anchor, artifacts)
    registry_entries, registry = registry_sources(
        registry_path, candidate, source_revision, release
    )
    source = load(models_path, SOURCE_FIELDS, "model-zoo source")
    if (
        source["schema"] != SOURCE_SCHEMA
        or source["release"] != release
        or source["source_revision"] != source_revision
    ):
        raise QualificationError("model-zoo source identity differs from candidate")
    models = source["models"]
    if not isinstance(models, list) or len(models) != len(EXPECTED_MODELS):
        raise QualificationError("model-zoo source must contain four frozen models")
    validate_review(review_path, source_revision, release)
    try:
        check_claim_documents(repo)
    except CLAIM_GENERATOR["ClaimGenerationError"] as error:
        raise QualificationError("generated release claim documents are stale") from error

    stage.mkdir()
    cards_dir = stage / "cards"
    support_dir = stage / "support"
    cards_dir.mkdir()
    support_dir.mkdir()
    common = {
        "result": "pass", "release": release, "source_revision": source_revision,
        "run_id": run_id, "candidate_manifest_sha256": sha256(candidate),
        "anchor_artifact": anchor_identity,
    }
    sealed_models = []
    model_evidence: list[str] = []
    for ordinal, expected in enumerate(EXPECTED_MODELS):
        model = object_(models[ordinal], SOURCE_MODEL_FIELDS, f"models[{ordinal}]")
        if (model["tier"], model["role"], model["model_id"]) != expected:
            raise QualificationError("model-zoo source order differs from policy")
        string(model["revision"], "model revision")
        digest(model["tokenizer_sha256"], "model tokenizer digest")
        string(model["license"], "model license")
        artifact_ids = model["artifact_ids"]
        evidence_ids = model["evidence_receipt_ids"]
        for values, label in ((artifact_ids, "artifact_ids"), (evidence_ids, "evidence_receipt_ids")):
            if (
                not isinstance(values, list) or not values
                or len(values) != len(set(values))
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise QualificationError(f"model {label} must be non-empty and unique")
        if any(artifact_id not in artifacts for artifact_id in artifact_ids):
            raise QualificationError("model references an artifact absent from candidate")
        for evidence_id in evidence_ids:
            digest(evidence_id, "model evidence receipt id")
            if evidence_id not in registry_entries:
                raise QualificationError("model evidence is absent from source registry")
            if registry_entries[evidence_id]["artifact_id"] not in artifact_ids:
                raise QualificationError("model evidence binds a different candidate artifact")
            if evidence_id not in model_evidence:
                model_evidence.append(evidence_id)
        source_card = contained(models_path.parent, model["card"], "model card")
        destination = cards_dir / f"{ordinal:02d}-{source_card.name}"
        shutil.copyfile(source_card, destination)
        sealed_models.append({
            **{key: model[key] for key in SOURCE_MODEL_FIELDS - {"card"}},
            "card": file_record(destination, destination.relative_to(stage).as_posix()),
        })

    zoo = seal({**common, "schema": ZOO_SCHEMA, "models": sealed_models})
    zoo_path = stage / "model-zoo.json"
    write_json(zoo_path, zoo)

    evidence_dir = support_dir / "evidence"
    evidence_dir.mkdir()
    source_entries = []
    for ordinal, evidence_id in enumerate(model_evidence):
        entry = registry_entries[evidence_id]
        source_receipt = contained(
            registry_path.parent, entry["path"], "model evidence receipt"
        )
        destination = evidence_dir / f"{ordinal:02d}.json"
        shutil.copyfile(source_receipt, destination)
        source_entries.append({
            "id": evidence_id, "kind": entry["kind"],
            "artifact_id": entry["artifact_id"],
            "receipt": file_record(
                destination, destination.relative_to(support_dir).as_posix()
            ),
        })
    registry_copy = support_dir / "source-registry.json"
    write_json(registry_copy, {
        "schema": CLAIM_SOURCE_SCHEMA, "release": release,
        "source_revision": source_revision,
        "candidate_manifest_sha256": sha256(candidate),
        "registry_sha256": sha256(ordinary(registry_path, "source registry")),
        "entries": source_entries,
    })
    generator_path = repo / "scripts" / "generate-release-claims.py"
    claims = seal({
        **common, "schema": CLAIMS_SCHEMA, "run_id": f"{run_id}-claims",
        "generator_id": GENERATOR_ID,
        "generator_file": file_record(
            generator_path, generator_path.relative_to(repo).as_posix()
        ),
        "source_registry": file_record(
            registry_copy, registry_copy.relative_to(stage).as_posix()
        ),
        "documents": [file_record(repo / path, path) for path in CLAIM_DOCUMENTS],
        "source_receipt_ids": [zoo["receipt_id"], *model_evidence],
    })
    claims_path = stage / "generated-claims.json"
    write_json(claims_path, claims)

    review_copy = support_dir / "governance-review.json"
    shutil.copyfile(ordinary(review_path, "governance review attestation"), review_copy)
    governance = seal({
        **common, "schema": GOVERNANCE_SCHEMA, "run_id": f"{run_id}-governance",
        "files": [file_record(repo / path, path) for path in GOVERNANCE_FILES],
        "review_attestation": file_record(
            review_copy, review_copy.relative_to(stage).as_posix()
        ),
        "repository_links_checked": True, "contacts_checked": True,
        "independent_policy_review": True,
        "unstaffed_channels_advertised": False,
    })
    governance_path = stage / "governance.json"
    write_json(governance_path, governance)

    validate_zoo(zoo_path, source_revision, release, candidate, anchor)
    validate_claims(
        claims_path, source_revision, release, candidate, anchor, repo
    )
    validate_governance(
        governance_path, source_revision, release, candidate, anchor, repo
    )
    fsync_tree(stage)
    return {"model_zoo": zoo, "generated_claims": claims, "governance": governance}


def qualify(
    output_dir: Path, *, repo: Path, candidate: Path, anchor: Path,
    registry_path: Path, models_path: Path, review_path: Path,
    source_revision: str, release: str, run_id: str,
) -> dict[str, dict[str, Any]]:
    if output_dir.exists() or output_dir.is_symlink():
        raise QualificationError(f"output directory already exists: {output_dir}")
    repo = repo.resolve(strict=True)
    require_clean_revision(repo, source_revision)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    stage.rmdir()
    try:
        receipts = assemble(
            stage, repo=repo, candidate=candidate, anchor=anchor,
            registry_path=registry_path, models_path=models_path,
            review_path=review_path, source_revision=source_revision,
            release=release, run_id=run_id,
        )
        os.replace(stage, output_dir)
        return receipts
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--anchor-wheel", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--governance-review", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipts = qualify(
        args.output_dir.absolute(), repo=args.repo, candidate=args.candidate.absolute(),
        anchor=args.anchor_wheel.absolute(), registry_path=args.registry.absolute(),
        models_path=args.models.absolute(), review_path=args.governance_review.absolute(),
        source_revision=args.source_revision, release=args.release, run_id=args.run_id,
    )
    print(json.dumps(receipts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
