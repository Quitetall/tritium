#!/usr/bin/env python3
"""Validate model-zoo, generated-claims, and governance release receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


ZOO_SCHEMA = "tritium.model-zoo-admission.v1"
CLAIMS_SCHEMA = "tritium.generated-claims.v1"
GOVERNANCE_SCHEMA = "tritium.governance-docs.v1"
COMMON_FIELDS = {
    "schema",
    "receipt_id",
    "result",
    "release",
    "source_revision",
    "run_id",
    "candidate_manifest_sha256",
    "anchor_artifact",
}
ZOO_FIELDS = COMMON_FIELDS | {"models"}
CLAIMS_FIELDS = COMMON_FIELDS | {"generator_id", "documents", "source_receipt_ids"}
GOVERNANCE_FIELDS = COMMON_FIELDS | {
    "files",
    "repository_links_checked",
    "contacts_checked",
    "independent_policy_review",
    "unstaffed_channels_advertised",
}
ARTIFACT_FIELDS = {"id", "kind", "name", "bytes", "sha256"}
MODEL_FIELDS = {
    "tier",
    "role",
    "model_id",
    "revision",
    "tokenizer_sha256",
    "license",
    "card",
    "artifact_ids",
    "evidence_receipt_ids",
}
FILE_FIELDS = {"path", "bytes", "sha256"}
EXPECTED_MODELS = (
    ("accessible", "tutorial", "HuggingFaceTB/SmolLM2-135M"),
    ("accessible", "recipe", "HuggingFaceTB/SmolLM2-1.7B"),
    ("native-reference", "native", "microsoft/bitnet-b1.58-2B-4T"),
    ("flagship", "language+mtp", "Qwen/Qwen3.6-27B"),
)
CLAIM_DOCUMENTS = (
    "README.md",
    "docs/book/src/model-zoo.md",
    "docs/book/src/benchmarks.md",
    "docs/compatibility.md",
)
GOVERNANCE_FILES = (
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "COMMUNITY.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "SECURITY.md",
    "SUPPORT.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/DISCUSSION_TEMPLATE/ideas.yml",
    ".github/DISCUSSION_TEMPLATE/q-a.yml",
    ".github/ISSUE_TEMPLATE/backend.yml",
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/estimator.yml",
    ".github/ISSUE_TEMPLATE/model-evidence.yml",
    ".github/ISSUE_TEMPLATE/question.yml",
)
HEX = frozenset("0123456789abcdef")
MAX_RECEIPT_BYTES = 8 * 1024 * 1024
MAX_SUPPORT_BYTES = 32 * 1024 * 1024


class ZooCommunityError(ValueError):
    """Zoo/community evidence is stale, incomplete, drifted, or overclaims."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def object_(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ZooCommunityError(f"{label} fields do not match frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ZooCommunityError(f"{label} must be non-empty")
    return value


def hex_(value: Any, length: int, label: str) -> str:
    text = string(value, label)
    if len(text) != length or any(character not in HEX for character in text):
        raise ZooCommunityError(
            f"{label} must be {length} lowercase hexadecimal characters"
        )
    return text


def digest(value: Any, label: str) -> str:
    text = string(value, label)
    if not text.startswith("sha256:"):
        raise ZooCommunityError(f"{label} must be a canonical digest")
    hex_(text.removeprefix("sha256:"), 64, label)
    return text


def load(path: Path, fields: set[str], label: str) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > MAX_RECEIPT_BYTES
    ):
        raise ZooCommunityError(f"{label} must be a bounded ordinary file")
    try:
        return object_(json.loads(path.read_bytes()), fields, label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ZooCommunityError(f"{label} must contain UTF-8 JSON") from error


def contained(root: Path, logical_value: Any, label: str) -> Path:
    text = string(logical_value, label)
    logical = PurePosixPath(text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in text:
        raise ZooCommunityError(f"{label} path is unsafe")
    cursor = root.resolve(strict=True)
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ZooCommunityError(f"{label} traverses a symlink")
    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ZooCommunityError(f"{label} escapes root") from error
    if (
        resolved.is_symlink()
        or not resolved.is_file()
        or resolved.stat().st_size > MAX_SUPPORT_BYTES
    ):
        raise ZooCommunityError(f"{label} must be a bounded ordinary file")
    return resolved


def validate_common(
    receipt: dict[str, Any],
    schema: str,
    revision: str,
    release: str,
    candidate: Path,
    anchor: Path,
) -> dict[str, Any]:
    if receipt["schema"] != schema or receipt["result"] != "pass":
        raise ZooCommunityError("receipt schema or result mismatch")
    if receipt["source_revision"] != revision or receipt["release"] != release:
        raise ZooCommunityError("receipt source or release is stale")
    hex_(revision, 40, "expected source revision")
    string(receipt["run_id"], "receipt.run_id")
    if candidate.is_symlink() or not candidate.is_file():
        raise ZooCommunityError("candidate manifest must be ordinary")
    if receipt["candidate_manifest_sha256"] != sha256(candidate):
        raise ZooCommunityError("receipt does not bind candidate manifest")
    record = object_(receipt["anchor_artifact"], ARTIFACT_FIELDS, "anchor artifact")
    if anchor.is_symlink() or not anchor.is_file():
        raise ZooCommunityError("anchor artifact must be ordinary")
    if (
        record["kind"] != "python-wheel"
        or record["name"] != anchor.name
        or record["bytes"] != anchor.stat().st_size
        or record["sha256"] != sha256(anchor)
    ):
        raise ZooCommunityError("receipt does not bind anchor wheel")
    return receipt


def validate_file(value: Any, root: Path, label: str) -> tuple[str, int, str]:
    record = object_(value, FILE_FIELDS, label)
    path = contained(root, record["path"], label)
    actual = (record["path"], path.stat().st_size, sha256(path))
    declared = (record["path"], record["bytes"], record["sha256"])
    if actual != declared:
        raise ZooCommunityError(f"{label} bytes drifted")
    return actual


def finish(receipt: dict[str, Any], label: str) -> dict[str, Any]:
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    if receipt_id != "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest():
        raise ZooCommunityError(f"{label} receipt identity mismatch")
    return receipt


def validate_zoo(
    receipt_path: Path,
    revision: str,
    release: str,
    candidate: Path,
    anchor: Path,
) -> dict[str, Any]:
    receipt = validate_common(
        load(receipt_path, ZOO_FIELDS, "model-zoo receipt"),
        ZOO_SCHEMA,
        revision,
        release,
        candidate,
        anchor,
    )
    models = receipt["models"]
    if not isinstance(models, list) or len(models) != len(EXPECTED_MODELS):
        raise ZooCommunityError("model zoo must contain four frozen entries")
    root = receipt_path.parent.resolve(strict=True)
    for ordinal, expected in enumerate(EXPECTED_MODELS):
        model = object_(models[ordinal], MODEL_FIELDS, f"models[{ordinal}]")
        if (model["tier"], model["role"], model["model_id"]) != expected:
            raise ZooCommunityError("model zoo tier/model order differs from policy")
        string(model["revision"], "model revision")
        digest(model["tokenizer_sha256"], "model tokenizer digest")
        string(model["license"], "model license")
        validate_file(model["card"], root, f"models[{ordinal}].card")
        for field in ("artifact_ids", "evidence_receipt_ids"):
            values = model[field]
            if (
                not isinstance(values, list)
                or not values
                or len(set(values)) != len(values)
                or any(not isinstance(item, str) or not item for item in values)
            ):
                raise ZooCommunityError(f"model {field} must be non-empty and unique")
        for value in model["evidence_receipt_ids"]:
            digest(value, "model evidence receipt id")
    return finish(receipt, "model-zoo")


def validate_claims(
    receipt_path: Path,
    revision: str,
    release: str,
    candidate: Path,
    anchor: Path,
    repo: Path,
) -> dict[str, Any]:
    receipt = validate_common(
        load(receipt_path, CLAIMS_FIELDS, "generated-claims receipt"),
        CLAIMS_SCHEMA,
        revision,
        release,
        candidate,
        anchor,
    )
    string(receipt["generator_id"], "receipt.generator_id")
    documents = receipt["documents"]
    if not isinstance(documents, list) or len(documents) != len(CLAIM_DOCUMENTS):
        raise ZooCommunityError("generated claim document inventory is incomplete")
    if [item.get("path") for item in documents if isinstance(item, dict)] != list(
        CLAIM_DOCUMENTS
    ):
        raise ZooCommunityError("generated claim document order differs from policy")
    for ordinal, document in enumerate(documents):
        validate_file(document, repo, f"documents[{ordinal}]")
    sources = receipt["source_receipt_ids"]
    if (
        not isinstance(sources, list)
        or not sources
        or len(set(sources)) != len(sources)
    ):
        raise ZooCommunityError("generated claims require unique source receipts")
    for value in sources:
        digest(value, "claim source receipt id")
    return finish(receipt, "generated-claims")


def validate_governance(
    receipt_path: Path,
    revision: str,
    release: str,
    candidate: Path,
    anchor: Path,
    repo: Path,
) -> dict[str, Any]:
    receipt = validate_common(
        load(receipt_path, GOVERNANCE_FIELDS, "governance receipt"),
        GOVERNANCE_SCHEMA,
        revision,
        release,
        candidate,
        anchor,
    )
    files = receipt["files"]
    if not isinstance(files, list) or len(files) != len(GOVERNANCE_FILES):
        raise ZooCommunityError("governance file inventory is incomplete")
    if [item.get("path") for item in files if isinstance(item, dict)] != list(
        GOVERNANCE_FILES
    ):
        raise ZooCommunityError("governance file order differs from policy")
    for ordinal, document in enumerate(files):
        validate_file(document, repo, f"files[{ordinal}]")
    for field in (
        "repository_links_checked",
        "contacts_checked",
        "independent_policy_review",
    ):
        if receipt[field] is not True:
            raise ZooCommunityError(f"governance {field} must pass")
    if receipt["unstaffed_channels_advertised"] is not False:
        raise ZooCommunityError("governance advertises an unstaffed channel")
    return finish(receipt, "governance")
