#!/usr/bin/env python3
"""Strict ADR 0033 release-evidence registry and local-RC gate report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy
from typing import Any


CUDA_RECEIPT = runpy.run_path(Path(__file__).with_name("verify-cuda-training-receipt.py"))
validate_cuda_receipt = CUDA_RECEIPT["validate"]

SCHEMA = "tritium.release-evidence-registry.v1"
REPORT_SCHEMA = "tritium.release-gate-report.v1"
TOP_FIELDS = {
    "schema", "release", "source_revision", "candidate_manifest_sha256", "receipts"
}
RECEIPT_FIELDS = {"id", "kind", "path", "sha256", "artifact_id", "parents"}
KNOWN_KINDS = frozenset({"cuda-training"})
HEX = frozenset("0123456789abcdef")
MAX_RECEIPT_BYTES = 32 * 1024 * 1024

# This policy is code, not registry input: a partial or adversarial registry cannot
# remove release gates. New receipt schemas become useful only after a validator lands.
GATES = (
    (
        "flagship-qwen",
        ("conversion-refinement", "quality", "task-retention", "runtime", "physical-bytes"),
    ),
    ("pytorch-hf", ("frontend-lifecycle", "distributed-training", "export-reload")),
    ("native-backends", ("backend-manifest", "cuda-training", "performance")),
    ("estimators-refinement", ("estimator-validation", "refinement", "baseline-ablation")),
    ("browser", ("browser-conformance",)),
    ("onnx", ("onnx-inference",)),
    ("packages", ("clean-install", "local-archive")),
    ("serving", ("serving-deployment",)),
    ("zoo-community", ("model-zoo", "generated-claims", "governance-docs")),
    ("reproduction-signoff", ("second-machine", "independent-review", "signature")),
)


class EvidenceError(ValueError):
    """Registry evidence is malformed, stale, duplicated, or unvalidated."""


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise EvidenceError(f"{label} fields do not match the frozen schema")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} must be a non-empty string")
    return value


def _hex(value: Any, length: int, label: str) -> str:
    text = _string(value, label)
    if len(text) != length or any(character not in HEX for character in text):
        raise EvidenceError(f"{label} must be {length} lowercase hexadecimal characters")
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _contained_file(root: Path, value: Any, label: str) -> Path:
    logical_text = _string(value, label)
    logical = PurePosixPath(logical_text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in logical_text:
        raise EvidenceError(f"{label} must be a contained POSIX path")
    cursor = root
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise EvidenceError(f"{label} must not traverse a symlink")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise EvidenceError(f"{label} is not contained below the registry") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise EvidenceError(f"{label} must name an ordinary file")
    return resolved


def _gate_row(
    gate_id: str, required: tuple[str, ...], evidence: dict[str, str]
) -> dict[str, Any]:
    satisfied = sorted(kind for kind in required if evidence.get(kind) == "empirical")
    structural = sorted(kind for kind in required if evidence.get(kind) == "structural")
    missing = sorted(kind for kind in required if kind not in evidence)
    if missing:
        status = "MISSING"
    elif structural:
        status = "STRUCTURAL_ONLY"
    else:
        status = "PASS"
    return {
        "id": gate_id,
        "status": status,
        "required_kinds": list(required),
        "satisfied_kinds": satisfied,
        "structural_kinds": structural,
        "missing_kinds": missing,
    }


def _check_ancestry(entries: dict[str, dict[str, Any]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(receipt_id: str) -> None:
        if receipt_id in visiting:
            raise EvidenceError("receipt ancestry contains a cycle")
        if receipt_id in visited:
            return
        visiting.add(receipt_id)
        for parent in entries[receipt_id]["parents"]:
            if parent not in entries:
                raise EvidenceError(f"receipt {receipt_id!r} has unknown parent {parent!r}")
            visit(parent)
        visiting.remove(receipt_id)
        visited.add(receipt_id)

    for receipt_id in entries:
        visit(receipt_id)


def evaluate(registry: Path, candidate: Path, candidate_document: dict[str, Any]) -> dict[str, Any]:
    if registry.is_symlink() or not registry.is_file():
        raise EvidenceError("registry must be an ordinary file")
    if registry.stat().st_size > MAX_RECEIPT_BYTES:
        raise EvidenceError("registry exceeds the metadata size limit")
    try:
        document = _object(json.loads(registry.read_bytes()), TOP_FIELDS, "registry")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError("registry must contain UTF-8 JSON") from error
    if document["schema"] != SCHEMA:
        raise EvidenceError(f"registry.schema must equal {SCHEMA!r}")
    release = _string(document["release"], "registry.release")
    revision = _hex(document["source_revision"], 40, "registry.source_revision")
    if release != candidate_document.get("release") or revision != candidate_document.get("source_revision"):
        raise EvidenceError("registry release identity does not match candidate")
    expected_candidate = _hex(
        document["candidate_manifest_sha256"], 64, "registry.candidate_manifest_sha256"
    )
    if expected_candidate != _sha256(candidate):
        raise EvidenceError("registry does not bind the exact candidate manifest")

    raw_receipts = document["receipts"]
    if not isinstance(raw_receipts, list):
        raise EvidenceError("registry.receipts must be an array")
    root = registry.parent.resolve(strict=True)
    entries: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    portable_paths: set[str] = set()
    run_ids: set[str] = set()
    artifacts = {
        artifact.get("id"): artifact
        for artifact in candidate_document.get("artifacts", [])
        if isinstance(artifact, dict)
    }
    evidence: dict[str, str] = {}
    for ordinal, raw in enumerate(raw_receipts):
        label = f"registry.receipts[{ordinal}]"
        entry = _object(raw, RECEIPT_FIELDS, label)
        receipt_id = _string(entry["id"], f"{label}.id")
        kind = _string(entry["kind"], f"{label}.kind")
        if kind not in KNOWN_KINDS:
            raise EvidenceError(f"{label}.kind has no release validator")
        if receipt_id in entries:
            raise EvidenceError(f"duplicate receipt id {receipt_id!r}")
        logical_path = _string(entry["path"], f"{label}.path")
        portable_path = logical_path.casefold()
        if logical_path in paths or portable_path in portable_paths:
            raise EvidenceError(f"duplicate receipt path {logical_path!r}")
        parents = entry["parents"]
        if not isinstance(parents, list) or len(set(parents)) != len(parents) or any(
            not isinstance(parent, str) or not parent for parent in parents
        ):
            raise EvidenceError(f"{label}.parents must be a unique string array")
        receipt_path = _contained_file(root, logical_path, f"{label}.path")
        if receipt_path.stat().st_size > MAX_RECEIPT_BYTES:
            raise EvidenceError(f"{label}.path exceeds the metadata size limit")
        if _sha256(receipt_path) != _hex(entry["sha256"], 64, f"{label}.sha256"):
            raise EvidenceError(f"{label}.sha256 does not match receipt bytes")
        artifact_id = _string(entry["artifact_id"], f"{label}.artifact_id")
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            raise EvidenceError(f"{label}.artifact_id is absent from candidate")
        if artifact.get("kind") != "python-wheel":
            raise EvidenceError("CUDA training evidence must bind a candidate Python wheel")
        artifact_path = candidate.parent / _string(artifact.get("path"), "candidate artifact path")
        try:
            receipt = validate_cuda_receipt(
                receipt_path, revision, release, artifact_path
            )
        except (OSError, ValueError) as error:
            raise EvidenceError(f"{label} failed CUDA receipt validation: {error}") from error
        if receipt["receipt_id"] != receipt_id:
            raise EvidenceError(f"{label}.id does not match the receipt identity")
        if receipt["artifact"]["kind"] != "python-wheel":
            raise EvidenceError("CUDA training receipt does not identify a Python wheel")
        run_id = receipt["run_id"]
        if run_id in run_ids:
            raise EvidenceError(f"duplicate run id {run_id!r}")
        run_ids.add(run_id)
        evidence[kind] = "empirical"
        entries[receipt_id] = {**entry, "parents": list(parents)}
        paths.add(logical_path)
        portable_paths.add(portable_path)
    _check_ancestry(entries)

    rows = [_gate_row(gate_id, required, evidence) for gate_id, required in GATES]
    ready = all(row["status"] == "PASS" for row in rows)
    return {
        "schema": REPORT_SCHEMA,
        "release": release,
        "source_revision": revision,
        "candidate_manifest_sha256": expected_candidate,
        "ready": ready,
        "rows": rows,
        "external_activation": "EXTERNAL_AUTH_REQUIRED",
    }


def render(report: dict[str, Any]) -> str:
    lines = ["STATUS           GATE                  MISSING"]
    for row in report["rows"]:
        missing = ",".join(row["missing_kinds"] + row["structural_kinds"]) or "-"
        lines.append(f"{row['status']:<16} {row['id']:<21} {missing}")
    lines.append(f"EXTERNAL_AUTH_REQUIRED public-activation     explicit-authorization")
    return "\n".join(lines)
