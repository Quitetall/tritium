#!/usr/bin/env python3
"""Validate estimator catalog, refinement lineage, and ablation evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any


ESTIMATOR_SCHEMA = "tritium.estimator-catalog-qualification.v1"
REFINEMENT_SCHEMA = "tritium.refinement-qualification.v1"
ABLATION_SCHEMA = "tritium.baseline-ablation-qualification.v1"
COMMON_FIELDS = {
    "schema", "receipt_id", "result", "release", "source_revision", "run_id",
    "candidate_manifest_sha256", "anchor_artifact",
}
ARTIFACT_FIELDS = {"id", "kind", "name", "bytes", "sha256"}
ESTIMATOR_FIELDS = COMMON_FIELDS | {"estimators", "external_plugin"}
ESTIMATOR_CASE_FIELDS = {
    "name", "algorithm_id", "schema_version", "physical_planes", "hard_trits_exact",
    "finite_nonnegative_scales", "master_gradients_finite", "state_gradients_finite",
    "state_roundtrip_exact", "tied_identity_preserved", "coverage_exact",
}
PLUGIN_FIELDS = {
    "registered", "duplicate_rejected", "contract_validated", "purity_opt_in_required",
    "invalid_projection_rejected",
}
REFINEMENT_FIELDS = COMMON_FIELDS | {
    "parent_artifact_id", "training_set_id", "validation_set_id", "splits_disjoint",
    "children",
}
CHILD_FIELDS = {
    "mode", "artifact", "parent_artifact_id", "work_id", "recipe_id", "ancestry_id",
    "trits_frozen", "allocation_frozen", "hard_candidates_held_out", "g128_aligned",
    "native_salt_package", "strict_reload", "latent_residuals",
}
ABLATION_FIELDS = COMMON_FIELDS | {
    "model_artifact_id", "evaluation_id", "baseline_set_id", "inventory_complete",
    "baselines",
}
BASELINE_FIELDS = {
    "method", "family", "artifact_bytes", "target_bytes", "rate_gap_bpw",
    "quality_score", "runtime_ms", "resident_bytes", "reproduced", "same_box",
    "publishable_recipe", "eligible_for_claim",
}
ESTIMATORS = (
    ("absmean-ste", "tritium.absmean-ste", 1),
    ("annealed-ste", "tritium.annealed-ste", 1),
    ("lsq", "tritium.lsq", 1),
    ("salt-ste", "tritium.salt-ste", 1),
    ("sparse-ternary", "tritium.sparse-ternary", 1),
    ("ttq", "tritium.ttq", 2),
    ("twn", "tritium.twn", 1),
)
CHILDREN = ("scale-only", "hard-pv", "s34")
BASELINES = (
    ("rtn-absmean", "ternary"),
    ("gptq-style", "global-low-bit"),
    ("awq-style", "global-low-bit"),
    ("salt-v1", "ternary"),
    ("no-curvature", "ablation"),
    ("no-rotation", "ablation"),
    ("greedy-salt-v1", "ablation"),
)
MAX_RECEIPT_BYTES = 32 * 1024 * 1024


class EstimatorRefinementError(ValueError):
    """Estimator/refinement evidence is stale, conflated, or incomplete."""


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
        raise EstimatorRefinementError(f"{label} fields do not match frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EstimatorRefinementError(f"{label} must be non-empty")
    return value


def digest(value: Any, label: str) -> str:
    text = string(value, label)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise EstimatorRefinementError(f"{label} must be a canonical SHA-256 digest")
    return text


def number(value: Any, label: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise EstimatorRefinementError(f"{label} must be finite and at least {minimum}")
    return float(value)


def contained(root: Path, value: Any) -> Path:
    text = string(value, "candidate artifact path")
    logical = PurePosixPath(text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in text:
        raise EstimatorRefinementError("candidate artifact path is unsafe")
    cursor = root.resolve(strict=True)
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise EstimatorRefinementError("candidate artifact path traverses a symlink")
    path = cursor.resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise EstimatorRefinementError("candidate artifact escapes root") from error
    if path.is_symlink() or not path.is_file():
        raise EstimatorRefinementError("candidate artifact must be ordinary")
    return path


def inventory(candidate: Path) -> dict[str, tuple[Any, ...]]:
    if candidate.is_symlink() or not candidate.is_file():
        raise EstimatorRefinementError("candidate manifest must be ordinary")
    try:
        document = json.loads(candidate.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EstimatorRefinementError("candidate manifest is malformed") from error
    values = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(values, list):
        raise EstimatorRefinementError("candidate artifact inventory is malformed")
    result = {}
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("identity"), dict):
            raise EstimatorRefinementError("candidate artifact is malformed")
        artifact_id = string(value.get("id"), "candidate artifact id")
        path = contained(candidate.parent, value.get("path"))
        actual = (
            artifact_id, value.get("kind"), path.name, path.stat().st_size, sha256(path)
        )
        declared = (
            artifact_id, value.get("kind"), path.name,
            value["identity"].get("bytes"), value["identity"].get("sha256"),
        )
        if artifact_id in result or actual != declared:
            raise EstimatorRefinementError("candidate artifact identity is duplicate or drifted")
        result[artifact_id] = actual
    return result


def bind(record: Any, artifacts: dict[str, tuple[Any, ...]], kind: str) -> dict[str, Any]:
    artifact = object_(record, ARTIFACT_FIELDS, "artifact")
    declared = (
        artifact["id"], artifact["kind"], artifact["name"], artifact["bytes"],
        artifact["sha256"],
    )
    if artifact["kind"] != kind or artifacts.get(artifact["id"]) != declared:
        raise EstimatorRefinementError("receipt does not bind candidate artifact")
    return artifact


def load_common(
    path: Path,
    fields: set[str],
    schema: str,
    revision: str,
    release: str,
    candidate: Path,
) -> tuple[dict[str, Any], dict[str, tuple[Any, ...]]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECEIPT_BYTES:
        raise EstimatorRefinementError("receipt must be a bounded ordinary file")
    try:
        receipt = object_(json.loads(path.read_bytes()), fields, "receipt")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EstimatorRefinementError("receipt must contain UTF-8 JSON") from error
    if receipt["schema"] != schema or receipt["result"] != "pass":
        raise EstimatorRefinementError("receipt schema or result mismatch")
    if receipt["source_revision"] != revision or receipt["release"] != release:
        raise EstimatorRefinementError("receipt source or release is stale")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise EstimatorRefinementError("expected revision is malformed")
    string(receipt["run_id"], "receipt.run_id")
    if receipt["candidate_manifest_sha256"] != sha256(candidate):
        raise EstimatorRefinementError("receipt does not bind candidate manifest")
    return receipt, inventory(candidate)


def finish(receipt: dict[str, Any], label: str) -> dict[str, Any]:
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    expected = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if receipt_id != expected:
        raise EstimatorRefinementError(f"{label} receipt identity mismatch")
    return receipt


def validate_estimators(
    path: Path, revision: str, release: str, candidate: Path
) -> dict[str, Any]:
    receipt, artifacts = load_common(
        path, ESTIMATOR_FIELDS, ESTIMATOR_SCHEMA, revision, release, candidate
    )
    bind(receipt["anchor_artifact"], artifacts, "python-wheel")
    cases = receipt["estimators"]
    if not isinstance(cases, list) or len(cases) != len(ESTIMATORS):
        raise EstimatorRefinementError("built-in estimator inventory is incomplete")
    for ordinal, expected in enumerate(ESTIMATORS):
        case = object_(cases[ordinal], ESTIMATOR_CASE_FIELDS, f"estimators[{ordinal}]")
        if (case["name"], case["algorithm_id"], case["physical_planes"]) != expected:
            raise EstimatorRefinementError("estimator identity or plane contract differs")
        if case["schema_version"] != 1 or any(
            case[field] is not True
            for field in (
                "hard_trits_exact", "finite_nonnegative_scales", "master_gradients_finite",
                "state_gradients_finite", "state_roundtrip_exact",
                "tied_identity_preserved", "coverage_exact",
            )
        ):
            raise EstimatorRefinementError("estimator differentiability/export gate failed")
    plugin = object_(receipt["external_plugin"], PLUGIN_FIELDS, "external plugin")
    if any(plugin[field] is not True for field in PLUGIN_FIELDS):
        raise EstimatorRefinementError("external estimator safety gate failed")
    return finish(receipt, "estimator")


def validate_refinement(
    path: Path, revision: str, release: str, candidate: Path
) -> dict[str, Any]:
    receipt, artifacts = load_common(
        path, REFINEMENT_FIELDS, REFINEMENT_SCHEMA, revision, release, candidate
    )
    bind(receipt["anchor_artifact"], artifacts, "model-bundle")
    parent = string(receipt["parent_artifact_id"], "parent artifact id")
    if artifacts.get(parent, (None, None))[1] != "model-bundle":
        raise EstimatorRefinementError("refinement parent is absent from candidate")
    digest(receipt["training_set_id"], "training set id")
    digest(receipt["validation_set_id"], "validation set id")
    if (
        receipt["training_set_id"] == receipt["validation_set_id"]
        or receipt["splits_disjoint"] is not True
    ):
        raise EstimatorRefinementError("refinement training/validation splits overlap")
    children = receipt["children"]
    if not isinstance(children, list) or len(children) != len(CHILDREN):
        raise EstimatorRefinementError("all three refinement children are required")
    seen = set()
    for ordinal, mode in enumerate(CHILDREN):
        child = object_(children[ordinal], CHILD_FIELDS, f"children[{ordinal}]")
        if child["mode"] != mode or child["parent_artifact_id"] != parent:
            raise EstimatorRefinementError("refinement child lineage differs from policy")
        artifact = bind(child["artifact"], artifacts, "model-bundle")
        if artifact["id"] in seen:
            raise EstimatorRefinementError("refinement children must bind distinct artifacts")
        seen.add(artifact["id"])
        for field in ("work_id", "recipe_id", "ancestry_id"):
            digest(child[field], f"child {field}")
        if (
            child["hard_candidates_held_out"] is not True
            or child["g128_aligned"] is not True
            or child["native_salt_package"] is not True
            or child["strict_reload"] is not True
            or child["latent_residuals"] != 0
        ):
            raise EstimatorRefinementError("refinement child is not deployable or held out")
        if mode == "scale-only":
            if child["trits_frozen"] is not True or child["allocation_frozen"] is not True:
                raise EstimatorRefinementError("scale-only refinement changed hard structure")
        elif child["trits_frozen"] is not False:
            raise EstimatorRefinementError("hard refinement did not declare assignment changes")
    if receipt["anchor_artifact"]["id"] != children[-1]["artifact"]["id"]:
        raise EstimatorRefinementError("refinement anchor is not the final S34 child")
    return finish(receipt, "refinement")


def validate_ablation(
    path: Path, revision: str, release: str, candidate: Path
) -> dict[str, Any]:
    receipt, artifacts = load_common(
        path, ABLATION_FIELDS, ABLATION_SCHEMA, revision, release, candidate
    )
    bind(receipt["anchor_artifact"], artifacts, "model-bundle")
    model_artifact_id = string(receipt["model_artifact_id"], "model artifact id")
    if artifacts.get(model_artifact_id, (None, None))[1] != "model-bundle":
        raise EstimatorRefinementError("ablation model artifact is absent")
    digest(receipt["evaluation_id"], "evaluation id")
    digest(receipt["baseline_set_id"], "baseline set id")
    if receipt["inventory_complete"] is not True:
        raise EstimatorRefinementError("baseline inventory is incomplete")
    baselines = receipt["baselines"]
    if not isinstance(baselines, list) or len(baselines) != len(BASELINES):
        raise EstimatorRefinementError("all frozen baselines and ablations are required")
    for ordinal, expected in enumerate(BASELINES):
        baseline = object_(baselines[ordinal], BASELINE_FIELDS, f"baselines[{ordinal}]")
        if (baseline["method"], baseline["family"]) != expected:
            raise EstimatorRefinementError("baseline identity/order differs from policy")
        artifact_bytes = number(baseline["artifact_bytes"], "artifact bytes", 1.0)
        target_bytes = number(baseline["target_bytes"], "target bytes", 1.0)
        gap = number(baseline["rate_gap_bpw"], "rate gap bpw")
        for field in ("quality_score", "runtime_ms", "resident_bytes"):
            number(baseline[field], f"baseline {field}", 1e-12)
        eligible = gap <= 0.05 and artifact_bytes <= target_bytes
        if (
            baseline["reproduced"] is not True
            or baseline["same_box"] is not True
            or baseline["publishable_recipe"] is not True
            or baseline["eligible_for_claim"] is not eligible
        ):
            raise EstimatorRefinementError("baseline reproduction or byte matching failed")
    return finish(receipt, "baseline-ablation")
