#!/usr/bin/env python3
"""Validate Qwen3.6 held-out quality and six-task retention evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any


QUALITY_SCHEMA = "tritium.qwen36-quality.v1"
TASK_SCHEMA = "tritium.qwen36-task-retention.v1"
MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
COMMON_FIELDS = {
    "schema", "receipt_id", "result", "release", "source_revision", "run_id",
    "candidate_manifest_sha256", "artifact", "model_id", "model_revision",
    "scope", "evaluation_id", "recipe_id",
}
ARTIFACT_FIELDS = {"id", "kind", "name", "bytes", "sha256"}
QUALITY_FIELDS = COMMON_FIELDS | {
    "dense_perplexity", "salt_v1_perplexity", "ptq_perplexity",
    "refined_perplexity", "ptq_gap_closed_fraction",
    "refined_relative_increase_pct", "refined_relative_ci95_upper_pct",
    "baseline_set_id", "baseline_inventory_complete", "baseline_comparisons",
    "near_zero_divergence", "additive_ternary_sota", "global_low_bit_pareto",
}
BASELINE_FIELDS = {
    "method", "family", "artifact_bytes", "resident_bytes", "quality_score",
    "runtime_ms", "reproduced", "matched_physical_bytes", "comparison_result",
}
TASK_FIELDS = COMMON_FIELDS | {"tasks", "mean_delta_pp", "mean_ci95_upper_pp"}
TASK_RESULT_FIELDS = {
    "name", "dense_accuracy_pct", "refined_accuracy_pct", "delta_pp",
    "ci95_upper_pp",
}
MAX_BYTES = 32 * 1024 * 1024


class FlagshipQualityError(ValueError):
    """Flagship quality evidence is stale, incomplete, or statistically invalid."""


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
        raise FlagshipQualityError(f"{label} fields do not match frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FlagshipQualityError(f"{label} must be non-empty")
    return value


def digest(value: Any, label: str) -> str:
    text = string(value, label)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise FlagshipQualityError(f"{label} must be a canonical SHA-256 digest")
    return text


def number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise FlagshipQualityError(f"{label} must be finite and at least {minimum}")
    return float(value)


def signed_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise FlagshipQualityError(f"{label} must be finite")
    return float(value)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def load(path: Path, fields: set[str], label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise FlagshipQualityError(f"{label} must be a bounded ordinary file")
    try:
        return object_(json.loads(path.read_bytes()), fields, label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FlagshipQualityError(f"{label} must contain UTF-8 JSON") from error


def contained(root: Path, value: Any) -> Path:
    text = string(value, "candidate artifact path")
    logical = PurePosixPath(text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in text:
        raise FlagshipQualityError("candidate artifact path is unsafe")
    cursor = root.resolve(strict=True)
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise FlagshipQualityError("candidate artifact path traverses a symlink")
    path = cursor.resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise FlagshipQualityError("candidate artifact escapes root") from error
    if path.is_symlink() or not path.is_file():
        raise FlagshipQualityError("candidate artifact must be ordinary")
    return path


def bind_artifact(receipt: dict[str, Any], candidate: Path) -> None:
    artifact = object_(receipt["artifact"], ARTIFACT_FIELDS, "artifact")
    if artifact["kind"] != "model-bundle":
        raise FlagshipQualityError("quality artifact must be a model bundle")
    try:
        document = json.loads(candidate.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FlagshipQualityError("candidate manifest is unreadable") from error
    values = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(values, list):
        raise FlagshipQualityError("candidate artifact inventory is malformed")
    matches = [value for value in values if isinstance(value, dict) and value.get("id") == artifact["id"]]
    if len(matches) != 1 or not isinstance(matches[0].get("identity"), dict):
        raise FlagshipQualityError("quality artifact is absent from candidate")
    value = matches[0]
    path = contained(candidate.parent, value.get("path"))
    actual = (artifact["id"], artifact["kind"], path.name, path.stat().st_size, sha256(path))
    declared = (artifact["id"], value.get("kind"), path.name, value["identity"].get("bytes"), value["identity"].get("sha256"))
    qualified = (artifact["id"], artifact["kind"], artifact["name"], artifact["bytes"], artifact["sha256"])
    if actual != declared or actual != qualified:
        raise FlagshipQualityError("quality artifact bytes contradict candidate")


def validate_common(
    receipt: dict[str, Any], schema: str, revision: str, release: str, candidate: Path
) -> None:
    if receipt["schema"] != schema or receipt["result"] != "pass":
        raise FlagshipQualityError("receipt schema or result mismatch")
    if receipt["source_revision"] != revision or receipt["release"] != release:
        raise FlagshipQualityError("receipt source or release is stale")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise FlagshipQualityError("expected source revision is malformed")
    string(receipt["run_id"], "receipt.run_id")
    if receipt["candidate_manifest_sha256"] != sha256(candidate):
        raise FlagshipQualityError("receipt does not bind candidate manifest")
    if (
        receipt["model_id"] != MODEL_ID
        or receipt["model_revision"] != MODEL_REVISION
        or receipt["scope"] != "language+mtp"
    ):
        raise FlagshipQualityError("receipt does not bind pinned language-plus-MTP scope")
    digest(receipt["evaluation_id"], "evaluation id")
    digest(receipt["recipe_id"], "recipe id")
    bind_artifact(receipt, candidate)


def finish(receipt: dict[str, Any], label: str) -> dict[str, Any]:
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    expected = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if receipt_id != expected:
        raise FlagshipQualityError(f"{label} receipt identity mismatch")
    return receipt


def validate_quality(
    receipt_path: Path, revision: str, release: str, candidate: Path
) -> dict[str, Any]:
    receipt = load(receipt_path, QUALITY_FIELDS, "quality receipt")
    validate_common(receipt, QUALITY_SCHEMA, revision, release, candidate)
    dense = number(receipt["dense_perplexity"], "dense perplexity", minimum=1e-12)
    salt_v1 = number(receipt["salt_v1_perplexity"], "SALT V1 perplexity", minimum=1e-12)
    ptq = number(receipt["ptq_perplexity"], "PTQ perplexity", minimum=1e-12)
    refined = number(receipt["refined_perplexity"], "refined perplexity", minimum=1e-12)
    if salt_v1 <= dense:
        raise FlagshipQualityError("SALT V1 must have a positive dense perplexity gap")
    expected_gap = (salt_v1 - ptq) / (salt_v1 - dense)
    gap = number(receipt["ptq_gap_closed_fraction"], "PTQ gap closure")
    if not close(gap, expected_gap) or gap < 0.5:
        raise FlagshipQualityError("PTQ does not close at least half the SALT V1 gap")
    expected_relative = (refined / dense - 1.0) * 100.0
    relative = signed_number(receipt["refined_relative_increase_pct"], "refined increase")
    upper = signed_number(receipt["refined_relative_ci95_upper_pct"], "refined CI upper")
    if not close(relative, expected_relative) or relative > 1.0 or upper > 1.0:
        raise FlagshipQualityError("refined perplexity or CI exceeds one percent")

    digest(receipt["baseline_set_id"], "baseline set id")
    if receipt["baseline_inventory_complete"] is not True:
        raise FlagshipQualityError("preregistered baseline inventory is incomplete")
    baselines = receipt["baseline_comparisons"]
    if not isinstance(baselines, list) or not baselines:
        raise FlagshipQualityError("quality receipt requires reproduced baselines")
    methods = set()
    for ordinal, value in enumerate(baselines):
        baseline = object_(value, BASELINE_FIELDS, f"baseline[{ordinal}]")
        method = string(baseline["method"], "baseline method")
        family = string(baseline["family"], "baseline family")
        if method in methods or family not in {"additive-ternary", "global-low-bit"}:
            raise FlagshipQualityError("baseline methods must be unique and classified")
        methods.add(method)
        for field in ("artifact_bytes", "resident_bytes", "quality_score", "runtime_ms"):
            number(baseline[field], f"baseline {field}", minimum=1e-12)
        if baseline["reproduced"] is not True or baseline["matched_physical_bytes"] is not True:
            raise FlagshipQualityError("baseline is not reproduced at matched physical bytes")
        comparison = baseline["comparison_result"]
        if comparison not in {"tritium-win", "tie", "tradeoff", "baseline-win"}:
            raise FlagshipQualityError("baseline comparison result is invalid")
        if family == "additive-ternary" and comparison != "tritium-win":
            raise FlagshipQualityError("Tritium did not strictly win an additive baseline")
        if family == "global-low-bit" and comparison == "baseline-win":
            raise FlagshipQualityError("a global low-bit baseline dominates Tritium")
    families = {baseline["family"] for baseline in baselines}
    if families != {"additive-ternary", "global-low-bit"}:
        raise FlagshipQualityError("both additive and global baseline families are required")
    if any(
        receipt[field] is not True
        for field in ("near_zero_divergence", "additive_ternary_sota", "global_low_bit_pareto")
    ):
        raise FlagshipQualityError("independent quality/SOTA verdicts must all pass")
    return finish(receipt, "quality")


def validate_tasks(
    receipt_path: Path, revision: str, release: str, candidate: Path
) -> dict[str, Any]:
    receipt = load(receipt_path, TASK_FIELDS, "task-retention receipt")
    validate_common(receipt, TASK_SCHEMA, revision, release, candidate)
    tasks = receipt["tasks"]
    if not isinstance(tasks, list) or len(tasks) != 6:
        raise FlagshipQualityError("task-retention receipt requires six tasks")
    names = set()
    deltas = []
    for ordinal, value in enumerate(tasks):
        task = object_(value, TASK_RESULT_FIELDS, f"tasks[{ordinal}]")
        name = string(task["name"], "task name")
        if name in names:
            raise FlagshipQualityError("task names must be unique")
        names.add(name)
        dense = number(task["dense_accuracy_pct"], "dense task accuracy")
        refined = number(task["refined_accuracy_pct"], "refined task accuracy")
        if dense > 100.0 or refined > 100.0:
            raise FlagshipQualityError("task accuracy exceeds 100 percent")
        delta = number(task["delta_pp"], "task delta")
        upper = number(task["ci95_upper_pp"], "task CI upper")
        if not close(delta, max(0.0, dense - refined)) or delta > 1.0 or upper > 1.0:
            raise FlagshipQualityError("individual task retention gate failed")
        deltas.append(delta)
    mean = sum(deltas) / len(deltas)
    declared_mean = number(receipt["mean_delta_pp"], "mean task delta")
    mean_upper = number(receipt["mean_ci95_upper_pp"], "mean task CI upper")
    if not close(mean, declared_mean) or mean > 0.5 or mean_upper > 0.5:
        raise FlagshipQualityError("mean task retention gate failed")
    return finish(receipt, "task-retention")
