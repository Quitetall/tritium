#!/usr/bin/env python3
"""Validate estimator catalog, refinement lineage, and ablation evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
from typing import Any


ESTIMATOR_SCHEMA = "tritium.estimator-catalog-qualification.v1"
REFINEMENT_SCHEMA = "tritium.refinement-qualification.v1"
ABLATION_SCHEMA = "tritium.baseline-ablation-qualification.v1"
COMMON_FIELDS = {
    "schema", "receipt_id", "result", "release", "source_revision", "run_id",
    "candidate_manifest_sha256", "anchor_artifact",
}
ARTIFACT_FIELDS = {"id", "kind", "name", "bytes", "sha256"}
ESTIMATOR_FIELDS = COMMON_FIELDS | {"estimators", "external_plugin", "trace"}
ESTIMATOR_CASE_FIELDS = {
    "name", "algorithm_id", "schema_version", "physical_planes", "hard_trits_exact",
    "finite_nonnegative_scales", "master_gradients_finite", "state_gradients_finite",
    "state_roundtrip_exact", "tied_identity_preserved", "coverage_exact",
}
PLUGIN_FIELDS = {
    "registered", "duplicate_rejected", "contract_validated", "purity_opt_in_required",
    "invalid_projection_rejected",
}
FILE_FIELDS = {"path", "bytes", "sha256"}
TRACE_SCHEMA = "tritium.estimator-catalog-execution.v1"
TRACE_FIELDS = {
    "schema", "result", "release", "source_revision", "run_id", "wheel",
    "environment", "estimators", "external_plugin",
}
TRACE_WHEEL_FIELDS = {"name", "bytes", "sha256"}
ENVIRONMENT_FIELDS = {"python", "torch", "tritium", "device"}
REFINEMENT_TRACE_SCHEMA = "tritium.refinement-execution.v1"
REFINEMENT_TRACE_FIELDS = {
    "schema", "result", "release", "source_revision", "run_id", "environment",
    "parent_artifact_id", "training_set_id", "training_members",
    "validation_set_id", "validation_members", "hard_candidate_set_id",
    "hard_candidate_members", "children",
}
REFINEMENT_TRACE_CHILD_FIELDS = {
    "mode", "artifact_id", "parent_artifact_id", "work_id", "recipe_id",
    "ancestry", "group_sizes", "packing", "package_artifact_id",
    "trits_changed", "allocations_changed", "reload_samples",
    "reload_max_abs_error", "reload_tolerance", "latent_residuals",
    "validation_loss_before", "validation_loss_after",
}
REFINEMENT_FIELDS = COMMON_FIELDS | {
    "parent_artifact_id", "training_set_id", "validation_set_id", "splits_disjoint",
    "children", "trace",
}
CHILD_FIELDS = {
    "mode", "artifact", "parent_artifact_id", "work_id", "recipe_id", "ancestry_id",
    "trits_frozen", "allocation_frozen", "hard_candidates_held_out", "g128_aligned",
    "native_salt_package", "strict_reload", "latent_residuals",
}
ABLATION_FIELDS = COMMON_FIELDS | {
    "model_artifact_id", "evaluation_id", "baseline_set_id", "inventory_complete",
    "baselines", "trace",
}
BASELINE_FIELDS = {
    "method", "family", "artifact_bytes", "target_bytes", "rate_gap_bpw",
    "quality_score", "runtime_ms", "resident_bytes", "reproduced", "same_box",
    "publishable_recipe", "eligible_for_claim",
}
ABLATION_TRACE_SCHEMA = "tritium.baseline-ablation-execution.v1"
ABLATION_TRACE_FIELDS = {
    "schema", "result", "release", "source_revision", "run_id", "environment",
    "model_artifact_id", "evaluation_id", "baseline_set_id", "target_bytes",
    "target_bpw", "baselines",
}
ABLATION_TRACE_BASELINE_FIELDS = {
    "method", "family", "recipe", "build_command", "evaluation_command",
    "artifact", "recipe_id", "artifact_bytes", "artifact_sha256",
    "parameter_count", "quality_score", "elapsed_samples_ms",
    "resident_samples_bytes", "physical_device", "reproduced",
    "publishable_recipe",
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
MAX_RELOAD_TOLERANCE = 1e-4


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


def command(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise EstimatorRefinementError(f"{label} must be a nonempty argv list")
    result = [string(item, f"{label} argument") for item in value]
    if any("\0" in item for item in result):
        raise EstimatorRefinementError(f"{label} arguments may not contain NUL")
    return result


def recipe_scope(raw: dict[str, Any]) -> dict[str, Any]:
    recipe = raw["recipe"]
    if not isinstance(recipe, dict) or not recipe:
        raise EstimatorRefinementError(
            "baseline trace recipe must be a nonempty JSON object"
        )
    try:
        json.dumps(
            recipe, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as error:
        raise EstimatorRefinementError(
            "baseline trace recipe must be canonical JSON"
        ) from error
    artifact = string(raw["artifact"], "baseline recipe artifact")
    if "\0" in artifact:
        raise EstimatorRefinementError(
            "baseline recipe artifact path may not contain NUL"
        )
    logical = PurePosixPath(artifact)
    if logical.is_absolute() or ".." in logical.parts or "\\" in artifact:
        raise EstimatorRefinementError("baseline recipe artifact path is unsafe")
    return {
        "method": raw["method"],
        "family": raw["family"],
        "recipe": recipe,
        "build_command": command(raw["build_command"], "baseline build command"),
        "evaluation_command": command(
            raw["evaluation_command"], "baseline evaluation command"
        ),
        "artifact": artifact,
    }


def number(value: Any, label: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise EstimatorRefinementError(f"{label} must be finite and at least {minimum}")
    return float(value)


def integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EstimatorRefinementError(f"{label} must be an integer at least {minimum}")
    return value


def ledger_id(members: Any, label: str) -> tuple[str, tuple[str, ...]]:
    if not isinstance(members, list) or not members:
        raise EstimatorRefinementError(f"{label} ledger must be non-empty")
    values = tuple(digest(value, f"{label} member") for value in members)
    if len(set(values)) != len(values):
        raise EstimatorRefinementError(f"{label} ledger contains duplicate members")
    value = "sha256:" + hashlib.sha256(canonical(list(values))).hexdigest()
    return value, values


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
    trace_record = object_(receipt["trace"], FILE_FIELDS, "estimator trace")
    trace_path = contained(path.parent, trace_record["path"])
    if (
        trace_path.stat().st_size <= 0
        or trace_path.stat().st_size > MAX_RECEIPT_BYTES
        or trace_path.stat().st_size != trace_record["bytes"]
        or sha256(trace_path) != trace_record["sha256"]
    ):
        raise EstimatorRefinementError("estimator trace bytes drifted")
    try:
        trace = object_(json.loads(trace_path.read_bytes()), TRACE_FIELDS, "estimator trace")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EstimatorRefinementError("estimator trace must contain UTF-8 JSON") from error
    wheel = object_(trace["wheel"], TRACE_WHEEL_FIELDS, "trace wheel")
    environment = object_(trace["environment"], ENVIRONMENT_FIELDS, "trace environment")
    if (
        trace["schema"] != TRACE_SCHEMA
        or trace["result"] != "pass"
        or trace["release"] != release
        or trace["source_revision"] != revision
        or trace["run_id"] != receipt["run_id"]
        or wheel != {
            "name": receipt["anchor_artifact"]["name"],
            "bytes": receipt["anchor_artifact"]["bytes"],
            "sha256": receipt["anchor_artifact"]["sha256"],
        }
        or trace["estimators"] != cases
        or trace["external_plugin"] != plugin
        or environment["device"] != "cpu"
        or environment["tritium"] != release.replace("-rc.", "rc")
    ):
        raise EstimatorRefinementError("estimator trace identity or results differ")
    for field in ("python", "torch"):
        string(environment[field], f"trace environment {field}")
    return finish(receipt, "estimator")


def derive_refinement_trace(
    trace_path: Path,
    *,
    receipt: dict[str, Any],
    artifacts: dict[str, tuple[Any, ...]],
    revision: str,
    release: str,
) -> dict[str, Any]:
    try:
        trace = object_(
            json.loads(trace_path.read_bytes()),
            REFINEMENT_TRACE_FIELDS,
            "refinement trace",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EstimatorRefinementError(
            "refinement trace must contain UTF-8 JSON"
        ) from error
    environment = object_(
        trace["environment"], ENVIRONMENT_FIELDS, "refinement environment"
    )
    if (
        trace["schema"] != REFINEMENT_TRACE_SCHEMA
        or trace["result"] != "pass"
        or trace["release"] != release
        or trace["source_revision"] != revision
        or trace["run_id"] != receipt["run_id"]
        or trace["parent_artifact_id"] != receipt["parent_artifact_id"]
        or environment["tritium"] != release.replace("-rc.", "rc")
    ):
        raise EstimatorRefinementError("refinement trace identity differs")
    for field in ("python", "torch", "device"):
        string(environment[field], f"refinement environment {field}")

    training_id, training = ledger_id(trace["training_members"], "training")
    validation_id, validation = ledger_id(trace["validation_members"], "validation")
    hard_id, hard_candidates = ledger_id(
        trace["hard_candidate_members"], "hard candidate"
    )
    if (
        trace["training_set_id"] != training_id
        or trace["validation_set_id"] != validation_id
        or trace["hard_candidate_set_id"] != hard_id
        or set(training) & set(validation)
        or set(hard_candidates) & (set(training) | set(validation))
    ):
        raise EstimatorRefinementError(
            "refinement data ledgers overlap or differ from their identities"
        )

    raw_children = trace["children"]
    if not isinstance(raw_children, list) or len(raw_children) != len(CHILDREN):
        raise EstimatorRefinementError("refinement trace requires all three children")
    children = []
    parent = receipt["parent_artifact_id"]
    for ordinal, mode in enumerate(CHILDREN):
        raw = object_(
            raw_children[ordinal],
            REFINEMENT_TRACE_CHILD_FIELDS,
            f"refinement trace children[{ordinal}]",
        )
        if (
            raw["mode"] != mode
            or raw["parent_artifact_id"] != parent
            or artifacts.get(raw["artifact_id"], (None, None))[1] != "model-bundle"
        ):
            raise EstimatorRefinementError("refinement trace child lineage differs")
        artifact = artifacts[raw["artifact_id"]]
        for field in ("work_id", "recipe_id", "package_artifact_id"):
            digest(raw[field], f"refinement child {field}")
        ancestry = raw["ancestry"]
        if (
            not isinstance(ancestry, list)
            or not ancestry
            or any(not isinstance(value, str) or not value for value in ancestry)
            or ancestry[-1] != parent
        ):
            raise EstimatorRefinementError("refinement child ancestry is incomplete")
        group_sizes = raw["group_sizes"]
        if (
            not isinstance(group_sizes, list)
            or not group_sizes
            or any(integer(value, "refinement group size", 1) != 128 for value in group_sizes)
        ):
            raise EstimatorRefinementError("refinement child is not G128 aligned")
        expected_packing = "s34" if mode == "s34" else "b3"
        if raw["packing"] != expected_packing:
            raise EstimatorRefinementError("refinement package packing differs")
        trits_changed = integer(raw["trits_changed"], "changed trits")
        allocations_changed = integer(
            raw["allocations_changed"], "changed allocations"
        )
        if (
            (mode == "scale-only" and (trits_changed != 0 or allocations_changed != 0))
            or (mode == "hard-pv" and trits_changed == 0)
            or (mode == "s34" and (trits_changed == 0 or allocations_changed == 0))
        ):
            raise EstimatorRefinementError("refinement hard-structure delta differs")
        reload_samples = integer(raw["reload_samples"], "reload samples", 32)
        reload_error = number(raw["reload_max_abs_error"], "reload error")
        reload_tolerance = number(raw["reload_tolerance"], "reload tolerance")
        before = number(raw["validation_loss_before"], "validation loss before")
        after = number(raw["validation_loss_after"], "validation loss after")
        latent_residuals = integer(raw["latent_residuals"], "latent residuals")
        if (
            reload_tolerance > MAX_RELOAD_TOLERANCE
            or reload_error > reload_tolerance
            or after > before
            or latent_residuals != 0
        ):
            raise EstimatorRefinementError(
                "refinement reload, validation, or hard-artifact gate failed"
            )
        children.append(
            {
                "mode": mode,
                "artifact": {
                    "id": artifact[0], "kind": artifact[1], "name": artifact[2],
                    "bytes": artifact[3], "sha256": artifact[4],
                },
                "parent_artifact_id": parent,
                "work_id": raw["work_id"],
                "recipe_id": raw["recipe_id"],
                "ancestry_id": "sha256:"
                + hashlib.sha256(canonical(ancestry)).hexdigest(),
                "trits_frozen": trits_changed == 0,
                "allocation_frozen": allocations_changed == 0,
                "hard_candidates_held_out": True,
                "g128_aligned": True,
                "native_salt_package": True,
                "strict_reload": reload_samples >= 32 and reload_error <= reload_tolerance,
                "latent_residuals": latent_residuals,
            }
        )
    return {
        "training_set_id": training_id,
        "validation_set_id": validation_id,
        "splits_disjoint": True,
        "children": children,
    }


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
    trace_record = object_(receipt["trace"], FILE_FIELDS, "refinement trace")
    trace_path = contained(path.parent, trace_record["path"])
    if (
        trace_path.stat().st_size <= 0
        or trace_path.stat().st_size > MAX_RECEIPT_BYTES
        or trace_path.stat().st_size != trace_record["bytes"]
        or sha256(trace_path) != trace_record["sha256"]
    ):
        raise EstimatorRefinementError("refinement trace bytes drifted")
    derived = derive_refinement_trace(
        trace_path,
        receipt=receipt,
        artifacts=artifacts,
        revision=revision,
        release=release,
    )
    if any(receipt[field] != derived[field] for field in derived):
        raise EstimatorRefinementError("refinement receipt differs from raw trace")
    return finish(receipt, "refinement")


def derive_ablation_trace(
    trace_path: Path,
    *,
    receipt: dict[str, Any],
    revision: str,
    release: str,
) -> dict[str, Any]:
    try:
        trace = object_(
            json.loads(trace_path.read_bytes()),
            ABLATION_TRACE_FIELDS,
            "baseline ablation trace",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EstimatorRefinementError(
            "baseline ablation trace must contain UTF-8 JSON"
        ) from error
    environment = object_(
        trace["environment"], ENVIRONMENT_FIELDS, "baseline environment"
    )
    if (
        trace["schema"] != ABLATION_TRACE_SCHEMA
        or trace["result"] != "pass"
        or trace["release"] != release
        or trace["source_revision"] != revision
        or trace["run_id"] != receipt["run_id"]
        or trace["model_artifact_id"] != receipt["model_artifact_id"]
        or environment["tritium"] != release.replace("-rc.", "rc")
    ):
        raise EstimatorRefinementError("baseline ablation trace identity differs")
    for field in ("python", "torch", "device"):
        string(environment[field], f"baseline environment {field}")
    evaluation_id = digest(trace["evaluation_id"], "baseline evaluation id")
    target_bytes = integer(trace["target_bytes"], "baseline target bytes", 1)
    target_bpw = number(trace["target_bpw"], "baseline target bpw", 1e-12)
    raw_baselines = trace["baselines"]
    if not isinstance(raw_baselines, list) or len(raw_baselines) != len(BASELINES):
        raise EstimatorRefinementError("baseline trace inventory is incomplete")
    baselines = []
    set_identity = []
    shared_parameter_count = None
    for ordinal, expected in enumerate(BASELINES):
        raw = object_(
            raw_baselines[ordinal],
            ABLATION_TRACE_BASELINE_FIELDS,
            f"baseline trace baselines[{ordinal}]",
        )
        if (raw["method"], raw["family"]) != expected:
            raise EstimatorRefinementError("baseline trace identity/order differs")
        recipe_id = digest(raw["recipe_id"], "baseline recipe id")
        expected_recipe_id = "sha256:" + hashlib.sha256(
            canonical(recipe_scope(raw))
        ).hexdigest()
        if recipe_id != expected_recipe_id:
            raise EstimatorRefinementError(
                "baseline recipe identity differs from retained recipe"
            )
        artifact_bytes = integer(raw["artifact_bytes"], "baseline artifact bytes", 1)
        artifact_sha256 = string(
            raw["artifact_sha256"], "baseline artifact SHA-256"
        )
        if re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None:
            raise EstimatorRefinementError(
                "baseline artifact SHA-256 must be lowercase hexadecimal"
            )
        parameter_count = integer(
            raw["parameter_count"], "baseline parameter count", 1
        )
        if shared_parameter_count is None:
            shared_parameter_count = parameter_count
        elif parameter_count != shared_parameter_count:
            raise EstimatorRefinementError("baseline parameter inventories differ")
        quality_score = number(raw["quality_score"], "baseline quality score", 1e-12)
        elapsed = raw["elapsed_samples_ms"]
        resident = raw["resident_samples_bytes"]
        if (
            not isinstance(elapsed, list)
            or len(elapsed) != 30
            or not isinstance(resident, list)
            or len(resident) != len(elapsed)
        ):
            raise EstimatorRefinementError(
                "baseline trace requires exactly thirty matched timing and residency samples"
            )
        elapsed_values = [
            number(value, "baseline elapsed sample", 1e-12) for value in elapsed
        ]
        resident_values = [
            integer(value, "baseline resident sample", 1) for value in resident
        ]
        if (
            raw["physical_device"] != environment["device"]
            or raw["reproduced"] is not True
            or raw["publishable_recipe"] is not True
        ):
            raise EstimatorRefinementError(
                "baseline trace is not reproduced on the frozen device"
            )
        rate_gap = abs(artifact_bytes * 8.0 / parameter_count - target_bpw)
        eligible = rate_gap <= 0.05 and artifact_bytes <= target_bytes
        baselines.append(
            {
                "method": raw["method"], "family": raw["family"],
                "artifact_bytes": artifact_bytes, "target_bytes": target_bytes,
                "rate_gap_bpw": rate_gap, "quality_score": quality_score,
                "runtime_ms": statistics.median(elapsed_values),
                "resident_bytes": max(resident_values),
                "reproduced": True, "same_box": True,
                "publishable_recipe": True, "eligible_for_claim": eligible,
            }
        )
        set_identity.append(
            {"method": raw["method"], "family": raw["family"], "recipe_id": recipe_id}
        )
    if not math.isclose(
        target_bpw,
        target_bytes * 8.0 / shared_parameter_count,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise EstimatorRefinementError("baseline byte and bpw targets differ")
    set_scope = {
        "model_artifact_id": trace["model_artifact_id"],
        "evaluation_id": evaluation_id,
        "target_bytes": target_bytes,
        "target_bpw": target_bpw,
        "recipes": set_identity,
    }
    baseline_set_id = "sha256:" + hashlib.sha256(canonical(set_scope)).hexdigest()
    if trace["baseline_set_id"] != baseline_set_id:
        raise EstimatorRefinementError("baseline set identity differs from recipes")
    return {
        "evaluation_id": evaluation_id,
        "baseline_set_id": baseline_set_id,
        "inventory_complete": True,
        "baselines": baselines,
    }


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
    trace_record = object_(receipt["trace"], FILE_FIELDS, "baseline ablation trace")
    trace_path = contained(path.parent, trace_record["path"])
    if (
        trace_path.stat().st_size <= 0
        or trace_path.stat().st_size > MAX_RECEIPT_BYTES
        or trace_path.stat().st_size != trace_record["bytes"]
        or sha256(trace_path) != trace_record["sha256"]
    ):
        raise EstimatorRefinementError("baseline ablation trace bytes drifted")
    derived = derive_ablation_trace(
        trace_path,
        receipt=receipt,
        revision=revision,
        release=release,
    )
    if any(receipt[field] != derived[field] for field in derived):
        raise EstimatorRefinementError("baseline ablation receipt differs from raw trace")
    return finish(receipt, "baseline-ablation")
