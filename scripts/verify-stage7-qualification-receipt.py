#!/usr/bin/env python3
"""Validate an authorized, candidate-bound Stage-7 recipe-freeze receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


SCHEMA = "tritium.stage7-qualification.v1"
MODEL_ID = "sha256:4be74d32a1a04f2984e9d118fdb165dd8cfbe972710796ab465a4c2152d58a08"
MODEL_REVISION = "effd688a12921b4cc83e3312b6feb579f70f9c71"
SMOKE_MODEL_ID = "sha256:18686427230dde98ee2926dafa133b5cb0c6f4de48eacd0a57e5d2ed76e15e57"
SMOKE_MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
FILE_FIELDS = {"path", "bytes", "sha256"}
CHECKPOINT_FIELDS = {
    "refinement_id", "tokens", "package_id", "artifact_sha256", "evaluation_sha256"
}
SOFT_AB_FIELDS = {
    "outcome", "winner", "ste_refinement_id", "hestia_refinement_id"
}
FIELDS = {
    "schema", "result", "release", "source_revision", "run_id", "model_id",
    "model_revision", "candidate_manifest_sha256", "campaign", "trace",
    "freeze_authorized", "freeze_reasons",
    "frozen_ptq_recipe_ids", "r4_control_recipe_id", "frozen_refined_recipe_id",
    "frozen_refined_checkpoint", "soft_method_ab", "receipt_id",
}
MAX_BYTES = 32 * 1024 * 1024
CAMPAIGN_FIELDS = {
    "schema", "release", "source_revision", "run_id", "model", "smoke_model",
    "smoke_provenance", "provenance", "thresholds", "recipe_count",
    "recipe_grid_id", "token_evidence_pack", "evidence",
}
MODEL_FIELDS = {"repo_id", "revision", "files"}
MODEL_FILE_NAMES = {
    ".gitattributes", "README.md", "config.json", "generation_config.json",
    "merges.txt", "model.safetensors", "special_tokens_map.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json",
}
PARTITION_FIELDS = {
    "id", "members", "datasets", "sampling_seed", "tokenizer_digest",
    "ordered_token_digest", "sequence_count", "tokens_per_sequence",
}
DATASET_FIELDS = {"repo_id", "revision", "fraction_ppm"}
DATASETS = (
    ("allenai/c4", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2", 500_000),
    ("open-web-math/open-web-math", "fde8ef8de2300f5e778f56261843dab89f230815", 250_000),
    ("bigcode/starcoderdata", "9fc30b578cedaec69e47302df72cf00feed7c8c4", 250_000),
)
TRACE_FIELDS = {
    "schema", "release", "source_revision", "run_id", "campaign_sha256",
    "stages", "baselines", "refinements",
}
STAGE_FIELDS = {"name", "input_ids", "measurements", "promoted_ids"}
MEASUREMENT_FIELDS = {
    "candidate_id", "track", "physical_bytes", "resident_bytes", "output_loss",
    "heldout_ppl", "task_metrics", "runtime_ms", "artifact", "physical_report",
    "correct",
}
REFINEMENT_FIELDS = {
    "refinement_id", "mode", "parent_candidate_id", "rate", "soft_method",
    "soft_policy", "refinement_corpus_id", "validation_id",
    "parent_validation_ppl", "checkpoints",
}
TRACE_CHECKPOINT_FIELDS = {
    "tokens", "validation_ppl", "teacher_kl", "artifact", "package_id",
    "codec", "serialized_bytes", "resident_bytes", "tensor_count",
    "trits_changed", "hard_reload_max_abs_error", "hard_reload_tolerance",
    "evaluation_receipt",
}
BASELINE_FIELDS = {"bf16", "salt_v1"}


def _raw_digest(value: Any, label: str) -> str:
    text = _string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise Stage7QualificationError(f"{label} must be a raw SHA-256 digest")
    return text


def _ids(value: Any, label: str, *, count: int | None = None) -> list[str]:
    if not isinstance(value, list):
        raise Stage7QualificationError(f"{label} must be an array")
    result = [_digest(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise Stage7QualificationError(f"{label} contains duplicate ids")
    if count is not None and len(result) != count:
        raise Stage7QualificationError(f"{label} must contain exactly {count} ids")
    return result


def _bounded_int(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0):
        raise Stage7QualificationError(f"{label} must be {'positive ' if positive else ''}integer")
    return value


def _file_shape(value: Any, label: str) -> None:
    record = _object(value, FILE_FIELDS, label)
    logical = PurePosixPath(_string(record["path"], f"{label}.path"))
    if (
        logical.is_absolute() or not logical.parts or ".." in logical.parts
        or "\\" in logical.as_posix() or logical.as_posix() != record["path"]
    ):
        raise Stage7QualificationError(f"{label}.path is unsafe")
    _bounded_int(record["bytes"], f"{label}.bytes", positive=True)
    _raw_digest(record["sha256"], f"{label}.sha256")


def _load_canonical_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage7QualificationError(f"{label} must contain UTF-8 JSON") from error
    if raw != canonical(value) + b"\n" or not isinstance(value, dict):
        raise Stage7QualificationError(f"{label} must be canonical JSON object")
    return value


def _envelope(path: Path, label: str, schema: str, receipt: dict[str, Any]) -> None:
    value = _load_canonical_json(path, label)
    if (
        value.get("schema") != schema
        or value.get("result") != "pass"
        or value.get("release") != receipt["release"]
        or value.get("source_revision") != receipt["source_revision"]
    ):
        raise Stage7QualificationError(f"{label} envelope differs")


def _validate_campaign_payload(
    campaign: Any, receipt: dict[str, Any], evidence_root: Path
) -> None:
    campaign = _object(campaign, CAMPAIGN_FIELDS, "campaign evidence")
    if campaign["schema"] != "tritium.stage7-campaign.v1":
        raise Stage7QualificationError("campaign schema mismatch")
    for field in ("release", "source_revision", "run_id"):
        if campaign[field] != receipt[field]:
            raise Stage7QualificationError(f"campaign {field} differs from receipt")
    _revision(campaign["source_revision"], "campaign source revision")
    for key, repo, revision in (
        ("model", "HuggingFaceTB/SmolLM2-1.7B", MODEL_REVISION),
        ("smoke_model", "HuggingFaceTB/SmolLM2-135M", SMOKE_MODEL_REVISION),
    ):
        model = _object(campaign[key], MODEL_FIELDS, f"campaign.{key}")
        if model["repo_id"] != repo or model["revision"] != revision:
            raise Stage7QualificationError(f"campaign.{key} identity differs")
        if not isinstance(model["files"], list) or {
            PurePosixPath(_string(item.get("path"), f"campaign.{key}.file.path")).name
            for item in model["files"] if isinstance(item, dict)
        } != MODEL_FILE_NAMES or len(model["files"]) != len(MODEL_FILE_NAMES):
            raise Stage7QualificationError(f"campaign.{key}.files inventory differs")
        for index, item in enumerate(model["files"]):
            _file_shape(item, f"campaign.{key}.files[{index}]")
            if item["path"] not in MODEL_FILE_NAMES:
                raise Stage7QualificationError(f"campaign.{key}.file path differs")
        expected_model_id = MODEL_ID if key == "model" else SMOKE_MODEL_ID
        if "sha256:" + hashlib.sha256(canonical(model)).hexdigest() != expected_model_id:
            raise Stage7QualificationError(f"campaign.{key} file identity differs")
    smoke = _object(campaign["smoke_provenance"], {
        "evaluation_id", "evaluation_members", "calibration_id", "dataset_repo_id",
        "dataset_revision", "sampling_seed", "tokenizer_digest", "ordered_token_digest",
        "sequence_count", "tokens_per_sequence", "prefix_start", "prefix_end",
    }, "campaign.smoke_provenance")
    _ids(smoke["evaluation_members"], "smoke evaluation members", count=128)
    if smoke["sequence_count"] != 128 or smoke["tokens_per_sequence"] != 2048:
        raise Stage7QualificationError("smoke provenance geometry differs")
    _digest(smoke["evaluation_id"], "smoke evaluation id")
    _digest(smoke["calibration_id"], "smoke calibration id")
    _digest(smoke["ordered_token_digest"], "smoke ordered token digest")
    _revision(smoke["dataset_revision"], "smoke dataset revision")
    _bounded_int(smoke["sampling_seed"], "smoke sampling seed")
    _digest(smoke["tokenizer_digest"], "smoke tokenizer digest")
    if smoke["evaluation_id"] != "sha256:" + hashlib.sha256(
        canonical(smoke["evaluation_members"])
    ).hexdigest() or smoke["ordered_token_digest"] != smoke["evaluation_id"]:
        raise Stage7QualificationError("smoke evaluation ids do not bind ordered members")
    if smoke["prefix_start"] != 0 or smoke["prefix_end"] != 128:
        raise Stage7QualificationError("smoke provenance prefix differs")
    provenance = _object(campaign["provenance"], {"calibration", "refinement", "validation", "evaluation"}, "campaign.provenance")
    for name, partition in provenance.items():
        partition = _object(partition, PARTITION_FIELDS, f"campaign.provenance.{name}")
        members = _ids(partition["members"], f"{name} members", count=512)
        if partition["sequence_count"] != 512 or partition["tokens_per_sequence"] != 2048:
            raise Stage7QualificationError(f"{name} provenance geometry differs")
        _bounded_int(partition["sampling_seed"], f"{name} sampling seed")
        _digest(partition["id"], f"{name} provenance id")
        _digest(partition["ordered_token_digest"], f"{name} ordered token digest")
        _digest(partition["tokenizer_digest"], f"{name} tokenizer digest")
        if partition["ordered_token_digest"] != "sha256:" + hashlib.sha256(
            canonical(members)
        ).hexdigest():
            raise Stage7QualificationError(f"{name} ordered digest differs from members")
        datasets = partition["datasets"]
        if not isinstance(datasets, list) or len(datasets) != len(DATASETS):
            raise Stage7QualificationError(f"{name} dataset inventory differs")
        for index, (repo, revision, fraction) in enumerate(DATASETS):
            dataset = _object(datasets[index], DATASET_FIELDS, f"{name}.datasets[{index}]")
            if dataset != {"repo_id": repo, "revision": revision, "fraction_ppm": fraction}:
                raise Stage7QualificationError(f"{name} dataset provenance differs")
        if len(set(members)) != 512:
            raise Stage7QualificationError(f"{name} provenance members are not unique")
    thresholds = _object(campaign["thresholds"], {
        "r3_gap_closure_min", "metadata_bpw_max", "scale_only_token_cap", "short_pv_token_cap",
    }, "campaign.thresholds")
    if thresholds != {"r3_gap_closure_min": 0.25, "metadata_bpw_max": 0.01, "scale_only_token_cap": 8_000_000, "short_pv_token_cap": 32_000_000}:
        raise Stage7QualificationError("campaign thresholds differ")
    if campaign["recipe_count"] != 1404:
        raise Stage7QualificationError("campaign recipe count differs")
    _digest(campaign["recipe_grid_id"], "campaign recipe grid id")
    _file_shape(campaign["token_evidence_pack"], "campaign token evidence pack")
    token_pack_path = _record(
        evidence_root, campaign["token_evidence_pack"], "campaign token evidence pack"
    )
    token_pack = _load_canonical_json(token_pack_path, "campaign token evidence pack")
    if set(token_pack) != {
        "schema", "pack_id", "tokenizer_digest", "tokenizer_vocab_size",
        "token_encoding", "tokens", "partitions",
    } or token_pack["schema"] != "tritium.stage7-token-evidence-pack.v1":
        raise Stage7QualificationError("campaign token evidence pack envelope differs")
    _digest(token_pack["pack_id"], "campaign token evidence pack.id")
    _digest(token_pack["tokenizer_digest"], "campaign token evidence pack.tokenizer_digest")
    _bounded_int(
        token_pack["tokenizer_vocab_size"],
        "campaign token evidence pack.tokenizer_vocab_size",
        positive=True,
    )
    if token_pack["token_encoding"] != "u32le":
        raise Stage7QualificationError("campaign token evidence pack encoding differs")
    _record(token_pack_path.parent, token_pack["tokens"], "campaign token evidence payload")
    partitions = token_pack["partitions"]
    if not isinstance(partitions, dict) or set(partitions) != {
        "calibration", "refinement", "validation", "evaluation",
    }:
        raise Stage7QualificationError("campaign token evidence partitions differ")
    for name, partition in partitions.items():
        if not isinstance(partition, dict) or set(partition) != {"sampling_seed", "sequences"}:
            raise Stage7QualificationError(
                f"campaign token evidence {name} partition differs"
            )
        _bounded_int(partition["sampling_seed"], f"campaign token evidence {name}.sampling_seed")
        if not isinstance(partition["sequences"], list) or len(partition["sequences"]) != 512:
            raise Stage7QualificationError(
                f"campaign token evidence {name} sequence inventory differs"
            )
    evidence = campaign["evidence"]
    if not isinstance(evidence, list) or len(evidence) != 3:
        raise Stage7QualificationError("campaign prerequisite evidence inventory differs")
    for index, kind in enumerate(("smoke", "native-kernels", "hestia-gate-c")):
        record = _object(evidence[index], FILE_FIELDS | {"kind"}, f"campaign.evidence[{index}]")
        if record["kind"] != kind:
            raise Stage7QualificationError("campaign prerequisite evidence order differs")
        _file_shape({field: record[field] for field in FILE_FIELDS}, f"campaign.evidence[{index}]")
        evidence_path = _record(
            evidence_root,
            {field: record[field] for field in FILE_FIELDS},
            f"campaign.evidence[{index}]",
        )
        _envelope(
            evidence_path,
            f"campaign.evidence[{index}]",
            {
                "smoke": "tritium.stage7-smoke.v2",
                "native-kernels": "tritium.stage7-native-kernels.v1",
                "hestia-gate-c": "tritium.stage7-hestia-gate-c.v1",
            }[kind],
            receipt,
        )


def _validate_trace_payload(
    trace: Any, receipt: dict[str, Any], campaign_path: Path, evidence_root: Path
) -> dict[str, Any]:
    trace = _object(trace, TRACE_FIELDS, "trace evidence")
    if trace["schema"] != "tritium.stage7-execution.v1":
        raise Stage7QualificationError("trace schema mismatch")
    for field in ("release", "source_revision", "run_id"):
        if trace[field] != receipt[field]:
            raise Stage7QualificationError(f"trace {field} differs from receipt")
    if trace["campaign_sha256"] != _sha256(campaign_path):
        raise Stage7QualificationError("trace does not bind campaign bytes")
    stages = trace["stages"]
    if not isinstance(stages, list) or len(stages) != 3:
        raise Stage7QualificationError("trace must contain three stages")
    full_promoted: set[str] = set()
    previous_promoted: set[str] | None = None
    refinement_ids: set[str] = set()
    refinement_checkpoints: dict[str, list[Any]] = {}
    for index, (name, required_count) in enumerate((("one-layer", 1404), ("four-layer", None), ("full-model", None))):
        stage = _object(stages[index], STAGE_FIELDS, f"trace.stages[{index}]")
        if stage["name"] != name:
            raise Stage7QualificationError("trace stage order differs")
        inputs = _ids(stage["input_ids"], f"{name} inputs", count=required_count)
        measurements = stage["measurements"]
        if not isinstance(measurements, list) or len(measurements) != len(inputs) or not measurements:
            raise Stage7QualificationError(f"{name} measurement coverage differs")
        seen: set[str] = set()
        for row_index, raw in enumerate(measurements):
            row = _object(raw, MEASUREMENT_FIELDS, f"{name}.measurements[{row_index}]")
            candidate_id = _digest(row["candidate_id"], f"{name}.candidate_id")
            if candidate_id not in inputs or candidate_id in seen or row["track"] != "ptq" or type(row["correct"]) is not bool:
                raise Stage7QualificationError(f"{name} measurement identity differs")
            _bounded_int(row["physical_bytes"], f"{name}.physical_bytes", positive=True)
            _bounded_int(row["resident_bytes"], f"{name}.resident_bytes", positive=True)
            if not isinstance(row["output_loss"], (int, float)) or isinstance(row["output_loss"], bool):
                raise Stage7QualificationError(f"{name}.output_loss must be numeric")
            if not isinstance(row["runtime_ms"], (int, float)) or isinstance(row["runtime_ms"], bool):
                raise Stage7QualificationError(f"{name}.runtime_ms must be numeric")
            if name != "full-model":
                if row["heldout_ppl"] is not None or row["task_metrics"] != {} or row["artifact"] is not None or row["physical_report"] is not None:
                    raise Stage7QualificationError(f"{name} leaks full-model evidence")
            else:
                if not isinstance(row["heldout_ppl"], (int, float)) or isinstance(row["heldout_ppl"], bool) or not isinstance(row["task_metrics"], dict) or not row["artifact"] or not row["physical_report"]:
                    raise Stage7QualificationError("full-model measurement lacks quality or physical evidence")
                _record(evidence_root, row["artifact"], f"{name}.artifact")
                physical_path = _record(
                    evidence_root, row["physical_report"], f"{name}.physical_report"
                )
                _envelope(
                    physical_path, f"{name}.physical_report",
                    "tritium.stage7-physical-report.v1", receipt,
                )
            seen.add(candidate_id)
        if set(seen) != set(inputs):
            raise Stage7QualificationError(f"{name} measurement coverage is incomplete")
        promoted = set(_ids(stage["promoted_ids"], f"{name} promotions"))
        if not promoted or not promoted <= seen:
            raise Stage7QualificationError(f"{name} promotions are not measured")
        if previous_promoted is not None and set(inputs) != previous_promoted:
            raise Stage7QualificationError(f"{name} inputs do not equal prior promotions")
        previous_promoted = promoted
        if name == "full-model":
            full_promoted = promoted
    baselines = _object(trace["baselines"], BASELINE_FIELDS, "trace.baselines")
    bf16 = _object(baselines["bf16"], {"heldout_ppl", "task_metrics"}, "trace.baselines.bf16")
    if not isinstance(bf16["heldout_ppl"], (int, float)) or not isinstance(bf16["task_metrics"], dict):
        raise Stage7QualificationError("bf16 baseline metrics are malformed")
    salt = baselines["salt_v1"]
    if not isinstance(salt, list) or len(salt) != 3:
        raise Stage7QualificationError("SALT baseline inventory differs")
    for index, row in enumerate(salt):
        if not isinstance(row, dict) or row.get("rate") not in ("R2", "R3", "R4") or not row.get("artifact") or not row.get("physical_report"):
            raise Stage7QualificationError(f"SALT baseline[{index}] is incomplete")
        _record(evidence_root, row["artifact"], f"SALT baseline[{index}].artifact")
        physical_path = _record(
            evidence_root, row["physical_report"], f"SALT baseline[{index}].physical_report"
        )
        _envelope(
            physical_path, f"SALT baseline[{index}].physical_report",
            "tritium.stage7-physical-report.v1", receipt,
        )
    refinements = trace["refinements"]
    if not isinstance(refinements, list) or len(refinements) != 5:
        raise Stage7QualificationError("refinement inventory differs")
    expected_refinement_modes = ("scale-only", "scale-only", "scale-only", "short-pv", "short-pv")
    expected_rates = ("R2", "R3", "R4", "R2", "R3")
    for index, row in enumerate(refinements):
        row = _object(row, REFINEMENT_FIELDS, f"trace.refinements[{index}]")
        refinement_id = _digest(row["refinement_id"], f"refinement[{index}].id")
        parent = _digest(row["parent_candidate_id"], f"refinement[{index}].parent")
        if parent not in full_promoted:
            raise Stage7QualificationError("refinement parent is not a full-model promotion")
        checkpoints = row["checkpoints"]
        if not isinstance(checkpoints, list) or len(checkpoints) not in (3, 4):
            raise Stage7QualificationError("refinement checkpoint curve differs")
        if row["mode"] != expected_refinement_modes[index] or row["rate"] != expected_rates[index]:
            raise Stage7QualificationError("refinement mode or rate inventory differs")
        refinement_ids.add(refinement_id)
        refinement_checkpoints[refinement_id] = checkpoints
    return {
        "full_promoted": full_promoted,
        "refinement_ids": refinement_ids,
        "refinement_checkpoints": refinement_checkpoints,
    }


class Stage7QualificationError(ValueError):
    """Stage-7 freeze evidence is stale, incomplete, or unauthorized."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise Stage7QualificationError(f"{label} fields do not match frozen schema")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise Stage7QualificationError(f"{label} must be non-empty")
    return value


def _digest(value: Any, label: str) -> str:
    text = _string(value, label)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise Stage7QualificationError(f"{label} must be a canonical SHA-256 digest")
    return text


def _revision(value: Any, label: str) -> str:
    text = _string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", text) is None:
        raise Stage7QualificationError(f"{label} must be a lowercase Git revision")
    return text


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise Stage7QualificationError("receipt must be a bounded ordinary file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage7QualificationError("receipt must contain UTF-8 JSON") from error
    if raw != canonical(value) + b"\n":
        raise Stage7QualificationError("receipt must be canonical JSON")
    return _object(value, FIELDS, "receipt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ordinary_candidate(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise Stage7QualificationError("candidate manifest must be an ordinary file")
    cursor = path.parent
    while True:
        if cursor.is_symlink():
            raise Stage7QualificationError("candidate manifest parent must not be a symlink")
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent


def _record(root: Path, value: Any, label: str) -> Path:
    record = _object(value, FILE_FIELDS, label)
    logical_text = _string(record["path"], f"{label}.path")
    logical = PurePosixPath(logical_text)
    if (
        logical.is_absolute()
        or ".." in logical.parts
        or "\\" in logical_text
        or logical.as_posix() != logical_text
    ):
        raise Stage7QualificationError(f"{label}.path is unsafe")
    root = root.resolve(strict=True)
    path = root
    for part in logical.parts:
        path = path / part
        if path.is_symlink():
            raise Stage7QualificationError(
                f"{label}.path traverses an intermediate symlink"
            )
    if not path.is_file():
        raise Stage7QualificationError(f"{label}.path must be an ordinary file")
    if isinstance(record["bytes"], bool) or not isinstance(record["bytes"], int) or record["bytes"] <= 0:
        raise Stage7QualificationError(f"{label}.bytes must be positive")
    digest = _digest("sha256:" + str(record["sha256"]), f"{label}.sha256")
    if path.stat().st_size != record["bytes"] or _sha256(path) != digest.removeprefix("sha256:"):
        raise Stage7QualificationError(f"{label} differs from opened file")
    return path


def _recipe_id(value: Any, label: str) -> str:
    return _digest(value, label)


def validate(receipt_path: Path, revision: str, release: str, candidate: Path) -> dict[str, Any]:
    receipt = _load(receipt_path)
    if receipt["schema"] != SCHEMA or receipt["result"] != "pass":
        raise Stage7QualificationError("receipt schema or result mismatch")
    if receipt["release"] != release or receipt["source_revision"] != revision:
        raise Stage7QualificationError("receipt source or release is stale")
    _revision(revision, "expected source revision")
    _string(receipt["run_id"], "receipt.run_id")
    if receipt["model_id"] != MODEL_ID or receipt["model_revision"] != MODEL_REVISION:
        raise Stage7QualificationError("receipt is not bound to frozen SmolLM2-1.7B")
    _ordinary_candidate(candidate)
    expected_candidate = _sha256(candidate)
    _raw_digest(receipt["candidate_manifest_sha256"], "receipt.candidate_manifest_sha256")
    if receipt["candidate_manifest_sha256"] != expected_candidate:
        raise Stage7QualificationError("receipt does not bind candidate manifest")
    if receipt["campaign"] is None or receipt["trace"] is None:
        raise Stage7QualificationError("campaign and trace evidence are required")
    campaign_path = _record(receipt_path.parent, receipt["campaign"], "campaign")
    trace_path = _record(receipt_path.parent, receipt["trace"], "trace")
    campaign = _load_canonical_json(campaign_path, "campaign")
    trace = _load_canonical_json(trace_path, "trace")
    _validate_campaign_payload(campaign, receipt, receipt_path.parent)
    trace_state = _validate_trace_payload(
        trace, receipt, campaign_path, receipt_path.parent
    )
    if receipt["freeze_authorized"] is not True or receipt["freeze_reasons"] != []:
        raise Stage7QualificationError("freeze is not authorized")
    selected = receipt["frozen_ptq_recipe_ids"]
    if not isinstance(selected, dict) or set(selected) != {"R2", "R3"}:
        raise Stage7QualificationError("publishable PTQ recipe selection is incomplete")
    for rate, value in selected.items():
        _recipe_id(value, f"frozen_ptq_recipe_ids.{rate}")
        if value not in trace_state["full_promoted"]:
            raise Stage7QualificationError(f"selected {rate} recipe is not a full-model promotion")
    _recipe_id(receipt["r4_control_recipe_id"], "r4_control_recipe_id")
    if receipt["r4_control_recipe_id"] not in trace_state["full_promoted"]:
        raise Stage7QualificationError("R4 control recipe is not a full-model promotion")
    _recipe_id(receipt["frozen_refined_recipe_id"], "frozen_refined_recipe_id")
    if receipt["frozen_refined_recipe_id"] not in trace_state["refinement_ids"]:
        raise Stage7QualificationError("frozen refinement is absent from trace")
    checkpoint = _object(receipt["frozen_refined_checkpoint"], CHECKPOINT_FIELDS, "frozen refined checkpoint")
    _recipe_id(checkpoint["refinement_id"], "checkpoint.refinement_id")
    if checkpoint["refinement_id"] != receipt["frozen_refined_recipe_id"]:
        raise Stage7QualificationError("checkpoint does not bind frozen refinement")
    if isinstance(checkpoint["tokens"], bool) or not isinstance(checkpoint["tokens"], int) or checkpoint["tokens"] <= 0:
        raise Stage7QualificationError("checkpoint.tokens must be positive")
    package_id = _string(checkpoint["package_id"], "checkpoint.package_id")
    if re.fullmatch(r"trp1_[0-9a-f]{64}", package_id) is None:
        raise Stage7QualificationError("checkpoint.package_id is malformed")
    _raw_digest(checkpoint["artifact_sha256"], "checkpoint.artifact_sha256")
    _raw_digest(checkpoint["evaluation_sha256"], "checkpoint.evaluation_sha256")
    selected_checkpoints = trace_state["refinement_checkpoints"].get(
        receipt["frozen_refined_recipe_id"], []
    )
    selected_trace_checkpoint = None
    for raw_checkpoint in selected_checkpoints:
        if isinstance(raw_checkpoint, dict) and raw_checkpoint.get("tokens") == checkpoint["tokens"]:
            selected_trace_checkpoint = raw_checkpoint
            break
    if selected_trace_checkpoint is None:
        raise Stage7QualificationError("frozen checkpoint is absent from trace")
    if set(selected_trace_checkpoint) != TRACE_CHECKPOINT_FIELDS:
        raise Stage7QualificationError("trace checkpoint fields differ")
    if selected_trace_checkpoint["package_id"] != checkpoint["package_id"]:
        raise Stage7QualificationError("frozen checkpoint package differs from trace")
    artifact_path = _record(receipt_path.parent, selected_trace_checkpoint["artifact"], "trace refinement artifact")
    evaluation_path = _record(
        receipt_path.parent,
        selected_trace_checkpoint["evaluation_receipt"],
        "trace refinement evaluation",
    )
    evaluation = _load_canonical_json(evaluation_path, "trace refinement evaluation")
    if (
        evaluation.get("tokens") != checkpoint["tokens"]
        or evaluation.get("package_id") != checkpoint["package_id"]
        or evaluation.get("artifact_sha256") != checkpoint["artifact_sha256"]
        or evaluation.get("result") != "pass"
    ):
        raise Stage7QualificationError("frozen checkpoint evaluation differs from trace")
    if _sha256(artifact_path) != checkpoint["artifact_sha256"]:
        raise Stage7QualificationError("frozen checkpoint artifact bytes differ from trace")
    if selected_trace_checkpoint["artifact"]["sha256"] != checkpoint["artifact_sha256"]:
        raise Stage7QualificationError("frozen checkpoint artifact differs from trace")
    if selected_trace_checkpoint["evaluation_receipt"]["sha256"] != checkpoint["evaluation_sha256"]:
        raise Stage7QualificationError("frozen checkpoint evaluation digest differs from trace")
    soft = _object(receipt["soft_method_ab"], SOFT_AB_FIELDS, "soft method A/B")
    _string(soft["outcome"], "soft method outcome")
    if soft["winner"] is not None:
        _string(soft["winner"], "soft method winner")
    _recipe_id(soft["ste_refinement_id"], "soft method STE id")
    _recipe_id(soft["hestia_refinement_id"], "soft method HESTIA id")
    if not {
        soft["ste_refinement_id"], soft["hestia_refinement_id"]
    } <= trace_state["refinement_ids"]:
        raise Stage7QualificationError("soft-method A/B ids are absent from trace")
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    if receipt_id != "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest():
        raise Stage7QualificationError("receipt identity mismatch")
    return receipt
