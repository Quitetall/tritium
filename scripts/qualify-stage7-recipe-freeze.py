#!/usr/bin/env python3
"""Qualify plan-0043 Stage-7 successive halving and terminal recipe freeze."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any, Iterable


CAMPAIGN_SCHEMA = "tritium.stage7-campaign.v1"
TRACE_SCHEMA = "tritium.stage7-execution.v1"
QUALIFICATION_SCHEMA = "tritium.stage7-qualification.v1"
SMOKE_SCHEMA = "tritium.stage7-smoke.v1"
SMOKE_EXECUTION_SCHEMA = "tritium.stage7-smoke-execution.v1"
NATIVE_SCHEMA = "tritium.stage7-native-kernels.v1"
PHYSICAL_SCHEMA = "tritium.stage7-physical-report.v1"
HESTIA_GATE_SCHEMA = "tritium.stage7-hestia-gate-c.v1"
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024
RATES = ("R2", "R3", "R4")
PUBLISHABLE_RATES = ("R2", "R3")
RATE_BPW = {"R2": 2.25, "R3": 3.50, "R4": 4.25}
RATE_PROFILE = {"R2": "CompactV1", "R3": "NearLosslessV1", "R4": "R4Control"}
GROUPS = (64, 128, 256)
CODECS = ("D2", "B3", "S34")
PLANES = (2, 3)
ROTATIONS = ("none", "signed-rht")
CURVATURES = ("input-hessian", "guided-fisher", "forward-kl-kronecker")
SOLVERS = (
    "greedy",
    "joint",
    "joint+feedback",
    "joint+feedback+output-recon",
    "+softened-relay-basin",
    "+modulated-basin",
)
TASKS = ("mmlu", "arc_challenge", "hellaswag", "boolq", "gsm8k", "math")
SOFT_METHODS = ("ste-soft", "hestia-relaxation")
STAGES = ("one-layer", "four-layer", "full-model")
GRID_COUNT = 1404
SMOLLM2_17B_REVISION = "effd688a12921b4cc83e3312b6feb579f70f9c71"
SMOLLM2_135M_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SMOLLM2_17B_MODEL_ID = "sha256:4be74d32a1a04f2984e9d118fdb165dd8cfbe972710796ab465a4c2152d58a08"
SMOLLM2_135M_MODEL_ID = "sha256:18686427230dde98ee2926dafa133b5cb0c6f4de48eacd0a57e5d2ed76e15e57"
MODEL_FILES = {
    ".gitattributes", "README.md", "config.json", "generation_config.json",
    "merges.txt", "model.safetensors", "special_tokens_map.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json",
}
FILE_FIELDS = {"path", "bytes", "sha256"}
CAMPAIGN_FIELDS = {
    "schema", "release", "source_revision", "run_id", "model", "smoke_model",
    "smoke_provenance", "provenance", "thresholds", "recipe_count",
    "recipe_grid_id", "evidence",
}
MODEL_FIELDS = {"repo_id", "revision", "files"}
PROVENANCE_FIELDS = {"calibration", "refinement", "validation", "evaluation"}
PARTITION_PROVENANCE_FIELDS = {
    "id", "members", "datasets", "sampling_seed", "tokenizer_digest",
    "ordered_token_digest", "sequence_count", "tokens_per_sequence",
}
DATASET_FIELDS = {"repo_id", "revision", "fraction_ppm"}
SMOKE_PROVENANCE_FIELDS = {
    "evaluation_id", "evaluation_members", "calibration_id", "dataset_repo_id",
    "dataset_revision", "sampling_seed", "tokenizer_digest",
    "ordered_token_digest", "sequence_count", "tokens_per_sequence",
    "prefix_start", "prefix_end",
}
THRESHOLD_FIELDS = {
    "r3_gap_closure_min", "metadata_bpw_max", "scale_only_token_cap",
    "short_pv_token_cap",
}
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
RECIPE_FIELDS = {
    "schema", "source_model_id", "calibration_id", "evaluation_id", "profile",
    "group_size", "allocation_macrotile", "min_planes", "max_planes", "codec",
    "matrix_byte_ceiling", "artifact_byte_ceiling", "curvature", "rotation",
    "solver", "refinement", "seed",
}
ROTATION_FIELDS = {"kind", "seed"}
SOLVER_FIELDS = {
    "variant", "em_restarts", "coordinate_sweeps", "ridge_condition_limit",
    "feedback", "output_reconstruction", "relay_basins",
}
REFINEMENT_POLICY_FIELDS = {"kind"}
BASELINE_FIELDS = {"bf16", "salt_v1"}
BF16_FIELDS = {"heldout_ppl", "task_metrics"}
SALT_FIELDS = {
    "rate", "codec", "baseline_id", "physical_bytes", "resident_bytes", "heldout_ppl",
    "task_metrics", "artifact", "physical_report",
}
REFINEMENT_FIELDS = {
    "refinement_id", "mode", "parent_candidate_id", "rate", "soft_method",
    "soft_policy", "refinement_corpus_id", "validation_id",
    "parent_validation_ppl", "checkpoints",
}
SENSITIVITY_FIELDS = {
    "schema", "result", "release", "source_revision", "model_id",
    "calibration_id", "parent_candidate_id", "sensitivity_method",
    "s2kf_source_model_digest", "s2kf_activation_cache_digest",
    "s2kf_token_stream_digest", "s2kf_record_digests", "s2kf_manifest_id",
    "tensor_scores", "evidence_id",
}
CHECKPOINT_FIELDS = {
    "tokens", "validation_ppl", "teacher_kl", "artifact", "package_id",
    "codec", "serialized_bytes", "resident_bytes", "tensor_count",
    "trits_changed", "hard_reload_max_abs_error", "hard_reload_tolerance",
    "evaluation_receipt",
}
REFINEMENT_EVALUATION_FIELDS = {
    "schema", "result", "release", "source_revision", "parent_candidate_id",
    "mode", "soft_method", "refinement_corpus_id", "validation_id", "tokens",
    "artifact_sha256", "package_id", "validation_ppl", "teacher_kl",
    "evaluation_id",
}
PHYSICAL_FIELDS = {
    "schema", "result", "release", "source_revision", "recipe_id",
    "artifact_sha256", "package_id", "codec", "tensor_count",
    "package_resident_bytes", "quantized_parameter_count", "components",
    "matrix_bytes", "artifact_bytes", "steady_resident_bytes",
    "peak_resident_bytes", "dense_materialized_bytes",
}
COMPONENT_FIELDS = {
    "trit_payload", "scales", "allocation_map", "transform", "padding",
    "tensor_headers", "container", "preserved_tensors",
}
SMOKE_FIELDS = {
    "schema", "result", "release", "source_revision", "model_id",
    "model_revision", "evaluation_id", "artifact", "package_id", "codec",
    "serialized_bytes", "resident_bytes", "tensor_count", "execution_log",
}
SMOKE_EXECUTION_FIELDS = {
    "schema", "result", "release", "source_revision", "model_id",
    "model_revision", "evaluation_id", "artifact_sha256", "stages",
}
SMOKE_STAGE_FIELDS = {"name", "result"}
NATIVE_FIELDS = {
    "schema", "result", "release", "source_revision", "model_revision",
    "physical_device", "driver", "sanitizer_version", "sanitizer_log", "cases",
}
NATIVE_CASE_FIELDS = {
    "codec", "group_size", "planes", "mode", "rows", "columns",
    "short_final_group", "plane_schedule", "packing", "cpu_max_abs_error",
    "cuda_max_abs_error", "tolerance", "dense_materialized_bytes",
}
HESTIA_GATE_FIELDS = {
    "schema", "result", "release", "source_revision", "gradcheck",
    "portable_cpu", "portable_cuda",
}
HESTIA_GRADCHECK_FIELDS = {
    "suite", "result", "inputs", "max_relative_error", "tolerance",
}
HESTIA_PORTABLE_FIELDS = {
    "backend", "result", "manifest_version", "operation", "vector_digest",
    "case_count", "physical_device", "driver",
}
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}


class Stage7Error(ValueError):
    """Stage-7 evidence is malformed, stale, cherry-picked, or threshold-drifted."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise Stage7Error(f"{label} fields do not match frozen schema")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise Stage7Error(f"{label} must be a nonempty string without NUL")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage7Error(f"{label} must be an integer at least {minimum}")
    return value


def _number(value: Any, label: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise Stage7Error(f"{label} must be finite and at least {minimum}")
    return float(value)


def _sha256(value: Any, label: str, *, prefixed: bool = False) -> str:
    text = _string(value, label)
    pattern = r"sha256:[0-9a-f]{64}" if prefixed else r"[0-9a-f]{64}"
    if re.fullmatch(pattern, text) is None:
        raise Stage7Error(f"{label} must be a canonical SHA-256 digest")
    return text


def _blake3(value: Any, label: str) -> str:
    text = _string(value, label)
    if re.fullmatch(r"blake3:[0-9a-f]{64}", text) is None:
        raise Stage7Error(f"{label} must be a canonical BLAKE3 digest")
    return text


def _revision(value: Any, label: str) -> str:
    text = _string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", text) is None:
        raise Stage7Error(f"{label} must be forty lowercase hexadecimal characters")
    return text


def _load(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size <= 0
        or path.stat().st_size > MAX_JSON_BYTES
    ):
        raise Stage7Error(f"{label} must be a bounded ordinary file")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Stage7Error(f"{label} must contain strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise Stage7Error(f"{label} must contain a JSON object")
    return value, raw


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_path(value: Any, label: str) -> PurePosixPath:
    text = _string(value, label)
    logical = PurePosixPath(text)
    if (
        not logical.parts
        or logical.is_absolute()
        or ".." in logical.parts
        or "\\" in text
    ):
        raise Stage7Error(f"{label} must be a contained nonempty POSIX path")
    return logical


def _verify_file(path: Path, record: dict[str, Any], label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise Stage7Error(f"{label}.path must resolve to an ordinary file")
    if path.stat().st_size != _integer(record["bytes"], f"{label}.bytes", 1):
        raise Stage7Error(f"{label}.bytes differs from opened file")
    if _hash_file(path) != _sha256(record["sha256"], f"{label}.sha256"):
        raise Stage7Error(f"{label}.sha256 differs from opened file")
    return path


def _open_record(
    root: Path, raw_record: Any, label: str, *, allow_hf_blob_link: bool = False
) -> Path:
    record = _object(raw_record, FILE_FIELDS, label)
    logical = _logical_path(record["path"], f"{label}.path")
    root = root.resolve(strict=True)
    cursor = root
    for part in logical.parts[:-1]:
        cursor /= part
        if cursor.is_symlink():
            raise Stage7Error(f"{label}.path traverses an intermediate symlink")
    candidate = cursor / logical.parts[-1]
    if not candidate.is_symlink():
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise Stage7Error(f"{label}.path escapes its root") from error
        return _verify_file(resolved, record, label)
    if not allow_hf_blob_link or root.parent.name != "snapshots":
        raise Stage7Error(f"{label}.path must not be a symlink")
    blob_root = root.parent.parent / "blobs"
    if blob_root.is_symlink() or not blob_root.is_dir():
        raise Stage7Error(f"{label}.path has no ordinary Hugging Face blob root")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(blob_root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise Stage7Error(f"{label}.path symlink escapes Hugging Face blob store") from error
    return _verify_file(resolved, record, label)


def _ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise Stage7Error(f"{label} must be a list")
    result = [_sha256(item, f"{label} item", prefixed=True) for item in value]
    if len(result) != len(set(result)):
        raise Stage7Error(f"{label} contains duplicates")
    return result


def _tasks(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(TASKS):
        raise Stage7Error(f"{label} must contain frozen six-task inventory")
    return {task: _number(value[task], f"{label}.{task}") for task in TASKS}


def _git_source(root: Path) -> str:
    script = Path(__file__).resolve(strict=True)
    try:
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.PIPE,
            timeout=30,
        ).strip()
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.PIPE,
            timeout=30,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        executing_top = subprocess.check_output(
            ["git", "-C", str(script.parent), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.PIPE,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise Stage7Error("source repository identity probe failed") from error
    _revision(revision, "source repository HEAD")
    resolved_root = root.resolve(strict=True)
    if Path(top).resolve(strict=True) != resolved_root:
        raise Stage7Error("source root must be the repository top level")
    if Path(executing_top).resolve(strict=True) != resolved_root:
        raise Stage7Error("source root differs from executing qualifier repository")
    relative_script = "scripts/qualify-stage7-recipe-freeze.py"
    try:
        committed_script = subprocess.check_output(
            ["git", "-C", str(resolved_root), "show", f"HEAD:{relative_script}"],
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise Stage7Error("executing qualifier is not tracked at source HEAD") from error
    if committed_script != script.read_bytes():
        raise Stage7Error("executing qualifier bytes differ from source HEAD")
    if status:
        raise Stage7Error("source repository must be clean for Stage-7 qualification")
    return revision


def _safetensors_geometry(
    path: Path,
) -> tuple[int, int, dict[str, tuple[int, int]]]:
    size = path.stat().st_size
    if size < 10:
        raise Stage7Error(f"safetensors file is truncated: {path.name}")
    with path.open("rb") as stream:
        header_bytes = int.from_bytes(stream.read(8), "little")
        if header_bytes <= 1 or header_bytes > MAX_SAFETENSORS_HEADER_BYTES:
            raise Stage7Error(f"safetensors header length is invalid: {path.name}")
        if 8 + header_bytes > size:
            raise Stage7Error(f"safetensors header exceeds file: {path.name}")
        try:
            header = json.loads(stream.read(header_bytes))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Stage7Error(f"safetensors header is invalid: {path.name}") from error
    if not isinstance(header, dict):
        raise Stage7Error(f"safetensors header must be an object: {path.name}")
    data_bytes = size - 8 - header_bytes
    intervals = []
    quantized = 0
    quantized_tensors = {}
    preserved = 0
    for name, raw_tensor in header.items():
        if name == "__metadata__":
            continue
        tensor = _object(raw_tensor, {"dtype", "shape", "data_offsets"}, name)
        dtype = tensor["dtype"]
        if dtype not in DTYPE_BYTES:
            raise Stage7Error(f"unsupported safetensors dtype {dtype}")
        shape = tensor["shape"]
        offsets = tensor["data_offsets"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise Stage7Error(f"safetensors tensor geometry is invalid: {name}")
        start = _integer(offsets[0], f"{name} start")
        end = _integer(offsets[1], f"{name} end", start)
        elements = math.prod(shape)
        expected = elements * DTYPE_BYTES[dtype]
        if end > data_bytes or end - start != expected:
            raise Stage7Error(f"safetensors tensor byte range is invalid: {name}")
        intervals.append((start, end, name))
        if len(shape) == 2:
            quantized += elements
            quantized_tensors[name] = (shape[0], shape[1])
        else:
            preserved += expected
    intervals.sort()
    if (
        not intervals
        or intervals[0][0] != 0
        or intervals[-1][1] != data_bytes
        or any(left[1] != right[0] for left, right in zip(intervals, intervals[1:]))
    ):
        raise Stage7Error(f"safetensors tensor ranges do not exactly cover data: {path.name}")
    return quantized, preserved, quantized_tensors


def _model_identity(
    model: dict[str, Any],
    model_root: Path,
    *,
    expected_repo: str,
    expected_revision: str,
    expected_model_id: str,
) -> tuple[str, int, int, tuple[str, ...]]:
    if model["repo_id"] != expected_repo:
        raise Stage7Error("Stage-7 model id differs from frozen rung")
    revision = _revision(model["revision"], "model revision")
    if revision != expected_revision:
        raise Stage7Error("Stage-7 model revision differs from frozen rung")
    if model_root.is_symlink() or not model_root.is_dir():
        raise Stage7Error("model root must be an ordinary directory")
    root = model_root.resolve(strict=True)
    if root.parent.name == "snapshots" and root.name != revision:
        raise Stage7Error("Hugging Face snapshot directory differs from model revision")
    files = model["files"]
    if not isinstance(files, list) or not files:
        raise Stage7Error("model file inventory must be nonempty")
    opened = []
    logical_paths = []
    for ordinal, record in enumerate(files):
        opened.append(_open_record(
            root, record, f"model.files[{ordinal}]", allow_hf_blob_link=True
        ))
        logical_paths.append(record["path"])
    if len(logical_paths) != len(set(logical_paths)):
        raise Stage7Error("model file inventory contains duplicates")
    if set(logical_paths) != MODEL_FILES:
        raise Stage7Error("model file inventory differs from frozen Hub revision")
    weight_files = [
        path for path, logical in zip(opened, logical_paths) if logical.endswith(".safetensors")
    ]
    if not weight_files:
        raise Stage7Error("model file inventory lacks safetensors weights")
    quantized = 0
    preserved = 0
    quantized_tensors = {}
    for path in weight_files:
        parameters, preserved_bytes, tensors = _safetensors_geometry(path)
        quantized += parameters
        preserved += preserved_bytes
        overlap = set(quantized_tensors) & set(tensors)
        if overlap:
            raise Stage7Error("rank-2 tensor inventory duplicates across weight shards")
        quantized_tensors.update(tensors)
    if quantized <= 0:
        raise Stage7Error("model contains no rank-2 quantization domain")
    scope = {"repo_id": model["repo_id"], "revision": revision, "files": files}
    model_id = "sha256:" + hashlib.sha256(canonical(scope)).hexdigest()
    if model_id != expected_model_id:
        raise Stage7Error("model file identities differ from frozen Hub revision")
    return model_id, quantized, preserved, tuple(sorted(quantized_tensors))


def _solver(variant: str) -> dict[str, Any]:
    joint = variant != "greedy"
    feedback = variant in {
        "joint+feedback", "joint+feedback+output-recon",
        "+softened-relay-basin", "+modulated-basin",
    }
    output = variant in {
        "joint+feedback+output-recon", "+softened-relay-basin", "+modulated-basin",
    }
    relay = {
        "softened": variant == "+softened-relay-basin",
        "modulated": variant == "+modulated-basin",
        "steps": 12,
        "step_size": 0.05,
        "initial_sharpness": 30.0,
        "sharpness_multiplier": 2.0,
        "sharpness_interval": 4,
        "scale_bounds": [0.001, 8.0],
        "threshold_bounds": [0.05, 0.95],
        "shift_bounds": [-2.0, 2.0],
        "softened_threshold": 0.5,
    }
    output_reconstruction = {
        "enabled": output,
        "schedule": "sliding-windows",
        "block_count": 24,
        "window_size": 3,
        "stride": 1,
        "scale_refit_starts": 4,
        "fixed_trits": True,
    }
    return {
        "variant": variant,
        "em_restarts": 4 if joint else 1,
        "coordinate_sweeps": 10 if joint else 1,
        "ridge_condition_limit": 1_000_000.0,
        "feedback": "block-ldlq-delta-corrected" if feedback else "none",
        "output_reconstruction": output_reconstruction,
        "relay_basins": relay,
    }


def recipe_grid(
    *,
    source_model_id: str,
    calibration_id: str,
    evaluation_id: str,
    quantized_parameters: int,
    preserved_bytes: int,
    metadata_bpw_max: float = 0.01,
) -> dict[str, dict[str, Any]]:
    """Derive complete frozen grid; callers cannot submit a candidate subset."""
    metadata_bytes = math.floor(metadata_bpw_max * quantized_parameters / 8)
    result = {}
    for rate in RATES:
        codecs = ("D2",) if rate == "R4" else CODECS
        planes = (2,) if rate == "R4" else PLANES
        matrix_ceiling = math.floor(RATE_BPW[rate] * quantized_parameters / 8)
        for group, codec, plane_cap, rotation, curvature, solver in itertools.product(
            GROUPS, codecs, planes, ROTATIONS, CURVATURES, SOLVERS
        ):
            recipe = {
                "schema": "tritium.salt-v2.recipe.v1",
                "source_model_id": source_model_id,
                "calibration_id": calibration_id,
                "evaluation_id": evaluation_id,
                "profile": RATE_PROFILE[rate],
                "group_size": group,
                "allocation_macrotile": 256,
                "min_planes": 1,
                "max_planes": plane_cap,
                "codec": codec,
                "matrix_byte_ceiling": matrix_ceiling,
                "artifact_byte_ceiling": matrix_ceiling + preserved_bytes + metadata_bytes,
                "curvature": curvature,
                "rotation": {"kind": rotation, "seed": 0},
                "solver": _solver(solver),
                "refinement": {"kind": "none"},
                "seed": 0,
            }
            candidate_id = "sha256:" + hashlib.sha256(canonical(recipe)).hexdigest()
            if candidate_id in result:
                raise Stage7Error("derived recipe grid contains a duplicate")
            result[candidate_id] = recipe
    if len(result) != GRID_COUNT:
        raise Stage7Error("derived recipe grid cardinality differs")
    return result


def _validate_smoke(
    path: Path, campaign: dict[str, Any], smoke_model_id: str,
    smoke_quantized_tensors: int,
) -> None:
    receipt, _ = _load(path, "Stage-7 smoke receipt")
    _object(receipt, SMOKE_FIELDS, "Stage-7 smoke receipt")
    expected = {
        "schema": SMOKE_SCHEMA,
        "result": "pass",
        "release": campaign["release"],
        "source_revision": campaign["source_revision"],
    }
    if any(receipt[field] != value for field, value in expected.items()):
        raise Stage7Error("Stage-7 smoke receipt envelope differs")
    if receipt["model_id"] != smoke_model_id:
        raise Stage7Error("smoke receipt does not bind frozen rung-1 model")
    if receipt["model_revision"] != campaign["smoke_model"]["revision"]:
        raise Stage7Error("smoke receipt model revision differs")
    if receipt["evaluation_id"] != campaign["smoke_provenance"]["evaluation_id"]:
        raise Stage7Error("smoke receipt evaluation provenance differs")
    artifact = _open_record(path.parent, receipt["artifact"], "smoke artifact")
    serialized = _integer(receipt["serialized_bytes"], "smoke serialized bytes", 1)
    resident = _integer(receipt["resident_bytes"], "smoke resident bytes", 1)
    tensors = _integer(receipt["tensor_count"], "smoke tensor count", 1)
    codec = _string(receipt["codec"], "smoke package codec")
    if (
        serialized != artifact.stat().st_size
        or tensors != smoke_quantized_tensors
        or codec not in {value.lower() for value in CODECS}
    ):
        raise Stage7Error("smoke package geometry differs")
    _verify_salt_v2_package(
        artifact,
        package_id=receipt["package_id"],
        codec=codec,
        serialized_bytes=serialized,
        resident_bytes=resident,
        tensors=tensors,
        source_revision=campaign["source_revision"],
    )
    execution_path = _open_record(
        path.parent, receipt["execution_log"], "smoke execution log"
    )
    execution, _ = _load(execution_path, "smoke execution log")
    _object(execution, SMOKE_EXECUTION_FIELDS, "smoke execution log")
    expected_execution = {
        "schema": SMOKE_EXECUTION_SCHEMA,
        "result": "pass",
        "release": campaign["release"],
        "source_revision": campaign["source_revision"],
        "model_id": smoke_model_id,
        "model_revision": campaign["smoke_model"]["revision"],
        "evaluation_id": campaign["smoke_provenance"]["evaluation_id"],
        "artifact_sha256": _hash_file(artifact),
    }
    if any(execution[field] != value for field, value in expected_execution.items()):
        raise Stage7Error("smoke execution identity or artifact differs")
    stages = execution["stages"]
    expected_stages = ("capture", "fit", "allocate", "package", "evaluate")
    if not isinstance(stages, list) or len(stages) != len(expected_stages):
        raise Stage7Error("smoke execution stage inventory is incomplete")
    for ordinal, name in enumerate(expected_stages):
        stage = _object(stages[ordinal], SMOKE_STAGE_FIELDS, f"smoke stage[{ordinal}]")
        if stage != {"name": name, "result": "pass"}:
            raise Stage7Error("smoke execution stage order or result differs")


def _validate_native(path: Path, campaign: dict[str, Any]) -> bool:
    receipt, _ = _load(path, "Stage-7 native-kernel receipt")
    _object(receipt, NATIVE_FIELDS, "Stage-7 native-kernel receipt")
    expected = {
        "schema": NATIVE_SCHEMA,
        "release": campaign["release"],
        "source_revision": campaign["source_revision"],
        "model_revision": campaign["model"]["revision"],
    }
    if any(receipt[field] != value for field, value in expected.items()):
        raise Stage7Error("Stage-7 native-kernel receipt envelope differs")
    if receipt["result"] not in {"pass", "negative"}:
        raise Stage7Error("Stage-7 native-kernel result differs")
    _string(receipt["physical_device"], "native physical device")
    _string(receipt["driver"], "native driver")
    _string(receipt["sanitizer_version"], "compute-sanitizer version")
    log = _open_record(path.parent, receipt["sanitizer_log"], "sanitizer log")
    text = log.read_text(encoding="utf-8")
    summaries = re.findall(r"ERROR SUMMARY: ([0-9]+) errors", text)
    if len(summaries) != 1:
        raise Stage7Error("compute-sanitizer log lacks exactly one error summary")
    gate_failed = int(summaries[0]) != 0
    cases = receipt["cases"]
    packing = {"D2": "direct-2bit", "B3": "radix-3", "S34": "s34"}
    expected_cases = set()
    for codec, group, planes, mode in itertools.product(
        CODECS, GROUPS, PLANES, ("exact", "fast")
    ):
        schedules = ("uniform", "mixed")
        for schedule, shape in itertools.product(schedules, ("aligned", "short")):
            expected_cases.add((
                codec,
                group,
                planes,
                mode,
                64 if shape == "aligned" else 3,
                group * 2 if shape == "aligned" else group + 17,
                shape == "short",
                schedule,
                packing[codec],
            ))
    if not isinstance(cases, list) or len(cases) != len(expected_cases):
        raise Stage7Error("native-kernel case inventory is incomplete")
    seen = set()
    for ordinal, raw_case in enumerate(cases):
        case = _object(raw_case, NATIVE_CASE_FIELDS, f"native cases[{ordinal}]")
        rows = _integer(case["rows"], "native rows", 1)
        columns = _integer(case["columns"], "native columns", 1)
        if type(case["short_final_group"]) is not bool:
            raise Stage7Error("native short-final-group flag must be boolean")
        _string(case["plane_schedule"], "native plane schedule")
        _string(case["packing"], "native packing")
        identity = (
            case["codec"], case["group_size"], case["planes"], case["mode"],
            rows, columns, case["short_final_group"],
            case["plane_schedule"], case["packing"],
        )
        if identity not in expected_cases or identity in seen:
            raise Stage7Error("native-kernel case identity differs or duplicates")
        seen.add(identity)
        tolerance = _number(case["tolerance"], "native tolerance", 1e-12)
        if (
            _number(case["cpu_max_abs_error"], "native CPU error") > tolerance
            or _number(case["cuda_max_abs_error"], "native CUDA error") > tolerance
            or _integer(case["dense_materialized_bytes"], "native dense bytes") != 0
        ):
            gate_failed = True
    if seen != expected_cases:
        raise Stage7Error("native-kernel case coverage differs")
    expected_result = "negative" if gate_failed else "pass"
    if receipt["result"] != expected_result:
        raise Stage7Error("native-kernel result contradicts measured gates")
    return not gate_failed


def _validate_hestia_gate(path: Path, campaign: dict[str, Any]) -> None:
    receipt, _ = _load(path, "HESTIA Gate-C receipt")
    _object(receipt, HESTIA_GATE_FIELDS, "HESTIA Gate-C receipt")
    expected = {
        "schema": HESTIA_GATE_SCHEMA,
        "result": "pass",
        "release": campaign["release"],
        "source_revision": campaign["source_revision"],
    }
    if any(receipt[field] != value for field, value in expected.items()):
        raise Stage7Error("HESTIA Gate-C receipt envelope differs")

    gradcheck = _object(
        receipt["gradcheck"], HESTIA_GRADCHECK_FIELDS, "HESTIA gradcheck"
    )
    tolerance = _number(gradcheck["tolerance"], "HESTIA gradcheck tolerance", 1e-12)
    if (
        gradcheck["suite"] != "tritium-train/gradcheck_hestia"
        or gradcheck["result"] != "pass"
        or gradcheck["inputs"] != ["weight", "temperature"]
        or tolerance != 2e-3
        or _number(
            gradcheck["max_relative_error"], "HESTIA gradcheck maximum error"
        ) > tolerance
    ):
        raise Stage7Error("HESTIA gradcheck receipt differs or fails")

    vector_digest = None
    vector_case_count = None
    for field, backend in (("portable_cpu", "cpu"), ("portable_cuda", "cuda")):
        portable = _object(
            receipt[field], HESTIA_PORTABLE_FIELDS, f"HESTIA portable {backend}"
        )
        digest = _sha256(
            portable["vector_digest"],
            f"HESTIA portable {backend} vector digest",
            prefixed=True,
        )
        case_count = _integer(
            portable["case_count"], f"HESTIA portable {backend} case count", 5
        )
        if (
            portable["backend"] != backend
            or portable["result"] != "pass"
            or portable["manifest_version"] != 3
            or portable["operation"] != "graph.hestia_relax"
        ):
            raise Stage7Error(f"HESTIA portable {backend} conformance differs")
        device = _string(
            portable["physical_device"], f"HESTIA portable {backend} physical device"
        )
        _string(portable["driver"], f"HESTIA portable {backend} driver")
        if (backend == "cpu" and device != "cpu") or (
            backend == "cuda" and not device.startswith("cuda:")
        ):
            raise Stage7Error(f"HESTIA portable {backend} device differs")
        if vector_digest is not None and digest != vector_digest:
            raise Stage7Error("HESTIA portable backends used different vectors")
        if vector_case_count is not None and case_count != vector_case_count:
            raise Stage7Error("HESTIA portable backends used different vector counts")
        vector_digest = digest
        vector_case_count = case_count


def _validate_partition_provenance(
    raw: Any, label: str
) -> tuple[str, list[str]]:
    partition = _object(raw, PARTITION_PROVENANCE_FIELDS, f"{label} provenance")
    members = _ids(partition["members"], f"{label} members")
    if len(members) != 512 or partition["sequence_count"] != 512:
        raise Stage7Error(f"{label} provenance must contain exactly 512 sequences")
    if partition["tokens_per_sequence"] != 2_048:
        raise Stage7Error(f"{label} provenance token geometry differs")
    _integer(partition["sampling_seed"], f"{label} sampling seed")
    _sha256(partition["tokenizer_digest"], f"{label} tokenizer digest", prefixed=True)
    ordered_id = "sha256:" + hashlib.sha256(canonical(members)).hexdigest()
    if partition["ordered_token_digest"] != ordered_id:
        raise Stage7Error(f"{label} ordered token digest differs")
    datasets = partition["datasets"]
    expected_datasets = (
        ("allenai/c4", 500_000),
        ("open-web-math/open-web-math", 250_000),
        ("bigcode/starcoderdata", 250_000),
    )
    if not isinstance(datasets, list) or len(datasets) != len(expected_datasets):
        raise Stage7Error(f"{label} dataset inventory differs")
    for ordinal, (repo_id, fraction) in enumerate(expected_datasets):
        dataset = _object(
            datasets[ordinal], DATASET_FIELDS, f"{label} dataset[{ordinal}]"
        )
        if dataset["repo_id"] != repo_id or dataset["fraction_ppm"] != fraction:
            raise Stage7Error(f"{label} dataset composition differs")
        _revision(dataset["revision"], f"{label} dataset revision")
    scope = {
        field: partition[field]
        for field in PARTITION_PROVENANCE_FIELDS - {"id"}
    }
    expected_id = "sha256:" + hashlib.sha256(canonical(scope)).hexdigest()
    if partition["id"] != expected_id:
        raise Stage7Error(f"{label} provenance id differs")
    return expected_id, members


def _validate_campaign(
    path: Path, *, model_root: Path, smoke_model_root: Path, source_root: Path
) -> tuple[
    dict[str, Any], bytes, dict[str, dict[str, Any]], str, int, int, int,
    tuple[str, ...], list[str],
]:
    campaign, raw = _load(path, "Stage-7 campaign")
    _object(campaign, CAMPAIGN_FIELDS, "Stage-7 campaign")
    if campaign["schema"] != CAMPAIGN_SCHEMA:
        raise Stage7Error("Stage-7 campaign schema differs")
    _string(campaign["release"], "campaign release")
    source_revision = _revision(campaign["source_revision"], "campaign source revision")
    if _git_source(source_root) != source_revision:
        raise Stage7Error("campaign source revision differs from clean repository HEAD")
    _string(campaign["run_id"], "campaign run id")
    model = _object(campaign["model"], MODEL_FIELDS, "campaign model")
    model_id, quantized, preserved, quantized_tensor_names = _model_identity(
        model,
        model_root,
        expected_repo="HuggingFaceTB/SmolLM2-1.7B",
        expected_revision=SMOLLM2_17B_REVISION,
        expected_model_id=SMOLLM2_17B_MODEL_ID,
    )
    smoke_model = _object(
        campaign["smoke_model"], MODEL_FIELDS, "campaign smoke model"
    )
    smoke_model_id, _, _, smoke_quantized_tensor_names = _model_identity(
        smoke_model,
        smoke_model_root,
        expected_repo="HuggingFaceTB/SmolLM2-135M",
        expected_revision=SMOLLM2_135M_REVISION,
        expected_model_id=SMOLLM2_135M_MODEL_ID,
    )

    smoke_provenance = _object(
        campaign["smoke_provenance"],
        SMOKE_PROVENANCE_FIELDS,
        "smoke provenance",
    )
    smoke_evaluation = _ids(
        smoke_provenance["evaluation_members"], "smoke evaluation members"
    )
    if (
        len(smoke_evaluation) != 128
        or smoke_provenance["sequence_count"] != 128
        or smoke_provenance["tokens_per_sequence"] != 2_048
        or smoke_provenance["prefix_start"] != 0
        or smoke_provenance["prefix_end"] != 128
    ):
        raise Stage7Error("smoke provenance must bind exact 128-sequence prefix")
    _string(smoke_provenance["dataset_repo_id"], "smoke dataset repo id")
    _revision(smoke_provenance["dataset_revision"], "smoke dataset revision")
    _integer(smoke_provenance["sampling_seed"], "smoke sampling seed")
    _sha256(
        smoke_provenance["tokenizer_digest"],
        "smoke tokenizer digest",
        prefixed=True,
    )
    smoke_evaluation_id = "sha256:" + hashlib.sha256(
        canonical(smoke_evaluation)
    ).hexdigest()
    if (
        smoke_provenance["evaluation_id"] != smoke_evaluation_id
        or smoke_provenance["ordered_token_digest"] != smoke_evaluation_id
    ):
        raise Stage7Error("smoke evaluation id does not bind ordered members")

    provenance = _object(campaign["provenance"], PROVENANCE_FIELDS, "provenance")
    validated_partitions = {
        kind: _validate_partition_provenance(provenance[kind], kind)
        for kind in ("calibration", "refinement", "validation", "evaluation")
    }
    partition_ids = {
        kind: validated[0] for kind, validated in validated_partitions.items()
    }
    partitions = {
        kind: validated[1] for kind, validated in validated_partitions.items()
    }
    member_sets = [set(members) for members in partitions.values()]
    if (
        any(not members for members in partitions.values())
        or len(set().union(*member_sets)) != sum(len(members) for members in member_sets)
    ):
        raise Stage7Error("four data partitions must be disjoint and nonempty")
    if smoke_evaluation != partitions["calibration"][:128]:
        raise Stage7Error("smoke provenance differs from frozen calibration prefix")
    calibration_id = partition_ids["calibration"]
    evaluation_id = partition_ids["evaluation"]
    calibration = provenance["calibration"]
    calibration_dataset = calibration["datasets"][0]
    if (
        smoke_provenance["calibration_id"] != calibration_id
        or smoke_provenance["dataset_repo_id"] != calibration_dataset["repo_id"]
        or smoke_provenance["dataset_revision"] != calibration_dataset["revision"]
        or smoke_provenance["sampling_seed"] != calibration["sampling_seed"]
        or smoke_provenance["tokenizer_digest"] != calibration["tokenizer_digest"]
    ):
        raise Stage7Error("smoke metadata differs from parent calibration provenance")

    thresholds = _object(campaign["thresholds"], THRESHOLD_FIELDS, "thresholds")
    if thresholds != {
        "r3_gap_closure_min": 0.25,
        "metadata_bpw_max": 0.01,
        "scale_only_token_cap": 8_000_000,
        "short_pv_token_cap": 32_000_000,
    }:
        raise Stage7Error("Stage-7 thresholds differ from plan 0043")
    grid = recipe_grid(
        source_model_id=model_id,
        calibration_id=calibration_id,
        evaluation_id=evaluation_id,
        quantized_parameters=quantized,
        preserved_bytes=preserved,
    )
    grid_id = "sha256:" + hashlib.sha256(canonical(sorted(grid))).hexdigest()
    if campaign["recipe_count"] != GRID_COUNT or campaign["recipe_grid_id"] != grid_id:
        raise Stage7Error("campaign does not bind complete frozen recipe grid")

    evidence = campaign["evidence"]
    if not isinstance(evidence, list) or len(evidence) != 3:
        raise Stage7Error("campaign prerequisite evidence inventory is incomplete")
    expected_kinds = ("smoke", "native-kernels", "hestia-gate-c")
    prerequisite_reasons = []
    for ordinal, kind in enumerate(expected_kinds):
        raw_record = evidence[ordinal]
        if not isinstance(raw_record, dict) or set(raw_record) != FILE_FIELDS | {"kind"}:
            raise Stage7Error(f"evidence[{ordinal}] fields do not match frozen schema")
        if raw_record["kind"] != kind:
            raise Stage7Error("campaign prerequisite evidence kind or order differs")
        receipt_path = _open_record(
            path.parent,
            {field: raw_record[field] for field in FILE_FIELDS},
            f"evidence[{ordinal}]",
        )
        if kind == "smoke":
            _validate_smoke(
                receipt_path,
                campaign,
                smoke_model_id,
                len(smoke_quantized_tensor_names),
            )
        elif kind == "native-kernels":
            if not _validate_native(receipt_path, campaign):
                prerequisite_reasons.append("native-kernel-gate-failed")
        else:
            _validate_hestia_gate(receipt_path, campaign)
    return (
        campaign, raw, grid, model_id, quantized, preserved,
        len(quantized_tensor_names), quantized_tensor_names, prerequisite_reasons,
    )


def _dominates(left: dict[str, Any], right: dict[str, Any], fields: Iterable[str]) -> bool:
    pairs = tuple((left[field], right[field]) for field in fields)
    return all(a <= b for a, b in pairs) and any(a < b for a, b in pairs)


def _pareto_front_ids(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> list[str]:
    return sorted(
        row["candidate_id"]
        for row in rows
        if not any(other is not row and _dominates(other, row, fields) for other in rows)
    )


def _pareto_ranked_half_ids(rows: list[dict[str, Any]]) -> list[str]:
    remaining = list(rows)
    ranked = []
    while remaining:
        ids = set(
            _pareto_front_ids(remaining, ("output_loss", "physical_bytes"))
        )
        layer = sorted(
            (row for row in remaining if row["candidate_id"] in ids),
            key=lambda row: (
                row["output_loss"], row["physical_bytes"], row["candidate_id"]
            ),
        )
        ranked.extend(layer)
        remaining = [row for row in remaining if row["candidate_id"] not in ids]
    return sorted(row["candidate_id"] for row in ranked[: math.ceil(len(rows) / 2)])


def _input_control_id(
    candidate_id: str, grid: dict[str, dict[str, Any]]
) -> str:
    recipe = json.loads(canonical(grid[candidate_id]))
    recipe["curvature"] = "input-hessian"
    control = "sha256:" + hashlib.sha256(canonical(recipe)).hexdigest()
    if control not in grid:
        raise Stage7Error("derived matched input-Hessian control is absent from grid")
    return control


def _with_controls(
    selected: Iterable[str], grid: dict[str, dict[str, Any]]
) -> list[str]:
    ids = set(selected)
    ids.update(_input_control_id(candidate_id, grid) for candidate_id in selected)
    return sorted(ids)


def _native_source_identity() -> str:
    from tritium import _tritium

    return _tritium.source_identity()


def _verify_salt_v2_package(
    path: Path,
    *,
    package_id: str,
    codec: str,
    serialized_bytes: int,
    resident_bytes: int,
    tensors: int,
    source_revision: str,
) -> None:
    if (
        not isinstance(package_id, str)
        or not package_id.startswith("trp1_")
        or len(package_id) != 69
        or any(character not in "0123456789abcdef" for character in package_id[5:])
    ):
        raise Stage7Error("SALT V2 package id is malformed")
    try:
        native_source = _native_source_identity()
    except Exception as error:
        raise Stage7Error("native package verifier source probe failed") from error
    if native_source != f"source-git:{source_revision}":
        raise Stage7Error("native package verifier source differs from campaign")
    try:
        from tritium import _tritium

        observed = _tritium.verify_salt_v2_package(
            str(path),
            package_id,
            serialized_bytes,
            resident_bytes,
            expected_tensors=tensors,
        )
    except Exception as error:
        raise Stage7Error("SALT V2 package semantic verification failed") from error
    expected = (package_id, codec, serialized_bytes, resident_bytes)
    if tuple(observed) != expected:
        raise Stage7Error("SALT V2 package semantic verification differs")


def _validate_physical_report(
    path: Path,
    *,
    artifact: Path,
    measurement: dict[str, Any],
    campaign: dict[str, Any],
    candidate_id: str,
    quantized: int,
    quantized_tensors: int,
    expected_codec: str,
    recipe: dict[str, Any],
    preserved: int,
) -> bool:
    report, _ = _load(path, "full-model physical report")
    _object(report, PHYSICAL_FIELDS, "full-model physical report")
    expected = {
        "schema": PHYSICAL_SCHEMA,
        "release": campaign["release"],
        "source_revision": campaign["source_revision"],
        "recipe_id": candidate_id,
        "artifact_sha256": _hash_file(artifact),
    }
    if any(report[field] != value for field, value in expected.items()):
        raise Stage7Error("full-model physical report envelope differs")
    if report["result"] not in {"pass", "negative"}:
        raise Stage7Error("full-model physical report result differs")
    if (
        _integer(report["quantized_parameter_count"], "quantized parameter count", 1)
        != quantized
        or _integer(report["tensor_count"], "quantized tensor count", 1)
        != quantized_tensors
        or _string(report["codec"], "SALT V2 package codec") != expected_codec
    ):
        raise Stage7Error("full-model package geometry differs")
    components = _object(report["components"], COMPONENT_FIELDS, "physical components")
    values = {
        field: _integer(components[field], f"physical component {field}")
        for field in COMPONENT_FIELDS
    }
    matrix = sum(values[field] for field in COMPONENT_FIELDS - {"preserved_tensors"})
    artifact_bytes = matrix + values["preserved_tensors"]
    package_resident = _integer(
        report["package_resident_bytes"], "package resident bytes", 1
    )
    if (
        values["preserved_tensors"] != preserved
        or report["matrix_bytes"] != matrix
        or report["artifact_bytes"] != artifact_bytes
        or artifact.stat().st_size != matrix
        or measurement["physical_bytes"] != matrix
        or report["steady_resident_bytes"] != package_resident + preserved
        or measurement["resident_bytes"] != package_resident + preserved
    ):
        raise Stage7Error("physical report totals differ from artifact or measurement")
    _verify_salt_v2_package(
        artifact,
        package_id=report["package_id"],
        codec=expected_codec,
        serialized_bytes=matrix,
        resident_bytes=package_resident,
        tensors=quantized_tensors,
        source_revision=campaign["source_revision"],
    )
    steady = _integer(report["steady_resident_bytes"], "steady resident bytes", 1)
    _integer(report["peak_resident_bytes"], "peak resident bytes", steady)
    dense_bytes = _integer(report["dense_materialized_bytes"], "dense materialized bytes")
    gate_pass = artifact_bytes <= recipe["artifact_byte_ceiling"] and dense_bytes == 0
    expected_result = "pass" if gate_pass else "negative"
    if report["result"] != expected_result:
        raise Stage7Error("physical report result contradicts measured gates")
    return gate_pass


def _measurement(
    raw: Any,
    *,
    full: bool,
    grid: dict[str, dict[str, Any]],
    campaign: dict[str, Any],
    trace_root: Path,
    quantized: int,
    quantized_tensors: int,
    preserved: int,
    label: str,
) -> tuple[dict[str, Any], bool]:
    row = _object(raw, MEASUREMENT_FIELDS, label)
    candidate_id = _sha256(row["candidate_id"], f"{label}.candidate_id", prefixed=True)
    if candidate_id not in grid:
        raise Stage7Error(f"{label} candidate is outside frozen grid")
    if row["track"] != "ptq" or type(row["correct"]) is not bool:
        raise Stage7Error(f"{label} must be a correctness-labeled PTQ measurement")
    physical = _integer(row["physical_bytes"], f"{label}.physical_bytes", 1)
    if physical > grid[candidate_id]["matrix_byte_ceiling"]:
        raise Stage7Error(f"{label} exceeds frozen matrix-byte ceiling")
    _integer(row["resident_bytes"], f"{label}.resident_bytes", physical)
    _number(row["output_loss"], f"{label}.output_loss")
    _number(row["runtime_ms"], f"{label}.runtime_ms", 1e-12)
    if not full:
        if (
            row["heldout_ppl"] is not None
            or row["task_metrics"] != {}
            or row["artifact"] is not None
            or row["physical_report"] is not None
        ):
            raise Stage7Error(f"{label} leaks held-out or artifact evidence before full model")
        return row, True
    _number(row["heldout_ppl"], f"{label}.heldout_ppl", 1e-12)
    _tasks(row["task_metrics"], f"{label}.task_metrics")
    artifact = _open_record(trace_root, row["artifact"], f"{label}.artifact")
    report = _open_record(
        trace_root, row["physical_report"], f"{label}.physical_report"
    )
    physical_pass = _validate_physical_report(
        report,
        artifact=artifact,
        measurement=row,
        campaign=campaign,
        candidate_id=candidate_id,
        quantized=quantized,
        quantized_tensors=quantized_tensors,
        expected_codec=grid[candidate_id]["codec"].lower(),
        recipe=grid[candidate_id],
        preserved=preserved,
    )
    if not physical_pass and row["correct"]:
        raise Stage7Error(f"{label} marks a physical-gate failure correct")
    return row, physical_pass


def _expected_promotions(
    name: str,
    rows: list[dict[str, Any]],
    grid: dict[str, dict[str, Any]],
) -> list[str]:
    output_aware = [
        row for row in rows
        if row["correct"] and grid[row["candidate_id"]]["curvature"] != "input-hessian"
    ]
    grouped = []
    for rate in RATES:
        codecs = ("D2",) if rate == "R4" else CODECS
        for codec in codecs:
            grouped.append([
                row
                for row in output_aware
                if grid[row["candidate_id"]]["profile"] == RATE_PROFILE[rate]
                and grid[row["candidate_id"]]["codec"] == codec
            ])
    if name == "one-layer":
        selected = set()
        for rows in grouped:
            selected.update(
                _pareto_front_ids(
                    rows, ("output_loss", "physical_bytes", "runtime_ms")
                )
            )
        return _with_controls(selected, grid)
    if name == "four-layer":
        selected = set()
        for rows in grouped:
            selected.update(_pareto_ranked_half_ids(rows))
        return _with_controls(selected, grid)
    selected = set()
    for rows in grouped:
        selected.update(
            _pareto_front_ids(
                rows, ("heldout_ppl", "physical_bytes", "runtime_ms")
            )
        )
    return sorted(selected)


def _validate_refinement_curve(
    row: dict[str, Any], *, cap: int, trace_root: Path, expected_codec: str,
    quantized_tensors: int, source_revision: str, release: str,
) -> tuple[float, float, int]:
    checkpoints = row["checkpoints"]
    if not isinstance(checkpoints, list) or len(checkpoints) not in {3, 4}:
        raise Stage7Error("refinement must contain governed three/four-point curve")
    schedule = [cap // 8, cap // 4, cap // 2, cap]
    best = (float(row["parent_validation_ppl"]), float("inf"), 0)
    consecutive = 0
    for ordinal, raw_checkpoint in enumerate(checkpoints):
        checkpoint = _object(
            raw_checkpoint, CHECKPOINT_FIELDS, f"refinement checkpoint[{ordinal}]"
        )
        if checkpoint["tokens"] != schedule[ordinal]:
            raise Stage7Error("refinement checkpoint schedule differs from frozen fractions")
        ppl = _number(checkpoint["validation_ppl"], "refinement validation perplexity", 1e-12)
        kl = _number(checkpoint["teacher_kl"], "refinement teacher KL")
        artifact = _open_record(
            trace_root, checkpoint["artifact"], "refinement artifact"
        )
        serialized = _integer(
            checkpoint["serialized_bytes"], "refinement serialized bytes", 1
        )
        resident = _integer(
            checkpoint["resident_bytes"], "refinement resident bytes", 1
        )
        tensors = _integer(
            checkpoint["tensor_count"], "refinement tensor count", 1
        )
        if (
            checkpoint["codec"] != expected_codec
            or serialized != artifact.stat().st_size
            or tensors != quantized_tensors
        ):
            raise Stage7Error("refinement package geometry differs")
        _verify_salt_v2_package(
            artifact,
            package_id=checkpoint["package_id"],
            codec=expected_codec,
            serialized_bytes=serialized,
            resident_bytes=resident,
            tensors=tensors,
            source_revision=source_revision,
        )
        evaluation_path = _open_record(
            trace_root,
            checkpoint["evaluation_receipt"],
            "refinement evaluation receipt",
        )
        evaluation, _ = _load(evaluation_path, "refinement evaluation receipt")
        _object(
            evaluation,
            REFINEMENT_EVALUATION_FIELDS,
            "refinement evaluation receipt",
        )
        expected_evaluation = {
            "schema": "tritium.stage7-refinement-evaluation.v1",
            "result": "pass",
            "release": release,
            "source_revision": source_revision,
            "parent_candidate_id": row["parent_candidate_id"],
            "mode": row["mode"],
            "soft_method": row["soft_method"],
            "refinement_corpus_id": row["refinement_corpus_id"],
            "validation_id": row["validation_id"],
            "tokens": checkpoint["tokens"],
            "artifact_sha256": _hash_file(artifact),
            "package_id": checkpoint["package_id"],
            "validation_ppl": checkpoint["validation_ppl"],
            "teacher_kl": checkpoint["teacher_kl"],
        }
        if any(
            evaluation[field] != value
            for field, value in expected_evaluation.items()
        ):
            raise Stage7Error(
                "refinement evaluation does not bind checkpoint artifact and metrics"
            )
        evaluation_id = "sha256:" + hashlib.sha256(
            canonical(expected_evaluation)
        ).hexdigest()
        if evaluation["evaluation_id"] != evaluation_id:
            raise Stage7Error("refinement evaluation id differs")
        if type(checkpoint["trits_changed"]) is not bool:
            raise Stage7Error("refinement trit-change flag must be boolean")
        error = _number(
            checkpoint["hard_reload_max_abs_error"], "hard reload max abs error"
        )
        tolerance = _number(
            checkpoint["hard_reload_tolerance"], "hard reload tolerance", 1e-12
        )
        if error > tolerance:
            raise Stage7Error("refinement hard reload exceeds tolerance")
        candidate = (ppl, kl, checkpoint["tokens"])
        if (
            candidate[0] <= best[0]
            and candidate[1] <= best[1]
            and candidate[:2] != best[:2]
        ):
            best = candidate
            consecutive = 0
        else:
            consecutive += 1
        if consecutive == 3 and ordinal != len(checkpoints) - 1:
            raise Stage7Error("refinement continued after three non-improving evaluations")
    if len(checkpoints) == 3 and consecutive != 3:
        raise Stage7Error("refinement stopped early without three non-improving evaluations")
    return best


def _soft_method_ab(
    pv: dict[str, tuple[dict[str, Any], tuple[float, float, int]]],
    fallback_id: str,
) -> tuple[str, dict[str, Any]]:
    ste, hestia = (pv[method] for method in SOFT_METHODS)
    ste_metrics = ste[1][:2]
    hestia_metrics = hestia[1][:2]
    ids = {method: pv[method][0]["refinement_id"] for method in SOFT_METHODS}
    if ste_metrics == hestia_metrics:
        selected = min(ids.values())
        outcome = "tie"
        winner = None
    elif all(left <= right for left, right in zip(ste_metrics, hestia_metrics)):
        selected = ids["ste-soft"]
        outcome = "ste-win"
        winner = "ste-soft"
    elif all(left <= right for left, right in zip(hestia_metrics, ste_metrics)):
        selected = ids["hestia-relaxation"]
        outcome = "hestia-win"
        winner = "hestia-relaxation"
    else:
        selected = fallback_id
        outcome = "tradeoff"
        winner = None
    return selected, {
        "outcome": outcome,
        "winner": winner,
        "ste_refinement_id": ids["ste-soft"],
        "hestia_refinement_id": ids["hestia-relaxation"],
    }


def _validate_hestia_policy(
    policy: Any,
    *,
    row: dict[str, Any],
    campaign: dict[str, Any],
    trace_root: Path,
    quantized_tensor_names: tuple[str, ...],
) -> None:
    expected_fields = {
        "kind", "tau_initial", "tau_floor", "schedule", "total_tokens",
        "sensitivity_alpha", "sensitivity_evidence", "floor_reached_by_fraction",
        "hard_boundary_fraction", "hard_export",
    }
    policy = _object(policy, expected_fields, "HESTIA soft policy")
    if (
        policy["kind"] != "hestia-relaxation"
        or policy["tau_initial"] != 1.0
        or policy["tau_floor"] != 0.01
        or policy["schedule"] != "exponential"
        or policy["total_tokens"] != campaign["thresholds"]["short_pv_token_cap"]
        or policy["sensitivity_alpha"] != 1.0
        or policy["floor_reached_by_fraction"] != 0.8
        or policy["hard_boundary_fraction"] != 0.8
        or policy["hard_export"] != "hard-trits-scale-only"
    ):
        raise Stage7Error("HESTIA soft policy differs from frozen schedule")
    evidence_path = _open_record(
        trace_root, policy["sensitivity_evidence"], "HESTIA sensitivity evidence"
    )
    evidence, _ = _load(evidence_path, "HESTIA sensitivity evidence")
    _object(evidence, SENSITIVITY_FIELDS, "HESTIA sensitivity evidence")
    model_id = "sha256:" + hashlib.sha256(canonical(campaign["model"])).hexdigest()
    expected = {
        "schema": "tritium.stage7-s2kf-sensitivity.v1",
        "result": "pass",
        "release": campaign["release"],
        "source_revision": campaign["source_revision"],
        "model_id": model_id,
        "calibration_id": campaign["provenance"]["calibration"]["id"],
        "parent_candidate_id": row["parent_candidate_id"],
    }
    if any(evidence[field] != value for field, value in expected.items()):
        raise Stage7Error("HESTIA sensitivity evidence envelope differs")
    if (
        evidence["sensitivity_method"]
        != "standardized-sigmoid(input-gram-trace*output-fisher-mean)"
        or evidence["s2kf_source_model_digest"] != model_id
        or evidence["s2kf_token_stream_digest"]
        != campaign["provenance"]["calibration"]["ordered_token_digest"]
    ):
        raise Stage7Error("HESTIA sensitivity S2KF provenance differs")
    _sha256(
        evidence["s2kf_activation_cache_digest"],
        "HESTIA S2KF activation-cache digest",
        prefixed=True,
    )
    record_digests = evidence["s2kf_record_digests"]
    if (
        not isinstance(record_digests, dict)
        or set(record_digests) != set(quantized_tensor_names)
    ):
        raise Stage7Error("HESTIA S2KF records differ from source rank-2 tensor inventory")
    for name, value in record_digests.items():
        _blake3(value, f"HESTIA S2KF record {name}")
    manifest_id = "sha256:" + hashlib.sha256(canonical(record_digests)).hexdigest()
    if evidence["s2kf_manifest_id"] != manifest_id:
        raise Stage7Error("HESTIA S2KF manifest id differs")
    scores = evidence["tensor_scores"]
    if (
        not isinstance(scores, dict)
        or not scores
        or any(not isinstance(name, str) or not name for name in scores)
    ):
        raise Stage7Error("HESTIA sensitivity scores are empty or malformed")
    if set(scores) != set(quantized_tensor_names):
        raise Stage7Error(
            "HESTIA sensitivity scores differ from source rank-2 tensor inventory"
        )
    values = [_number(value, f"HESTIA sensitivity {name}") for name, value in scores.items()]
    if any(value > 1.0 for value in values):
        raise Stage7Error("HESTIA standardized sensitivity exceeds one")
    if not any(value > 0.0 for value in values):
        raise Stage7Error("HESTIA sensitivity scores contain no positive signal")
    scope = {field: evidence[field] for field in SENSITIVITY_FIELDS - {"evidence_id"}}
    expected_id = "sha256:" + hashlib.sha256(canonical(scope)).hexdigest()
    if evidence["evidence_id"] != expected_id:
        raise Stage7Error("HESTIA sensitivity evidence id differs")


def _validate_refinements(
    rows: Any,
    *,
    selected: dict[str, str],
    campaign: dict[str, Any],
    grid: dict[str, dict[str, Any]],
    quantized_tensors: int,
    quantized_tensor_names: tuple[str, ...],
    trace_root: Path,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    if not isinstance(rows, list) or len(rows) != 5:
        raise Stage7Error(
            "refinement inventory must contain three scale curves and two PV A/B curves"
        )
    scale: dict[str, tuple[dict[str, Any], tuple[float, float, int]]] = {}
    pv = {}
    for ordinal, raw_row in enumerate(rows):
        row = _object(raw_row, REFINEMENT_FIELDS, f"refinements[{ordinal}]")
        scope = {field: row[field] for field in REFINEMENT_FIELDS - {"refinement_id"}}
        expected_id = "sha256:" + hashlib.sha256(canonical(scope)).hexdigest()
        if row["refinement_id"] != expected_id:
            raise Stage7Error("refinement id does not bind complete curve and policy")
        mode = row["mode"]
        rate = row["rate"]
        parent = row["parent_candidate_id"]
        provenance = campaign["provenance"]
        if (
            row["refinement_corpus_id"] != provenance["refinement"]["id"]
            or row["validation_id"] != provenance["validation"]["id"]
        ):
            raise Stage7Error("refinement curve provenance differs from campaign")
        if rate not in RATES or selected.get(rate) != parent:
            raise Stage7Error("refinement does not descend from selected PTQ point")
        expected_codec = grid[parent]["codec"].lower()
        parent_ppl = _number(
            row["parent_validation_ppl"], "refinement parent validation perplexity", 1e-12
        )
        if mode == "scale-only":
            if row["soft_method"] is not None or rate in scale:
                raise Stage7Error("scale-only refinement method or rate duplicates")
            if row["soft_policy"] != {"kind": "none"}:
                raise Stage7Error("scale-only soft policy differs")
            best = _validate_refinement_curve(
                row,
                cap=campaign["thresholds"]["scale_only_token_cap"],
                trace_root=trace_root,
                expected_codec=expected_codec,
                quantized_tensors=quantized_tensors,
                source_revision=campaign["source_revision"],
                release=campaign["release"],
            )
            if any(checkpoint["trits_changed"] for checkpoint in row["checkpoints"]):
                raise Stage7Error("scale-only refinement changed trits")
            scale[rate] = (row, best)
        elif mode == "short-pv":
            method = row["soft_method"]
            if method not in SOFT_METHODS or method in pv:
                raise Stage7Error("short-PV soft-method A/B inventory differs")
            if method == "ste-soft":
                if row["soft_policy"] != {
                    "kind": "ste-soft",
                    "hard_boundary_fraction": 0.8,
                    "hard_export": "hard-trits-scale-only",
                }:
                    raise Stage7Error("STE soft policy differs from frozen schedule")
            else:
                _validate_hestia_policy(
                    row["soft_policy"],
                    row=row,
                    campaign=campaign,
                    trace_root=trace_root,
                    quantized_tensor_names=quantized_tensor_names,
                )
            best = _validate_refinement_curve(
                row,
                cap=campaign["thresholds"]["short_pv_token_cap"],
                trace_root=trace_root,
                expected_codec=expected_codec,
                quantized_tensors=quantized_tensors,
                source_revision=campaign["source_revision"],
                release=campaign["release"],
            )
            if not all(checkpoint["trits_changed"] for checkpoint in row["checkpoints"]):
                raise Stage7Error("short-PV curve lacks hard-trit exports")
            pv[method] = (row, best)
        else:
            raise Stage7Error("refinement mode differs")
        del parent_ppl
    if set(scale) != set(RATES) or set(pv) != set(SOFT_METHODS):
        raise Stage7Error("refinement rate or soft-method inventory is incomplete")
    primary_scale = scale["R3"]
    best_publishable = primary_scale[0]["parent_candidate_id"]
    if {row[0]["parent_candidate_id"] for row in pv.values()} != {best_publishable}:
        raise Stage7Error("short-PV A/B does not share best publishable scale-only parent")
    refined_id, soft_method_ab = _soft_method_ab(
        pv, primary_scale[0]["refinement_id"]
    )
    curves = [*scale.values(), *pv.values()]
    selected_row, selected_metrics = next(
        curve for curve in curves if curve[0]["refinement_id"] == refined_id
    )
    if selected_metrics[2] == 0:
        return refined_id, soft_method_ab, None
    selected_checkpoint = next(
        checkpoint
        for checkpoint in selected_row["checkpoints"]
        if checkpoint["tokens"] == selected_metrics[2]
    )
    frozen_checkpoint = {
        "refinement_id": refined_id,
        "tokens": selected_checkpoint["tokens"],
        "package_id": selected_checkpoint["package_id"],
        "artifact_sha256": selected_checkpoint["artifact"]["sha256"],
        "evaluation_sha256": selected_checkpoint["evaluation_receipt"]["sha256"],
    }
    return refined_id, soft_method_ab, frozen_checkpoint


def _validate_trace(
    path: Path,
    *,
    campaign_path: Path,
    campaign: dict[str, Any],
    grid: dict[str, dict[str, Any]],
    quantized: int,
    quantized_tensors: int,
    quantized_tensor_names: tuple[str, ...],
    preserved: int,
) -> tuple[dict[str, Any], bytes, dict[str, dict[str, Any]], dict[str, Any]]:
    trace, raw = _load(path, "Stage-7 execution trace")
    _object(trace, TRACE_FIELDS, "Stage-7 execution trace")
    if trace["schema"] != TRACE_SCHEMA:
        raise Stage7Error("Stage-7 execution schema differs")
    for field in ("release", "source_revision", "run_id"):
        if trace[field] != campaign[field]:
            raise Stage7Error(f"execution {field} differs from campaign")
    if trace["campaign_sha256"] != _hash_file(campaign_path):
        raise Stage7Error("execution trace does not bind exact campaign bytes")
    stages = trace["stages"]
    if not isinstance(stages, list) or len(stages) != len(STAGES):
        raise Stage7Error("execution trace must contain three successive-halving stages")
    previous = sorted(grid)
    full_rows = {}
    physical_gate_failed = False
    for ordinal, name in enumerate(STAGES):
        stage = _object(stages[ordinal], STAGE_FIELDS, f"stages[{ordinal}]")
        if stage["name"] != name:
            raise Stage7Error("successive-halving stage order differs")
        inputs = _ids(stage["input_ids"], f"{name} inputs")
        if sorted(inputs) != previous:
            raise Stage7Error(f"{name} inputs differ from complete grid or prior promotions")
        raw_rows = stage["measurements"]
        if not isinstance(raw_rows, list) or len(raw_rows) != len(inputs):
            raise Stage7Error(f"{name} must measure every admitted input exactly once")
        rows = []
        seen = set()
        for row_ordinal, raw_row in enumerate(raw_rows):
            row, physical_pass = _measurement(
                raw_row,
                full=name == "full-model",
                grid=grid,
                campaign=campaign,
                trace_root=path.parent,
                quantized=quantized,
                quantized_tensors=quantized_tensors,
                preserved=preserved,
                label=f"{name}.measurements[{row_ordinal}]",
            )
            if not physical_pass:
                physical_gate_failed = True
            candidate_id = row["candidate_id"]
            if candidate_id not in inputs or candidate_id in seen:
                raise Stage7Error(f"{name} measurement coverage differs")
            seen.add(candidate_id)
            rows.append(row)
        if seen != set(inputs):
            raise Stage7Error(f"{name} measurement coverage is incomplete")
        promoted = _ids(stage["promoted_ids"], f"{name} promotions")
        expected = _expected_promotions(name, rows, grid)
        if sorted(promoted) != expected:
            raise Stage7Error(f"{name} promotions differ from frozen selection rule")
        previous = sorted(promoted)
        if name == "full-model":
            full_rows = {row["candidate_id"]: row for row in rows}

    baselines = _object(trace["baselines"], BASELINE_FIELDS, "baselines")
    bf16 = _object(baselines["bf16"], BF16_FIELDS, "bf16 baseline")
    _number(bf16["heldout_ppl"], "bf16 heldout perplexity", 1e-12)
    _tasks(bf16["task_metrics"], "bf16 task metrics")
    salt_rows = baselines["salt_v1"]
    if not isinstance(salt_rows, list) or len(salt_rows) != len(RATES):
        raise Stage7Error("SALT V1 baseline inventory is incomplete")
    salt = {}
    for ordinal, rate in enumerate(RATES):
        row = _object(salt_rows[ordinal], SALT_FIELDS, f"salt_v1[{ordinal}]")
        if row["rate"] != rate:
            raise Stage7Error("SALT V1 baseline rate order differs")
        codec = _string(row["codec"], "SALT V1 codec")
        if codec not in {value.lower() for value in CODECS}:
            raise Stage7Error("SALT V1 codec differs")
        source_recipe = next(iter(grid.values()))
        expected_baseline_id = "sha256:" + hashlib.sha256(canonical({
            "method": "salt-v1",
            "rate": rate,
            "codec": codec,
            "source_model_id": source_recipe["source_model_id"],
            "evaluation_id": source_recipe["evaluation_id"],
        })).hexdigest()
        if row["baseline_id"] != expected_baseline_id:
            raise Stage7Error("SALT V1 baseline id does not bind method/rate/evaluation")
        physical = _integer(row["physical_bytes"], "SALT V1 physical bytes", 1)
        resident = _integer(row["resident_bytes"], "SALT V1 resident bytes", physical)
        _number(row["heldout_ppl"], "SALT V1 heldout perplexity", 1e-12)
        _tasks(row["task_metrics"], "SALT V1 task metrics")
        artifact = _open_record(path.parent, row["artifact"], "SALT V1 artifact")
        physical_report = _open_record(
            path.parent, row["physical_report"], "SALT V1 physical report"
        )
        matrix_ceiling = math.floor(RATE_BPW[rate] * quantized / 8)
        metadata = math.floor(campaign["thresholds"]["metadata_bpw_max"] * quantized / 8)
        if not _validate_physical_report(
            physical_report,
            artifact=artifact,
            measurement={"physical_bytes": physical, "resident_bytes": resident},
            campaign=campaign,
            candidate_id=expected_baseline_id,
            quantized=quantized,
            quantized_tensors=quantized_tensors,
            expected_codec=codec,
            recipe={"artifact_byte_ceiling": matrix_ceiling + preserved + metadata},
            preserved=preserved,
        ):
            physical_gate_failed = True
        salt[rate] = row

    promoted = set(stages[-1]["promoted_ids"])
    selected = {}
    for rate in RATES:
        rows = [
            full_rows[candidate_id]
            for candidate_id in promoted
            if grid[candidate_id]["profile"] == RATE_PROFILE[rate]
        ]
        if rows:
            selected[rate] = min(
                rows,
                key=lambda row: (
                    row["heldout_ppl"], row["physical_bytes"], row["runtime_ms"],
                    row["candidate_id"],
                ),
            )["candidate_id"]
    complete_selection = set(selected) == set(RATES)
    if complete_selection:
        if any(
            salt[rate]["physical_bytes"] != full_rows[selected[rate]]["physical_bytes"]
            for rate in RATES
        ):
            raise Stage7Error("SALT V1 baseline is not matched to selected physical bytes")
        refined_id, soft_method_ab, refined_checkpoint = _validate_refinements(
            trace["refinements"],
            selected=selected,
            campaign=campaign,
            grid=grid,
            quantized_tensors=quantized_tensors,
            quantized_tensor_names=quantized_tensor_names,
            trace_root=path.parent,
        )
    else:
        if trace["refinements"] != []:
            raise Stage7Error("refinements must be absent when one PTQ rate is missing")
        refined_id = None
        refined_checkpoint = None
        soft_method_ab = {
            "outcome": "not-run", "winner": None, "reason": "missing-rate",
        }
    return trace, raw, full_rows, {
        "bf16": bf16, "salt_v1": salt, "selected": selected,
        "refined_id": refined_id, "soft_method_ab": soft_method_ab,
        "refined_checkpoint": refined_checkpoint, "promoted": promoted,
        "physical_gate_failed": physical_gate_failed,
    }


def _gate_reasons(
    grid: dict[str, dict[str, Any]],
    full_rows: dict[str, dict[str, Any]],
    state: dict[str, Any],
) -> list[str]:
    reasons = []
    selected = state["selected"]
    if state["physical_gate_failed"]:
        reasons.append("physical-accounting-gate-failed")
    if set(selected) != set(RATES):
        reasons.append("missing-nondominated-ptq-rate")
        return reasons
    if state["refined_checkpoint"] is None:
        reasons.append("no-refinement-checkpoint-improves-parent")
    r3_id = selected["R3"]
    if grid[r3_id]["solver"]["variant"] == "greedy":
        reasons.append("r3-selected-method-is-not-joint")
    salt_r3 = state["salt_v1"]["R3"]
    denominator = salt_r3["heldout_ppl"] - state["bf16"]["heldout_ppl"]
    closure = (
        (salt_r3["heldout_ppl"] - full_rows[r3_id]["heldout_ppl"]) / denominator
        if denominator > 0
        else float("-inf")
    )
    if closure < 0.25:
        reasons.append("r3-gap-closure-below-25-percent")
    if any(
        full_rows[r3_id]["task_metrics"][task] < salt_r3["task_metrics"][task]
        for task in TASKS
    ):
        reasons.append("r3-task-regression-versus-salt-v1")
    matched_win = False
    for candidate_id in state["promoted"]:
        control_id = _input_control_id(candidate_id, grid)
        candidate = full_rows[candidate_id]
        control = full_rows.get(control_id)
        if (
            control is not None
            and control["correct"]
            and candidate["correct"]
            and candidate["physical_bytes"] == control["physical_bytes"]
            and candidate["heldout_ppl"] < control["heldout_ppl"]
        ):
            matched_win = True
            break
    if not matched_win:
        reasons.append("no-configuration-matched-output-aware-curvature-win")
    return reasons


def _record(path: Path, raw: bytes) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Stage7Error(f"output already exists: {path}")
    parent = path.parent.resolve(strict=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise Stage7Error(f"output already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def qualify(
    campaign_path: Path,
    trace_path: Path,
    *,
    model_root: Path,
    smoke_model_root: Path,
    source_root: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    """Validate raw Stage-7 evidence and emit pass or terminal-negative receipt."""
    campaign_path = campaign_path.resolve(strict=True)
    trace_path = trace_path.resolve(strict=True)
    if campaign_path.parent != trace_path.parent:
        raise Stage7Error("campaign and trace must share one evidence directory")
    if output is not None and output.parent.resolve(strict=True) != campaign_path.parent:
        raise Stage7Error("qualification output must share campaign evidence directory")
    (
        campaign, campaign_raw, grid, model_id, quantized, preserved,
        quantized_tensors, quantized_tensor_names, prerequisite_reasons,
    ) = _validate_campaign(
        campaign_path,
        model_root=model_root,
        smoke_model_root=smoke_model_root,
        source_root=source_root,
    )
    _, trace_raw, full_rows, state = _validate_trace(
        trace_path,
        campaign_path=campaign_path,
        campaign=campaign,
        grid=grid,
        quantized=quantized,
        quantized_tensors=quantized_tensors,
        quantized_tensor_names=quantized_tensor_names,
        preserved=preserved,
    )
    reasons = prerequisite_reasons + _gate_reasons(grid, full_rows, state)
    authorized = not reasons
    receipt = {
        "schema": QUALIFICATION_SCHEMA,
        "result": "pass" if authorized else "negative",
        "release": campaign["release"],
        "source_revision": campaign["source_revision"],
        "run_id": campaign["run_id"],
        "model_id": model_id,
        "model_revision": campaign["model"]["revision"],
        "campaign": _record(campaign_path, campaign_raw),
        "trace": _record(trace_path, trace_raw),
        "freeze_authorized": authorized,
        "freeze_reasons": reasons,
        "frozen_ptq_recipe_ids": (
            {rate: state["selected"][rate] for rate in PUBLISHABLE_RATES}
            if authorized else {}
        ),
        "r4_control_recipe_id": state["selected"].get("R4") if authorized else None,
        "frozen_refined_recipe_id": state["refined_id"] if authorized else None,
        "frozen_refined_checkpoint": (
            state["refined_checkpoint"] if authorized else None
        ),
        "soft_method_ab": state["soft_method_ab"],
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    if output is not None:
        _write_atomic(output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qualify plan-0043 Stage-7 recipe freeze or terminal negative"
    )
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--smoke-model-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        receipt = qualify(
            args.campaign,
            args.trace,
            model_root=args.model_root,
            smoke_model_root=args.smoke_model_root,
            source_root=args.source_root,
            output=args.output,
        )
    except (OSError, Stage7Error) as error:
        parser.error(str(error))
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt["freeze_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
