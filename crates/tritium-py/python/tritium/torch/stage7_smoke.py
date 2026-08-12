"""Resumable, evidence-bound Stage-7 model smoke execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Union

import torch
from torch import nn

from .config import TernaryConfig
from .conversion import prepare
from .module_artifacts import load_module_conversion, load_packed_module
from .ptq import (
    _hash_value,
    _source_model_digest,
    calibrate,
    convert,
    load_activation_calibration,
    load_quantized_module,
)
from .stage7 import Stage7CausalData

Pathish = Union[str, os.PathLike[str]]
_SCHEMA = "tritium.stage7-smoke-model.v1"
_REQUEST_SCHEMA = "tritium.stage7-smoke-model-request.v1"
_ALLOCATION_SCHEMA = "tritium.stage7-smoke-allocation.v1"
_EVALUATION_SCHEMA = "tritium.stage7-smoke-evaluation.v1"
_SMOKE_SCHEMA = "tritium.stage7-smoke.v2"
_SMOKE_EXECUTION_SCHEMA = "tritium.stage7-smoke-execution.v2"
_STAGES = ("capture", "fit", "allocate", "package", "evaluate")
_MAX_JSON_BYTES = 8 * 1024 * 1024
_TOP_LEVEL_FILES = {
    "request.json",
    "calibration",
    "conversion",
    "allocation.json",
    "packed",
    "evaluation.json",
    "result.json",
}
_CAMPAIGN_FIELDS = {
    "schema",
    "release",
    "source_revision",
    "run_id",
    "model",
    "smoke_model",
    "smoke_provenance",
    "provenance",
    "thresholds",
    "recipe_count",
    "recipe_grid_id",
    "token_evidence_pack",
    "evidence",
}
_MODEL_FIELDS = {"repo_id", "revision", "files"}
_FILE_FIELDS = {"path", "bytes", "sha256"}
_TOKEN_MANIFEST_FIELDS = {
    "schema",
    "pack_id",
    "tokenizer_digest",
    "tokenizer_vocab_size",
    "token_encoding",
    "tokens",
    "partitions",
}
_SMOKE_PROVENANCE_FIELDS = {
    "evaluation_id",
    "evaluation_members",
    "calibration_id",
    "dataset_repo_id",
    "dataset_revision",
    "sampling_seed",
    "tokenizer_digest",
    "ordered_token_digest",
    "sequence_count",
    "tokens_per_sequence",
    "prefix_start",
    "prefix_end",
}
_PROVENANCE_FIELDS = {"calibration", "refinement", "validation", "evaluation"}
_PARTITION_PROVENANCE_FIELDS = {
    "id",
    "members",
    "datasets",
    "sampling_seed",
    "tokenizer_digest",
    "ordered_token_digest",
    "sequence_count",
    "tokens_per_sequence",
}
_MODEL_FILES = {
    ".gitattributes",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
_TOKENIZER_FILES = {
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
SMOLLM2_135M_REPO_ID = "HuggingFaceTB/SmolLM2-135M"
SMOLLM2_135M_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SMOLLM2_135M_MODEL_ID = (
    "sha256:18686427230dde98ee2926dafa133b5cb0c6f4de48eacd0a57e5d2ed76e15e57"
)
SMOLLM2_C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"


@dataclass(frozen=True)
class Stage7SmokeModelResult:
    """Strictly reopened result of one real five-stage model smoke."""

    schema: str
    result: str
    artifact_dir: Path
    artifact_path: Path
    request_id: str
    source_model_digest: str
    evidence_id: str
    conversion_artifact_id: str
    allocation_id: str
    packed_artifact_id: str
    package_id: str
    packing: str
    serialized_bytes: int
    resident_bytes: int
    tensor_count: int
    evaluation_receipt_id: str
    mean_loss: float
    evaluated_tokens: int
    stage_names: tuple[str, ...]
    stage_ids: tuple[str, ...]
    terminal_validated: bool


@dataclass(frozen=True)
class Stage7SmolLM2SmokeResult:
    """Frozen SmolLM2 smoke plus qualifier-compatible receipt paths."""

    model: Stage7SmokeModelResult
    smoke_receipt_path: Path
    execution_log_path: Path
    model_id: str
    model_revision: str
    evaluation_id: str


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _pairs_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _load_json(path: Path, *, fields: set[str], label: str) -> dict[str, Any]:
    metadata = path.lstat()
    if path.is_symlink() or not path.is_file() or metadata.st_size > _MAX_JSON_BYTES:
        raise ValueError(f"{label} must be a bounded ordinary file")
    try:
        encoded = path.read_bytes()
        value = json.loads(
            encoded,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} fields differ from schema")
    if encoded != _canonical(value) + b"\n":
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_or_match(path: Path, value: dict[str, Any], *, label: str) -> None:
    if path.exists() or path.is_symlink():
        observed = _load_json(path, fields=set(value), label=label)
        if observed != value:
            raise ValueError(f"{label} differs from current smoke request")
        return
    _atomic_json(path, value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _open_record(
    base: Path,
    record: Any,
    *,
    label: str,
    allow_symlink: bool = False,
) -> Path:
    if not isinstance(record, dict) or set(record) != _FILE_FIELDS:
        raise ValueError(f"{label} fields differ from file-record schema")
    logical = record["path"]
    if not isinstance(logical, str) or not logical:
        raise ValueError(f"{label}.path must be a nonempty string")
    parsed = PurePosixPath(logical)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != logical:
        raise ValueError(f"{label}.path must be a canonical relative path")
    size = record["bytes"]
    digest = record["sha256"]
    if (
        type(size) is not int
        or size < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} byte or digest ledger is invalid")
    path = base.joinpath(*parsed.parts)
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing") from error
    if (path.is_symlink() and not allow_symlink) or not path.is_file():
        raise ValueError(f"{label} must be an ordinary file")
    if path.stat().st_size != size or _sha256_file(path) != digest:
        raise ValueError(f"{label} differs from its exact file record")
    del metadata
    return path


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    metadata = path.lstat()
    if path.is_symlink() or not path.is_file() or metadata.st_size > _MAX_JSON_BYTES:
        raise ValueError(f"{label} must be a bounded ordinary file")
    try:
        value = json.loads(
            path.read_bytes(), object_pairs_hook=_pairs_without_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _smollm_scope(
    campaign: dict[str, Any], model_dir: Path
) -> tuple[str, str, int, int]:
    model = campaign["smoke_model"]
    if not isinstance(model, dict) or set(model) != _MODEL_FIELDS:
        raise ValueError("Stage-7 smoke model fields differ from schema")
    if (
        model["repo_id"] != SMOLLM2_135M_REPO_ID
        or model["revision"] != SMOLLM2_135M_REVISION
    ):
        raise ValueError("Stage-7 smoke model differs from frozen SmolLM2-135M")
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise ValueError("Stage-7 smoke model root must be an ordinary directory")
    resolved = model_dir.resolve(strict=True)
    if resolved.parent.name == "snapshots" and resolved.name != SMOLLM2_135M_REVISION:
        raise ValueError("Hugging Face snapshot path differs from frozen model revision")
    records = model["files"]
    file_paths = {
        item.get("path") for item in records if isinstance(item, dict)
    } if isinstance(records, list) else set()
    if not isinstance(records, list) or file_paths != _MODEL_FILES:
        raise ValueError("Stage-7 smoke model file inventory differs from frozen revision")
    if len(records) != len(_MODEL_FILES):
        raise ValueError("Stage-7 smoke model file inventory contains duplicates")
    opened = {
        record["path"]: _open_record(
            resolved,
            record,
            label=f"smoke_model.files[{index}]",
            allow_symlink=True,
        )
        for index, record in enumerate(records)
    }
    model_id = _digest(model)
    if model_id != SMOLLM2_135M_MODEL_ID:
        raise ValueError("Stage-7 smoke model file identities differ from frozen revision")
    tokenizer_records = sorted(
        (record for record in records if record["path"] in _TOKENIZER_FILES),
        key=lambda record: record["path"],
    )
    tokenizer_digest = _digest(tokenizer_records)
    config = _load_object(opened["config.json"], label="SmolLM config")
    vocab_size = config.get("vocab_size")
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("SmolLM config vocab_size is invalid")
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("Stage-7 SmolLM smoke requires safetensors") from error
    rank_two = set()
    for logical, path in opened.items():
        if not logical.endswith(".safetensors"):
            continue
        with safe_open(path, framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                shape = tensors.get_slice(name).get_shape()
                if len(shape) == 2:
                    if name in rank_two:
                        raise ValueError("Stage-7 smoke rank-2 tensor inventory duplicates")
                    rank_two.add(name)
    if not rank_two:
        raise ValueError("Stage-7 smoke model contains no rank-2 tensors")
    return model_id, tokenizer_digest, vocab_size, len(rank_two)


def _campaign_smoke_data(
    campaign_path: Path,
    campaign: dict[str, Any],
    *,
    tokenizer_digest: str,
    vocab_size: int,
    batch_sequences: int,
    device: str,
) -> tuple[Stage7CausalData, str]:
    provenance = campaign["smoke_provenance"]
    if not isinstance(provenance, dict) or set(provenance) != _SMOKE_PROVENANCE_FIELDS:
        raise ValueError("Stage-7 smoke provenance fields differ from schema")
    members = provenance["evaluation_members"]
    if (
        not isinstance(members, list)
        or len(members) != 128
        or any(not isinstance(member, str) for member in members)
        or provenance["sequence_count"] != 128
        or provenance["tokens_per_sequence"] != 2_048
        or provenance["prefix_start"] != 0
        or provenance["prefix_end"] != 128
    ):
        raise ValueError("Stage-7 smoke provenance must bind exact 128-sequence prefix")
    evaluation_id = _digest(members)
    if (
        provenance["evaluation_id"] != evaluation_id
        or provenance["ordered_token_digest"] != evaluation_id
        or provenance["tokenizer_digest"] != tokenizer_digest
        or provenance["dataset_repo_id"] != "allenai/c4"
        or provenance["dataset_revision"] != SMOLLM2_C4_REVISION
    ):
        raise ValueError("Stage-7 smoke provenance differs from frozen C4 prefix")
    partitions = campaign["provenance"]
    if not isinstance(partitions, dict) or set(partitions) != _PROVENANCE_FIELDS:
        raise ValueError("Stage-7 campaign partition inventory differs from schema")
    calibration = partitions["calibration"]
    if (
        not isinstance(calibration, dict)
        or set(calibration) != _PARTITION_PROVENANCE_FIELDS
    ):
        raise ValueError("Stage-7 calibration provenance fields differ from schema")
    calibration_identity = {
        field: calibration[field]
        for field in _PARTITION_PROVENANCE_FIELDS - {"id"}
    }
    calibration_members = calibration["members"]
    calibration_datasets = calibration["datasets"]
    if (
        not isinstance(calibration_members, list)
        or len(calibration_members) != 512
        or not isinstance(calibration_datasets, list)
        or not calibration_datasets
        or not isinstance(calibration_datasets[0], dict)
        or calibration_datasets[0].get("repo_id") != "allenai/c4"
        or calibration_datasets[0].get("revision") != SMOLLM2_C4_REVISION
        or calibration["id"] != _digest(calibration_identity)
        or provenance["calibration_id"] != calibration["id"]
        or calibration_members[:128] != members
        or calibration["sampling_seed"] != provenance["sampling_seed"]
        or calibration["tokenizer_digest"] != tokenizer_digest
        or calibration["sequence_count"] != 512
        or calibration["tokens_per_sequence"] != 2_048
    ):
        raise ValueError("Stage-7 smoke provenance differs from calibration parent")
    manifest_path = _open_record(
        campaign_path.parent,
        campaign["token_evidence_pack"],
        label="campaign token evidence pack",
    )
    manifest = _load_json(
        manifest_path,
        fields=_TOKEN_MANIFEST_FIELDS,
        label="Stage-7 token evidence manifest",
    )
    if (
        manifest["schema"] != "tritium.stage7-token-evidence-pack.v1"
        or manifest["tokenizer_digest"] != tokenizer_digest
        or manifest["tokenizer_vocab_size"] != vocab_size
        or manifest["token_encoding"] != "u32le"
    ):
        raise ValueError("Stage-7 token evidence differs from SmolLM tokenizer")
    data = Stage7CausalData.open(
        manifest_path,
        expected_pack_id=manifest["pack_id"],
        expected_tokenizer_digest=tokenizer_digest,
        expected_tokenizer_vocab_size=vocab_size,
        partition="calibration",
        start_sequence=0,
        sequence_count=128,
        batch_sequences=batch_sequences,
        device=device,
    )
    if (
        list(data.receipt.sequence_ids) != members
        or data.receipt.ordered_members_sha256 != evaluation_id
        or data.receipt.sampling_seed != provenance["sampling_seed"]
    ):
        raise ValueError("Stage-7 token window differs from frozen smoke members")
    return data, evaluation_id


def _stream_digest(data: Stage7CausalData) -> str:
    digest = hashlib.sha256()
    batches = 0
    for batches, batch in enumerate(data, 1):
        _hash_value(digest, f"batch[{batches - 1}]", batch)
    if batches == 0:
        raise ValueError("Stage-7 smoke data must yield at least one batch")
    return "sha256:" + digest.hexdigest()


def _scale_group_size(shape: tuple[int, ...]) -> int:
    if len(shape) != 2 or shape[1] <= 0:
        raise ValueError("Stage-7 packed weight must have positive rank-2 geometry")
    if shape[1] % 128 == 0:
        return 128
    if shape[1] % 64 == 0:
        return 64
    raise ValueError("Stage-7 packed weight columns must be G64-aligned")


def _validate_ptq_options(profile: str, target_bpw: float | None) -> None:
    if profile not in {"compact-v1", "near-lossless-v1"}:
        raise ValueError("profile must be 'compact-v1' or 'near-lossless-v1'")
    if target_bpw is not None and (
        isinstance(target_bpw, bool)
        or not isinstance(target_bpw, (int, float))
        or not math.isfinite(float(target_bpw))
        or target_bpw <= 0
    ):
        raise ValueError("target_bpw must be finite and positive when provided")


def _model_output(output: Any, field: str) -> Any:
    return output.get(field) if isinstance(output, Mapping) else getattr(output, field, None)


def _evaluate(model: nn.Module, data: Stage7CausalData) -> tuple[float, int]:
    model.eval()
    loss_sum = 0.0
    tokens = 0
    with torch.no_grad():
        for batch in data:
            output = model(**batch)
            labels = batch["labels"]
            mask = batch["attention_mask"]
            selected = labels[..., 1:].ne(-100) & mask[..., 1:].bool()
            count = int(selected.count_nonzero().item())
            if count == 0:
                raise ValueError("Stage-7 evaluation batch has no causal targets")
            loss = _model_output(output, "loss")
            if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                logits = _model_output(output, "logits")
                if not isinstance(logits, torch.Tensor) or logits.ndim < 3:
                    raise ValueError("model evaluation must expose scalar loss or causal logits")
                shifted_logits = logits[..., :-1, :][selected]
                shifted_labels = labels[..., 1:][selected]
                loss = torch.nn.functional.cross_entropy(shifted_logits, shifted_labels)
            scalar = float(loss.detach().to(device="cpu", dtype=torch.float64))
            if not math.isfinite(scalar) or scalar < 0:
                raise ValueError("Stage-7 evaluation produced an invalid loss")
            loss_sum += scalar * count
            tokens += count
    if tokens == 0:
        raise ValueError("Stage-7 evaluation consumed no causal targets")
    return loss_sum / tokens, tokens


def _result_from_value(target: Path, value: dict[str, Any]) -> Stage7SmokeModelResult:
    identity = dict(value)
    result_id = identity.pop("result_id")
    if result_id != _digest(identity):
        raise ValueError("Stage-7 smoke result identity differs")
    package = value["package"]
    evaluation = value["evaluation"]
    stages = value["stages"]
    if [stage["name"] for stage in stages] != list(_STAGES):
        raise ValueError("Stage-7 smoke stage order differs")
    if any(set(stage) != {"name", "id"} for stage in stages):
        raise ValueError("Stage-7 smoke stage fields differ")
    return Stage7SmokeModelResult(
        schema=value["schema"],
        result=value["result"],
        artifact_dir=target,
        artifact_path=target / package["path"],
        request_id=value["request_id"],
        source_model_digest=value["source_model_digest"],
        evidence_id=value["evidence_id"],
        conversion_artifact_id=value["conversion_artifact_id"],
        allocation_id=value["allocation_id"],
        packed_artifact_id=value["packed_artifact_id"],
        package_id=package["package_id"],
        packing=package["packing"],
        serialized_bytes=package["serialized_bytes"],
        resident_bytes=package["resident_bytes"],
        tensor_count=package["tensor_count"],
        evaluation_receipt_id=evaluation["evaluation_id"],
        mean_loss=evaluation["mean_loss"],
        evaluated_tokens=evaluation["evaluated_tokens"],
        stage_names=tuple(stage["name"] for stage in stages),
        stage_ids=tuple(stage["id"] for stage in stages),
        terminal_validated=value["terminal_validated"],
    )


def run_stage7_smoke_model(
    model: nn.Module,
    data: Stage7CausalData,
    output_dir: Pathish,
    *,
    packing: str = "b3",
    profile: str = "compact-v1",
    target_bpw: float | None = None,
    max_evidence_bytes: int = 64 * 1024 * 1024,
    max_working_bytes: int = 512 * 1024 * 1024,
    max_payload_bytes: int = 8 * 1024 * 1024 * 1024,
) -> Stage7SmokeModelResult:
    """Execute or strictly resume real capture, fit, allocation, package and evaluation.

    This model-level primitive emits no release-qualified Stage-7 claim. Frozen
    SmolLM model/campaign binding is owned by ``run_stage7_smollm2_smoke``.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(data, Stage7CausalData):
        raise TypeError("data must be terminally validated Stage7CausalData")
    if not data.receipt.terminal_validated:
        raise ValueError("Stage-7 smoke data lacks terminal validation")
    if packing not in {"d2", "b3", "s34"}:
        raise ValueError("packing must be 'd2', 'b3', or 's34'")
    _validate_ptq_options(profile, target_bpw)
    for label, value in (
        ("max_evidence_bytes", max_evidence_bytes),
        ("max_working_bytes", max_working_bytes),
        ("max_payload_bytes", max_payload_bytes),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")

    target = Path(output_dir).absolute()
    if target.is_symlink():
        raise ValueError("Stage-7 smoke output must not be a symlink")
    target.mkdir(parents=True, exist_ok=True)
    if not target.is_dir():
        raise ValueError("Stage-7 smoke output must be an ordinary directory")
    unknown = {child.name for child in target.iterdir()} - _TOP_LEVEL_FILES
    if unknown:
        raise ValueError(f"Stage-7 smoke output contains unknown entries: {sorted(unknown)}")

    config = TernaryConfig.ptq(
        profile=profile,
        target_modules=("Linear", "Embedding"),
        target_bpw=None if target_bpw is None else float(target_bpw),
    )
    source_digest = _source_model_digest(model)
    token_stream_digest = _stream_digest(data)
    data_identity = json.loads(_canonical(asdict(data.receipt)))
    request = {
        "schema": _REQUEST_SCHEMA,
        "source_model_digest": source_digest,
        "data": data_identity,
        "config": config.to_dict(),
        "packing": packing,
        "max_evidence_bytes": max_evidence_bytes,
        "max_working_bytes": max_working_bytes,
        "max_payload_bytes": max_payload_bytes,
        "token_stream_digest": token_stream_digest,
    }
    request["request_id"] = _digest(request)
    _write_or_match(target / "request.json", request, label="Stage-7 smoke request")

    prepared = prepare(model, config, inplace=True)
    calibration_path = target / "calibration"
    if calibration_path.exists() or calibration_path.is_symlink():
        calibration = load_activation_calibration(
            calibration_path, max_evidence_bytes=max_evidence_bytes
        )
    else:
        calibration = calibrate(
            prepared,
            data,
            evidence_dir=calibration_path,
            max_evidence_bytes=max_evidence_bytes,
        )
    if (
        calibration.source_model_digest != source_digest
        or calibration.token_stream_digest != token_stream_digest
    ):
        raise ValueError("Stage-7 capture evidence differs from current source or tokens")

    conversion_path = target / "conversion"
    if conversion_path.exists() or conversion_path.is_symlink():
        conversion = load_module_conversion(conversion_path)
    else:
        conversion = convert(
            prepared,
            calibration,
            work_dir=conversion_path,
            max_working_bytes=max_working_bytes,
        )
    if (
        conversion.source_model_digest != source_digest
        or conversion.evidence_id != calibration.evidence_id
    ):
        raise ValueError("Stage-7 fitted conversion differs from capture evidence")

    allocation = {
        "schema": _ALLOCATION_SCHEMA,
        "request_id": request["request_id"],
        "conversion_artifact_id": conversion.artifact_id,
        "packing": packing,
        "tensors": [
            {
                "path": weight.path,
                "shape": list(weight.shape),
                "planes": len(weight.planes),
                "scale_group_size": _scale_group_size(weight.shape),
            }
            for weight in conversion.weights
        ],
    }
    allocation["allocation_id"] = _digest(allocation)
    _write_or_match(
        target / "allocation.json", allocation, label="Stage-7 smoke allocation"
    )

    packed_path = target / "packed"
    if packed_path.exists() or packed_path.is_symlink():
        packed = load_packed_module(packed_path)
    else:
        packed = conversion.pack_native(
            packed_path,
            packing=packing,
            max_payload_bytes=max_payload_bytes,
        )
    if (
        packed.conversion_artifact_id != conversion.artifact_id
        or packed.packing != packing
        or packed.tensors != len(allocation["tensors"])
    ):
        raise ValueError("Stage-7 package differs from fitted allocation")

    evaluation_path = target / "evaluation.json"
    evaluation_fields = {
        "schema",
        "request_id",
        "conversion_artifact_id",
        "packed_artifact_id",
        "package_id",
        "data_pack_id",
        "ordered_members_sha256",
        "mean_loss",
        "evaluated_tokens",
        "evaluation_id",
    }
    if evaluation_path.exists() or evaluation_path.is_symlink():
        evaluation = _load_json(
            evaluation_path,
            fields=evaluation_fields,
            label="Stage-7 smoke evaluation",
        )
        identity = dict(evaluation)
        evaluation_id = identity.pop("evaluation_id")
        if evaluation_id != _digest(identity):
            raise ValueError("Stage-7 smoke evaluation identity differs")
    else:
        quantized = load_quantized_module(model, conversion, inplace=False)
        mean_loss, evaluated_tokens = _evaluate(quantized, data)
        evaluation = {
            "schema": _EVALUATION_SCHEMA,
            "request_id": request["request_id"],
            "conversion_artifact_id": conversion.artifact_id,
            "packed_artifact_id": packed.artifact_id,
            "package_id": packed.package_id,
            "data_pack_id": data.receipt.pack_id,
            "ordered_members_sha256": data.receipt.ordered_members_sha256,
            "mean_loss": mean_loss,
            "evaluated_tokens": evaluated_tokens,
        }
        evaluation["evaluation_id"] = _digest(evaluation)
        _atomic_json(evaluation_path, evaluation)
    expected_evaluation = {
        "schema": _EVALUATION_SCHEMA,
        "request_id": request["request_id"],
        "conversion_artifact_id": conversion.artifact_id,
        "packed_artifact_id": packed.artifact_id,
        "package_id": packed.package_id,
        "data_pack_id": data.receipt.pack_id,
        "ordered_members_sha256": data.receipt.ordered_members_sha256,
    }
    if any(evaluation[field] != value for field, value in expected_evaluation.items()):
        raise ValueError("Stage-7 evaluation differs from current model, package, or data")
    if (
        not isinstance(evaluation["evaluated_tokens"], int)
        or evaluation["evaluated_tokens"] <= 0
        or not isinstance(evaluation["mean_loss"], (int, float))
        or not math.isfinite(float(evaluation["mean_loss"]))
        or float(evaluation["mean_loss"]) < 0
    ):
        raise ValueError("Stage-7 evaluation metrics are invalid")

    stages = [
        {"name": "capture", "id": calibration.evidence_id},
        {"name": "fit", "id": conversion.artifact_id},
        {"name": "allocate", "id": allocation["allocation_id"]},
        {"name": "package", "id": packed.artifact_id},
        {"name": "evaluate", "id": evaluation["evaluation_id"]},
    ]
    result_value = {
        "schema": _SCHEMA,
        "result": "pass",
        "request_id": request["request_id"],
        "source_model_digest": source_digest,
        "evidence_id": calibration.evidence_id,
        "conversion_artifact_id": conversion.artifact_id,
        "allocation_id": allocation["allocation_id"],
        "packed_artifact_id": packed.artifact_id,
        "package": {
            "path": "packed/weights.tsalt2",
            "package_id": packed.package_id,
            "packing": packed.packing,
            "serialized_bytes": packed.serialized_bytes,
            "resident_bytes": packed.resident_bytes,
            "tensor_count": packed.tensors,
        },
        "evaluation": {
            "evaluation_id": evaluation["evaluation_id"],
            "mean_loss": float(evaluation["mean_loss"]),
            "evaluated_tokens": evaluation["evaluated_tokens"],
        },
        "stages": stages,
        "terminal_validated": True,
    }
    result_value["result_id"] = _digest(result_value)
    result_path = target / "result.json"
    _write_or_match(result_path, result_value, label="Stage-7 smoke result")
    reopened = _load_json(
        result_path, fields=set(result_value), label="Stage-7 smoke result"
    )
    result = _result_from_value(target, reopened)
    if (
        not result.artifact_path.is_file()
        or result.artifact_path.stat().st_size != packed.serialized_bytes
    ):
        raise ValueError("Stage-7 smoke package disappeared after result publication")
    return result


def run_stage7_smollm2_smoke(
    campaign_path: Pathish,
    model_dir: Pathish,
    output_dir: Pathish,
    *,
    device: str = "cuda",
    batch_sequences: int = 1,
    packing: str = "b3",
    profile: str = "compact-v1",
    target_bpw: float | None = None,
    max_evidence_bytes: int = 64 * 1024 * 1024,
    max_working_bytes: int = 512 * 1024 * 1024,
    max_payload_bytes: int = 8 * 1024 * 1024 * 1024,
) -> Stage7SmolLM2SmokeResult:
    """Run frozen 135M smoke and emit qualifier-compatible evidence.

    Campaign input may be a pre-evidence template: smoke evidence inventory is
    not trusted here. Final Stage-7 qualifier independently validates resulting
    receipt through clean source, model, token, and native package boundaries.
    """

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA Stage-7 smoke requested but CUDA is unavailable")
    if type(batch_sequences) is not int or not 1 <= batch_sequences <= 128:
        raise ValueError("batch_sequences must be an integer between 1 and 128")
    _validate_ptq_options(profile, target_bpw)
    requested_campaign = Path(campaign_path)
    if requested_campaign.is_symlink():
        raise ValueError("Stage-7 campaign path must not be a symlink")
    campaign_file = requested_campaign.resolve(strict=True)
    campaign = _load_json(
        campaign_file, fields=_CAMPAIGN_FIELDS, label="Stage-7 campaign template"
    )
    if campaign["schema"] != "tritium.stage7-campaign.v1":
        raise ValueError("Stage-7 campaign schema differs")
    release = campaign["release"]
    source_revision = campaign["source_revision"]
    if not isinstance(release, str) or not release:
        raise ValueError("Stage-7 campaign release must be nonempty")
    if (
        not isinstance(source_revision, str)
        or len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise ValueError("Stage-7 campaign source revision must be a full lowercase Git ID")
    model_root = Path(model_dir)
    model_id, tokenizer_digest, vocab_size, expected_tensors = _smollm_scope(
        campaign, model_root
    )
    data, evaluation_id = _campaign_smoke_data(
        campaign_file,
        campaign,
        tokenizer_digest=tokenizer_digest,
        vocab_size=vocab_size,
        batch_sequences=batch_sequences,
        device=device,
    )

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("Stage-7 SmolLM smoke requires transformers") from error
    source = AutoModelForCausalLM.from_pretrained(
        model_root.resolve(strict=True),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
    ).eval()
    if getattr(source.config, "vocab_size", None) != vocab_size:
        raise ValueError("loaded SmolLM vocabulary differs from frozen source")
    source.to(device)

    target = Path(output_dir).absolute()
    if target.is_symlink():
        raise ValueError("Stage-7 SmolLM smoke output must not be a symlink")
    if target.exists():
        if not target.is_dir():
            raise ValueError("Stage-7 SmolLM smoke output must be a directory")
        allowed = {"model", "execution.json", "smoke-receipt.json"}
        unknown = {child.name for child in target.iterdir()} - allowed
        if unknown:
            raise ValueError(
                f"Stage-7 SmolLM smoke output contains unknown entries: {sorted(unknown)}"
            )
    model_result = run_stage7_smoke_model(
        source,
        data,
        target / "model",
        packing=packing,
        profile=profile,
        target_bpw=target_bpw,
        max_evidence_bytes=max_evidence_bytes,
        max_working_bytes=max_working_bytes,
        max_payload_bytes=max_payload_bytes,
    )
    if model_result.tensor_count != expected_tensors:
        raise ValueError(
            "Stage-7 smoke package does not cover exact source-derived rank-2 inventory"
        )
    artifact_sha256 = _sha256_file(model_result.artifact_path)
    execution = {
        "schema": _SMOKE_EXECUTION_SCHEMA,
        "result": "pass",
        "release": release,
        "source_revision": source_revision,
        "model_id": model_id,
        "model_revision": SMOLLM2_135M_REVISION,
        "evaluation_id": evaluation_id,
        "profile": profile,
        "target_bpw": target_bpw,
        "artifact_sha256": artifact_sha256,
        "stages": [
            {"name": name, "result": "pass"} for name in model_result.stage_names
        ],
    }
    execution_path = target / "execution.json"
    _write_or_match(
        execution_path, execution, label="Stage-7 SmolLM smoke execution log"
    )
    execution_record = {
        "path": "execution.json",
        "bytes": execution_path.stat().st_size,
        "sha256": _sha256_file(execution_path),
    }
    artifact_record = {
        "path": "model/packed/weights.tsalt2",
        "bytes": model_result.serialized_bytes,
        "sha256": artifact_sha256,
    }
    smoke = {
        "schema": _SMOKE_SCHEMA,
        "result": "pass",
        "release": release,
        "source_revision": source_revision,
        "model_id": model_id,
        "model_revision": SMOLLM2_135M_REVISION,
        "evaluation_id": evaluation_id,
        "profile": profile,
        "target_bpw": target_bpw,
        "artifact": artifact_record,
        "package_id": model_result.package_id,
        "codec": model_result.packing,
        "serialized_bytes": model_result.serialized_bytes,
        "resident_bytes": model_result.resident_bytes,
        "tensor_count": model_result.tensor_count,
        "execution_log": execution_record,
    }
    smoke_path = target / "smoke-receipt.json"
    _write_or_match(smoke_path, smoke, label="Stage-7 SmolLM smoke receipt")
    return Stage7SmolLM2SmokeResult(
        model=model_result,
        smoke_receipt_path=smoke_path,
        execution_log_path=execution_path,
        model_id=model_id,
        model_revision=SMOLLM2_135M_REVISION,
        evaluation_id=evaluation_id,
    )


__all__ = [
    "SMOLLM2_135M_MODEL_ID",
    "SMOLLM2_135M_REPO_ID",
    "SMOLLM2_135M_REVISION",
    "Stage7SmokeModelResult",
    "Stage7SmolLM2SmokeResult",
    "run_stage7_smoke_model",
    "run_stage7_smollm2_smoke",
]
