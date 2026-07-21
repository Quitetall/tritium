"""Resumable phased Qwen3.6 PTQ lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import nn

from .. import _tritium
from ..salt import reconcile_qwen36_ptq_packages
from .artifacts import QuantizationResult, load
from .config import TernaryConfig
from .conversion import PreparedModel, prepare
from .errors import TritiumError
from .module_artifacts import (
    FittedWeight,
    ModuleQuantizationResult,
    load_module_conversion,
    module_recipe_id,
    seal_module_conversion,
)
from .projection import TernaryPlane, TernaryProjection, validate_projection

Pathish = Union[str, os.PathLike[str]]


@dataclass(frozen=True)
class CalibrationReceipt:
    """Content-bound admission of a complete canonical PTQ evidence namespace."""

    evidence_dir: Path
    evidence_id: str
    curvature: str
    record_count: int
    source_model_digest: str
    activation_cache_digest: str
    token_stream_digest: str
    max_evidence_bytes: int
    schema_version: int = 1


@dataclass(frozen=True)
class ActivationRecord:
    """One bounded diagonal-curvature record for a selected module input."""

    module: str
    weight_aliases: Tuple[str, ...]
    features: int
    samples: int
    file: str
    digest: str
    bytes: int


@dataclass(frozen=True)
class ActivationCalibrationReceipt:
    """Strict receipt for streamed activation evidence from a live module."""

    evidence_dir: Path
    evidence_id: str
    curvature: str
    record_count: int
    source_model_digest: str
    activation_cache_digest: str
    token_stream_digest: str
    max_evidence_bytes: int
    records: Tuple[ActivationRecord, ...]
    schema_version: int = 1


def _hash_field(digest: "hashlib._Hash", tag: str, payload: bytes) -> None:
    encoded = tag.encode("utf-8")
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _hash_tensor(digest: "hashlib._Hash", tag: str, value: torch.Tensor) -> None:
    tensor = value.detach()
    _hash_field(digest, f"{tag}:dtype", str(tensor.dtype).encode("ascii"))
    _hash_field(
        digest,
        f"{tag}:shape",
        json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"),
    )
    flat = tensor.contiguous().view(-1)
    chunk_elements = max(1, (1024 * 1024) // max(1, flat.element_size()))
    for offset in range(0, flat.numel(), chunk_elements):
        chunk = flat[offset : offset + chunk_elements].cpu().numpy().tobytes()
        _hash_field(digest, f"{tag}:chunk", chunk)


def _hash_value(digest: "hashlib._Hash", tag: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        _hash_tensor(digest, tag, value)
    elif isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("calibration batch mappings require string keys")
        for key in sorted(value):
            _hash_value(digest, f"{tag}.{key}", value[key])
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _hash_value(digest, f"{tag}[{index}]", item)
    elif value is None or type(value) in {bool, int, float, str}:
        _hash_field(
            digest,
            tag,
            json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8"),
        )
    else:
        raise TypeError(
            f"unsupported calibration batch value at {tag}: {type(value).__name__}"
        )


def _source_model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        _hash_tensor(digest, f"state.{name}", value)
    return f"sha256:{digest.hexdigest()}"


def _selected_linear_modules(
    prepared: PreparedModel,
) -> Tuple[Tuple[str, nn.Linear, Tuple[str, ...]], ...]:
    assert isinstance(prepared.model, nn.Module)
    selected = {
        alias
        for entry in prepared.coverage.entries
        if entry.disposition == "selected"
        for alias in entry.aliases
    }
    records = []
    seen = set()
    weight_owners = {}
    for path, module in prepared.model.named_modules(remove_duplicate=False):
        weight_name = f"{path}.weight" if path else "weight"
        if not isinstance(module, nn.Linear) or weight_name not in selected:
            continue
        key = id(module)
        if key in seen:
            continue
        seen.add(key)
        prior_owner = weight_owners.get(id(module.weight))
        if prior_owner is not None:
            raise TritiumError(
                "raw diagonal calibration does not yet merge distinct Linear modules sharing one weight",
                code="unsupported_shared_parameter",
                stage="calibrate",
                details={"modules": [prior_owner, path]},
            )
        weight_owners[id(module.weight)] = path
        aliases = next(
            entry.aliases
            for entry in prepared.coverage.entries
            if weight_name in entry.aliases
        )
        records.append((path, module, aliases))
    if not records:
        raise TritiumError(
            "raw calibration currently requires at least one selected Linear module",
            code="unsupported_module",
            stage="calibrate",
        )
    covered = {alias for _, _, aliases in records for alias in aliases}
    unsupported = sorted(
        entry.path
        for entry in prepared.coverage.entries
        if entry.disposition == "selected"
        and not covered.intersection(entry.aliases)
    )
    if unsupported:
        raise TritiumError(
            "raw diagonal calibration cannot cover selected non-Linear weights",
            code="unsupported_module",
            stage="calibrate",
            details={"parameters": unsupported},
        )
    return tuple(records)


def _invoke_model(model: nn.Module, batch: Any) -> Any:
    if isinstance(batch, Mapping):
        return model(**batch)
    if isinstance(batch, (tuple, list)):
        return model(*batch)
    return model(batch)


def _json_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate calibration manifest field {key!r}")
        value[key] = item
    return value


def load_activation_calibration(
    evidence_dir: Pathish, *, max_evidence_bytes: int = 64 * 1024 * 1024
) -> ActivationCalibrationReceipt:
    """Strictly reopen and rehash streamed diagonal-curvature evidence."""

    if max_evidence_bytes <= 0:
        raise ValueError("max_evidence_bytes must be positive")
    requested = Path(evidence_dir)
    if requested.is_symlink():
        raise ValueError("activation evidence directory must not be a symlink")
    directory = requested.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("activation evidence must be an ordinary directory")
    manifest_path = directory / "calibration.json"
    metadata = manifest_path.lstat()
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or metadata.st_size > 1024 * 1024
    ):
        raise ValueError(
            "calibration.json must be an ordinary manifest no larger than 1 MiB"
        )
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream, object_pairs_hook=_json_without_duplicates)
    fields = {
        "schema_version",
        "curvature",
        "source_model_digest",
        "activation_cache_digest",
        "token_stream_digest",
        "record_count",
        "records",
        "evidence_id",
    }
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ValueError("calibration manifest fields do not match schema version 1")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported activation calibration schema_version")
    if manifest["curvature"] != "diagonal-second-moment-f64le":
        raise ValueError("unsupported activation curvature representation")
    for field in (
        "source_model_digest",
        "activation_cache_digest",
        "token_stream_digest",
        "evidence_id",
    ):
        value = manifest[field]
        if (
            not isinstance(value, str)
            or len(value) != 71
            or not value.startswith("sha256:")
        ):
            raise ValueError(f"invalid {field}")
        try:
            bytes.fromhex(value[7:])
        except ValueError as error:
            raise ValueError(f"invalid {field}") from error
    values = manifest["records"]
    if (
        type(manifest["record_count"]) is not int
        or manifest["record_count"] <= 0
        or not isinstance(values, list)
        or len(values) != manifest["record_count"]
    ):
        raise ValueError("activation record_count does not match records")
    record_fields = {
        "module",
        "weight_aliases",
        "features",
        "samples",
        "file",
        "digest",
        "bytes",
    }
    records = []
    cache_digest = hashlib.sha256()
    total_bytes = 0
    expected_files = {"calibration.json"}
    for index, item in enumerate(values):
        if not isinstance(item, dict) or set(item) != record_fields:
            raise ValueError("activation record fields do not match schema version 1")
        filename = f"curvature-{index:05d}.f64le"
        if item["file"] != filename:
            raise ValueError(
                "activation records are missing, duplicated, or out of order"
            )
        aliases = item["weight_aliases"]
        if (
            not isinstance(item["module"], str)
            or not isinstance(aliases, list)
            or not aliases
            or any(not isinstance(alias, str) or not alias for alias in aliases)
        ):
            raise ValueError("activation record module aliases are invalid")
        if (
            type(item["features"]) is not int
            or item["features"] <= 0
            or type(item["samples"]) is not int
            or item["samples"] <= 0
            or type(item["bytes"]) is not int
            or item["bytes"] != item["features"] * 8
        ):
            raise ValueError("activation record geometry or byte ledger is invalid")
        path = directory / filename
        file_metadata = path.lstat()
        if (
            path.is_symlink()
            or not path.is_file()
            or file_metadata.st_size != item["bytes"]
        ):
            raise ValueError("activation record is not an exact ordinary file")
        payload = path.read_bytes()
        digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        if item["digest"] != digest:
            raise ValueError("activation record digest mismatch")
        _hash_field(cache_digest, filename, payload)
        total_bytes += len(payload)
        if total_bytes > max_evidence_bytes:
            raise ValueError("activation evidence exceeds max_evidence_bytes")
        expected_files.add(filename)
        records.append(
            ActivationRecord(
                module=item["module"],
                weight_aliases=tuple(aliases),
                features=item["features"],
                samples=item["samples"],
                file=filename,
                digest=digest,
                bytes=item["bytes"],
            )
        )
    if {child.name for child in directory.iterdir()} != expected_files:
        raise ValueError("activation evidence directory contains unknown files")
    actual_cache_digest = f"sha256:{cache_digest.hexdigest()}"
    if manifest["activation_cache_digest"] != actual_cache_digest:
        raise ValueError("activation cache digest mismatch")
    identity = dict(manifest)
    evidence_id = identity.pop("evidence_id")
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if evidence_id != f"sha256:{hashlib.sha256(canonical).hexdigest()}":
        raise ValueError("activation evidence identity mismatch")
    return ActivationCalibrationReceipt(
        evidence_dir=directory,
        evidence_id=evidence_id,
        curvature=manifest["curvature"],
        record_count=len(records),
        source_model_digest=manifest["source_model_digest"],
        activation_cache_digest=actual_cache_digest,
        token_stream_digest=manifest["token_stream_digest"],
        max_evidence_bytes=max_evidence_bytes,
        records=tuple(records),
    )


def _collect_activations(
    prepared: PreparedModel,
    data: Iterable[Any],
    evidence_dir: Path,
    max_evidence_bytes: int,
) -> ActivationCalibrationReceipt:
    if max_evidence_bytes <= 0:
        raise ValueError("max_evidence_bytes must be positive")
    if evidence_dir.exists() or evidence_dir.is_symlink():
        raise FileExistsError(f"calibration evidence already exists: {evidence_dir}")
    modules = _selected_linear_modules(prepared)
    required_bytes = sum(module.in_features * 8 for _, module, _ in modules)
    if required_bytes > max_evidence_bytes:
        raise TritiumError(
            "activation evidence exceeds max_evidence_bytes",
            code="evidence_too_large",
            stage="calibrate",
            details={
                "required_bytes": required_bytes,
                "max_evidence_bytes": max_evidence_bytes,
            },
        )
    accumulators: Dict[str, torch.Tensor] = {
        path: torch.zeros(module.in_features, dtype=torch.float64)
        for path, module, _ in modules
    }
    samples = {path: 0 for path, _, _ in modules}
    handles = []

    def hook_for(path: str):
        def capture(_module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                raise TritiumError(
                    "selected Linear module did not receive a tensor input",
                    code="invalid_calibration_batch",
                    stage="calibrate",
                    module=path,
                )
            value = args[0].detach()
            if value.shape[-1] != accumulators[path].numel():
                raise TritiumError(
                    "Linear calibration input width changed",
                    code="invalid_calibration_batch",
                    stage="calibrate",
                    module=path,
                )
            rows = value.reshape(-1, value.shape[-1])
            chunk_rows = max(1, (1024 * 1024) // rows.shape[-1])
            for offset in range(0, rows.shape[0], chunk_rows):
                chunk = rows[offset : offset + chunk_rows].to(torch.float64)
                accumulators[path].add_(chunk.square().sum(dim=0).cpu())
            samples[path] += rows.shape[0]

        return capture

    for path, module, _ in modules:
        handles.append(module.register_forward_pre_hook(hook_for(path)))

    model = prepared.model
    was_training = model.training
    token_digest = hashlib.sha256()
    batches = 0
    try:
        model.eval()
        with torch.no_grad():
            for batches, batch in enumerate(data, 1):
                _hash_value(token_digest, f"batch[{batches - 1}]", batch)
                _invoke_model(model, batch)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if batches == 0:
        raise TritiumError(
            "calibration data must yield at least one batch",
            code="invalid_calibration_batch",
            stage="calibrate",
        )
    empty = [path for path, count in samples.items() if count == 0]
    if empty:
        raise TritiumError(
            "selected modules were not all exercised by calibration data",
            code="incomplete_coverage",
            stage="calibrate",
            details={"modules": empty},
        )

    parent = evidence_dir.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / evidence_dir.name
    staging = Path(
        tempfile.mkdtemp(prefix=f".{evidence_dir.name}.", dir=str(parent))
    )
    activation_digest = hashlib.sha256()
    record_values = []
    records = []
    try:
        for index, (path, _, aliases) in enumerate(modules):
            payload = accumulators[path].numpy().astype("<f8", copy=False).tobytes()
            if len(payload) > max_evidence_bytes:
                raise TritiumError(
                    "activation evidence exceeds max_evidence_bytes",
                    code="evidence_too_large",
                    stage="calibrate",
                )
            filename = f"curvature-{index:05d}.f64le"
            (staging / filename).write_bytes(payload)
            digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
            _hash_field(activation_digest, filename, payload)
            record = ActivationRecord(
                module=path,
                weight_aliases=tuple(aliases),
                features=accumulators[path].numel(),
                samples=samples[path],
                file=filename,
                digest=digest,
                bytes=len(payload),
            )
            records.append(record)
            record_values.append(
                {
                    "module": record.module,
                    "weight_aliases": list(record.weight_aliases),
                    "features": record.features,
                    "samples": record.samples,
                    "file": record.file,
                    "digest": record.digest,
                    "bytes": record.bytes,
                }
            )
        total_bytes = sum(record.bytes for record in records)
        if total_bytes > max_evidence_bytes:
            raise TritiumError(
                "activation evidence exceeds max_evidence_bytes",
                code="evidence_too_large",
                stage="calibrate",
                details={
                    "required_bytes": total_bytes,
                    "max_evidence_bytes": max_evidence_bytes,
                },
            )
        source_digest = _source_model_digest(model)
        cache_digest = f"sha256:{activation_digest.hexdigest()}"
        stream_digest = f"sha256:{token_digest.hexdigest()}"
        manifest = {
            "schema_version": 1,
            "curvature": "diagonal-second-moment-f64le",
            "source_model_digest": source_digest,
            "activation_cache_digest": cache_digest,
            "token_stream_digest": stream_digest,
            "record_count": len(records),
            "records": record_values,
        }
        canonical = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        evidence_id = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
        manifest["evidence_id"] = evidence_id
        (staging / "calibration.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _tritium.publish_directory_noreplace(str(staging), str(target))
    except BaseException:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise
    return load_activation_calibration(
        target, max_evidence_bytes=max_evidence_bytes
    )


def _diagonal_additive_projection(
    master: torch.Tensor, curvature: torch.Tensor, planes: int
) -> TernaryProjection:
    if master.ndim != 2 or curvature.ndim != 1 or curvature.numel() != master.shape[1]:
        raise TritiumError(
            "calibration curvature does not match the selected weight",
            code="evidence_geometry_mismatch",
            stage="convert",
        )
    master_f64 = master.detach().to(dtype=torch.float64, device="cpu")
    diagonal = curvature.to(dtype=torch.float64, device="cpu")
    mean = diagonal.mean()
    if not bool(torch.isfinite(diagonal).all()) or bool((diagonal < 0).any()):
        raise TritiumError(
            "calibration curvature must be finite and nonnegative",
            code="invalid_evidence",
            stage="convert",
        )
    diagonal = (
        torch.ones_like(diagonal)
        if float(mean) == 0.0
        else diagonal + mean * 1e-4
    )
    residual = master_f64
    decoded = torch.zeros_like(master_f64)
    fitted_planes = []
    for _ in range(planes):
        initial_scale = (residual.abs() * diagonal).sum(dim=1, keepdim=True)
        initial_scale = initial_scale / diagonal.sum().clamp_min(
            torch.finfo(torch.float64).tiny
        )
        nonzero_scale = initial_scale.clamp_min(torch.finfo(torch.float64).tiny)
        trits = (residual / nonzero_scale).round().clamp(-1, 1).to(torch.int8)
        trits_f64 = trits.to(torch.float64)
        denominator = (trits_f64.square() * diagonal).sum(dim=1, keepdim=True)
        numerator = (residual * trits_f64 * diagonal).sum(dim=1, keepdim=True)
        scale = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny),
            torch.zeros_like(numerator),
        ).clamp_min(0)
        plane = TernaryPlane(
            trits=trits,
            scales=scale.to(torch.float16),
            group_size=master.shape[1],
        )
        fitted_planes.append(plane)
        stored_scale_f64 = plane.scales.to(torch.float64)
        decoded = decoded + trits_f64 * stored_scale_f64
        residual = master_f64 - decoded
    dense = torch.zeros_like(master, device="cpu")
    for plane in fitted_planes:
        dense = dense + plane.trits.to(master.dtype) * plane.scales
    projection = TernaryProjection(
        dense=dense,
        planes=tuple(fitted_planes),
        algorithm_id=_diagonal_algorithm_id(planes),
        schema_version=1,
    )
    validate_projection(
        projection,
        master.detach().to(device="cpu"),
        algorithm_id=projection.algorithm_id,
        schema_version=1,
    )
    return projection


def _diagonal_algorithm_id(planes: int) -> str:
    return f"tritium.diagonal-additive-{planes}@1"


def _fit_module(
    prepared: PreparedModel,
    calibration: ActivationCalibrationReceipt,
    work_dir: Pathish,
) -> ModuleQuantizationResult:
    """Fit hard additive planes from strict live-module calibration evidence."""

    if not isinstance(prepared, PreparedModel) or not isinstance(
        prepared.model, nn.Module
    ):
        raise TypeError("module conversion requires a live-module PreparedModel")
    if prepared.config.mode != "ptq":
        raise TritiumError(
            "module conversion requires PTQ preparation",
            code="invalid_phase",
            stage="convert",
        )
    if not isinstance(calibration, ActivationCalibrationReceipt):
        raise TypeError("module conversion requires an ActivationCalibrationReceipt")
    if prepared.config.target_bpw is not None:
        raise TritiumError(
            "generic diagonal PTQ does not yet implement target_bpw allocation",
            code="unsupported_recipe",
            stage="convert",
            details={"target_bpw": prepared.config.target_bpw},
        )
    if prepared.coverage is None:
        raise TritiumError(
            "module conversion requires exact prepared coverage",
            code="coverage_missing",
            stage="convert",
        )
    reopened = load_activation_calibration(
        calibration.evidence_dir,
        max_evidence_bytes=calibration.max_evidence_bytes,
    )
    if reopened != calibration:
        raise TritiumError(
            "activation calibration changed after admission",
            code="evidence_changed",
            stage="convert",
        )
    source_digest = _source_model_digest(prepared.model)
    if source_digest != calibration.source_model_digest:
        raise TritiumError(
            "prepared model changed after calibration",
            code="source_changed",
            stage="convert",
        )
    parameters = dict(prepared.model.named_parameters(remove_duplicate=False))
    algorithm_id = _diagonal_algorithm_id(prepared.config.planes)
    recipe_id = module_recipe_id(
        source_digest,
        calibration.evidence_id,
        algorithm_id,
        prepared.config,
        prepared.coverage,
    )

    def fit_weight(record: ActivationRecord) -> FittedWeight:
        try:
            master = parameters[record.weight_aliases[0]]
        except KeyError as error:
            raise TritiumError(
                "calibration refers to a missing source parameter",
                code="evidence_geometry_mismatch",
                stage="convert",
                module=record.module,
            ) from error
        payload = (calibration.evidence_dir / record.file).read_bytes()
        curvature = torch.frombuffer(bytearray(payload), dtype=torch.float64)
        curvature = curvature / record.samples
        projection = _diagonal_additive_projection(
            master, curvature, prepared.config.planes
        )
        error = (
            master.detach().cpu().to(torch.float64)
            - projection.dense.to(torch.float64)
        ).square()
        denominator = curvature.sum().clamp_min(1e-30) * master.shape[0]
        weighted_mse = float((error * curvature).sum() / denominator)
        return FittedWeight(
            path=record.weight_aliases[0],
            aliases=record.weight_aliases,
            planes=projection.planes,
            weighted_mse=weighted_mse,
        )

    return seal_module_conversion(
        work_dir,
        source_model_digest=source_digest,
        evidence_id=calibration.evidence_id,
        algorithm_id=algorithm_id,
        recipe_id=recipe_id,
        config=prepared.config,
        coverage=prepared.coverage,
        records=calibration.records,
        fit_weight=fit_weight,
    )


def calibrate(
    prepared: PreparedModel,
    data: Any = None,
    *,
    evidence_dir: Pathish,
    max_evidence_bytes: int = 64 * 1024 * 1024,
) -> Union[CalibrationReceipt, ActivationCalibrationReceipt]:
    """Admit precomputed Qwen3.6 S2KF evidence for PTQ conversion.

    Local Qwen sources admit the canonical 506-record S2KF namespace. Live
    PyTorch modules instead stream bounded diagonal second moments from
    ``data`` into a separately typed evidence namespace.
    """

    if not isinstance(prepared, PreparedModel):
        raise TypeError("calibrate requires a PreparedModel")
    if prepared.config.mode != "ptq":
        raise TritiumError(
            "QAT preparation does not use the PTQ calibration phase",
            code="invalid_phase",
            stage="calibrate",
        )
    if isinstance(prepared.model, nn.Module):
        if data is None:
            raise TritiumError(
                "live-module calibration requires an iterable of batches",
                code="calibration_data_required",
                stage="calibrate",
            )
        return _collect_activations(
            prepared, data, Path(evidence_dir), max_evidence_bytes
        )
    if data is not None:
        raise TritiumError(
            "raw calibration collection is not available; provide evidence_dir only",
            code="evidence_required",
            stage="calibrate",
        )
    requested = Path(evidence_dir)
    if requested.is_symlink():
        raise TritiumError(
            "PTQ evidence directory must not be a symlink",
            code="invalid_evidence_path",
            stage="calibrate",
        )
    directory = requested.resolve(strict=True)
    values = _tritium.inspect_qwen36_ptq_evidence(
        str(directory), max_evidence_bytes=max_evidence_bytes
    )
    return CalibrationReceipt(
        evidence_dir=directory,
        evidence_id=values[0],
        curvature=values[1],
        record_count=values[2],
        source_model_digest=values[3],
        activation_cache_digest=values[4],
        token_stream_digest=values[5],
        max_evidence_bytes=max_evidence_bytes,
    )


def convert(
    prepared: PreparedModel,
    calibration: Union[CalibrationReceipt, ActivationCalibrationReceipt],
    *,
    revision: Optional[str] = None,
    work_dir: Optional[Pathish] = None,
    output_dir: Optional[Pathish] = None,
    compact_max_bytes: Optional[int] = None,
    compact_max_resident_bytes: Optional[int] = None,
    near_lossless_max_bytes: Optional[int] = None,
    near_lossless_max_resident_bytes: Optional[int] = None,
    packing: str = "b3",
) -> Union[QuantizationResult, ModuleQuantizationResult]:
    """Run resumable PTQ, exact allocation, admission, and atomic export."""

    if not isinstance(prepared, PreparedModel):
        raise TypeError("convert requires a PreparedModel")
    if isinstance(prepared.model, nn.Module):
        if work_dir is None:
            raise TypeError("live-module convert requires work_dir")
        qwen_arguments = {
            "revision": revision,
            "output_dir": output_dir,
            "compact_max_bytes": compact_max_bytes,
            "compact_max_resident_bytes": compact_max_resident_bytes,
            "near_lossless_max_bytes": near_lossless_max_bytes,
            "near_lossless_max_resident_bytes": near_lossless_max_resident_bytes,
        }
        supplied = sorted(
            name for name, value in qwen_arguments.items() if value is not None
        )
        if supplied:
            raise TypeError(
                "live-module convert does not accept Qwen package arguments: "
                + ", ".join(supplied)
            )
        return _fit_module(prepared, calibration, work_dir)
    if not isinstance(calibration, CalibrationReceipt):
        raise TypeError("Qwen convert requires a CalibrationReceipt")
    if prepared.config.mode != "ptq" or not isinstance(prepared.model, Path):
        raise TritiumError(
            "Qwen3.6 PTQ conversion requires a prepared local source directory",
            code="invalid_phase",
            stage="convert",
        )
    if prepared.config.target_bpw is not None:
        raise TritiumError(
            "Qwen3.6 conversion currently requires exact byte ceilings; target_bpw is not silently approximated",
            code="unsupported_recipe",
            stage="convert",
            details={"target_bpw": prepared.config.target_bpw},
        )
    required = {
        "revision": revision,
        "work_dir": work_dir,
        "output_dir": output_dir,
        "compact_max_bytes": compact_max_bytes,
        "compact_max_resident_bytes": compact_max_resident_bytes,
        "near_lossless_max_bytes": near_lossless_max_bytes,
        "near_lossless_max_resident_bytes": near_lossless_max_resident_bytes,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise TypeError(
            "Qwen convert missing required arguments: " + ", ".join(missing)
        )
    current = calibrate(
        prepared,
        evidence_dir=calibration.evidence_dir,
        max_evidence_bytes=calibration.max_evidence_bytes,
    )
    if current != calibration:
        raise TritiumError(
            "calibration evidence changed after admission",
            code="evidence_changed",
            stage="convert",
        )
    native = reconcile_qwen36_ptq_packages(
        prepared.model,
        revision=revision,
        work_dir=work_dir,
        evidence_dir=calibration.evidence_dir,
        output_dir=output_dir,
        compact_max_bytes=compact_max_bytes,
        compact_max_resident_bytes=compact_max_resident_bytes,
        near_lossless_max_bytes=near_lossless_max_bytes,
        near_lossless_max_resident_bytes=near_lossless_max_resident_bytes,
        packing=packing,
        max_evidence_bytes=calibration.max_evidence_bytes,
    )
    return load(native.artifact_dir)


def quantize(
    model_or_id: Pathish,
    config: TernaryConfig,
    *,
    revision: str,
    work_dir: Pathish,
    evidence_dir: Pathish,
    output_dir: Pathish,
    compact_max_bytes: int,
    compact_max_resident_bytes: int,
    near_lossless_max_bytes: int,
    near_lossless_max_resident_bytes: int,
    packing: str = "b3",
    max_evidence_bytes: int = 64 * 1024 * 1024,
) -> QuantizationResult:
    """Compose exact PTQ ``prepare`` → ``calibrate`` → ``convert`` phases."""

    prepared = prepare(model_or_id, config, inplace=False)
    calibration = calibrate(
        prepared,
        evidence_dir=evidence_dir,
        max_evidence_bytes=max_evidence_bytes,
    )
    return convert(
        prepared,
        calibration,
        revision=revision,
        work_dir=work_dir,
        output_dir=output_dir,
        compact_max_bytes=compact_max_bytes,
        compact_max_resident_bytes=compact_max_resident_bytes,
        near_lossless_max_bytes=near_lossless_max_bytes,
        near_lossless_max_resident_bytes=near_lossless_max_resident_bytes,
        packing=packing,
    )


__all__ = [
    "ActivationCalibrationReceipt",
    "ActivationRecord",
    "CalibrationReceipt",
    "FittedWeight",
    "ModuleQuantizationResult",
    "calibrate",
    "convert",
    "load_activation_calibration",
    "load_module_conversion",
    "quantize",
]
