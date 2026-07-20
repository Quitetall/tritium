"""Resumable phased Qwen3.6 PTQ lifecycle."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from .. import _tritium
from ..salt import reconcile_qwen36_ptq_packages
from .artifacts import QuantizationResult, load
from .config import TernaryConfig
from .conversion import PreparedModel, prepare
from .errors import TritiumError

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


def calibrate(
    prepared: PreparedModel,
    data: Any = None,
    *,
    evidence_dir: Pathish,
    max_evidence_bytes: int = 64 * 1024 * 1024,
) -> CalibrationReceipt:
    """Admit precomputed Qwen3.6 S2KF evidence for PTQ conversion.

    Raw activation collection is intentionally not simulated. Until the
    streaming collector lands, callers supply the exact canonical 506-record
    evidence directory and leave ``data`` as ``None``.
    """

    if not isinstance(prepared, PreparedModel):
        raise TypeError("calibrate requires a PreparedModel")
    if prepared.config.mode != "ptq":
        raise TritiumError(
            "QAT preparation does not use the PTQ calibration phase",
            code="invalid_phase",
            stage="calibrate",
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
    calibration: CalibrationReceipt,
    *,
    revision: str,
    work_dir: Pathish,
    output_dir: Pathish,
    compact_max_bytes: int,
    compact_max_resident_bytes: int,
    near_lossless_max_bytes: int,
    near_lossless_max_resident_bytes: int,
    packing: str = "b3",
) -> QuantizationResult:
    """Run resumable PTQ, exact allocation, admission, and atomic export."""

    if not isinstance(prepared, PreparedModel):
        raise TypeError("convert requires a PreparedModel")
    if not isinstance(calibration, CalibrationReceipt):
        raise TypeError("convert requires a CalibrationReceipt")
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


__all__ = ["CalibrationReceipt", "calibrate", "convert", "quantize"]
