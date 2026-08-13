"""Portable estimator contract shared by reference and optimized adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from .. import _tritium
from .errors import TritiumError


def expand_plane_scales(
    scales: torch.Tensor, *, rows: int, columns: int, group_size: int
) -> torch.Tensor:
    """Expand per-group scales to coefficient geometry without a dense shadow."""

    if type(group_size) is not int or group_size <= 0:
        raise ValueError("plane group_size must be a positive integer")
    groups = (columns + group_size - 1) // group_size
    if tuple(scales.shape) != (rows, groups):
        raise ValueError("plane scales shape does not match group geometry")
    return scales.repeat_interleave(group_size, dim=1)[..., :columns]


@dataclass(frozen=True)
class ProjectionContext:
    """Deterministic inputs that may affect a ternary projection."""

    step: int = 0
    training: bool = True
    role: str = "weight"
    seed: int = 0


@dataclass(frozen=True)
class TernaryPlane:
    """One exportable hard ternary plane."""

    trits: torch.Tensor
    scales: torch.Tensor
    group_size: int
    structure: str = "dense"


@dataclass(frozen=True)
class TernaryProjection:
    """One-to-three hard planes plus a differentiable training value."""

    dense: torch.Tensor
    planes: Tuple[TernaryPlane, ...]
    algorithm_id: str
    schema_version: int
    exportable: bool = True

    @property
    def trits(self) -> torch.Tensor:
        """Single-plane compatibility view."""

        if len(self.planes) != 1:
            raise TritiumError(
                "multi-plane projection has no singular trits tensor",
                code="estimator_contract",
                stage="project",
            )
        return self.planes[0].trits

    @property
    def scales(self) -> torch.Tensor:
        """Single-plane compatibility view."""

        if len(self.planes) != 1:
            raise TritiumError(
                "multi-plane projection has no singular scales tensor",
                code="estimator_contract",
                stage="project",
            )
        return self.planes[0].scales

    @property
    def group_size(self) -> int:
        """Single-plane compatibility view."""

        if len(self.planes) != 1:
            raise TritiumError(
                "multi-plane projection has no singular group size",
                code="estimator_contract",
                stage="project",
            )
        return self.planes[0].group_size


@dataclass(frozen=True)
class DenseGroupFit:
    """Native dense-curvature fit and its final weighted objective."""

    projection: TernaryProjection
    objective: float


@dataclass(frozen=True)
class KroneckerGroupFit:
    """Native S2KF output-aware fit with bound record identity."""

    projection: TernaryProjection
    objective: float
    record_digest: str


def fit_dense_ternary_group(
    weights: torch.Tensor,
    metric: torch.Tensor,
    *,
    planes: int = 3,
    max_iterations: int = 16,
    ridge: float = 1e-8,
    em_restarts: int = 4,
    ridge_condition_limit: float = 1e6,
    scale_precision: str = "f16",
    softened_relay: bool = False,
    modulated_relay: bool = False,
) -> DenseGroupFit:
    """Fit one dense PSD-curvature group with native SALT joint optimization.

    ``weights`` and ``metric`` are one-dimensional group weights and a square
    row-major PSD metric. This helper is the bridge for output-aware/K-FAC PTQ
    drivers; it does not capture calibration or manufacture provenance. Callers
    must bind returned projection metadata to their calibration receipt.
    """

    if not isinstance(weights, torch.Tensor) or weights.ndim != 1:
        raise TypeError("weights must be a rank-1 tensor")
    if not weights.dtype.is_floating_point:
        raise TypeError("weights must use a floating dtype")
    if not isinstance(metric, torch.Tensor) or metric.ndim != 2:
        raise TypeError("metric must be a rank-2 tensor")
    if tuple(metric.shape) != (weights.numel(), weights.numel()):
        raise ValueError("metric must be square with one row per weight")
    if not metric.dtype.is_floating_point:
        raise TypeError("metric must use a floating dtype")
    if weights.device.type != "cpu" or metric.device.type != "cpu":
        raise ValueError("native dense fitting currently requires CPU tensors")
    if not torch.isfinite(weights).all() or not torch.isfinite(metric).all():
        raise ValueError("weights and metric must be finite")
    scales, trits, reconstruction, _objective = _tritium.fit_joint_ternary_dense(
        weights.to(torch.float32).tolist(),
        metric.to(torch.float64).reshape(-1).tolist(),
        planes,
        max_iterations,
        ridge,
        em_restarts,
        ridge_condition_limit,
        scale_precision,
        softened_relay,
        modulated_relay,
    )
    scale_dtype = torch.float16 if scale_precision == "f16" else torch.float32
    planes_out = tuple(
        TernaryPlane(
            trits=torch.tensor(trit_values, dtype=torch.int8).reshape(1, -1),
            scales=torch.tensor([[scale]], dtype=scale_dtype),
            group_size=weights.numel(),
        )
        for scale, trit_values in zip(scales, trits)
    )
    dense = torch.tensor(reconstruction, dtype=weights.dtype).reshape(1, -1)
    projection = TernaryProjection(
        dense=dense,
        planes=planes_out,
        algorithm_id="tritium.salt-v2-joint-dense@1",
        schema_version=1,
    )
    validate_projection(projection, weights.reshape(1, -1))
    return DenseGroupFit(projection=projection, objective=float(_objective))


def fit_kronecker_group(
    weights: torch.Tensor,
    evidence: Union[bytes, bytearray, memoryview, str, Path],
    *,
    planes: int = 3,
    max_iterations: int = 16,
    ridge: float = 1e-8,
    em_restarts: int = 4,
    ridge_condition_limit: float = 1e6,
    scale_precision: str = "f16",
    softened_relay: bool = False,
    modulated_relay: bool = False,
    max_evidence_bytes: int = 64 * 1024 * 1024,
    row_start: int = 0,
    row_count: Optional[int] = None,
) -> KroneckerGroupFit:
    """Fit rank-2 weights against canonical S2KF K-FAC evidence.

    Evidence is either canonical S2KF bytes or a regular file containing one
    canonical record. Native code validates record checksum, geometry, PSD
    factors, and source-bound record identity before fitting. This function is
    a compute primitive: callers still own source-model and token-cache
    admission and must persist returned ``record_digest`` in their receipt.
    """

    if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
        raise TypeError("weights must be a rank-2 tensor")
    if not weights.dtype.is_floating_point:
        raise TypeError("weights must use a floating dtype")
    if weights.device.type != "cpu":
        raise ValueError("native Kronecker fitting currently requires CPU tensors")
    if type(row_start) is not int or row_start < 0:
        raise ValueError("row_start must be a nonnegative integer")
    if row_count is not None and (type(row_count) is not int or row_count <= 0):
        raise ValueError("row_count must be a positive integer when provided")
    if type(max_evidence_bytes) is not int or max_evidence_bytes <= 0:
        raise ValueError("max_evidence_bytes must be a positive integer")
    if isinstance(evidence, (str, Path)):
        path = Path(evidence)
        if path.is_symlink() or not path.is_file():
            raise ValueError("S2KF evidence must be a regular non-symlink file")
        if path.stat().st_size > max_evidence_bytes:
            raise ValueError("S2KF evidence exceeds max_evidence_bytes")
        evidence_bytes = path.read_bytes()
    else:
        evidence_bytes = bytes(evidence)
    if not evidence_bytes:
        raise ValueError("S2KF evidence must not be empty")
    if len(evidence_bytes) > max_evidence_bytes:
        raise ValueError("S2KF evidence exceeds max_evidence_bytes")
    weight_bytes = (
        weights.detach()
        .to(dtype=torch.float32, device="cpu")
        .contiguous()
        .numpy()
        .tobytes()
    )
    rows, columns, scales, trits, reconstruction, objective, record_digest = (
        _tritium.fit_kronecker_ternary(
            weight_bytes,
            evidence_bytes,
            planes=planes,
            max_iterations=max_iterations,
            ridge=ridge,
            em_restarts=em_restarts,
            ridge_condition_limit=ridge_condition_limit,
            scale_precision=scale_precision,
            softened_relay=softened_relay,
            modulated_relay=modulated_relay,
            row_start=row_start,
            row_count=row_count,
        )
    )
    group_count = columns // 128
    scale_dtype = torch.float16 if scale_precision == "f16" else torch.float32
    planes_out = tuple(
        TernaryPlane(
            trits=torch.tensor(values, dtype=torch.int8).reshape(rows, columns),
            scales=torch.tensor(values_scale, dtype=scale_dtype).reshape(
                rows, group_count
            ),
            group_size=128,
        )
        for values, values_scale in zip(trits, scales)
    )
    dense = torch.tensor(reconstruction, dtype=weights.dtype).reshape(rows, columns)
    projection = TernaryProjection(
        dense=dense,
        planes=planes_out,
        algorithm_id="tritium.salt-v2-kfac-joint@1",
        schema_version=1,
    )
    validate_projection(projection, weights)
    return KroneckerGroupFit(
        projection=projection,
        objective=float(objective),
        record_digest=record_digest,
    )


def _require_tensor_contract(
    condition: torch.Tensor,
    message: str,
    *,
    details=None,
) -> None:
    if torch.compiler.is_compiling():
        torch._assert_async(condition, message)
        return
    if not bool(condition):
        raise TritiumError(
            message,
            code="estimator_contract",
            stage="project",
            details=details,
        )


def validate_projection(
    projection: TernaryProjection,
    master: torch.Tensor,
    *,
    algorithm_id: str = "",
    schema_version: int = 0,
) -> None:
    """Fail closed when an estimator violates Tritium's exportable contract."""

    if projection.schema_version < 1:
        raise TritiumError(
            "estimator schema_version must be positive",
            code="estimator_contract",
            stage="project",
        )
    if algorithm_id and projection.algorithm_id != algorithm_id:
        raise TritiumError(
            "projection algorithm identity does not match estimator",
            code="estimator_contract",
            stage="project",
            details={"expected": algorithm_id, "observed": projection.algorithm_id},
        )
    if schema_version and projection.schema_version != schema_version:
        raise TritiumError(
            "projection schema identity does not match estimator",
            code="estimator_contract",
            stage="project",
            details={"expected": schema_version, "observed": projection.schema_version},
        )
    if projection.dense.shape != master.shape:
        raise TritiumError(
            "projection shape does not match master weight",
            code="estimator_contract",
            stage="project",
            details={
                "master_shape": tuple(master.shape),
                "dense_shape": tuple(projection.dense.shape),
            },
        )
    if not 1 <= len(projection.planes) <= 3:
        raise TritiumError(
            "projection must contain between one and three planes",
            code="estimator_contract",
            stage="project",
        )

    decoded = torch.zeros_like(master)
    for index, plane in enumerate(projection.planes):
        if plane.trits.shape != master.shape:
            raise TritiumError(
                "projection trit shape does not match master weight",
                code="estimator_contract",
                stage="project",
                details={"plane": index, "trit_shape": tuple(plane.trits.shape)},
            )
        if plane.trits.dtype != torch.int8:
            raise TritiumError(
                "projection trits must use torch.int8",
                code="estimator_contract",
                stage="project",
                details={"plane": index},
            )
        _require_tensor_contract(
            torch.all((plane.trits >= -1) & (plane.trits <= 1)),
            "projection contains a value outside {-1, 0, +1}",
            details={"plane": index},
        )
        _require_tensor_contract(
            torch.isfinite(plane.scales).all() & torch.all(plane.scales >= 0),
            "projection scales must be finite and nonnegative",
            details={"plane": index},
        )
        if plane.group_size <= 0:
            raise TritiumError(
                "projection group_size must be positive",
                code="estimator_contract",
                stage="project",
                details={"plane": index},
            )
        if plane.structure not in {"dense", "s34"}:
            raise TritiumError(
                "projection structure must be 'dense' or 's34'",
                code="estimator_contract",
                stage="project",
                details={"plane": index},
            )
        try:
            expanded_scales = expand_plane_scales(
                plane.scales,
                rows=master.shape[0],
                columns=master.shape[1],
                group_size=plane.group_size,
            )
            decoded = decoded + plane.trits.to(master.dtype) * expanded_scales.to(
                master.dtype
            )
        except (RuntimeError, ValueError) as error:
            raise TritiumError(
                "projection scales are not broadcastable over trits",
                code="estimator_contract",
                stage="project",
                details={"plane": index},
            ) from error
    _require_tensor_contract(
        torch.isfinite(projection.dense).all(),
        "projection forward value must be finite",
    )
    if projection.exportable:
        _require_tensor_contract(
            torch.all(projection.dense.detach() == decoded),
            "projection forward value is not its canonical ternary decode",
        )
