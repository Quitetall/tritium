"""Portable estimator contract shared by reference and optimized adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .errors import TritiumError


@dataclass(frozen=True)
class ProjectionContext:
    """Deterministic inputs that may affect a ternary projection."""

    step: int = 0
    training: bool = True
    role: str = "weight"
    seed: int = 0


@dataclass(frozen=True)
class TernaryProjection:
    """Validated hard ternary planes and differentiable decoded forward value."""

    dense: torch.Tensor
    trits: torch.Tensor
    scales: torch.Tensor
    group_size: int
    algorithm_id: str
    schema_version: int


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
    if projection.dense.shape != master.shape or projection.trits.shape != master.shape:
        raise TritiumError(
            "projection shape does not match master weight",
            code="estimator_contract",
            stage="project",
            details={
                "master_shape": tuple(master.shape),
                "dense_shape": tuple(projection.dense.shape),
                "trit_shape": tuple(projection.trits.shape),
            },
        )
    if projection.trits.dtype != torch.int8:
        raise TritiumError(
            "projection trits must use torch.int8",
            code="estimator_contract",
            stage="project",
        )
    if not torch.all((projection.trits >= -1) & (projection.trits <= 1)):
        raise TritiumError(
            "projection contains a value outside {-1, 0, +1}",
            code="estimator_contract",
            stage="project",
        )
    if not torch.isfinite(projection.scales).all() or torch.any(projection.scales < 0):
        raise TritiumError(
            "projection scales must be finite and nonnegative",
            code="estimator_contract",
            stage="project",
        )
    if projection.group_size <= 0:
        raise TritiumError(
            "projection group_size must be positive",
            code="estimator_contract",
            stage="project",
        )

    try:
        decoded = projection.trits.to(master.dtype) * projection.scales
    except RuntimeError as error:
        raise TritiumError(
            "projection scales are not broadcastable over trits",
            code="estimator_contract",
            stage="project",
        ) from error
    if not torch.equal(projection.dense.detach(), decoded):
        raise TritiumError(
            "projection forward value is not its canonical ternary decode",
            code="estimator_contract",
            stage="project",
        )
