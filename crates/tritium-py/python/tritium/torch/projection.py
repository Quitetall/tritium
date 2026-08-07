"""Portable estimator contract shared by reference and optimized adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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
            decoded = decoded + plane.trits.to(master.dtype) * plane.scales.to(
                master.dtype
            )
        except RuntimeError as error:
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
