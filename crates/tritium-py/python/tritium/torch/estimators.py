"""Reference ternary estimators implemented with device-resident PyTorch ops."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Callable, Dict, Tuple

import torch
from torch import nn

from .projection import (
    ProjectionContext,
    TernaryPlane,
    TernaryProjection,
    validate_projection,
)
from .errors import TritiumError


class Estimator(nn.Module, ABC):
    """One-method extension seam for ternary projection research."""

    algorithm_id: str
    schema_version: int = 1

    @abstractmethod
    def project(
        self, master: torch.Tensor, *, context: ProjectionContext
    ) -> TernaryProjection:
        """Project a latent master into an exportable ternary representation."""


class AbsMeanSTE(Estimator):
    """Per-row AbsMean ternary projection with masked straight-through gradient."""

    algorithm_id = "tritium.absmean-ste"
    schema_version = 1

    def project(
        self, master: torch.Tensor, *, context: ProjectionContext
    ) -> TernaryProjection:
        del context
        if master.ndim != 2:
            raise ValueError("AbsMeanSTE requires a rank-2 master weight")

        accumulation_dtype = (
            torch.float32 if master.dtype in {torch.float16, torch.bfloat16} else master.dtype
        )
        scales = (
            master.detach()
            .to(accumulation_dtype)
            .abs()
            .mean(dim=1, keepdim=True)
            .to(master.dtype)
        )
        safe_scales = scales.clamp_min(torch.finfo(master.dtype).tiny)
        normalized = master / safe_scales
        hard = normalized.round().clamp(-1.0, 1.0)
        decoded = hard.detach() * scales
        # Rust's STE contract uses a strict |w/s| < 1 mask and returns zero for
        # a degenerate all-zero row. Keep that exact backward while the forward
        # remains the hard canonical decode.
        mask = ((normalized.abs() < 1.0) & (scales > 0)).to(master.dtype)
        dense = decoded + (master - master.detach()) * mask
        return TernaryProjection(
            dense=dense,
            planes=(
                TernaryPlane(
                    trits=hard.detach().to(torch.int8),
                    scales=scales,
                    group_size=master.shape[1],
                ),
            ),
            algorithm_id=self.algorithm_id,
            schema_version=self.schema_version,
        )


class SaltSTE(AbsMeanSTE):
    """SALT QAT base estimator, composable into residual additive planes."""

    algorithm_id = "tritium.salt-ste"


def _rank2(master: torch.Tensor, name: str) -> None:
    if master.ndim != 2:
        raise ValueError(f"{name} requires a rank-2 master weight")
    if not master.dtype.is_floating_point:
        raise TypeError(f"{name} requires a floating master weight")


def _absmean_scale(master: torch.Tensor) -> torch.Tensor:
    accumulation_dtype = (
        torch.float32 if master.dtype in {torch.float16, torch.bfloat16} else master.dtype
    )
    return master.detach().to(accumulation_dtype).abs().mean(dim=1, keepdim=True).to(master.dtype)


def _projection(
    master: torch.Tensor,
    trits: torch.Tensor,
    scales: torch.Tensor,
    surrogate: torch.Tensor,
    estimator: Estimator,
) -> TernaryProjection:
    decoded = trits.to(master.dtype) * scales
    # Form the zero-valued STE term before adding it. Reassociating this as
    # `(decoded + surrogate) - surrogate.detach()` leaves rounding residue and
    # violates the bit-exact hard-forward contract.
    dense = decoded.detach() + (surrogate - surrogate.detach())
    return TernaryProjection(
        dense=dense,
        planes=(
            TernaryPlane(
                trits=trits.detach().to(torch.int8),
                scales=scales,
                group_size=master.shape[1],
            ),
        ),
        algorithm_id=estimator.algorithm_id,
        schema_version=estimator.schema_version,
    )


class AdditiveEstimator(Estimator):
    """Residual composition of two or three independently stateful estimators."""

    schema_version = 1

    def __init__(self, estimators: Tuple[Estimator, ...]) -> None:
        super().__init__()
        if not 2 <= len(estimators) <= 3:
            raise ValueError("AdditiveEstimator requires two or three planes")
        first = estimators[0]
        if any(
            estimator.algorithm_id != first.algorithm_id
            or estimator.schema_version != first.schema_version
            for estimator in estimators
        ):
            raise ValueError(
                "additive plane estimators must share one algorithm schema"
            )
        self.estimators = nn.ModuleList(estimators)
        self.algorithm_id = (
            f"tritium.additive-{len(estimators)}/"
            f"{first.algorithm_id}@{first.schema_version}"
        )

    def project(
        self, master: torch.Tensor, *, context: ProjectionContext
    ) -> TernaryProjection:
        residual = master
        dense = torch.zeros_like(master)
        planes = []
        for estimator in self.estimators:
            projection = estimator.project(residual, context=context)
            validate_projection(
                projection,
                residual,
                algorithm_id=estimator.algorithm_id,
                schema_version=estimator.schema_version,
            )
            if len(projection.planes) != 1:
                raise TritiumError(
                    "nested additive estimators are not supported",
                    code="estimator_contract",
                    stage="project",
                )
            planes.append(projection.planes[0])
            dense = dense + projection.dense
            residual = master - dense
        return TernaryProjection(
            dense=dense,
            planes=tuple(planes),
            algorithm_id=self.algorithm_id,
            schema_version=self.schema_version,
        )


class AnnealedSTE(Estimator):
    """Hard AbsMean forward with a progressively sharper tanh surrogate."""

    algorithm_id = "tritium.annealed-ste"
    schema_version = 1

    def __init__(
        self,
        initial_temperature: float = 1.0,
        growth: float = 1.01,
        max_temperature: float = 32.0,
    ) -> None:
        super().__init__()
        if initial_temperature <= 0 or growth < 1 or max_temperature < initial_temperature:
            raise ValueError("invalid annealed STE temperature schedule")
        self.initial_temperature = float(initial_temperature)
        self.growth = float(growth)
        self.max_temperature = float(max_temperature)

    def project(
        self, master: torch.Tensor, *, context: ProjectionContext
    ) -> TernaryProjection:
        _rank2(master, "AnnealedSTE")
        scales = _absmean_scale(master)
        safe = scales.clamp_min(torch.finfo(master.dtype).tiny)
        normalized = master / safe
        trits = normalized.round().clamp(-1, 1).to(torch.int8)
        if self.growth == 1:
            temperature = self.initial_temperature
        else:
            max_step = math.ceil(
                math.log(self.max_temperature / self.initial_temperature) / math.log(self.growth)
            )
            temperature = min(
                self.max_temperature,
                self.initial_temperature * math.pow(self.growth, min(max_step, max(0, context.step))),
            )
        denominator = math.tanh(temperature)
        surrogate = scales * torch.tanh(normalized * temperature) / denominator
        return _projection(master, trits, scales, surrogate, self)


class LSQEstimator(Estimator):
    """Learned step-size ternary quantization with an STE-clipped code."""

    algorithm_id = "tritium.lsq"
    schema_version = 1

    def __init__(self, initial_scale: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(initial_scale) or initial_scale <= 0:
            raise ValueError("LSQ initial_scale must be finite and positive")
        self.log_scale = nn.Parameter(torch.tensor(math.log(math.expm1(initial_scale))))

    def project(
        self, master: torch.Tensor, *, context: ProjectionContext
    ) -> TernaryProjection:
        del context
        _rank2(master, "LSQEstimator")
        scale = torch.nn.functional.softplus(self.log_scale).to(master.dtype)
        scale = scale + torch.finfo(master.dtype).eps
        scales = scale.expand(master.shape[0], 1)
        normalized = (master / scale).clamp(-1, 1)
        trits = normalized.detach().round().to(torch.int8)
        straight_code = normalized + (normalized.round() - normalized).detach()
        return _projection(master, trits, scales, straight_code * scale, self)


class TWNEstimator(Estimator):
    """Ternary Weight Networks thresholding with per-row nonzero scales."""

    algorithm_id = "tritium.twn"
    schema_version = 1

    def __init__(self, threshold_factor: float = 0.7) -> None:
        super().__init__()
        if not math.isfinite(threshold_factor) or threshold_factor <= 0:
            raise ValueError("TWN threshold_factor must be finite and positive")
        self.threshold_factor = float(threshold_factor)

    def project(
        self, master: torch.Tensor, *, context: ProjectionContext
    ) -> TernaryProjection:
        del context
        _rank2(master, "TWNEstimator")
        absolute = master.detach().abs()
        threshold = absolute.mean(dim=1, keepdim=True) * self.threshold_factor
        selected = absolute > threshold
        trits = torch.where(selected, master.detach().sign(), torch.zeros_like(master)).to(
            torch.int8
        )
        count = selected.sum(dim=1, keepdim=True).clamp_min(1)
        scales = (absolute * selected).sum(dim=1, keepdim=True) / count
        return _projection(master, trits, scales, master, self)


class TTQEstimator(Estimator):
    """Trained ternary quantization with learned sign scales and threshold."""

    algorithm_id = "tritium.ttq"
    schema_version = 1

    def __init__(self, initial_scale: float = 1.0, initial_threshold: float = 0.05) -> None:
        super().__init__()
        if not math.isfinite(initial_scale) or initial_scale <= 0:
            raise ValueError("TTQ initial_scale must be finite and positive")
        if not 0 < initial_threshold < 1:
            raise ValueError("TTQ initial_threshold must be in (0, 1)")
        raw_scale = math.log(math.expm1(initial_scale))
        self.positive_log_scale = nn.Parameter(torch.tensor(raw_scale))
        self.negative_log_scale = nn.Parameter(torch.tensor(raw_scale))
        self.threshold_logit = nn.Parameter(
            torch.tensor(math.log(initial_threshold / (1 - initial_threshold)))
        )

    def project(
        self, master: torch.Tensor, *, context: ProjectionContext
    ) -> TernaryProjection:
        del context
        _rank2(master, "TTQEstimator")
        positive_scale = torch.nn.functional.softplus(self.positive_log_scale).to(master.dtype)
        negative_scale = torch.nn.functional.softplus(self.negative_log_scale).to(master.dtype)
        positive_scale = positive_scale + torch.finfo(master.dtype).eps
        negative_scale = negative_scale + torch.finfo(master.dtype).eps
        row_magnitude = _absmean_scale(master)
        threshold_ratio = torch.sigmoid(self.threshold_logit).to(master.dtype)
        threshold = row_magnitude * threshold_ratio
        detached = master.detach()
        trits = torch.where(
            detached > threshold,
            torch.ones_like(detached),
            torch.where(detached < -threshold, -torch.ones_like(detached), torch.zeros_like(detached)),
        ).to(torch.int8)
        scales = torch.where(
            trits > 0,
            positive_scale,
            torch.where(trits < 0, negative_scale, torch.zeros_like(master)),
        )
        sharpness = 10.0
        positive_gate = torch.sigmoid(sharpness * (master - threshold))
        negative_gate = torch.sigmoid(sharpness * (-master - threshold))
        surrogate = positive_scale * positive_gate - negative_scale * negative_gate
        return _projection(master, trits, scales, surrogate, self)


class SparseTernaryEstimator(Estimator):
    """Explicit magnitude-sparse ternary projection with per-row scaling."""

    algorithm_id = "tritium.sparse-ternary"
    schema_version = 1

    def __init__(self, target_sparsity: float = 0.5) -> None:
        super().__init__()
        if not 0 <= target_sparsity < 1:
            raise ValueError("target_sparsity must be in [0, 1)")
        self.target_sparsity = float(target_sparsity)

    def project(
        self, master: torch.Tensor, *, context: ProjectionContext
    ) -> TernaryProjection:
        del context
        _rank2(master, "SparseTernaryEstimator")
        absolute = master.detach().abs()
        keep = max(1, math.ceil(master.shape[1] * (1 - self.target_sparsity)))
        selected_indices = torch.topk(absolute, k=keep, dim=1, sorted=False).indices
        selected = torch.zeros_like(absolute, dtype=torch.bool).scatter_(
            1, selected_indices, True
        )
        trits = torch.where(selected, master.detach().sign(), torch.zeros_like(master)).to(
            torch.int8
        )
        count = selected.sum(dim=1, keepdim=True).clamp_min(1)
        scales = (absolute * selected).sum(dim=1, keepdim=True) / count
        return _projection(master, trits, scales, master, self)


EstimatorFactory = Callable[[], Estimator]
_ESTIMATORS: Dict[str, EstimatorFactory] = {
    "absmean-ste": AbsMeanSTE,
    "salt-ste": SaltSTE,
    "annealed-ste": AnnealedSTE,
    "lsq": LSQEstimator,
    "twn": TWNEstimator,
    "ttq": TTQEstimator,
    "sparse-ternary": SparseTernaryEstimator,
}


def register_estimator(name: str, factory: EstimatorFactory) -> None:
    """Register one explicit estimator factory; duplicates fail closed."""

    if not name or name.strip() != name:
        raise TritiumError(
            "estimator registry name must be nonempty and canonical",
            code="estimator_registry",
            stage="register",
        )
    if name in _ESTIMATORS:
        raise TritiumError(
            f"estimator {name!r} is already registered",
            code="estimator_registry",
            stage="register",
        )
    if not callable(factory):
        raise TritiumError(
            "estimator factory must be callable",
            code="estimator_registry",
            stage="register",
        )
    _ESTIMATORS[name] = factory


def create_estimator(name: str) -> Estimator:
    """Construct and validate a registered estimator."""

    factory = _ESTIMATORS.get(name)
    if factory is None:
        raise TritiumError(
            f"unknown estimator {name!r}",
            code="unsupported_recipe",
            stage="inspect",
        )
    estimator = factory()
    if not isinstance(estimator, Estimator) or not estimator.algorithm_id:
        raise TritiumError(
            f"factory for {name!r} did not return a valid Estimator",
            code="estimator_registry",
            stage="create",
        )
    return estimator


def registered_estimators() -> Tuple[str, ...]:
    """Return the canonical deterministic registry inventory."""

    return tuple(sorted(_ESTIMATORS))
