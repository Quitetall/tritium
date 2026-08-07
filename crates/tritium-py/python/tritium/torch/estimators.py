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
    physical_planes: int = 1

    @property
    def projection_step(self) -> int:
        """Schedule position product paths must place in projection context."""

        return 0

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
        stored_scales = scales.to(torch.float16)
        decoded = hard.detach() * stored_scales.to(master.dtype)
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
                    scales=stored_scales,
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
    return _multi_projection(master, ((trits, scales),), surrogate, estimator)


def _soft_projection(
    master: torch.Tensor,
    trits: torch.Tensor,
    scales: torch.Tensor,
    soft: torch.Tensor,
    estimator: Estimator,
    *,
    exportable: bool,
) -> TernaryProjection:
    """Build HESTIA's soft forward without weakening hard-plane validation."""

    stored_scales = scales.to(torch.float16)
    decoded = trits.to(master.dtype) * stored_scales.to(master.dtype)
    dense = decoded.detach() + (soft - soft.detach()) if exportable else soft
    return TernaryProjection(
        dense=dense,
        planes=(
            TernaryPlane(
                trits=trits.detach().to(torch.int8),
                scales=stored_scales,
                group_size=master.shape[1],
            ),
        ),
        algorithm_id=estimator.algorithm_id,
        schema_version=estimator.schema_version,
        exportable=exportable,
    )


def hestia_soft_expectation(
    master: torch.Tensor,
    scales: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Evaluate HESTIA's differentiable ternary expectation oracle."""

    _rank2(master, "hestia_soft_expectation")
    if scales.shape not in {(master.shape[0],), (master.shape[0], 1)}:
        raise ValueError("HESTIA scales must contain one value per row")
    if temperature.numel() != 1 or not temperature.dtype.is_floating_point:
        raise ValueError("HESTIA temperature must be one floating scalar")
    valid_temperature = torch.isfinite(temperature).all() & (temperature > 0).all()
    valid_scales = torch.isfinite(scales).all() & (scales >= 0).all()
    valid_master = torch.isfinite(master).all()
    if torch.compiler.is_compiling():
        torch._assert_async(
            valid_temperature,
            "HESTIA temperature must be finite and positive",
        )
        torch._assert_async(valid_scales, "HESTIA scales must be finite and nonnegative")
        torch._assert_async(valid_master, "HESTIA master must be finite")
    elif not bool(valid_temperature):
        raise ValueError("HESTIA temperature must be finite and positive")
    elif not bool(valid_scales):
        raise ValueError("HESTIA scales must be finite and nonnegative")
    elif not bool(valid_master):
        raise ValueError("HESTIA master must be finite")

    accumulation_dtype = (
        torch.float32 if master.dtype in {torch.float16, torch.bfloat16} else master.dtype
    )
    work = master.to(accumulation_dtype)
    scale = scales.detach().reshape(master.shape[0], 1).to(accumulation_dtype)
    tau = temperature.reshape(()).to(device=master.device, dtype=accumulation_dtype)
    min_differentiable_tau = math.sqrt(torch.finfo(accumulation_dtype).tiny)
    representable = (
        torch.isfinite(scale).all()
        & torch.isfinite(tau)
        & (tau >= min_differentiable_tau)
    )
    if torch.compiler.is_compiling():
        torch._assert_async(
            representable,
            "HESTIA inputs are not representable in accumulation dtype",
        )
    elif not bool(representable):
        raise ValueError("HESTIA inputs are not representable in accumulation dtype")
    live_rows = scale > 0
    divisor = torch.where(live_rows, scale, torch.ones_like(scale))
    raw_normalized = torch.where(live_rows, work.div(divisor), torch.zeros_like(work))
    finite_normalized = torch.isfinite(raw_normalized).all()
    if torch.compiler.is_compiling():
        torch._assert_async(
            finite_normalized,
            "HESTIA normalized weights are not representable",
        )
    elif not bool(finite_normalized):
        raise ValueError("HESTIA normalized weights are not representable")
    grid = work.new_tensor((-1.0, 0.0, 1.0))
    sqrt_max = math.sqrt(torch.finfo(accumulation_dtype).max)
    direct_magnitude = raw_normalized.abs() + 1
    direct_usable = (direct_magnitude <= sqrt_max) & (
        direct_magnitude / sqrt_max <= torch.sqrt(tau.detach())
    )
    direct_normalized = torch.where(
        direct_usable, raw_normalized, torch.zeros_like(raw_normalized)
    ).unsqueeze(-1)
    direct_distance = (direct_normalized - grid).square()
    direct_logits = -direct_distance / tau

    max_logit = -math.log(torch.finfo(accumulation_dtype).tiny)
    if accumulation_dtype == torch.float64:
        fallback_representable = direct_usable.all()
        if torch.compiler.is_compiling():
            torch._assert_async(
                fallback_representable,
                "HESTIA float64 fallback is not representable",
            )
        elif not bool(fallback_representable):
            raise ValueError("HESTIA float64 fallback is not representable")
        stable_logits = torch.zeros_like(direct_logits)
    else:
        # f64 provides ample headroom for every finite f16/bf16/f32 input once
        # tau passed the differentiable floor. Clamp dimensionless logits after
        # division; never multiply tau by a saturation threshold.
        fallback_normalized = raw_normalized.to(torch.float64).unsqueeze(-1)
        fallback_grid = grid.to(torch.float64)
        reference = torch.where(
            fallback_normalized < -0.5,
            fallback_normalized.new_tensor(-1.0),
            torch.where(
                fallback_normalized > 0.5,
                fallback_normalized.new_tensor(1.0),
                fallback_normalized.new_tensor(0.0),
            ),
        )
        difference = reference - fallback_grid
        relative_distance = difference * (fallback_normalized - fallback_grid)
        relative_distance = relative_distance + difference * (
            fallback_normalized - reference
        )
        scaled_relative = relative_distance / tau.to(torch.float64)
        stable_logits = -scaled_relative.clamp(min=0, max=max_logit).to(
            accumulation_dtype
        )
    logits = torch.where(direct_usable.unsqueeze(-1), direct_logits, stable_logits)
    expected = (torch.softmax(logits, dim=-1) * grid).sum(dim=-1)
    output = (scale * expected).to(master.dtype)
    finite_output = torch.isfinite(output).all()
    if torch.compiler.is_compiling():
        torch._assert_async(finite_output, "HESTIA expectation must be finite")
    elif not bool(finite_output):
        raise ValueError("HESTIA expectation must be finite")
    return output


def _multi_projection(
    master: torch.Tensor,
    values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    surrogate: torch.Tensor,
    estimator: Estimator,
) -> TernaryProjection:
    decoded = torch.zeros_like(master)
    planes = []
    for trits, scales in values:
        stored_scales = scales.to(torch.float16)
        decoded = decoded + trits.to(master.dtype) * stored_scales.to(master.dtype)
        planes.append(
            TernaryPlane(
                trits=trits.detach().to(torch.int8),
                scales=stored_scales,
                group_size=master.shape[1],
            )
        )
    # Form the zero-valued STE term before adding it. Reassociating this as
    # `(decoded + surrogate) - surrogate.detach()` leaves rounding residue and
    # violates the bit-exact hard-forward contract.
    dense = decoded.detach() + (surrogate - surrogate.detach())
    return TernaryProjection(
        dense=dense,
        planes=tuple(planes),
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
        exportable = True
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
            exportable = exportable and projection.exportable
        return TernaryProjection(
            dense=dense,
            planes=tuple(planes),
            algorithm_id=self.algorithm_id,
            schema_version=self.schema_version,
            exportable=exportable,
        )

    def set_step(self, step: int) -> "AdditiveEstimator":
        """Advance every child schedule as one additive projection."""

        setters = [getattr(estimator, "set_step", None) for estimator in self.estimators]
        if any(setter is None for setter in setters):
            raise TritiumError(
                "additive estimator children do not expose a shared schedule",
                code="unsupported_recipe",
                stage="schedule",
            )
        for setter in setters:
            setter(step)
        return self

    @property
    def projection_step(self) -> int:
        steps = {estimator.projection_step for estimator in self.estimators}
        if len(steps) != 1:
            raise TritiumError(
                "additive estimator child schedules differ",
                code="estimator_contract",
                stage="schedule",
            )
        return steps.pop()


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


class HestiaEstimator(Estimator):
    """Soft ternary expectation with exponential temperature decay."""

    algorithm_id = "tritium.hestia"
    schema_version = 1

    def __init__(
        self,
        initial_temperature: float = 1.0,
        temperature_floor: float = 0.01,
        total_steps: int = 100,
    ) -> None:
        super().__init__()
        if (
            not math.isfinite(initial_temperature)
            or not math.isfinite(temperature_floor)
            or initial_temperature <= 0
            or temperature_floor <= 0
            or initial_temperature < temperature_floor
            or isinstance(total_steps, bool)
            or not isinstance(total_steps, int)
            or total_steps <= 0
        ):
            raise ValueError("invalid HESTIA temperature schedule")
        self.initial_temperature = float(initial_temperature)
        self.temperature_floor = float(temperature_floor)
        self.total_steps = total_steps
        self.register_buffer("_schedule_step", torch.tensor(0, dtype=torch.int64))
        self._schedule_step_value = 0

    @property
    def schedule_step(self) -> int:
        """Current explicit schedule position used by module forwards and export."""

        return self._schedule_step_value

    @property
    def projection_step(self) -> int:
        return self.schedule_step

    def set_step(self, step: int) -> "HestiaEstimator":
        """Set schedule position; training loops call this before each forward."""

        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("HESTIA schedule step must be a nonnegative integer")
        self._schedule_step.fill_(step)
        self._schedule_step_value = step
        return self

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._schedule_step_value = int(self._schedule_step.item())

    def temperature(self, step: int) -> float:
        """Return schedule temperature, clamped to exact endpoint values."""

        if step <= 0:
            return self.initial_temperature
        if step >= self.total_steps:
            return self.temperature_floor
        progress = step / self.total_steps
        log_tau = math.log(self.initial_temperature) + progress * (
            math.log(self.temperature_floor) - math.log(self.initial_temperature)
        )
        return max(self.temperature_floor, math.exp(log_tau))

    def project(
        self, master: torch.Tensor, *, context: ProjectionContext
    ) -> TernaryProjection:
        _rank2(master, "HestiaEstimator")
        tau = self.temperature(context.step)
        scales = _absmean_scale(master)
        safe = scales.clamp_min(torch.finfo(master.dtype).tiny)
        normalized = master / safe
        soft = hestia_soft_expectation(master, scales, master.new_tensor(tau))
        trits = normalized.detach().round().clamp(-1, 1).to(torch.int8)
        exportable = tau <= self.temperature_floor
        return _soft_projection(
            master,
            trits,
            scales,
            soft,
            self,
            exportable=exportable,
        )


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
    physical_planes = 2

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
        positive_trits = (detached > threshold).to(torch.int8)
        negative_trits = -(detached < -threshold).to(torch.int8)
        positive_scales = positive_scale.expand(master.shape[0], 1)
        negative_scales = negative_scale.expand(master.shape[0], 1)
        sharpness = 10.0
        positive_gate = torch.sigmoid(sharpness * (master - threshold))
        negative_gate = torch.sigmoid(sharpness * (-master - threshold))
        surrogate = positive_scale * positive_gate - negative_scale * negative_gate
        return _multi_projection(
            master,
            (
                (positive_trits, positive_scales),
                (negative_trits, negative_scales),
            ),
            surrogate,
            self,
        )


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
    "hestia": HestiaEstimator,
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
