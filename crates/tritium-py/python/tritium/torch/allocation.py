"""Native rate-distortion allocation for additive ternary planes."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Sequence

from .. import _tritium


@dataclass(frozen=True)
class PlaneAllocation:
    """Deterministic plane counts selected from measured error curves."""

    plane_counts: tuple[int, ...]
    target_bpw: float
    achieved_bpw: float
    total_weights: int


def _materialize(value: Sequence[object], label: str) -> tuple[object, ...]:
    try:
        return tuple(value)
    except TypeError as error:
        raise ValueError(f"{label} must be an iterable") from error


def _positive_size(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{label} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _plane_bound(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a nonnegative integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{label} must be a nonnegative integer") from error
    if result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _finite_float(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite numeric evidence")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be finite numeric evidence") from error
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        suffix = " finite and nonnegative" if nonnegative else " finite"
        raise ValueError(f"{label} must be{suffix}")
    return result


def allocate_planes(
    group_sizes: Sequence[int],
    sensitivities: Sequence[float],
    error_curves: Sequence[Sequence[float]],
    target_bpw: float,
    *,
    t_min: int = 1,
    t_max: int = 3,
) -> PlaneAllocation:
    """Allocate planes with native deterministic rate-distortion water-filling.

    ``error_curves[g]`` must contain measured nonnegative ``err(0..=t_max)``
    values for group ``g``. Returned ``achieved_bpw`` never exceeds the native
    budget except for allocator floating-point tolerance.
    """

    raw_sizes = _materialize(group_sizes, "group_sizes")
    raw_sensitivities = _materialize(sensitivities, "sensitivities")
    raw_curves = _materialize(error_curves, "error_curves")
    if not raw_sizes:
        raise ValueError("group_sizes must not be empty")
    if not (
        len(raw_sizes) == len(raw_sensitivities) == len(raw_curves)
    ):
        raise ValueError(
            "group_sizes, sensitivities, and error_curves must have equal lengths"
        )
    sizes = tuple(
        _positive_size(value, f"group_sizes[{index}]")
        for index, value in enumerate(raw_sizes)
    )
    normalized_sensitivities = tuple(
        _finite_float(value, f"sensitivities[{index}]", nonnegative=True)
        for index, value in enumerate(raw_sensitivities)
    )
    normalized_curves: tuple[tuple[float, ...], ...] = tuple(
        tuple(
            _finite_float(value, f"error_curves[{group}][{index}]", nonnegative=True)
            for index, value in enumerate(_materialize(curve, f"error_curves[{group}]"))
        )
        for group, curve in enumerate(raw_curves)
    )
    normalized_target = _finite_float(target_bpw, "target_bpw")
    if normalized_target <= 0.0:
        raise ValueError("target_bpw must be finite and positive")
    normalized_t_min = _plane_bound(t_min, "t_min")
    normalized_t_max = _plane_bound(t_max, "t_max")
    if normalized_t_min > normalized_t_max:
        raise ValueError("t_min must not exceed t_max")
    for group, curve in enumerate(normalized_curves):
        if len(curve) <= normalized_t_max:
            raise ValueError(
                f"error_curves[{group}] must contain t_max + 1 values"
            )

    counts, achieved_bpw = _tritium.allocate_planes(
        list(sizes),
        list(normalized_sensitivities),
        [list(curve) for curve in normalized_curves],
        normalized_target,
        normalized_t_min,
        normalized_t_max,
    )
    if len(counts) != len(sizes) or any(
        count < normalized_t_min or count > normalized_t_max for count in counts
    ):
        raise RuntimeError("native allocator returned invalid plane counts")
    return PlaneAllocation(
        plane_counts=tuple(int(count) for count in counts),
        target_bpw=normalized_target,
        achieved_bpw=float(achieved_bpw),
        total_weights=sum(sizes),
    )
