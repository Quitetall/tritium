"""Bounded hard-discrete refinement primitives for additive ternary weights."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from .config import RefinementConfig
from .projection import TernaryPlane


@dataclass(frozen=True)
class RefinedWeight:
    """One hard child weight plus exact parent/child reconstruction losses."""

    planes: Tuple[TernaryPlane, ...]
    parent_weighted_mse: float
    refined_weighted_mse: float
    iterations: int
    kind: str
    structure: str


def _validate(
    master: torch.Tensor,
    planes: Sequence[TernaryPlane],
    metric: torch.Tensor,
    config: RefinementConfig,
    iterations: int,
    max_working_bytes: int,
) -> None:
    if (
        not isinstance(master, torch.Tensor)
        or master.ndim != 2
        or not master.dtype.is_floating_point
        or not bool(torch.isfinite(master).all())
    ):
        raise ValueError("refinement master must be one finite floating matrix")
    if not 1 <= len(planes) <= 3:
        raise ValueError("refinement requires one to three parent planes")
    if (
        not isinstance(metric, torch.Tensor)
        or metric.ndim != 1
        or metric.numel() != master.shape[1]
        or not metric.dtype.is_floating_point
        or not bool(torch.isfinite(metric).all())
        or bool((metric < 0).any())
        or not bool((metric > 0).any())
    ):
        raise ValueError("refinement metric must be finite nonnegative input curvature")
    for plane in planes:
        if (
            plane.trits.dtype != torch.int8
            or tuple(plane.trits.shape) != tuple(master.shape)
            or plane.trits.device != master.device
            or plane.scales.dtype != torch.float16
            or tuple(plane.scales.shape) != (master.shape[0], 1)
            or plane.scales.device != master.device
            or plane.group_size != master.shape[1]
            or plane.structure not in {"dense", "s34"}
            or not bool(torch.all((plane.trits >= -1) & (plane.trits <= 1)))
            or not bool(torch.isfinite(plane.scales).all())
            or bool((plane.scales < 0).any())
        ):
            raise ValueError("refinement parent plane is not deployable row-scale SALT")
        if plane.structure == "s34" and (
            master.shape[1] % 4
            or not bool(
                torch.all(
                    torch.count_nonzero(
                        plane.trits.reshape(master.shape[0], -1, 4) == 0,
                        dim=2,
                    )
                    == 1
                )
            )
        ):
            raise ValueError("refinement parent S34 plane violates one-zero groups")
    if not isinstance(config, RefinementConfig):
        raise TypeError("config must be RefinementConfig")
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if type(max_working_bytes) is not int or max_working_bytes < 1024:
        raise ValueError("max_working_bytes must be at least 1024")
    if config.structure == "s34" and master.shape[1] % 4:
        raise ValueError("S34 refinement requires input width divisible by four")
    if config.kind == "scale-only" and any(
        plane.structure != config.structure for plane in planes
    ):
        raise ValueError("scale-only refinement must preserve parent structure")
    if metric.device != master.device:
        raise ValueError("refinement metric must share the master device")


def _decoded(
    trits: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.zeros_like(trits[0], dtype=dtype)
    for codes, row_scales in zip(trits, scales):
        value.add_(codes.to(dtype) * row_scales.to(dtype))
    return value


def _loss_rows(
    master: torch.Tensor,
    trits: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
    metric: torch.Tensor,
) -> torch.Tensor:
    dtype = torch.float64
    residual = master.to(dtype) - _decoded(trits, scales, dtype)
    return (residual.square() * metric.to(dtype)).sum(dim=1)


def _solve_scales(
    master: torch.Tensor,
    trits: Sequence[torch.Tensor],
    metric: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Exact small-K nonnegative least squares, then deployment f16 rounding."""

    dtype = torch.float64
    codes = torch.stack(tuple(value.to(dtype) for value in trits), dim=1)
    curvature = metric.to(dtype)
    target = master.to(dtype)
    gram = torch.einsum("rpi,i,rqi->rpq", codes, curvature, codes)
    rhs = torch.einsum("rpi,i,ri->rp", codes, curvature, target)
    rows, plane_count = rhs.shape
    best = torch.zeros((rows, plane_count), dtype=dtype, device=master.device)
    best_loss = (target.square() * curvature).sum(dim=1)
    for mask in range(1, 1 << plane_count):
        active = [index for index in range(plane_count) if mask & (1 << index)]
        sub_gram = gram[:, active][:, :, active]
        sub_rhs = rhs[:, active]
        solution = torch.linalg.pinv(sub_gram) @ sub_rhs.unsqueeze(-1)
        solution = solution.squeeze(-1)
        feasible = torch.isfinite(solution).all(dim=1) & (solution >= 0).all(dim=1)
        candidate = torch.zeros_like(best)
        candidate[:, active] = solution
        decoded = torch.einsum("rpi,rp->ri", codes, candidate)
        loss = ((target - decoded).square() * curvature).sum(dim=1)
        accept = feasible & (loss < best_loss)
        best[accept] = candidate[accept]
        best_loss[accept] = loss[accept]
    stored = best.to(torch.float16)
    return tuple(stored[:, index : index + 1] for index in range(plane_count))


def _dense_sweep(
    master: torch.Tensor,
    trits: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    dtype = torch.float64
    result = [value.clone() for value in trits]
    for plane_index, scale in enumerate(scales):
        other = _decoded(
            tuple(
                value
                for index, value in enumerate(result)
                if index != plane_index
            ),
            tuple(
                value
                for index, value in enumerate(scales)
                if index != plane_index
            ),
            dtype,
        ) if len(result) > 1 else torch.zeros_like(master, dtype=dtype)
        row_scale = scale.to(dtype)
        residual = master.to(dtype) - other
        negative = (residual + row_scale).square()
        zero = residual.square()
        positive = (residual - row_scale).square()
        selected = (
            torch.stack((negative, zero, positive), dim=0).argmin(dim=0) - 1
        ).to(torch.int8)
        selected[row_scale.squeeze(1) == 0] = 0
        result[plane_index] = selected
    return tuple(result)


def _s34_patterns(device: torch.device) -> torch.Tensor:
    values = [
        pattern
        for pattern in itertools.product((-1, 0, 1), repeat=4)
        if pattern.count(0) == 1 and all(value != 0 for value in pattern if value)
    ]
    return torch.tensor(values, dtype=torch.float64, device=device)


def _s34_sweep(
    master: torch.Tensor,
    trits: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
    metric: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    dtype = torch.float64
    patterns = _s34_patterns(master.device)
    rows, columns = master.shape
    groups = columns // 4
    curvature = metric.to(dtype).reshape(1, groups, 4)
    result = [value.clone() for value in trits]
    for plane_index, scale in enumerate(scales):
        other = _decoded(
            tuple(
                value
                for index, value in enumerate(result)
                if index != plane_index
            ),
            tuple(
                value
                for index, value in enumerate(scales)
                if index != plane_index
            ),
            dtype,
        ) if len(result) > 1 else torch.zeros_like(master, dtype=dtype)
        residual = (master.to(dtype) - other).reshape(rows, groups, 4)
        row_scale = scale.to(dtype).reshape(rows, 1, 1)
        best_error = torch.full(
            (rows, groups), math.inf, dtype=dtype, device=master.device
        )
        best_index = torch.zeros(
            (rows, groups), dtype=torch.int64, device=master.device
        )
        for index, pattern in enumerate(patterns):
            error = (
                (residual - row_scale * pattern.reshape(1, 1, 4)).square()
                * curvature
            ).sum(dim=2)
            accept = error < best_error
            best_error[accept] = error[accept]
            best_index[accept] = index
        result[plane_index] = patterns[best_index].reshape(rows, columns).to(
            torch.int8
        )
    return tuple(result)


def refine_weight_diagonal(
    master: torch.Tensor,
    planes: Sequence[TernaryPlane],
    metric: torch.Tensor,
    config: RefinementConfig,
    *,
    iterations: int = 4,
    max_working_bytes: int = 64 * 1024 * 1024,
) -> RefinedWeight:
    """Refine one additive weight under streamed diagonal input curvature."""

    _validate(master, planes, metric, config, iterations, max_working_bytes)
    rows, columns = master.shape
    plane_count = len(planes)
    bytes_per_row = max(1, columns * (24 + 8 * plane_count))
    chunk_rows = max(1, min(rows, max_working_bytes // bytes_per_row))
    output_trits = [torch.empty_like(plane.trits) for plane in planes]
    output_scales = [
        torch.empty_like(plane.scales, dtype=torch.float16) for plane in planes
    ]
    parent_total = 0.0
    refined_total = 0.0
    denominator = float(metric.to(torch.float64).sum()) * rows
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        chunk_master = master[start:end]
        chunk_trits = tuple(plane.trits[start:end].clone() for plane in planes)
        chunk_scales = tuple(plane.scales[start:end].clone() for plane in planes)
        parent_loss = _loss_rows(chunk_master, chunk_trits, chunk_scales, metric)
        if config.kind == "scale-only":
            candidate_trits = chunk_trits
            candidate_scales = _solve_scales(chunk_master, candidate_trits, metric)
        else:
            candidate_trits = chunk_trits
            candidate_scales = chunk_scales
            for _ in range(iterations):
                if config.structure == "s34":
                    candidate_trits = _s34_sweep(
                        chunk_master, candidate_trits, candidate_scales, metric
                    )
                else:
                    candidate_trits = _dense_sweep(
                        chunk_master, candidate_trits, candidate_scales
                    )
                candidate_scales = _solve_scales(
                    chunk_master, candidate_trits, metric
                )
        candidate_loss = _loss_rows(
            chunk_master, candidate_trits, candidate_scales, metric
        )
        if config.structure == "dense":
            accept = candidate_loss <= parent_loss
            for index in range(plane_count):
                candidate_trits[index][~accept] = chunk_trits[index][~accept]
                candidate_scales[index][~accept] = chunk_scales[index][~accept]
            candidate_loss = torch.minimum(candidate_loss, parent_loss)
        for index in range(plane_count):
            output_trits[index][start:end] = candidate_trits[index]
            output_scales[index][start:end] = candidate_scales[index]
        parent_total += float(parent_loss.sum())
        refined_total += float(candidate_loss.sum())
    structure = config.structure
    refined_planes = tuple(
        TernaryPlane(
            trits=trits,
            scales=scales,
            group_size=columns,
            structure=structure,
        )
        for trits, scales in zip(output_trits, output_scales)
    )
    return RefinedWeight(
        planes=refined_planes,
        parent_weighted_mse=parent_total / denominator,
        refined_weighted_mse=refined_total / denominator,
        iterations=0 if config.kind == "scale-only" else iterations,
        kind=config.kind,
        structure=structure,
    )


__all__ = ["RefinedWeight", "refine_weight_diagonal"]
