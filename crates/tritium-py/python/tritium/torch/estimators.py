"""Reference ternary estimators implemented with device-resident PyTorch ops."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from .projection import ProjectionContext, TernaryProjection


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
            trits=hard.detach().to(torch.int8),
            scales=scales,
            group_size=master.shape[1],
            algorithm_id=self.algorithm_id,
            schema_version=self.schema_version,
        )


class SaltSTE(AbsMeanSTE):
    """Single-plane SALT QAT reference; additive planes land in plan 0048."""

    algorithm_id = "tritium.salt-ste"
