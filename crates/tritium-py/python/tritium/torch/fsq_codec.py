"""Unified, swappable scalar-quantizer for the LMQ neural codec.

ONE home for FSQ. Before this module the level set ``[2,3,5,8,16,32]`` and the
``step = 2/L`` round-grid were inlined in three places in ``encoder.py`` plus a
separate ``MultiScaleFSQ`` that used a *different* grid (floor + half-step
centers, no point at 0) — four implementations waiting to drift. See ADR 0068.

Design:
  * ``fsq_quantize`` / ``fsq_dither`` / ``fsq_dropout_ste`` / ``fsq_infer`` are
    the canonical primitives. Every quantization path in the codec routes
    through them, so the grid is defined exactly once.
  * ``Quantizer`` is the swappable interface; ``ScalarFSQ`` and ``ResidualFSQ``
    conform to it. Future quantizers (e.g. BSQ — ADR 0068 §SOTA) implement the
    same ``forward`` contract and drop in without touching call sites.
  * The straight-through estimator is **pluggable** (``STE`` enum). Default
    ``HARD`` reproduces the historical behaviour bit-for-bit (round-to-nearest
    grid, identity-gradient STE). ``SOFT`` is the annealed soft-round staircase
    (β→∞ ⇒ exact hard FSQ ⇒ train == deploy); its temperature schedule is wired
    in W2. ``STOCHASTIC`` is unbiased stochastic rounding.

THE GRID (canonical, matches the deployed codec ``_maybe_infer_fsq``):
    step = 2 / L ;  index = round(value / step) ;  dequantized = index * step
on the CDF-normalized [-1, 1] latent. Indexing is dimension-agnostic (no
coupling to LATENT_DIM); ``L`` is the per-window operating rate (~log2(L) b/val).
"""
from __future__ import annotations

import enum
import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Ported verbatim from LamQuant lamquant_neural/models/quantizer.py by its
# copyright holder (ADR 0037 Stage 4); FSQ_LEVELS inlined from its
# constants.py single source (guarded there by a single-source test).
FSQ_LEVELS: tuple[int, ...] = (2, 3, 5, 8, 16, 32)

__all__ = [
    "STE", "fsq_step", "soft_round", "fsq_quantize", "fsq_dither",
    "fsq_dither_seeded", "fsq_dropout_ste", "fsq_dropout_ste_seeded",
    "fsq_infer", "Quantizer", "ScalarFSQ", "FSQ_LEVELS",
]


class STE(enum.Enum):
    """Straight-through-estimator strategy for the hard FSQ round."""
    HARD = "hard"          # forward=round, grad=identity (historical default)
    SOFT = "soft"          # annealed soft-round; differentiable, β→∞ == HARD
    STOCHASTIC = "stochastic"  # unbiased stochastic rounding


def fsq_step(level: int) -> float:
    """The quantization step for ``level`` FSQ resolutions on [-1, 1].

    Precondition: ``level >= 2`` (a single-level grid is meaningless).
    """
    if level < 2:
        raise ValueError(f"FSQ level must be >= 2, got {level}")
    return 2.0 / level


def soft_round(y: torch.Tensor, beta: float) -> torch.Tensor:
    """Differentiable approximation of ``round(y)`` (Agustsson & Theis, 2020).

    ``soft_round(y, β) = ⌊y⌋ + ½ + ½·tanh(β·(frac−½)) / tanh(β/2)``.
    β→∞ ⇒ exact ``round`` (the deployed grid); β→0 ⇒ identity. Used by the
    annealed STE so the *training* objective converges to the *deployed* hard
    quantizer rather than optimizing a continuous proxy (ADR 0068 §2).
    """
    if beta <= 0:
        return y
    floor = torch.floor(y)
    frac = y - floor
    denom = torch.tanh(torch.tensor(beta / 2.0, device=y.device, dtype=y.dtype))
    return floor + 0.5 + 0.5 * torch.tanh(beta * (frac - 0.5)) / denom


def fsq_quantize(
    x: torch.Tensor, level: int, *, ste: STE = STE.HARD, beta: float = 8.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hard FSQ on the canonical round grid, with a pluggable STE.

    Args:
        x:     latent in [-1, 1] (post-CDF). Any shape.
        level: number of quantization resolutions L (the operating rate).
        ste:   gradient strategy (see ``STE``).
        beta:  temperature for ``STE.SOFT`` (ignored otherwise).

    Returns:
        ``(dequantized, indices)`` — ``dequantized`` carries the chosen STE
        gradient; ``indices`` are the integer codes (``round(x/step)``, detached).
    """
    step = fsq_step(level)
    y = x / step
    idx = torch.round(y)
    q = idx * step
    if ste is STE.HARD:
        deq = x + (q - x).detach()                      # forward=q, grad=1
    elif ste is STE.SOFT:
        deq = soft_round(y, beta) * step                # differentiable, →q
    elif ste is STE.STOCHASTIC:
        noisy = y + (torch.rand_like(y) - 0.5)          # unbiased rounding
        idx_s = torch.round(noisy)
        deq = x + (idx_s * step - x).detach()
        idx = idx_s
    else:  # pragma: no cover - exhaustive
        raise ValueError(f"unknown STE {ste!r}")
    return deq, idx.detach()


def fsq_dither(x: torch.Tensor, level: int) -> torch.Tensor:
    """Additive uniform dither of width ``step`` — no quantization.

    The light training regularizer historically inlined in
    ``encoder.encode_stage3``: it perturbs the latent within one quantization
    cell so the decoder learns robustness to the eventual rounding, without
    committing to a hard grid during training.
    """
    step = fsq_step(level)
    return x + (torch.rand_like(x) - 0.5) * step


def _compiled_generators(
    x: torch.Tensor,
    stochastic_seeds,
    *,
    domain: str,
):
    seeds = tuple(stochastic_seeds)
    if len(seeds) != x.shape[0]:
        raise ValueError(
            "stochastic_seeds must contain one compiled seed per row"
        )
    generators = []
    for seed in seeds:
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or seed > (1 << 64) - 1
        ):
            raise ValueError("stochastic seed must be a valid u64")
        digest = hashlib.sha256(
            b"org.quitetall.lamquant.fsq-training-v1\0"
            + domain.encode("ascii")
            + b"\0"
            + seed.to_bytes(8, "little")
        ).digest()
        generator = torch.Generator(device=x.device)
        generator.manual_seed(int.from_bytes(digest[:8], "little"))
        generators.append(generator)
    return generators


def fsq_dither_seeded(
    x: torch.Tensor,
    stochastic_seeds,
) -> torch.Tensor:
    """Apply row-local random-level FSQ dither from compiled seeds."""
    if x.ndim != 3:
        raise ValueError(f"seeded FSQ expects [B, C, T], got {tuple(x.shape)}")
    rows = []
    for row, generator in zip(
        x,
        _compiled_generators(x, stochastic_seeds, domain="dither"),
        strict=True,
    ):
        level_index = int(
            torch.randint(
                0,
                len(FSQ_LEVELS),
                (),
                generator=generator,
                device=x.device,
            ).item()
        )
        step = fsq_step(int(FSQ_LEVELS[level_index]))
        noise = torch.rand(
            row.shape,
            dtype=row.dtype,
            device=row.device,
            generator=generator,
        )
        rows.append(row + (noise - 0.5) * step)
    return torch.stack(rows, dim=0)


def fsq_dropout_ste(x: torch.Tensor, level: int) -> torch.Tensor:
    """StableCodec dithered-FSQ dropout (per-sample 50% pass / 25% dither / 25% hard).

    Reproduces, exactly, the historical ``encoder.encode`` / V2 training path:
    a per-sample Bernoulli mask routes each window to passthrough, additive
    dither, or hard-STQ quantization. Source: stable_codec/fsq.py.
    """
    assert x.ndim == 3, f"fsq_dropout_ste expects [B, C, T], got {tuple(x.shape)}"
    step = fsq_step(level)
    b = x.shape[0]
    mask_pass = torch.bernoulli(torch.full((b, 1, 1), 0.5, device=x.device)).bool()
    mask_noise = torch.bernoulli(torch.full((b, 1, 1), 0.5, device=x.device)).bool()
    noise = (torch.rand_like(x) - 0.5) * step
    q_hard = x + (torch.round(x / step) * step - x).detach()
    return torch.where(
        mask_pass.expand_as(x), x,
        torch.where(mask_noise.expand_as(x), x + noise, q_hard),
    )


def fsq_dropout_ste_seeded(
    x: torch.Tensor,
    stochastic_seeds,
) -> torch.Tensor:
    """Apply row-local random-level FSQ dropout from compiled seeds."""
    if x.ndim != 3:
        raise ValueError(f"seeded FSQ expects [B, C, T], got {tuple(x.shape)}")
    rows = []
    for row, generator in zip(
        x,
        _compiled_generators(x, stochastic_seeds, domain="dropout-ste"),
        strict=True,
    ):
        level_index = int(
            torch.randint(
                0,
                len(FSQ_LEVELS),
                (),
                generator=generator,
                device=x.device,
            ).item()
        )
        step = fsq_step(int(FSQ_LEVELS[level_index]))
        route = float(
            torch.rand(
                (),
                generator=generator,
                device=x.device,
            ).item()
        )
        if route < 0.5:
            rows.append(row)
        elif route < 0.75:
            noise = torch.rand(
                row.shape,
                dtype=row.dtype,
                device=row.device,
                generator=generator,
            )
            rows.append(row + (noise - 0.5) * step)
        else:
            rows.append(row + (torch.round(row / step) * step - row).detach())
    return torch.stack(rows, dim=0)


def fsq_infer(x: torch.Tensor, level: int) -> torch.Tensor:
    """Eval-time hard FSQ — the REAL deployed codec quantization (no STE).

    ``round(x/step)*step`` on the round grid. Forward-identical to
    ``fsq_quantize(x, level, ste=HARD)`` but without the gradient passthrough,
    matching the historical ``encoder._maybe_infer_fsq``.
    """
    step = fsq_step(level)
    return torch.round(x / step) * step


# ── Swappable quantizer interface ──────────────────────────────────────────

class Quantizer(nn.Module, ABC):
    """A latent quantizer. Implementations are interchangeable at call sites.

    ``forward`` returns ``(reconstructed_latent, indices, aux)`` where ``aux``
    is a dict (e.g. residual quant loss, per-scale tokens). ``training`` /
    ``self.training`` selects the stochastic train path vs the deterministic
    eval grid, so callers never branch on mode themselves.
    """

    @abstractmethod
    def forward(self, latent: torch.Tensor, level: Optional[int] = None):
        ...


class ScalarFSQ(Quantizer):
    """Flat scalar FSQ at a single resolution — the production quantizer.

    train: ``fsq_dropout_ste`` (or ``fsq_dither`` if ``light_dither``).
    eval:  ``fsq_infer`` at ``level`` (or pass-through if ``level is None``,
           preserving the historical L=∞ continuous-eval default).
    """

    def __init__(self, level: Optional[int] = None, *,
                 ste: STE = STE.HARD, light_dither: bool = False):
        super().__init__()
        self.level = level
        self.ste = ste
        self.light_dither = light_dither

    def forward(self, latent: torch.Tensor, level: Optional[int] = None):
        L = level if level is not None else self.level
        if self.training:
            if L is None:
                L = int(FSQ_LEVELS[torch.randint(0, len(FSQ_LEVELS), (1,)).item()])
            q = fsq_dither(latent, L) if self.light_dither else fsq_dropout_ste(latent, L)
            return q, None, {}
        if L is None:
            return latent, None, {}        # continuous eval (L=∞ ceiling)
        step = fsq_step(L)
        idx = torch.round(latent / step).detach()
        return fsq_infer(latent, L), idx, {}


# NOTE: the SNAC-style residual / multi-scale quantizer currently lives in
# ``lamquant.student.multiscale_fsq.MultiScaleFSQ`` (not in production). It uses
# its own historical floor-centre index grid; ADR 0068 Phase 6 (residual-FSQ +
# SNN per-window level) migrates it onto these canonical primitives and the
# ``Quantizer`` interface. Kept out of this module until then so there is a
# single residual implementation, not two.
