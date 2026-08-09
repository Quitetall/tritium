# Ported verbatim from LamQuant (lamquant/ingredients/optimizers/esoap.py) into
# tritium.torch by the LamQuant copyright holder; relicensed to this
# repository's Apache-2.0 terms. Original header retained below.
#
# ESOAP — Eigen-Split adaptive optimizer: SOAP/Adam preconditioning on the
# LEADING eigensubspace + Muon orthogonalization on the TAIL eigensubspace.
#
# CLEAN-ROOM reimplementation of the *principle* described in the COSMOS paper
# (arXiv:2502.17410): "apply SOAP to the leading eigensubspace, which captures
# the primary optimization dynamics, and Muon to the remaining eigensubspace."
# Implemented from the paper's stated principle ONLY — no code from the
# unlicensed upstream repo (github.com/lliu606/COSMOS) was read or copied; that
# repo stays REFERENCE-ONLY in reference/REGISTRY.yaml and the trainer's
# `--optimizer cosmos` remains hard-gated. This is our own design and may differ
# from COSMOS in side-handling, rank split, and scaling.
#
# Method components (both OSI-permissive lineage):
#   - SOAP eigenbasis idea (Gram eigvecs) — public; our own implementation.
#   - Muon Newton-Schulz orthogonalizer — modded-nanogpt (MIT), standard NS5.
#
# Per 2-D weight W[m,n] with raw grad G and Nesterov momentum M:
#   1. EMA right-Gram  C = β·C + (1-β)·GᵀG   (n×n)
#   2. eigvecs V of C, descending eigenvalue
#   3. rotate momentum into the column eigenbasis: A = M·V   (m×n)
#   4. SPLIT columns by eigenvalue rank: lead = A[:, :r], tail = A[:, r:]
#   5. LEAD → Adam: v ← β₂v + (1-β₂)lead²;  upd_lead = lead / (√v + ε)
#   6. TAIL → Muon: upd_tail = NewtonSchulz5(tail)   (orthogonalize)
#   7. recombine + rotate back: U = [upd_lead | upd_tail]·Vᵀ
#   8. RMS-normalize U, apply  p ← p − lr·U   (decoupled wd optional)
#
# Single torch.optim.Optimizer with method-tagged groups (same contract as
# sinksoaph.py): method="esoap" on 2-D matrices, method="adamw" on the rest.
# Single-GPU / single-process.

from __future__ import annotations

import torch
from torch import Tensor

from tritium.torch.optim_cautious import cautious_decoupled_wd_

__all__ = ["ESOAP"]


def _gram_eigenbasis(C: Tensor, eps: float) -> Tensor:
    """Descending-eigenvalue eigenvectors of a symmetric PSD Gram matrix."""
    C = C.float()
    C = 0.5 * (C + C.T)
    if eps > 0.0:
        C = C + eps * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
    _evals, evecs = torch.linalg.eigh(C)
    return evecs.flip(-1)


def _newton_schulz5(G: Tensor, steps: int, eps: float) -> Tensor:
    """Quintic Newton-Schulz orthogonalization (Muon). Drives the singular
    values of ``G`` toward 1 (≈ U Vᵀ of its SVD) without an explicit SVD.

    Standard Muon NS5 coefficients (modded-nanogpt, MIT). Operates in float32;
    the iteration is row/column-symmetric so we transpose tall inputs to keep
    the inner products small.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    norm = X.norm()
    # Threshold, not == 0: a subnormal-but-nonzero norm would make X/(norm+eps)
    # explode to O(1e8) and the quintic NS iterations then diverge to NaN.
    if float(norm) < eps:
        return torch.zeros_like(X)
    X = X / (norm + eps)
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X


def _esoap_direction(
    grad: Tensor,
    momentum: Tensor,
    gram: Tensor,
    v_lead: Tensor,
    rank: int,
    mu: float,
    gram_beta: float,
    beta2: float,
    nesterov: bool,
    ns_steps: int,
    eps: float,
) -> Tensor:
    """ESOAP update direction for one 2-D matrix (RMS-normalized).

    Mutates ``momentum`` / ``gram`` / ``v_lead`` in place (EMA state). ``rank``
    is the number of leading eigen-columns that get Adam preconditioning; the
    remaining ``n - rank`` columns get Muon orthogonalization.
    """
    G = grad.float()
    m, n = G.shape
    assert v_lead.shape[1] == max(1, min(rank, n)), (
        f"v_lead width {v_lead.shape[1]} != rank {rank} (rank_frac changed "
        "mid-training? state shape is fixed at init)")

    # A non-finite grad would poison the Gram/v_lead/momentum EMAs permanently
    # (the eigenbasis never recovers). Skip the update on this matrix instead.
    if not torch.isfinite(G).all():
        return torch.zeros_like(G)

    # First-order momentum + Nesterov-style (Muon-convention) blend: this is the
    # Muon momentum mix, not textbook Nesterov lookahead.
    momentum.lerp_(G, 1.0 - mu)
    M = G.lerp(momentum, mu) if nesterov else momentum.clone()

    # EMA right-Gram + its eigenbasis (column eigenspace).
    gram.lerp_(G.T @ G, 1.0 - gram_beta)
    V = _gram_eigenbasis(gram, eps)                 # [n, n]

    # Rotate momentum into the column eigenbasis, split lead | tail.
    A = M @ V                                       # [m, n]
    r = max(1, min(rank, n))
    lead = A[:, :r]                                 # [m, r] dominant dynamics
    tail = A[:, r:]                                 # [m, n-r] remainder

    # LEAD: Adam second-moment preconditioning (SOAP-style, in rotated coords).
    v_lead.mul_(beta2).addcmul_(lead, lead, value=1.0 - beta2)
    upd_lead = lead / (v_lead.sqrt() + eps)

    # TAIL: Muon orthogonalization (cheap, no second moment).
    if tail.shape[1] > 0:
        upd_tail = _newton_schulz5(tail, steps=ns_steps, eps=eps)
        A_upd = torch.cat([upd_lead, upd_tail], dim=1)
    else:
        A_upd = upd_lead

    # Rotate back, RMS-normalize so the step scale is lr-comparable to AdamW.
    U = A_upd @ V.T                                 # [m, n]
    rms = U.square().mean().sqrt().clamp_min(eps)
    return U / rms


def _adamw_apply_(
    p: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor, step: int,
    lr: float, beta1: float, beta2: float, eps: float, weight_decay: float,
    cautious_wd: bool = False,
) -> None:
    """Standard AdamW step (in place) — for the non-esoap group.

    ``cautious_wd`` (ADR 0030, SPECULATIVE, default off): decay only entries
    where the AdamW update already agrees in sign with the param. WD on ``p``
    is independent of the EMA/denom computation, so the plain path below is
    byte-identical to applying decay first.
    """
    exp_avg.lerp_(grad, 1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bias1 = 1.0 - beta1 ** step
    bias2 = 1.0 - beta2 ** step
    denom = (exp_avg_sq.sqrt() / (bias2 ** 0.5)).add_(eps)
    if weight_decay != 0.0 and cautious_wd:
        update = (exp_avg / denom).mul_(lr / bias1)
        cautious_decoupled_wd_(p, update, lr, weight_decay)
    else:
        if weight_decay != 0.0:
            p.mul_(1.0 - lr * weight_decay)
        p.addcdiv_(exp_avg, denom, value=-lr / bias1)


class ESOAP(torch.optim.Optimizer):
    """Eigen-Split optimizer: SOAP/Adam lead + Muon tail (COSMOS-principle).

    Param groups routed by a ``method`` tag:
      * ``"esoap"`` — 2-D params only. State: ``momentum [m,n]``, ``gram [n,n]``,
        ``v_lead [m,r]``. ``weight_decay`` applies as decoupled decay.
      * ``"adamw"`` — any shape. State: ``step``, ``exp_avg``, ``exp_avg_sq``.

    ``rank_frac`` sets the leading-subspace size: ``r = max(1, round(n·frac))``.
    The WSDScheduler drives ``group["lr"]`` for both methods.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        *,
        rank_frac: float = 0.5,
        mu: float = 0.95,
        gram_beta: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.95),
        ns_steps: int = 5,
        nesterov: bool = True,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        cautious_wd: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 < rank_frac <= 1.0:
            raise ValueError(f"rank_frac must be in (0,1], got {rank_frac}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")
        defaults = dict(
            lr=lr, method="adamw", rank_frac=rank_frac, mu=mu,
            gram_beta=gram_beta, betas=betas, ns_steps=ns_steps,
            nesterov=nesterov, eps=eps, weight_decay=weight_decay,
            cautious_wd=cautious_wd,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            method = group.get("method", "adamw")
            if method not in ("esoap", "adamw"):
                raise ValueError(
                    f"unknown param-group method {method!r} "
                    "(expected 'esoap' or 'adamw')")
            if method == "esoap":
                for p in group["params"]:
                    if p.ndim != 2:
                        raise ValueError(
                            "method='esoap' only supports 2-D parameters; got "
                            f"shape {tuple(p.shape)}. Route non-2-D to 'adamw'.")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            method = group.get("method", "adamw")
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ESOAP does not support sparse grads")
                state = self.state[p]

                if method == "esoap":
                    m, n = p.shape
                    r = max(1, round(n * group["rank_frac"]))
                    r = min(r, n)
                    if len(state) == 0:
                        # All EMA state in float32 (momentum included) so a
                        # future bf16/half run can't accumulate truncation error
                        # in the momentum tracker — matches gram/v_lead + Muon.
                        state["momentum"] = torch.zeros_like(
                            p, dtype=torch.float32)
                        state["gram"] = torch.zeros(
                            (n, n), device=p.device, dtype=torch.float32)
                        state["v_lead"] = torch.zeros(
                            (m, r), device=p.device, dtype=torch.float32)
                    beta1, beta2 = group["betas"]
                    direction = _esoap_direction(
                        grad=grad,
                        momentum=state["momentum"],
                        gram=state["gram"],
                        v_lead=state["v_lead"],
                        rank=r,
                        mu=group["mu"],
                        gram_beta=group["gram_beta"],
                        beta2=beta2,
                        nesterov=group["nesterov"],
                        ns_steps=group["ns_steps"],
                        eps=eps,
                    )
                    direction_p = direction.to(p.dtype)
                    if wd != 0.0 and group.get("cautious_wd", False):
                        # Cautious decoupled WD (ADR 0030, SPECULATIVE, default
                        # off): decay only where the step agrees in sign with
                        # the param, so it never fights the update. A/B-gated.
                        upd = direction_p.mul(lr)
                        cautious_decoupled_wd_(p, upd, lr, wd)
                    else:
                        if wd != 0.0:
                            p.mul_(1.0 - lr * wd)
                        p.add_(direction_p, alpha=-lr)
                else:  # adamw
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    beta1, beta2 = group["betas"]
                    _adamw_apply_(
                        p, grad, exp_avg=state["exp_avg"],
                        exp_avg_sq=state["exp_avg_sq"], step=state["step"],
                        lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=wd,
                        cautious_wd=group.get("cautious_wd", False))

        return loss
