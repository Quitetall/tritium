# Ported verbatim from LamQuant (lamquant/ingredients/optimizers/sinksoaph.py) into
# tritium.torch by the LamQuant copyright holder; relicensed to this
# repository's Apache-2.0 terms. Original header retained below.
#
# SinkSOAPH — Gram-Sinkhorn direction + hyperball post-step, for PyTorch.
#
# Method provenance (both OSI-approved, GPL-3.0-compatible):
#   * SinkSOAP direction — KellerJordan/modded-nanogpt PR #298 (MIT):
#       Gram-eigenbasis rotation of the momentum, then a Sinkhorn-Knopp
#       row/column energy balance of the rotated update (no second-order
#       moment), rotated back.
#   * Hyperball post-step — marin-community/marin experiments/grug/moe
#       (Apache-2.0), `adamh.py::_scale_invariant_2d`: a scale-invariant,
#       norm-preserving sphere step. The applied delta has magnitude
#       ``lr * ||p||`` in the (unit) update direction, after which the
#       parameter is re-projected back onto the sphere of radius ``||p||``.
#
# "SinkSOAPH" = SinkSOAP direction with the hyperball post-step substituted
# for SinkSOAP's Muon-Frobenius scaling + NorMuon post-conditioner. Because
# the hyperball is the scale controller, neither the Muon-Frobenius rescale
# nor the NorMuon branch is used here.
#
# This is a single ``torch.optim.Optimizer`` that routes each param group by
# a ``method`` tag so the LamQuant SNN trainer can keep its single-optimizer
# contract (one ``.step()`` / ``.zero_grad()``; the WSDScheduler scales every
# group's ``lr``):
#   * ``method="sinksoaph"`` — Gram-Sinkhorn + hyperball. 2-D params only.
#   * ``method="adamw"``     — decoupled-weight-decay AdamW. Any shape.
#
# Single-GPU / single-process only — no torch.distributed sharding (the SNN
# is 57 K params; the upstream all_gather param-shard loop is unnecessary).

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["SinkSOAPH"]


# --------------------------------------------------------------------------- #
# Gram-Sinkhorn direction primitives                                          #
# --------------------------------------------------------------------------- #
def _gram_eigenbasis(C: Tensor, eps: float) -> Tensor:
    """Eigenvectors of a symmetric PSD Gram matrix, sorted by descending
    eigenvalue. ``C`` is symmetrised and Tikhonov-regularised before eigh."""
    C = C.float()
    C = 0.5 * (C + C.T)
    if eps > 0.0:
        C = C + eps * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
    evals, evecs = torch.linalg.eigh(C)
    # eigh returns ascending; flip to descending.
    return evecs.flip(-1)


def _sinkhorn_energy_balance(A: Tensor, steps: int, eps: float) -> Tensor:
    """Sinkhorn-Knopp balance of the elementwise energy ``A**2`` toward
    *uniform* row/column marginals (1/m, 1/n), then lift the diagonal
    scalings back onto ``A``.

    For ``A`` in R^{m x n}::

        B   = A**2 + eps
        r   = (1/m) / (B  @ c)     # row scale
        c   = (1/n) / (B.T @ r)    # col scale   (repeated `steps` times)
        A_bal = sqrt(r) * A * sqrt(c)

    No Frobenius/RMS rescaling is applied — the hyperball post-step owns the
    final magnitude.
    """
    assert A.ndim == 2
    out_dtype = A.dtype
    A = A.float()
    m, n = A.shape

    B = A.square()
    # Scale-RELATIVE smoothing. `eps` is a relative factor (not absolute): the
    # floor is `eps · mean(A²)`. A near-dead row/column (energy → 0) would
    # otherwise be amplified by ~√(1/eps_abs) and — because the hyperball
    # post-step renormalises the whole direction — that one row would hijack the
    # entire update. Tying the floor to the matrix's own mean energy bounds the
    # amplification relative to the matrix scale, while staying negligible for
    # well-conditioned A (so the balanced marginals are unchanged in the common
    # case). The +1e-30 guards an all-zero A (step 0, zero grad).
    smooth = eps * B.mean().clamp_min(1e-12) + 1e-30
    B = B + smooth
    target_r = torch.full((m,), 1.0 / m, device=A.device, dtype=torch.float32)
    target_c = torch.full((n,), 1.0 / n, device=A.device, dtype=torch.float32)
    r = torch.ones(m, device=A.device, dtype=torch.float32)
    c = torch.ones(n, device=A.device, dtype=torch.float32)

    for _ in range(steps):
        r = target_r / (B @ c + smooth)
        c = target_c / (B.T @ r + smooth)

    A_bal = r.sqrt()[:, None] * A * c.sqrt()[None, :]
    return A_bal.to(out_dtype)


def _sinksoaph_direction(
    grad: Tensor,
    momentum: Tensor,
    left_gram: Tensor,
    right_gram: Tensor,
    mu: float,
    gram_beta: float,
    nesterov: bool,
    sinkhorn_steps: int,
    eps: float,
    sinkhorn_eps: float,
) -> Tensor:
    """Compute the (unscaled) SinkSOAP update direction for one 2-D matrix.

    Mutates ``momentum``/``left_gram``/``right_gram`` in place (EMA state).
    Returns ``O = U @ A_bal @ V.T`` — the magnitude is intentionally NOT
    controlled here; the hyperball post-step sets the final scale.
    """
    G = grad.float()

    # A non-finite grad would permanently poison the Gram/momentum EMAs (the
    # eigenbasis never recovers); skip this matrix's update instead.
    if not torch.isfinite(G).all():
        return torch.zeros_like(G)

    # First-order momentum EMA + optional Nesterov-style (Muon-convention) blend.
    momentum.lerp_(G, 1.0 - mu)
    M = G.lerp(momentum, mu) if nesterov else momentum.clone()

    # EMA Gram matrices from the raw gradient.
    left_gram.lerp_(G @ G.T, 1.0 - gram_beta)
    right_gram.lerp_(G.T @ G, 1.0 - gram_beta)

    # Eigenbases, rotate momentum, Sinkhorn-balance, rotate back.
    U = _gram_eigenbasis(left_gram, eps)
    V = _gram_eigenbasis(right_gram, eps)
    A = U.T @ M @ V
    A_bal = _sinkhorn_energy_balance(
        A, steps=sinkhorn_steps, eps=sinkhorn_eps).float()
    return U @ A_bal @ V.T


def _hyperball_apply_(p: Tensor, direction: Tensor, lr: float, eps: float) -> None:
    """Scale-invariant, norm-preserving sphere step (in place on ``p``).

    Mirrors marin ``adamh.py::_scale_invariant_2d``:

        p_norm = ||p||
        new_p  = p - lr * direction * p_norm / max(||direction||, eps)
        new_p  = new_p / ||new_p|| * p_norm        # re-project to ||p|| sphere

    ``direction`` is a descent direction (subtracted). The parameter norm is
    preserved across the step, so no separate weight decay is applied to a
    hyperball group.
    """
    p_norm = p.norm()
    # Skip the sphere step for a (near-)zero parameter: re-projecting to a tiny
    # radius would amplify numerical noise. Unreachable in practice (hyperball
    # preserves ||p|| from init), but cheap to guard.
    if float(p_norm) < eps:
        return
    d = direction.float()
    d_norm = d.norm().clamp_min(eps)
    new_p = p.float() - lr * d * (p_norm / d_norm)
    new_p = new_p / new_p.norm().clamp_min(eps) * p_norm
    p.copy_(new_p.to(p.dtype))


# --------------------------------------------------------------------------- #
# AdamW (decoupled weight decay) — for the non-hyperball group                 #
# --------------------------------------------------------------------------- #
def _adamw_apply_(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> None:
    """Standard AdamW step (in place on ``p``). Matches the adamw A/B arm's
    betas=(0.9, 0.95) so the non-matrix params train identically across arms.
    """
    if weight_decay != 0.0:
        p.mul_(1.0 - lr * weight_decay)
    exp_avg.lerp_(grad, 1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bias1 = 1.0 - beta1 ** step
    bias2 = 1.0 - beta2 ** step
    denom = (exp_avg_sq.sqrt() / (bias2 ** 0.5)).add_(eps)
    step_size = lr / bias1
    p.addcdiv_(exp_avg, denom, value=-step_size)


# --------------------------------------------------------------------------- #
# Optimizer                                                                    #
# --------------------------------------------------------------------------- #
class SinkSOAPH(torch.optim.Optimizer):
    """Hybrid Gram-Sinkhorn-hyperball + AdamW optimizer.

    Param groups are routed by a ``method`` key:

      * ``"sinksoaph"`` — every parameter MUST be 2-D. State per param:
        ``momentum [m,n]``, ``left_gram [m,m]``, ``right_gram [n,n]``.
        ``weight_decay`` is ignored (hyperball preserves ``||p||``).
      * ``"adamw"`` — any shape. State: ``step``, ``exp_avg``, ``exp_avg_sq``.

    The WSDScheduler sets ``group["lr"]`` each epoch; both methods read it, so
    the matrix group's hyperball fraction and the AdamW lr share one schedule
    (clean controlled swap vs the all-AdamW baseline).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        *,
        mu: float = 0.95,
        gram_beta: float = 0.95,
        sinkhorn_steps: int = 10,
        nesterov: bool = True,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        sinkhorn_eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if sinkhorn_steps < 1:
            raise ValueError(f"sinkhorn_steps must be >= 1, got {sinkhorn_steps}")
        defaults = dict(
            lr=lr, method="adamw", mu=mu, gram_beta=gram_beta,
            sinkhorn_steps=sinkhorn_steps, nesterov=nesterov, betas=betas,
            sinkhorn_eps=sinkhorn_eps,
            eps=eps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            method = group.get("method", "adamw")
            if method not in ("sinksoaph", "adamw"):
                raise ValueError(
                    f"unknown param-group method {method!r} "
                    "(expected 'sinksoaph' or 'adamw')")
            if method == "sinksoaph":
                for p in group["params"]:
                    if p.ndim != 2:
                        raise ValueError(
                            "method='sinksoaph' only supports 2-D parameters; "
                            f"got shape {tuple(p.shape)}. Route non-2-D params "
                            "to an 'adamw' group.")

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

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SinkSOAPH does not support sparse grads")
                state = self.state[p]

                if method == "sinksoaph":
                    if len(state) == 0:
                        m, n = p.shape
                        # float32 momentum (matches grams) — no bf16 EMA drift.
                        state["momentum"] = torch.zeros_like(
                            p, dtype=torch.float32)
                        state["left_gram"] = torch.zeros(
                            (m, m), device=p.device, dtype=torch.float32)
                        state["right_gram"] = torch.zeros(
                            (n, n), device=p.device, dtype=torch.float32)
                    direction = _sinksoaph_direction(
                        grad=grad,
                        momentum=state["momentum"],
                        left_gram=state["left_gram"],
                        right_gram=state["right_gram"],
                        mu=group["mu"],
                        gram_beta=group["gram_beta"],
                        nesterov=group["nesterov"],
                        sinkhorn_steps=group["sinkhorn_steps"],
                        eps=eps,
                        sinkhorn_eps=group["sinkhorn_eps"],
                    )
                    _hyperball_apply_(p, direction, lr=lr, eps=eps)
                else:  # adamw
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    beta1, beta2 = group["betas"]
                    _adamw_apply_(
                        p, grad,
                        exp_avg=state["exp_avg"],
                        exp_avg_sq=state["exp_avg_sq"],
                        step=state["step"],
                        lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=group["weight_decay"],
                    )

        return loss
