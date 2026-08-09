# Ported verbatim from LamQuant (ingredients/optimizers/cautious_wd.py) into
# tritium.torch by the LamQuant copyright holder; relicensed to this
# repository's Apache-2.0 terms.
"""Cautious decoupled weight decay (ADR 0030) — shared optimizer kernel.

SPECULATIVE, flag-gated default-off. The same per-entry fold is used by SOAP
and ESOAP (Adam + esoap paths); kept here once so the three call sites cannot
drift. `soap_optimizer.py` is a vendored upstream file, so it imports this
neutral helper rather than carrying its own copy.
"""
from __future__ import annotations

import torch


def cautious_decoupled_wd_(p: torch.Tensor, update: torch.Tensor,
                           lr: float, weight_decay: float) -> None:
    """Apply `p -= update`, folding decoupled WD (lr*wd*p) into `update` only
    on entries where it agrees in sign with the step (update*p > 0) — so decay
    never fights the update. `update` must already be the lr-scaled step. Both
    `p` and `update` are modified in place.
    """
    mask = (update * p) > 0
    update.add_(p * mask, alpha=lr * weight_decay)
    p.add_(update, alpha=-1.0)
