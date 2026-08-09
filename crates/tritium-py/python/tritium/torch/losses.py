"""Preference-optimization losses (ADR 0037 Stage 1). Pure torch.

DPO per Rafailov et al. 2023 (arXiv:2305.18290), the `trl.DPOTrainer`
convention BLUT's pipelines use: sigmoid loss over the beta-scaled policy/
reference log-ratio difference on (chosen, rejected) pairs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["dpo_loss"]


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DPO loss over per-sequence summed log-probs (shape ``[batch]`` each).

    Returns ``(loss, chosen_rewards, rejected_rewards)`` — the mean loss plus
    the detached beta-scaled implicit rewards, matching trl's reporting so
    migrated pipelines keep their reward-margin metrics.

    ``label_smoothing`` implements the conservative-DPO variant (Eq. 3 of the
    cDPO note): ``(1-e)·-logσ(β·Δ) + e·-logσ(-β·Δ)``; ``0.0`` is exact DPO.
    """
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError(f"label_smoothing must be in [0, 0.5), got {label_smoothing}")
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    loss = (
        -F.logsigmoid(logits) * (1.0 - label_smoothing)
        - F.logsigmoid(-logits) * label_smoothing
    ).mean()
    chosen_rewards = (beta * (policy_chosen_logps - reference_chosen_logps)).detach()
    rejected_rewards = (beta * (policy_rejected_logps - reference_rejected_logps)).detach()
    return loss, chosen_rewards, rejected_rewards
