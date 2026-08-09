"""LR schedules for tritium.torch — WSD/WSD-infinity (ported verbatim from
LamQuant blut_core scheduler ingredient by its copyright holder) plus factory
helpers over torch built-ins.

Relocated verbatim from ``lamquant/student/train_joint.py`` — it was defined in
the codec trainer and cross-imported by the SNN trainers
(``pretrain_ssl_tueg`` / ``train_4state_controller``), the worst coupling in the
training tree. It now lives in its own home; the SNN trainers import it here
instead of reaching into a sibling trainer.
"""
from __future__ import annotations

import math


class WSDScheduler:
    """Warmup-Stable-Decay scheduler for continual training.

    Three phases (user direction 2026-05-21, cosine-then-WSD):
      1. Warmup (cosine ramp): LR 0 → peak via half-cosine, smoother
         than linear at the start and end of the warmup window.
      2. Stable (constant): LR = peak, shippable any time.
      3. Decay (cosine): LR peak → min_lr over decay_epochs.

    Pass ``warmup_kind="linear"`` to restore the legacy linear ramp.

    Infinite mode (decay_frac=0):
      Stable phase runs forever. The model trains at peak LR
      indefinitely — every checkpoint is shippable. When you
      want to finalize, call trigger_decay(n_epochs) to start
      the cosine cooldown manually.

    For continual training: resume from any stable-phase checkpoint.
    No re-warming disruption since stable phase is at full LR.
    """

    def __init__(self, optimizer, total_epochs: int, peak_lr: float,
                 warmup_frac: float = 0.05, decay_frac: float = 0.10,
                 min_lr: float = 1e-6, warmup_kind: str = "cosine"):
        if warmup_kind not in ("cosine", "linear"):
            raise ValueError(f"warmup_kind must be 'cosine' or 'linear', got {warmup_kind!r}")
        self.warmup_kind = warmup_kind
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_epochs = max(1, int(total_epochs * warmup_frac))

        # decay_frac=0 → infinite stable phase (no automatic decay)
        if decay_frac <= 0:
            self.decay_epochs = 0
            self.decay_start = total_epochs + 1  # never reached
            self._infinite = True
        else:
            self.decay_epochs = max(1, int(total_epochs * decay_frac))
            self.decay_start = total_epochs - self.decay_epochs
            self._infinite = False

        self.stable_start = self.warmup_epochs
        self._last_lr = [peak_lr] * len(optimizer.param_groups)
        # Preserve each param group's relative LR at construction time (e.g.
        # an encoder group scaled to a fraction of peak_lr via
        # encoder_lr_scale before this scheduler is built) — step() applies
        # the schedule's absolute lr scaled by this per-group ratio instead
        # of overwriting every group with the identical value, which used to
        # silently discard any differential per-group rate from epoch 2
        # onward. A caller that never differentiates groups gets scale=1.0
        # everywhere (all groups start at peak_lr), so this is a no-op for
        # every existing single-rate use.
        self._group_scale = [
            (pg['lr'] / peak_lr) if peak_lr else 1.0
            for pg in optimizer.param_groups
        ]
        # Sanity-check the implicit contract this inference depends on: no
        # group should start ABOVE peak_lr unless a caller deliberately
        # wants a super-peak group. In practice this catches the far more
        # likely mistake -- passing a peak_lr that doesn't match how the
        # optimizer was actually constructed -- which would otherwise run
        # the entire schedule at a silently wrong rate with no signal.
        if peak_lr and any(s > 1.0 + 1e-6 for s in self._group_scale):
            print(
                f"[WSD] warning: a param group's initial lr exceeds "
                f"peak_lr={peak_lr:.2e} (scales={[round(s, 3) for s in self._group_scale]}) "
                f"-- if this wasn't a deliberate super-peak group, peak_lr likely "
                f"doesn't match how the optimizer was constructed"
            )
        self.epoch = 0
        self._decay_triggered = False
        self._decay_trigger_epoch = None

    def trigger_decay(self, n_epochs: int = 40):
        """Manually trigger the cosine decay phase.

        Call this when you want to finalize the model. The decay
        starts at the current epoch and runs for n_epochs.
        """
        self._decay_triggered = True
        self._decay_trigger_epoch = self.epoch
        self.decay_epochs = n_epochs
        self.decay_start = self.epoch
        print(f"[WSD] Decay triggered at epoch {self.epoch}, "
              f"will decay over {n_epochs} epochs to lr={self.min_lr:.1e}")

    def step(self, epoch=None):
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1

        if self.epoch <= self.warmup_epochs:
            # progress in [0, 1] across warmup window
            p = self.epoch / max(self.warmup_epochs, 1)
            if self.warmup_kind == "cosine":
                # Half-cosine ramp: 0 → peak via 0.5*(1 - cos(pi*p)).
                # Same start/end values as linear but smoother derivative
                # at both edges (avoids the optimizer-state shock that
                # a sharp linear corner can trigger right at peak LR).
                lr = self.peak_lr * 0.5 * (1.0 - math.cos(math.pi * p))
            else:  # "linear"
                lr = self.peak_lr * p
        elif not self._decay_triggered and self._infinite:
            # Infinite stable — runs forever at peak LR
            lr = self.peak_lr
        elif self.epoch < self.decay_start:
            lr = self.peak_lr
        else:
            # Cosine decay
            progress = (self.epoch - self.decay_start) / max(self.decay_epochs, 1)
            progress = min(progress, 1.0)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        self._last_lr = []
        for pg, scale in zip(self.optimizer.param_groups, self._group_scale):
            pg['lr'] = lr * scale
            self._last_lr.append(pg['lr'])

    def get_last_lr(self):
        return self._last_lr

    @property
    def phase(self) -> str:
        if self.epoch <= self.warmup_epochs:
            return 'warmup'
        elif not self._decay_triggered and self._infinite:
            return 'stable∞'
        elif self.epoch < self.decay_start:
            return 'stable'
        else:
            return 'decay'


def create_schedule(name: str, optimizer, **cfg):
    """Build an LR schedule by registry name.

    - ``wsd`` — :class:`WSDScheduler` (``decay_frac=0`` gives the infinite
      stable phase; call ``trigger_decay(n)`` to finalize).
    - ``cosine`` — torch ``CosineAnnealingLR``.
    - ``sgdr`` — torch ``CosineAnnealingWarmRestarts`` (SGDR).
    - ``schedule_free`` — no schedule object: pair with the
      ``schedule_free_adamw`` optimizer from :mod:`tritium.torch.optim` and
      drive its train()/eval() mode switches instead.
    """
    import torch as _torch

    name = name.lower().replace("-", "_")
    if name == "wsd":
        return WSDScheduler(optimizer, **cfg)
    if name == "cosine":
        return _torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **cfg)
    if name == "sgdr":
        return _torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **cfg)
    if name == "schedule_free":
        raise ValueError(
            "schedule_free is an optimizer property, not a schedule: use "
            "tritium.torch.optim.create_optimizer('schedule_free_adamw', ...) "
            "and call .train()/.eval() on it"
        )
    raise ValueError(f"unknown schedule {name!r} (wsd, cosine, sgdr)")


def wsd_state_dict(sched: WSDScheduler) -> dict:
    """Serializable WSD state for the checkpoint envelope's ``sched`` key —
    WSD-infinity is stateful (``trigger_decay``), so resume must carry this."""
    return {
        "epoch": sched.epoch,
        "decay_triggered": sched._decay_triggered,
        "decay_trigger_epoch": sched._decay_trigger_epoch,
        "decay_epochs": sched.decay_epochs,
        "decay_start": sched.decay_start,
        "last_lr": list(sched._last_lr),
    }


def wsd_load_state_dict(sched: WSDScheduler, state: dict) -> None:
    """Restore :func:`wsd_state_dict` — exact continuation incl. a triggered
    decay in flight."""
    sched.epoch = state["epoch"]
    sched._decay_triggered = state["decay_triggered"]
    sched._decay_trigger_epoch = state["decay_trigger_epoch"]
    sched.decay_epochs = state["decay_epochs"]
    sched.decay_start = state["decay_start"]
    sched._last_lr = list(state["last_lr"])
