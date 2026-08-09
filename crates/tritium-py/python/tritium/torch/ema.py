"""EMA-as-artifact (ADR 0037 Stage 1): exponential moving average of weights
with the LamQuant conventions — torch ``AveragedModel`` with a decay
multi-avg fn, explicit non-parameter buffer syncing, and envelope
integration so the EMA weights can BE the shipped checkpoint.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

__all__ = ["Ema"]


class Ema:
    """Weight EMA wrapper.

    - ``update(model)`` after each optimizer step.
    - ``sync_buffers(model, names)`` hand-syncs non-parameter buffers the
      average does not track (LamQuant's CDF-breakpoint pattern: fitted state
      refit mid-training must be copied into the EMA twin explicitly).
    - ``state_dict()`` / ``load_state_dict()`` round-trip for the envelope's
      ``ema`` key; ``averaged_state_dict()`` is what ships when the EMA
      weights are the promoted artifact.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0,1), got {decay}")
        self.decay = decay
        self._avg = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))

    def update(self, model: torch.nn.Module) -> None:
        self._avg.update_parameters(model)

    def sync_buffers(self, model: torch.nn.Module, names: Iterable[str]) -> None:
        """Copy named buffers from the live model into the EMA twin verbatim
        (no averaging — fitted state is replaced, not blended)."""
        live = dict(model.named_buffers())
        twin = dict(self._avg.module.named_buffers())
        for name in names:
            if name not in live or name not in twin:
                raise KeyError(f"buffer {name!r} not present on both models")
            twin[name].copy_(live[name])

    @property
    def module(self) -> torch.nn.Module:
        return self._avg.module

    def averaged_state_dict(self) -> dict:
        return self._avg.module.state_dict()

    def state_dict(self) -> dict:
        return {"decay": self.decay, "avg": self._avg.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if state["decay"] != self.decay:
            raise ValueError(
                f"EMA decay mismatch: checkpoint {state['decay']} vs configured {self.decay}"
            )
        self._avg.load_state_dict(state["avg"])
