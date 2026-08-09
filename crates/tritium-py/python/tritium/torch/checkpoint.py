"""Checkpoint envelope v1 (CONTRACT.md §4) — save/load with RNG state, atomic
publish, alias-tolerant reads, and the post-save reload smoke check.

Canonical keys: model, opt, config, step, rng (+ optional ema, sched,
manifest_ref). Readers accept the v1 aliases LamQuant already writes
(state_dict/optimizer/rng_state/training_config_hash) and preserve extra keys.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Callable

import torch

__all__ = [
    "capture_rng",
    "restore_rng",
    "save_envelope",
    "load_envelope",
    "ENVELOPE_ALIASES",
]

ENVELOPE_ALIASES = {
    "model": ("state_dict",),
    "opt": ("optimizer",),
    "rng": ("rng_state",),
    "config": ("training_config_hash",),
}


def capture_rng() -> dict[str, Any]:
    """Snapshot python/numpy/torch (+cuda when present) RNG state."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    """Restore a :func:`capture_rng` snapshot. Missing sub-states are skipped
    (a CPU resume of a CUDA checkpoint restores what applies)."""
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(s, dtype=torch.uint8) for s in state["cuda"]]
        )


def save_envelope(
    path: str | os.PathLike,
    *,
    model: dict,
    opt: dict,
    config: dict,
    step: int,
    rng: dict | None = None,
    ema: dict | None = None,
    sched: dict | None = None,
    manifest_ref: str | None = None,
    extra: dict | None = None,
    smoke_check: Callable[[dict], None] | None = None,
) -> Path:
    """Atomically write a v1 envelope: tmp + fsync + ``os.replace``, rotating
    any existing checkpoint to ``.prev`` first so a crash mid-publish never
    destroys the last good one. ``rng=None`` captures the live RNG state.

    ``smoke_check`` receives the RELOADED payload after publish (default: a
    structural check that every canonical key round-tripped); pass a callback
    that runs a forward pass for the full LamQuant-style check.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model,
        "opt": opt,
        "config": config,
        "step": int(step),
        "rng": capture_rng() if rng is None else rng,
    }
    if ema is not None:
        payload["ema"] = ema
    if sched is not None:
        payload["sched"] = sched
    if manifest_ref is not None:
        payload["manifest_ref"] = manifest_ref
    if extra:
        overlap = payload.keys() & extra.keys()
        if overlap:
            raise ValueError(f"extra keys collide with envelope keys: {sorted(overlap)}")
        payload.update(extra)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        prev = path.with_suffix(path.suffix + ".prev")
        os.replace(path, prev)
    os.replace(tmp, path)

    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    if smoke_check is not None:
        smoke_check(reloaded)
    else:
        missing = [k for k in ("model", "opt", "config", "step", "rng") if k not in reloaded]
        if missing:
            raise RuntimeError(f"post-save smoke check: envelope missing {missing} after reload")
    return path


def load_envelope(path: str | os.PathLike, map_location: str = "cpu") -> dict[str, Any]:
    """Load an envelope, normalizing v1 aliases to canonical keys (aliases are
    kept alongside so alias-era callers keep working). Raises if any canonical
    key is unsatisfiable."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    for canonical, aliases in ENVELOPE_ALIASES.items():
        if canonical not in payload:
            for alias in aliases:
                if alias in payload:
                    payload[canonical] = payload[alias]
                    break
    missing = [k for k in ("model", "opt", "config", "step") if k not in payload]
    if missing:
        raise ValueError(f"not a v1 envelope (missing {missing}): {path}")
    return payload
