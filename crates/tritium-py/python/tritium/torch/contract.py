"""Trainer contract v1 reference implementation (blut CONTRACT.md).

Public, stdlib+torch-free on purpose (importable in any trainer process):
status emitters for both stdout channels, the heartbeat writer, and the
resume-key derivation. The authoritative contract text and golden fixtures
live in the blut repo; this module is the implementation trainers import.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

__all__ = [
    "CONTRACT_VERSION",
    "HEARTBEAT_INTERVAL",
    "announce",
    "emit",
    "emit_metric",
    "step",
    "eval_",
    "saved",
    "done",
    "failed",
    "heartbeat",
    "write_heartbeat_file",
    "resume_key",
]

CONTRACT_VERSION = "1"
# Must equal the Rust engine's resume::HEARTBEAT_INTERVAL_SECS.
HEARTBEAT_INTERVAL = 60

_KINDS = {"step", "eval", "saved", "done", "failed", "heartbeat"}


def _println(line: str) -> None:
    print(line, flush=True)


def announce() -> None:
    """Emit the contract announcement — call once, first, before any status."""
    _println(f"BLUT_CONTRACT {CONTRACT_VERSION}")


def emit(kind: str, **fields: Any) -> None:
    """Emit one control-channel line. Prefer the typed helpers below."""
    if kind not in _KINDS:
        raise ValueError(f"unknown status kind {kind!r} (CONTRACT.md §1)")
    _println(json.dumps({"kind": kind, **fields}))


def emit_metric(metrics: dict, *, kind: str = "epoch", phase: str | None = None) -> str:
    """Emit one ``BLUT_METRIC <json>`` observability line (numeric values
    only, bools excluded — non-numeric entries are dropped, matching the
    LamQuant emitter this freezes)."""
    payload = {
        k: v
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    payload["kind"] = kind
    if phase is not None:
        payload["phase"] = phase
    line = "BLUT_METRIC " + json.dumps(payload)
    _println(line)
    return line


def step(step: int, total: int, loss: float, lr: float, vram_mb: int) -> None:
    emit("step", step=int(step), total=int(total), loss=float(loss), lr=float(lr), vram_mb=int(vram_mb))


def eval_(step: int, eval_loss: float) -> None:
    emit("eval", step=int(step), eval_loss=float(eval_loss))


def saved(path: str | os.PathLike) -> None:
    emit("saved", path=str(path))


def done(final_loss: float, checkpoint_dir: str | os.PathLike) -> None:
    emit("done", final_loss=float(final_loss), checkpoint_dir=str(checkpoint_dir))


def failed(error: str) -> None:
    """Emit the terminal failure event. The caller must then exit non-zero."""
    emit("failed", error=str(error))


def heartbeat(phase: str | None = None, vram_mb: int | None = None) -> None:
    fields: dict[str, Any] = {}
    if phase is not None:
        fields["phase"] = phase
    if vram_mb is not None:
        fields["vram_mb"] = int(vram_mb)
    emit("heartbeat", **fields)


def write_heartbeat_file(state_json: str | os.PathLike, state: dict | None = None) -> None:
    """Durable liveness (CONTRACT.md §3): atomically publish ``state.json``
    with a fresh ``heartbeat_unix``. Call at least every HEARTBEAT_INTERVAL
    seconds; the Rust engine — never the trainer — judges staleness."""
    path = Path(state_json)
    payload = dict(state or {})
    payload["heartbeat_unix"] = int(time.time())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    os.replace(tmp, path)


def _canonical_json(obj: Any) -> str:
    """The PINNED canonical-JSON dialect (CONTRACT.md §5): sorted keys,
    ``,``/``:`` separators, ASCII-escaped non-ASCII (``\\uXXXX``), floats via
    Python ``repr`` shortest round-trip, NaN/Inf rejected. Any other
    implementation (the Rust engine included) must byte-match THIS output —
    note ``1`` and ``1.0`` canonicalize differently by design; configs are
    single-sourced so type-stable."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def resume_key(config: dict, data_digest: str) -> str:
    """CONTRACT.md §5, pinned composition::

        config_hash = blake3(canonical_json(config)).hex
        key         = blake3(config_hash + "\\x1f" + data_digest
                             + "\\x1f" + CONTRACT_VERSION).hex

    (0x1f unit separators; all pieces UTF-8.) Requires the ``blake3`` package —
    the contract pins the algorithm, so there is deliberately no silent
    fallback to another hash."""
    try:
        from blake3 import blake3
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "resume_key requires the 'blake3' package (pip install blake3); "
            "the contract pins BLAKE3 and no fallback hash is permitted"
        ) from e
    config_hash = blake3(_canonical_json(config).encode()).hexdigest()
    material = f"{config_hash}\x1f{data_digest}\x1f{CONTRACT_VERSION}".encode()
    return blake3(material).hexdigest()


def _self_check() -> int:  # pragma: no cover - manual utility
    """`python -m tritium.torch.contract` emits a demo stream (for eyeballing
    against blut's golden fixture)."""
    announce()
    heartbeat(phase="model-load")
    step(1, 100, 2.34, 2e-4, 8123)
    saved("ckpt/step_1.pt")
    eval_(1, 2.31)
    done(2.34, "ckpt")
    return 0


if __name__ == "__main__":
    sys.exit(_self_check())
