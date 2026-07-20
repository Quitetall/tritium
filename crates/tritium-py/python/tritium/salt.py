"""Production SALT V2 orchestration primitives.

This module deliberately distinguishes a sealed rate-free master campaign from
the final deployable model returned by the future high-level ``quantize`` API.
"""

from __future__ import annotations

from os import PathLike
from typing import Union

from ._tritium import (
    Qwen36PtqMasterReceipt,
    reconcile_qwen36_ptq_masters as _reconcile_qwen36_ptq_masters,
)

Pathish = Union[str, PathLike[str]]


def reconcile_qwen36_ptq_masters(
    model_dir: Pathish,
    *,
    revision: str,
    work_dir: Pathish,
    evidence_dir: Pathish,
    packing: str = "b3",
    max_evidence_bytes: int = 64 * 1024 * 1024,
) -> Qwen36PtqMasterReceipt:
    """Resume and seal the pinned Qwen3.6 rate-free PTQ master campaign.

    ``evidence_dir`` must be a complete canonical ``S2KF`` namespace. The call
    performs source admission, exact-BF16 preservation, one-matrix-at-a-time
    fitting, atomic content-addressed installation, strict resume, and final
    campaign sealing in Rust without moving model weights through Python lists.

    The returned receipt is not a deployable model. Physical profile allocation,
    package assembly, evaluation, and export remain later governed stages.
    """

    return _reconcile_qwen36_ptq_masters(
        str(model_dir),
        revision,
        str(work_dir),
        str(evidence_dir),
        packing=packing,
        max_evidence_bytes=max_evidence_bytes,
    )


__all__ = ["Qwen36PtqMasterReceipt", "reconcile_qwen36_ptq_masters"]
