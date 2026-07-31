"""Production SALT V2 orchestration primitives.

This module deliberately distinguishes a sealed rate-free master campaign from
the final deployable model returned by the future high-level ``quantize`` API.
"""

from __future__ import annotations

from os import PathLike
from typing import Union

from ._tritium import (
    Qwen36PtqMasterReceipt,
    Qwen36PtqPackageReceipt,
    reconcile_qwen36_ptq_masters as _reconcile_qwen36_ptq_masters,
)

try:
    from ._tritium import (
        reconcile_qwen36_ptq_packages as _reconcile_qwen36_ptq_packages,
    )
except ImportError:  # non-unix wheels omit the native symbol (unix-only staging)
    _reconcile_qwen36_ptq_packages = None

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


def reconcile_qwen36_ptq_packages(
    model_dir: Pathish,
    *,
    revision: str,
    work_dir: Pathish,
    evidence_dir: Pathish,
    output_dir: Pathish,
    compact_max_bytes: int,
    compact_max_resident_bytes: int,
    near_lossless_max_bytes: int,
    near_lossless_max_resident_bytes: int,
    packing: str = "b3",
    max_evidence_bytes: int = 64 * 1024 * 1024,
) -> Qwen36PtqPackageReceipt:
    """Resume PTQ and atomically publish two admitted matrix packages.

    The output directory must not exist. Tritium stages both exact SALT V2
    profiles, the exact preserved BF16 safetensors companion, and their receipt
    manifest beside the destination. It syncs and strictly reopens them before
    publishing the complete directory with one rename. Schema-v3 bundles include
    bounded, content-bound Hugging Face language configuration and tokenizer
    assets. Qwen3.6 runtime wiring remains a separate release gate.
    """

    if _reconcile_qwen36_ptq_packages is None:
        raise RuntimeError(
            "reconcile_qwen36_ptq_packages requires a unix platform; this wheel "
            "was built without the native symbol"
        )
    return _reconcile_qwen36_ptq_packages(
        str(model_dir),
        revision,
        str(work_dir),
        str(evidence_dir),
        str(output_dir),
        compact_max_bytes=compact_max_bytes,
        compact_max_resident_bytes=compact_max_resident_bytes,
        near_lossless_max_bytes=near_lossless_max_bytes,
        near_lossless_max_resident_bytes=near_lossless_max_resident_bytes,
        packing=packing,
        max_evidence_bytes=max_evidence_bytes,
    )


__all__ = [
    "Qwen36PtqMasterReceipt",
    "Qwen36PtqPackageReceipt",
    "reconcile_qwen36_ptq_masters",
    "reconcile_qwen36_ptq_packages",
]
