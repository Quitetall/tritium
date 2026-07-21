"""Authenticated Qwen language-plus-MTP ONNX bundle admission.

These APIs require digests from an independently trusted package manifest. They
never treat hashes calculated from candidate files as admission authority.
"""

from __future__ import annotations

from os import PathLike
from typing import Any

from . import _tritium


def verify_qwen35_bundle(
    language_path: str | PathLike[str],
    mtp_path: str | PathLike[str],
    weights_path: str | PathLike[str],
    *,
    language_blake3: str,
    mtp_blake3: str,
    weights_blake3: str,
    max_graph_bytes: int = 256 * 1024 * 1024,
    max_weights_bytes: int = 64 * 1024 * 1024 * 1024,
) -> Any:
    """Verify graph structure, shared external ranges, identity, and exact digests."""
    return _tritium.verify_qwen35_onnx_bundle(
        str(language_path),
        str(mtp_path),
        str(weights_path),
        language_blake3=language_blake3,
        mtp_blake3=mtp_blake3,
        weights_blake3=weights_blake3,
        max_graph_bytes=max_graph_bytes,
        max_weights_bytes=max_weights_bytes,
    )


def stage_qwen35_bundle(
    language_path: str | PathLike[str],
    mtp_path: str | PathLike[str],
    weights_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    language_blake3: str,
    mtp_blake3: str,
    weights_blake3: str,
    max_graph_bytes: int = 256 * 1024 * 1024,
    max_weights_bytes: int = 64 * 1024 * 1024 * 1024,
) -> Any:
    """Verify and atomically publish a durable canonical four-file bundle."""
    return _tritium.stage_qwen35_onnx_bundle(
        str(language_path),
        str(mtp_path),
        str(weights_path),
        str(output_dir),
        language_blake3=language_blake3,
        mtp_blake3=mtp_blake3,
        weights_blake3=weights_blake3,
        max_graph_bytes=max_graph_bytes,
        max_weights_bytes=max_weights_bytes,
    )


__all__ = ["stage_qwen35_bundle", "verify_qwen35_bundle"]
