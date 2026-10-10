#!/usr/bin/env python3
"""Run the pinned five-minute SmolLM2 tutorial through an installed CPU wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

SMOLLM2_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOLLM2_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate_receipt(receipt: dict[str, Any], *, max_seconds: float) -> None:
    if receipt.get("schema") != "tritium.smollm2-five-minute.v2":
        raise ValueError("SmolLM2 tutorial emitted an unsupported receipt schema")
    if receipt.get("passed") is not True:
        raise ValueError("SmolLM2 tutorial did not pass")
    if receipt.get("model_id") != SMOLLM2_MODEL_ID:
        raise ValueError("tutorial receipt model identity differs from the pinned model")
    if receipt.get("source_revision") != SMOLLM2_REVISION:
        raise ValueError("tutorial receipt model revision differs from the pinned revision")
    if receipt.get("device") != "cpu":
        raise ValueError("tutorial receipt is not CPU evidence")
    elapsed = receipt.get("elapsed_seconds_excluding_download")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or not 0 <= elapsed < max_seconds
    ):
        raise ValueError("tutorial receipt exceeds its measured wall-time limit")
    coverage = receipt.get("coverage")
    selected_parameters = (
        coverage.get("selected_parameters") if isinstance(coverage, dict) else None
    )
    if (
        not isinstance(selected_parameters, int)
        or isinstance(selected_parameters, bool)
        or selected_parameters <= 0
    ):
        raise ValueError("tutorial receipt contains no PTQ-converted parameters")
    storage = receipt.get("storage")
    if not isinstance(storage, dict):
        raise ValueError("tutorial receipt has no physical storage accounting")
    dense_bytes = storage.get("selected_dense_bytes")
    packed_bytes = storage.get("compact_checkpoint_bytes")
    if (
        not isinstance(dense_bytes, int)
        or isinstance(dense_bytes, bool)
        or not isinstance(packed_bytes, int)
        or isinstance(packed_bytes, bool)
    ):
        raise ValueError("tutorial receipt byte accounting is invalid")
    if dense_bytes <= 0 or packed_bytes <= 0 or packed_bytes >= dense_bytes:
        raise ValueError("tutorial checkpoint did not reduce selected weight bytes")
    if not isinstance(receipt.get("onnx_artifact_id"), str) or not receipt["onnx_artifact_id"]:
        raise ValueError("tutorial receipt has no ONNX artifact identity")
    if receipt.get("onnx_graph_optimization_level") != "ORT_DISABLE_ALL":
        raise ValueError("tutorial ONNX runtime may have expanded packed weights")
    if receipt.get("onnx_parity_rtol") != 1e-4 or receipt.get("onnx_parity_atol") != 1e-4:
        raise ValueError("tutorial ONNX parity tolerances differ from the frozen gate")
    max_error = receipt.get("onnx_replay_max_abs_error")
    tolerance_ratio = receipt.get("onnx_replay_max_tolerance_ratio")
    if (
        not isinstance(max_error, (int, float))
        or isinstance(max_error, bool)
        or not math.isfinite(max_error)
        or max_error < 0
        or not isinstance(tolerance_ratio, (int, float))
        or isinstance(tolerance_ratio, bool)
        or not math.isfinite(tolerance_ratio)
        or not 0 <= tolerance_ratio <= 1
    ):
        raise ValueError("tutorial ONNX replay did not satisfy measured parity")
    optimizer_state_entries = receipt.get("qat_optimizer_state_entries")
    if (
        not isinstance(optimizer_state_entries, int)
        or isinstance(optimizer_state_entries, bool)
        or optimizer_state_entries <= 0
    ):
        raise ValueError("tutorial receipt shows no resumed QAT optimizer state")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.wheel.is_symlink() or not args.wheel.is_file():
        raise ValueError("candidate wheel must be an ordinary .whl file")
    wheel = args.wheel.resolve(strict=True)
    if wheel.suffix != ".whl":
        raise ValueError("candidate wheel must have a .whl filename")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        raise ValueError("source revision must be a full lowercase Git object ID")
    if not args.release or not args.run_id:
        raise ValueError("release and run ID must be non-empty")

    from tritium.torch import run_smollm2_release_demo
    from tritium.torch import tutorial
    from tritium.torch._installed_candidate import verify_installed_candidate

    version, module = verify_installed_candidate(
        wheel_artifact=wheel, source_revision=args.source_revision,
        release=args.release, executing_files=(Path(tutorial.__file__),),
    )
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("tutorial output directory must not already exist")

    max_seconds = 300.0
    receipt = run_smollm2_release_demo(
        args.output_dir,
        model_id=SMOLLM2_MODEL_ID,
        revision=SMOLLM2_REVISION,
        device="cpu",
        max_seconds=max_seconds,
    )
    validate_receipt(receipt, max_seconds=max_seconds)
    summary = {
        "run_id": args.run_id,
        "candidate_source_revision": args.source_revision,
        "candidate_release": args.release,
        "candidate_wheel": wheel.name,
        "candidate_wheel_sha256": sha256_file(wheel),
        "candidate_wheel_bytes": wheel.stat().st_size,
        "installed_distribution_version": version,
        "installed_module": str(module),
        "tutorial_receipt": str(args.output_dir / "receipt.json"),
        "tutorial_receipt_sha256": sha256_file(args.output_dir / "receipt.json"),
        "model_id": SMOLLM2_MODEL_ID,
        "model_revision": SMOLLM2_REVISION,
        "elapsed_seconds_excluding_download": receipt[
            "elapsed_seconds_excluding_download"
        ],
    }
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
