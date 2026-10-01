#!/usr/bin/env python3
"""Capture pinned Qwen3.6 S2KF records from the verified calibration pack.

Without ``--execute`` this performs source/pack/replay preflight only. Loading
the 27B checkpoint and executing the capture requires the explicit flag.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import runpy
import sys
from typing import Any


REPLAY = runpy.run_path(Path(__file__).with_name("qwen36_calibration_replay.py"))
PINNED_REVISION = REPLAY["_VERIFIER"]["REVISION"]


def _parse_max_memory(values: list[str]) -> dict[Any, str]:
    result: dict[Any, str] = {}
    for value in values:
        device, separator, limit = value.partition("=")
        if not separator or not device or not limit:
            raise ValueError("--max-memory must use DEVICE=LIMIT syntax")
        key: Any = int(device) if device.isdecimal() else device
        if key in result:
            raise ValueError(f"--max-memory repeats device {device!r}")
        result[key] = limit
    return result


def _validate_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"{label} must be 64 hexadecimal characters")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--official-source-identity", required=True, type=Path)
    parser.add_argument("--pack-receipt", required=True, type=Path)
    parser.add_argument("--replay-contract", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--capture-binding-output", required=True, type=Path)
    parser.add_argument("--declared-revision", default=PINNED_REVISION)
    parser.add_argument(
        "--activation-cache-digest",
        required=True,
        help="64-hex digest from the frozen activation-cache recipe; never inferred",
    )
    parser.add_argument(
        "--curvature",
        required=True,
        choices=("input-hessian", "guided-fisher", "forward-kl-kronecker"),
    )
    parser.add_argument("--damping", required=True, type=float)
    parser.add_argument(
        "--guided-loss-reduction",
        choices=("sum", "mean-attention-mask", "mean-valid-causal-labels"),
    )
    parser.add_argument("--max-shared-modules", type=int, default=8)
    parser.add_argument("--max-evidence-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-batch-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-capture-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-objective-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory", action="append", default=[])
    parser.add_argument("--offload-folder", type=Path)
    parser.add_argument("--input-device")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="load the local 27B checkpoint and begin/resume empirical capture",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        replay_module = REPLAY["Qwen36CalibrationReplay"]
        replay = replay_module.open(
            args.manifest,
            args.model_dir,
            args.official_source_identity,
            args.pack_receipt,
        )
        saved_contract = REPLAY["_VERIFIER"]["validate_replay_contract"](
            args.replay_contract
        )
        if saved_contract != replay.contract:
            raise ValueError("saved replay contract differs from verified pack inputs")
        max_memory = _parse_max_memory(args.max_memory)
        if args.declared_revision != replay.receipt["revision"]:
            raise ValueError("declared revision differs from verified Qwen source")
        _validate_digest(args.activation_cache_digest, "activation-cache digest")
        if args.curvature == "guided-fisher" and args.guided_loss_reduction is None:
            raise ValueError("guided-fisher requires --guided-loss-reduction")
        if args.curvature != "guided-fisher" and args.guided_loss_reduction is not None:
            raise ValueError("--guided-loss-reduction is valid only for guided-fisher")
        if args.max_shared_modules <= 0 or min(
            args.max_evidence_bytes,
            args.max_batch_bytes,
            args.max_capture_bytes,
            args.max_objective_bytes,
        ) <= 0:
            raise ValueError("capture limits must be positive")
        if not math.isfinite(args.damping) or args.damping < 0:
            raise ValueError("damping must be finite and nonnegative")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(
        "PREFLIGHT PASS "
        f"pack={replay.receipt['receipt_id']} "
        f"replay={replay.contract['contract_id']} "
        f"batch={replay.token_stream_digest}"
    )
    if not args.execute:
        print("NOT STARTED: model load and capture require --execute")
        return 0

    if not args.offload_folder:
        parser.error("--execute requires an explicit --offload-folder")
    try:
        import torch
        from transformers import AutoModelForImageTextToText
        from tritium.torch.qwen36 import attach_qwen36_mtp, capture_qwen36_components

        if not torch.cuda.is_available() and not args.allow_cpu:
            raise RuntimeError("no CUDA device detected; pass --allow-cpu to opt in")
        args.offload_folder.mkdir(parents=True, exist_ok=True)
        model = AutoModelForImageTextToText.from_pretrained(
            str(args.model_dir),
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
            max_memory=max_memory or None,
            offload_folder=str(args.offload_folder),
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        attach_qwen36_mtp(model, args.model_dir)
        embedding_weight = model.model.language_model.embed_tokens.weight
        input_device = args.input_device or str(embedding_weight.device)
        if input_device == "meta":
            raise RuntimeError(
                "input embedding is disk-offloaded; specify --input-device explicitly"
            )
        tensor_factory = REPLAY["torch_int64_tensor_factory"](input_device)
        native_receipt = capture_qwen36_components(
            model,
            replay.data_factory(tensor_factory),
            model_dir=args.model_dir,
            declared_revision=args.declared_revision,
            work_dir=args.work_dir,
            evidence_dir=args.evidence_dir,
            curvature=args.curvature,
            activation_cache_digest=args.activation_cache_digest,
            token_stream_digest=replay.token_stream_digest,
            damping=args.damping,
            guided_loss_reduction=args.guided_loss_reduction,
            max_evidence_bytes=args.max_evidence_bytes,
            max_batch_bytes=args.max_batch_bytes,
            max_capture_bytes=args.max_capture_bytes,
            max_objective_bytes=args.max_objective_bytes,
            max_shared_modules=args.max_shared_modules,
        )
        binding = replay.capture_binding(
            native_receipt,
            max_evidence_bytes=args.max_evidence_bytes,
        )
        REPLAY["write_capture_binding"](args.capture_binding_output, binding)
    except Exception as error:
        print(f"qwen36 capture failed: {error}", file=sys.stderr)
        return 1
    print(
        f"CAPTURE RECEIPT {binding['binding_id']} "
        f"evidence_set={binding['evidence_set_digest']} records={binding['records']}"
    )
    print("Next: run scripts/verify-qwen36-capture-binding.py with the same paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
