#!/usr/bin/env python3
"""Independently reopen and verify pack-linked Qwen3.6 S2KF capture evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy


REPLAY = runpy.run_path(Path(__file__).with_name("qwen36_calibration_replay.py"))
PINNED_REVISION = REPLAY["_VERIFIER"]["REVISION"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--official-source-identity", required=True, type=Path)
    parser.add_argument("--pack-receipt", required=True, type=Path)
    parser.add_argument("--replay-contract", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--declared-revision", default=PINNED_REVISION)
    args = parser.parse_args()
    try:
        receipt = REPLAY["reopen_capture_binding"](
            args.binding,
            manifest_path=args.manifest,
            official_source_identity_path=args.official_source_identity,
            pack_receipt_path=args.pack_receipt,
            replay_contract_path=args.replay_contract,
            model_dir=args.model_dir,
            work_dir=args.work_dir,
            evidence_dir=args.evidence_dir,
            declared_revision=args.declared_revision,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(
        f"PASS {receipt['binding_id']} "
        f"records={receipt['records']} evidence_set={receipt['evidence_set_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
