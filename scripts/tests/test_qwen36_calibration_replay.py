from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "qwen36_calibration_replay.py"
SPEC = importlib.util.spec_from_file_location("qwen36_calibration_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QwenCalibrationReplayTests(unittest.TestCase):
    def test_batches_follow_the_frozen_single_sequence_policy(self):
        token_bytes = b"\x07\x00\x00\x00\x09\x00\x00\x00"
        with patch.dict(
            MODULE._VERIFIER,
            {"TOKENS_PER_SEQUENCE": 2, "SEQUENCES_PER_PARTITION": 1},
        ):
            batches = list(
                MODULE.iter_capture_batches(
                    token_bytes,
                    lambda values, shape: (tuple(values), shape),
                )
            )
        self.assertEqual(
            batches,
            [
                {
                    "attention_mask": ((1, 1), (1, 2)),
                    "input_ids": ((7, 9), (1, 2)),
                }
            ],
        )

    def test_reader_reopens_only_the_receipt_bound_calibration_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = b"\x07\x00\x00\x00\x09\x00\x00\x00"
            (root / "stage7.u32le").write_bytes(tokens)
            manifest = {
                "tokens": {"path": "stage7.u32le"},
                "partitions": {
                    "calibration": {
                        "sequences": [
                            {
                                "token_offset": 0,
                                "token_count": 2,
                                "token_sha256": "sha256:"
                                + hashlib.sha256(tokens).hexdigest(),
                            }
                        ]
                    }
                },
            }
            manifest["pack_id"] = "sha256:" + hashlib.sha256(
                MODULE._VERIFIER["canonical"](manifest)
            ).hexdigest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            receipt = {
                "pack_id": manifest["pack_id"],
                "calibration": {
                    "ordered_token_sha256": "sha256:"
                    + hashlib.sha256(tokens).hexdigest()
                },
            }
            with patch.dict(
                MODULE._VERIFIER,
                {"TOKENS_PER_SEQUENCE": 2, "SEQUENCES_PER_PARTITION": 1},
            ):
                self.assertEqual(
                    MODULE._read_calibration_tokens(manifest_path, receipt),
                    tokens,
                )
                receipt["calibration"]["ordered_token_sha256"] = "sha256:" + "0" * 64
                with self.assertRaisesRegex(
                    ValueError,
                    "differs from verified pack receipt",
                ):
                    MODULE._read_calibration_tokens(manifest_path, receipt)

    def test_replay_factory_is_fresh_and_rejects_wrong_geometry(self):
        token_bytes = b"\x07\x00\x00\x00\x09\x00\x00\x00"
        receipt = {"receipt_id": "pack"}
        contract = {"capture_batch_sha256": "sha256:" + "a" * 64}
        with patch.dict(
            MODULE._VERIFIER,
            {
                "validate_receipt": lambda _path: receipt,
                "verify_pack": lambda *_args: receipt,
                "make_replay_contract": lambda *_args: contract,
            }
        ), patch.object(MODULE, "_read_calibration_tokens", return_value=token_bytes):
            replay = MODULE.Qwen36CalibrationReplay.open(
                Path("manifest"),
                Path("model"),
                Path("official-identity"),
                Path("pack-receipt"),
            )
        factory = replay.data_factory(lambda values, shape: (tuple(values), shape))
        with patch.dict(
            MODULE._VERIFIER,
            {"TOKENS_PER_SEQUENCE": 2, "SEQUENCES_PER_PARTITION": 1},
        ):
            first = list(factory(object()))
            second = list(factory(object()))
        self.assertEqual(first, second)
        self.assertIsNot(first[0], second[0])
        self.assertEqual(replay.token_stream_digest, contract["capture_batch_sha256"])
        with patch.dict(
            MODULE._VERIFIER,
            {"TOKENS_PER_SEQUENCE": 2, "SEQUENCES_PER_PARTITION": 2},
        ), self.assertRaisesRegex(ValueError, "frozen replay geometry"):
            list(
                MODULE.iter_capture_batches(
                    token_bytes, lambda values, shape: values
                )
            )

    def test_unverified_constructor_is_not_available(self):
        with self.assertRaisesRegex(TypeError, "use Qwen36CalibrationReplay.open"):
            MODULE.Qwen36CalibrationReplay(b"", {}, {})


if __name__ == "__main__":
    unittest.main()
