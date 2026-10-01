from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "verify-qwen36-calibration-pack.py"
SPEC = importlib.util.spec_from_file_location("qwen36_calibration_pack", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QwenCalibrationPackTests(unittest.TestCase):
    def test_tokenizer_fingerprint_matches_canonical_asset_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contents = {
                "config.json": b'{"text_config":{"vocab_size":248320}}',
                "merges.txt": b"merge fixture\n",
                "tokenizer.json": b"tokenizer fixture\n",
                "tokenizer_config.json": b"config fixture\n",
                "vocab.json": b"vocab fixture\n",
            }
            files = []
            for name, payload in contents.items():
                (root / name).write_bytes(payload)
                files.append(
                    {
                        "name": name,
                        "size": len(payload),
                        "algorithm": "sha256",
                        "digest": hashlib.sha256(payload).hexdigest(),
                    }
                )
            identity = {
                "repository": MODULE.REPOSITORY,
                "revision": MODULE.REVISION,
                "result": "pass",
                "files": files,
            }

            self.assertEqual(
                MODULE._tokenizer_digest(root, identity),
                "sha256:8fa674b21219452bf3a0d819e5367e3a8ec43820b70a446e4d9fe72f8e143f5e",
            )

    def test_tokenizer_fingerprint_rejects_unpinned_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field in ("repository", "revision", "result"):
                identity = {
                    "repository": MODULE.REPOSITORY,
                    "revision": MODULE.REVISION,
                    "result": "pass",
                    "files": [],
                }
                identity[field] = "untrusted"
                with self.subTest(field=field):
                    with self.assertRaisesRegex(
                        MODULE.CalibrationPackError, "official source identity"
                    ):
                        MODULE._tokenizer_digest(root, identity)

    def test_pack_receipt_binds_the_exact_calibration_window(self):
        dataset_layout = {
            "dataset": {
                "revision": "f" * 40,
                "config": "default",
                "data_dir": None,
                "split": "train",
                "text_field": "text",
                "sequences": 1,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            pack_dir = root / "pack"
            model_dir.mkdir()
            pack_dir.mkdir()
            contents = {
                "config.json": b'{"text_config":{"vocab_size":248320}}',
                "merges.txt": b"merge fixture\n",
                "tokenizer.json": b"tokenizer fixture\n",
                "tokenizer_config.json": b"config fixture\n",
                "vocab.json": b"vocab fixture\n",
            }
            files = []
            for name, payload in contents.items():
                (model_dir / name).write_bytes(payload)
                files.append(
                    {
                        "name": name,
                        "size": len(payload),
                        "algorithm": "sha256",
                        "digest": hashlib.sha256(payload).hexdigest(),
                    }
                )
            identity = {
                "repository": MODULE.REPOSITORY,
                "revision": MODULE.REVISION,
                "result": "pass",
                "receipt_id": "sha256:" + "1" * 64,
                "source_model_id": MODULE.OFFICIAL_IDENTITY_MODULE[
                    "SOURCE_MODEL_ID"
                ],
                "files": files,
            }
            payload = (1).to_bytes(4, "little") + (2).to_bytes(4, "little")
            (pack_dir / "stage7.u32le").write_bytes(payload)
            sequence = {
                "dataset_repo_id": "dataset",
                "dataset_revision": "f" * 40,
                "dataset_config": "default",
                "dataset_data_dir": None,
                "dataset_split": "train",
                "source_rows": [
                    {
                        "row_index": 0,
                        "text_field": "text",
                        "content_sha256": "3" * 64,
                    }
                ],
                "token_offset": 0,
                "token_count": 2,
                "token_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
            sequence["id"] = "sha256:" + hashlib.sha256(
                MODULE.canonical(sequence)
            ).hexdigest()
            tokenizer_digest = MODULE._tokenizer_digest(model_dir, identity)
            manifest = {
                "schema": MODULE.PACK_SCHEMA,
                "tokenizer_digest": tokenizer_digest,
                "tokenizer_vocab_size": MODULE.VOCAB_SIZE,
                "token_encoding": "u32le",
                "tokens": {
                    "path": "stage7.u32le",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "partitions": {
                    "calibration": {"sampling_seed": 7, "sequences": [sequence]}
                },
            }
            manifest["pack_id"] = "sha256:" + hashlib.sha256(
                MODULE.canonical(manifest)
            ).hexdigest()
            manifest_path = pack_dir / "manifest.json"
            manifest_path.write_bytes(MODULE.canonical(manifest))

            with patch.dict(
                MODULE.OFFICIAL_IDENTITY_MODULE,
                {"validate_identity_receipt": lambda _: identity},
            ), patch.multiple(
                MODULE,
                PARTITIONS=("calibration",),
                SEQUENCES_PER_PARTITION=1,
                TOKENS_PER_SEQUENCE=2,
                TOKENIZER_FILES=(
                    "merges.txt",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                ),
                TOKEN_PAYLOAD_BYTES=8,
                DATASETS=dataset_layout,
            ):
                receipt = MODULE.verify_pack(
                    manifest_path, model_dir, root / "identity.json"
                )

            self.assertEqual(receipt["result"], "pass")
            self.assertEqual(receipt["pack_id"], manifest["pack_id"])
            self.assertEqual(receipt["calibration"]["token_count"], 2)
            self.assertEqual(
                receipt["calibration"]["ordered_token_sha256"],
                "sha256:" + hashlib.sha256(payload).hexdigest(),
            )
            receipt_path = root / "pack-receipt.json"
            receipt_path.write_bytes(MODULE.canonical(receipt))
            with patch.multiple(
                MODULE,
                SEQUENCES_PER_PARTITION=1,
                TOKENS_PER_SEQUENCE=2,
                TOKEN_PAYLOAD_BYTES=8,
                DATASETS=dataset_layout,
            ):
                self.assertEqual(
                    MODULE.validate_receipt(receipt_path)["receipt_id"],
                    receipt["receipt_id"],
                )
                receipt["pack_id"] = "sha256:" + "4" * 64
                receipt_path.write_bytes(MODULE.canonical(receipt))
                with self.assertRaisesRegex(
                    MODULE.CalibrationPackError, "receipt ID"
                ):
                    MODULE.validate_receipt(receipt_path)


if __name__ == "__main__":
    unittest.main()
