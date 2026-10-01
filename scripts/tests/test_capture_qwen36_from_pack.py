from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "capture-qwen36-from-pack.py"
SPEC = importlib.util.spec_from_file_location("capture_qwen36_from_pack", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CaptureQwenFromPackTests(unittest.TestCase):
    def _base_args(self):
        return [
            "--manifest", "manifest.json",
            "--model-dir", "model",
            "--official-source-identity", "identity.json",
            "--pack-receipt", "pack.json",
            "--replay-contract", "contract.json",
            "--work-dir", "work",
            "--evidence-dir", "evidence",
            "--capture-binding-output", "binding.json",
        ]

    def test_activation_cache_identity_must_be_a_sha256_hex_digest(self):
        MODULE._validate_digest("a" * 64, "activation-cache digest")
        for invalid in ("a" * 63, "a" * 65, "g" * 64, "sha256:" + "a" * 64):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
                    MODULE._validate_digest(invalid, "activation-cache digest")

    def test_preflight_does_not_require_capture_recipe(self):
        args = MODULE._parser().parse_args(self._base_args())
        self.assertIsNone(args.activation_cache_digest)
        self.assertIsNone(args.curvature)
        self.assertIsNone(args.damping)

    def test_execute_requires_stage7_receipt_before_model_loading(self):
        replay = type("Replay", (), {
            "receipt": {"receipt_id": "pack", "revision": MODULE.PINNED_REVISION},
            "contract": {"contract_id": "contract"},
            "token_stream_digest": "batch",
        })()
        replay_module = type("ReplayModule", (), {
            "open": staticmethod(lambda *_args: replay),
        })
        verifier = {"validate_replay_contract": lambda _path: replay.contract}
        stderr = io.StringIO()
        with (
            mock.patch.object(MODULE, "REPLAY", {
                "Qwen36CalibrationReplay": replay_module,
                "_VERIFIER": verifier,
            }),
            mock.patch.object(sys, "argv", [
                str(SCRIPT), *self._base_args(), "--execute",
                "--curvature", "input-hessian", "--damping", "0.01",
                "--activation-cache-digest", "a" * 64,
                "--offload-folder", "offload",
            ]),
            contextlib.redirect_stderr(stderr),
        ):
            with self.assertRaises(SystemExit) as raised:
                MODULE.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--stage7-qualification-receipt", stderr.getvalue())

    def test_execute_validates_freeze_against_candidate_and_checkout_revision(self):
        replay = type("Replay", (), {
            "receipt": {"receipt_id": "pack", "revision": MODULE.PINNED_REVISION},
            "contract": {"contract_id": "contract"},
            "token_stream_digest": "batch",
        })()
        replay_module = type("ReplayModule", (), {
            "open": staticmethod(lambda *_args: replay),
        })
        verifier = {"validate_replay_contract": lambda _path: replay.contract}
        revision = "b" * 40
        checked = []
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate.json"
            candidate.write_text(json.dumps({
                "schema": "tritium.release-candidate.v1",
                "release": "1.1.0-rc.9",
                "source_revision": revision,
            }))
            receipt = root / "stage7.json"
            receipt.write_text("{}\n")

            def validate(receipt_path, actual_revision, release, candidate_path):
                checked.append((receipt_path, actual_revision, release, candidate_path))
                return {"receipt_id": "sha256:" + "c" * 64}

            with (
                mock.patch.object(MODULE, "REPLAY", {
                    "Qwen36CalibrationReplay": replay_module,
                    "_VERIFIER": verifier,
                }),
                mock.patch.dict(MODULE.STAGE7, {"validate": validate}),
                mock.patch.object(
                    MODULE.subprocess,
                    "run",
                    return_value=type("Completed", (), {"stdout": revision + "\n"})(),
                ),
                mock.patch.object(sys, "argv", [
                    str(SCRIPT), *self._base_args(), "--execute",
                    "--release-candidate-manifest", str(candidate),
                    "--stage7-qualification-receipt", str(receipt),
                ]),
                contextlib.redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as raised:
                    MODULE.main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires --curvature", stderr.getvalue())
        self.assertEqual(checked, [(receipt, revision, "1.1.0-rc.9", candidate)])

    def test_execution_recipe_requires_frozen_cache_digest(self):
        args = MODULE._parser().parse_args(
            self._base_args() + ["--curvature", "input-hessian", "--damping", "0.01"]
        )
        with self.assertRaisesRegex(ValueError, "requires --activation-cache-digest"):
            MODULE._validate_capture_recipe(args)

    def test_execution_recipe_rejects_non_finite_damping(self):
        args = MODULE._parser().parse_args(
            self._base_args()
            + [
                "--curvature", "input-hessian",
                "--damping", "nan",
                "--activation-cache-digest", "a" * 64,
            ]
        )
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            MODULE._validate_capture_recipe(args)

    def test_max_memory_is_parsed_as_device_limits(self):
        self.assertEqual(
            MODULE._parse_max_memory(["0=20GiB", "cpu=48GiB"]),
            {0: "20GiB", "cpu": "48GiB"},
        )

    def test_max_memory_rejects_duplicates_and_malformed_limits(self):
        for values in (["0=20GiB", "0=21GiB"], ["0"], ["cpu="]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    MODULE._parse_max_memory(values)


if __name__ == "__main__":
    unittest.main()
