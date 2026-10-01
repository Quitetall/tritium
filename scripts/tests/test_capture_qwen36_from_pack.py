from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "capture-qwen36-from-pack.py"
SPEC = importlib.util.spec_from_file_location("capture_qwen36_from_pack", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CaptureQwenFromPackTests(unittest.TestCase):
    def test_activation_cache_identity_must_be_a_sha256_hex_digest(self):
        MODULE._validate_digest("a" * 64, "activation-cache digest")
        for invalid in ("a" * 63, "a" * 65, "g" * 64, "sha256:" + "a" * 64):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
                    MODULE._validate_digest(invalid, "activation-cache digest")

    def test_activation_cache_digest_is_required_by_cli(self):
        parser = MODULE._parser()
        with self.assertRaises(SystemExit) as raised:
            parser.parse_args([])
        self.assertEqual(raised.exception.code, 2)

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
