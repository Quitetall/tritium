import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "qwen36_ptq_status", ROOT / "scripts" / "qwen36-ptq-status.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Qwen36PtqStatusTests(unittest.TestCase):
    def test_idle_snapshot_is_non_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = MODULE.inspect(Path(directory))
        self.assertEqual(snapshot["schema"], "tritium.qwen36-ptq-status.v1")
        self.assertEqual(snapshot["status"], "idle")
        self.assertIsNone(snapshot["staged_record"])
        self.assertEqual(snapshot["published_master_count"], 0)

    def test_newest_temp_and_published_state_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "record.tmp.999999.1.0"
            old.write_bytes(b"old")
            os.utime(old, (1, 1))
            nested = root / "nested" / "tensor-work" / ".tmp"
            nested.mkdir(parents=True)
            current = nested / f"record.tmp.{os.getpid()}.2.0"
            current.write_bytes(b"current")
            objects = root / "nested" / "objects"
            objects.mkdir(parents=True)
            (objects / "master.s2kf").write_bytes(b"master")
            snapshot = MODULE.inspect(root)
        self.assertEqual(snapshot["status"], "running")
        self.assertEqual(
            snapshot["staged_record"]["path"],
            "nested/tensor-work/.tmp/" + current.name,
        )
        self.assertEqual(snapshot["staged_record"]["bytes"], 7)
        self.assertTrue(snapshot["staged_record"]["pid_alive"])
        self.assertEqual(snapshot["published_master_count"], 1)

    def test_seal_is_reported_complete_even_without_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.complete.tq36c").write_bytes(b"seal")
            snapshot = MODULE.inspect(root)
        self.assertEqual(snapshot["status"], "complete")
        self.assertEqual(snapshot["seal_count"], 1)

    def test_dead_owner_is_reported_stalled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "record.tmp.4000000000.1.0").write_bytes(b"orphan")
            snapshot = MODULE.inspect(root)
        self.assertEqual(snapshot["status"], "stalled")
        self.assertFalse(snapshot["staged_record"]["pid_alive"])

    def test_json_cli_output_is_parseable(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts" / "qwen36-ptq-status.py"),
                    "--work-dir",
                    str(directory),
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            output = result.stdout
        self.assertEqual(json.loads(output)["status"], "idle")

    def test_rate_eta_is_bounded_by_sampled_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / f"record.tmp.{os.getpid()}.2.0"
            current.write_bytes(b"x" * 100)

            def grow_record(_seconds):
                current.write_bytes(b"x" * 200)

            with mock.patch.object(MODULE.time, "sleep", side_effect=grow_record):
                snapshot = MODULE.inspect(root, sample_seconds=10, target_bytes=300)

        self.assertEqual(snapshot["target_bytes"], 300)
        self.assertEqual(snapshot["bytes_per_second"], 10.0)
        self.assertEqual(snapshot["estimated_seconds_remaining"], 10.0)

    def test_campaign_descriptor_reports_full_expected_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign_dir = root / "master-campaigns"
            staging = campaign_dir / ".tmp"
            staging.mkdir(parents=True)
            current = staging / f"record.tmp.{os.getpid()}.2.0"
            current.write_bytes(b"x")

            metadata = bytearray(420)
            metadata[314:318] = (1).to_bytes(4, "little")
            metadata[318] = ord("x")
            metadata[319:323] = (1).to_bytes(4, "little")
            metadata[-40:-32] = (10).to_bytes(8, "little")
            catalog = (
                b"TSQ36SC\x00"
                + (1).to_bytes(2, "little")
                + b"\x00\x00"
                + (1).to_bytes(4, "little")
                + len(metadata).to_bytes(4, "little")
                + metadata
            )
            descriptor = bytearray(313)
            descriptor[:8] = b"TSQ36CP\x00"
            descriptor[309:313] = len(catalog).to_bytes(4, "little")
            (campaign_dir / "campaign.tq36p").write_bytes(
                bytes(descriptor) + catalog + b"\x00" * 32
            )

            snapshot = MODULE.inspect(root)

        self.assertEqual(snapshot["campaign_tensor_count"], 1)
        self.assertEqual(snapshot["campaign_expected_payload_bytes"], 10)
        self.assertEqual(snapshot["campaign_expected_record_bytes"], 607)
        self.assertIsNone(snapshot["campaign_estimated_seconds_remaining"])

    def test_negative_target_bytes_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(MODULE.StatusError):
                MODULE.inspect(Path(directory), target_bytes=-1)


if __name__ == "__main__":
    unittest.main()
