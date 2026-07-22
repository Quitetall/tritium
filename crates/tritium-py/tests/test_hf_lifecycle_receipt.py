import json
import tempfile
import unittest
from pathlib import Path

from tritium.torch.hf_lifecycle import (
    run_hf_lifecycle,
    validate_hf_lifecycle_receipt,
)


class HuggingFaceLifecycleReceiptTests(unittest.TestCase):
    def _run(self, root: Path):
        wheel = root / "tritium_torch-1.1.0rc0-cp39-abi3-linux_x86_64.whl"
        wheel.write_bytes(b"exact candidate wheel")
        output = root / "hf-lifecycle"
        receipt = run_hf_lifecycle(
            output,
            wheel_artifact=wheel,
            source_revision="a" * 40,
            release="1.1.0-rc.0",
            run_id="hf-lifecycle-run-1",
        )
        path = output / "receipt.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        return output, wheel, receipt

    def test_receipt_survives_relocation_and_reloads_through_auto_model(self):
        with tempfile.TemporaryDirectory() as raw:
            output, wheel, receipt = self._run(Path(raw))
            relocated = Path(raw) / "relocated"
            output.rename(relocated)
            path = relocated / "receipt.json"

            self.assertEqual(
                validate_hf_lifecycle_receipt(
                    path,
                    expected_wheel=wheel,
                    expected_source_revision="a" * 40,
                    expected_release="1.1.0-rc.0",
                ),
                receipt,
            )
            self.assertEqual(receipt["schema"], "tritium.hf-lifecycle.v1")
            self.assertTrue(receipt["tied_before_save"])
            self.assertTrue(receipt["tied_after_reload"])
            self.assertTrue(receipt["safe_serialization"])
            self.assertEqual(receipt["checkpoint_dir"], "hf-checkpoint")

    def test_rejects_equal_size_checkpoint_tampering(self):
        with tempfile.TemporaryDirectory() as raw:
            output, _, _ = self._run(Path(raw))
            path = output / "receipt.json"
            checkpoint_file = next(
                item for item in (output / "hf-checkpoint").rglob("*") if item.is_file()
            )
            payload = bytearray(checkpoint_file.read_bytes())
            payload[-1] ^= 1
            checkpoint_file.write_bytes(payload)

            with self.assertRaisesRegex(ValueError, "checkpoint tree identity"):
                validate_hf_lifecycle_receipt(path)


if __name__ == "__main__":
    unittest.main()
