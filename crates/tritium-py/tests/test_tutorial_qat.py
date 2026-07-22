import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tritium.torch.tutorial_qat import (
    run_installed_qat_tutorial,
    validate_tutorial_receipt,
)


def _rehash(receipt):
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(payload).hexdigest()


class InstalledQatTutorialReceiptTests(unittest.TestCase):
    def test_result_strictly_reopens(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "tutorial"
            receipt = run_installed_qat_tutorial(output, device_name="cpu")
            path = output / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")

            self.assertEqual(
                validate_tutorial_receipt(path, expected_device="cpu"), receipt
            )
            self.assertGreater(receipt["checkpoint_model_bytes"], 0)
            self.assertTrue(receipt["checkpoint_model_sha256"].startswith("sha256:"))
            self.assertGreater(receipt["checkpoint_optimizer_bytes"], 0)
            self.assertTrue(
                receipt["checkpoint_optimizer_sha256"].startswith("sha256:")
            )
            self.assertEqual(receipt["optimizer_state_entries"], 1)
            self.assertEqual(receipt["resume_steps"], 1)
            self.assertTrue(
                (output / "latent-checkpoint/model.safetensors").is_file()
            )
            self.assertTrue((output / "latent-checkpoint/optimizer.pt").is_file())

    def test_rejects_equal_size_checkpoint_tampering(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "tutorial"
            receipt = run_installed_qat_tutorial(output, device_name="cpu")
            path = output / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            optimizer = output / "latent-checkpoint/optimizer.pt"
            payload = bytearray(optimizer.read_bytes())
            payload[-1] ^= 1
            optimizer.write_bytes(payload)

            with self.assertRaisesRegex(ValueError, "checkpoint_optimizer_sha256"):
                validate_tutorial_receipt(path, expected_device="cpu")

    def test_rejects_symlinked_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "tutorial"
            receipt = run_installed_qat_tutorial(output, device_name="cpu")
            path = output / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            checkpoint = output / "latent-checkpoint"
            external = output.parent / "external-checkpoint"
            checkpoint.rename(external)
            checkpoint.symlink_to(external, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "ordinary directory"):
                validate_tutorial_receipt(path, expected_device="cpu")

    def test_rejects_rehashed_claim_drift(self):
        mutations = [
            ("converted_parameters", 2, "coverage"),
            ("aliases", ["embed.weight"], "aliases"),
            ("algorithm_id", "tritium.salt-ste", "estimator"),
            ("planes", 1, "plane"),
            ("device", "cuda:0", "device"),
            ("optimizer_state_entries", 0, "optimizer state"),
            ("resume_steps", 0, "resume"),
        ]
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                output = Path(raw) / "tutorial"
                receipt = run_installed_qat_tutorial(output, device_name="cpu")
                receipt[field] = value
                _rehash(receipt)
                path = output / "receipt.json"
                path.write_text(json.dumps(receipt), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, message):
                    validate_tutorial_receipt(path, expected_device="cpu")


if __name__ == "__main__":
    unittest.main()
