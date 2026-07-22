from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "verify-cuda-training-receipt.py")
ReceiptError = MODULE["ReceiptError"]
validate = MODULE["validate"]


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def fixture(path: Path, artifact: Path) -> dict:
    value = {
        "schema": "tritium.cuda-training-qualification.v1",
        "source_revision": "a" * 40,
        "release": "1.1.0-rc.0",
        "run_id": "run-17",
        "started_at_utc": "2026-07-21T12:00:00Z",
        "duration_ms": 4000.0,
        "command": ["python", "hf_cuda_worker.py"],
        "artifact": {
            "kind": "python-wheel",
            "name": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        "machine": {
            "machine_id": "sha256:" + "b" * 64,
            "system": "Linux",
            "architecture": "x86_64",
        },
        "environment": {
            "python_version": "3.13.5",
            "torch_version": "2.11.0",
            "transformers_version": "5.5.3",
            "accelerate_version": "1.10.0",
            "cuda_runtime": "13.0",
            "cuda_driver": "610.43.03",
        },
        "device": {
            "index": 0,
            "uuid": "GPU-physical",
            "name": "NVIDIA GeForce RTX 4090",
            "compute_capability": "8.9",
            "total_memory_bytes": 25_000_000_000,
        },
        "workload": {
            "seed": 401,
            "mixed_precision": "fp16",
            "steps": 5,
            "batch_size": 1,
            "sequence_length": 8,
            "model_config_sha256": "c" * 64,
        },
        "measurements": {"elapsed_ms": 250.0, "steps_per_second": 20.0},
        "invariants": {
            "ternary_operator_host_transfers": 0,
            "ternary_operator_dtype": "torch.float16",
            "checkpoint_exact": True,
        },
        "result": "pass",
    }
    value["receipt_id"] = "sha256:" + hashlib.sha256(canonical(value)).hexdigest()
    path.write_bytes(canonical(value) + b"\n")
    return value


class VerifyCudaTrainingReceiptTests(unittest.TestCase):
    def test_accepts_bound_physical_fp16_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "receipt.json"
            artifact = Path(raw) / "candidate.whl"
            artifact.write_bytes(b"wheel")
            expected = fixture(path, artifact)
            self.assertEqual(
                validate(path, "a" * 40, "1.1.0-rc.0", artifact), expected
            )

    def test_rejects_revision_result_measurement_and_digest_drift(self):
        mutations = (
            ("source_revision", "b" * 40),
            ("result", "fail"),
            ("measurements", {"elapsed_ms": 250.0, "steps_per_second": 21.0}),
            (
                "invariants",
                {
                    "ternary_operator_host_transfers": 1,
                    "ternary_operator_dtype": "torch.float16",
                    "checkpoint_exact": True,
                },
            ),
        )
        for field, replacement in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                path = Path(raw) / "receipt.json"
                artifact = Path(raw) / "candidate.whl"
                artifact.write_bytes(b"wheel")
                value = fixture(path, artifact)
                value[field] = replacement
                path.write_bytes(canonical(value) + b"\n")
                with self.assertRaises(ReceiptError):
                    validate(path, "a" * 40, "1.1.0-rc.0", artifact)

    def test_rejects_unknown_fields_and_nonfinite_numbers(self):
        for field, replacement in (("unknown", True), ("duration_ms", float("nan"))):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                path = Path(raw) / "receipt.json"
                artifact = Path(raw) / "candidate.whl"
                artifact.write_bytes(b"wheel")
                value = fixture(path, artifact)
                value[field] = replacement
                unsigned = dict(value)
                unsigned.pop("receipt_id")
                value["receipt_id"] = "sha256:" + hashlib.sha256(
                    canonical(unsigned)
                ).hexdigest()
                path.write_bytes(canonical(value) + b"\n")
                with self.assertRaises(ReceiptError):
                    validate(path, "a" * 40, "1.1.0-rc.0", artifact)

    def test_rejects_qualified_artifact_byte_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "receipt.json"
            artifact = Path(raw) / "candidate.whl"
            artifact.write_bytes(b"wheel")
            fixture(path, artifact)
            artifact.write_bytes(b"changed")
            with self.assertRaises(ReceiptError):
                validate(path, "a" * 40, "1.1.0-rc.0", artifact)


if __name__ == "__main__":
    unittest.main()
