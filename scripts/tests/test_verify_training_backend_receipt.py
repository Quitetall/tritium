import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "verify-training-backend-receipt.py")
canonical = MODULE["canonical"]
validate = MODULE["validate"]
TrainingBackendReceiptError = MODULE["TrainingBackendReceiptError"]


def fixture(root: Path):
    manifest = json.loads((ROOT / "spec/training/v2/manifest.json").read_bytes())
    vectors = json.loads((ROOT / "spec/training/v2/vectors/v2.json").read_bytes())
    operations = [item["id"] for item in manifest["operations"]]
    artifacts = []
    bundles = []
    physical = {
        "cpu": "cpu:x86_64", "cuda": "cuda:0:GPU-1", "rocm": "rocm:0:GPU-2",
        "metal": "metal:Apple-M4", "wgpu": "vulkan:GPU-1",
        "wasi": "wasmtime:46.0.0:x86_64", "mcu": "stm32h7:board-1",
    }
    for ordinal, family in enumerate(MODULE["FAMILIES"]):
        prefix = MODULE["BACKEND_PREFIXES"][family]
        backend_id = prefix + ("device-0" if prefix.endswith(":") else "")
        cases = []
        for case in vectors["cases"]:
            success = case["expected"]["kind"] == "success"
            cases.append(
                {
                    "case_id": case["case_id"],
                    "receipt": {
                        "operation": case["operation"], "execution": case["execution"],
                        "dtype": "f32", "input_digest": f"{ordinal + 1:x}" * 64,
                        "output_digest": f"{ordinal + 2:x}" * 64,
                        "peak_resident_bytes": 4,
                        "scratch_bytes": 0, "host_transfers": 0,
                        "device_resident": True,
                    } if success else None,
                }
            )
        wire = {
            "schema_id": "tritium.training_receipts", "schema_version": 1,
            "backend_id": backend_id,
            "backend_build": f"tritium-{family}+source-git:" + "a" * 40,
            "physical_device": physical[family],
            "manifest_digest": MODULE["MANIFEST_BLAKE3"],
            "vector_digest": MODULE["VECTOR_BLAKE3"],
            "supported_operations": operations, "dtypes": ["f32", "u32", "bytes"],
            "limits": {"max_rank": 8, "max_elements": 1_000_000, "max_bytes": 4_000_000},
            "device_resident": True, "cases": cases,
        }
        path = root / f"{family}.training-receipts.json"
        path.write_text(json.dumps(wire, indent=2) + "\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact_id = f"training-{family}"
        blake3 = f"{ordinal + 3:x}" * 64
        artifacts.append(
            {
                "id": artifact_id, "kind": "training-receipt-bundle", "path": path.name,
                "identity": {"bytes": path.stat().st_size, "sha256": digest, "blake3": blake3},
            }
        )
        bundles.append(
            {
                "family": family,
                "artifact": {
                    "id": artifact_id, "kind": "training-receipt-bundle", "name": path.name,
                    "bytes": path.stat().st_size, "sha256": digest, "blake3": blake3,
                },
            }
        )
    candidate = root / "manifest.json"
    candidate.write_bytes(canonical({"artifacts": artifacts}))
    receipt = {
        "schema": MODULE["SCHEMA"], "result": "pass",
        "release": "1.1.0-rc.0", "source_revision": "a" * 40,
        "run_id": "training-backends-1",
        "candidate_manifest_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "manifest_sha256": MODULE["MANIFEST_SHA256"],
        "vectors_sha256": MODULE["VECTOR_SHA256"], "bundles": bundles,
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    receipt_path = root / "receipt.json"
    receipt_path.write_bytes(canonical(receipt))
    return candidate, receipt_path, receipt


class TrainingBackendReceiptTests(unittest.TestCase):
    def test_accepts_all_physical_backend_bundles(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, path, receipt = fixture(Path(raw))
            self.assertEqual(
                validate(path, "a" * 40, "1.1.0-rc.0", candidate, ROOT), receipt
            )

    def test_rejects_missing_backend(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, path, receipt = fixture(Path(raw))
            receipt["bundles"].pop()
            unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
            path.write_bytes(canonical(receipt))
            with self.assertRaisesRegex(TrainingBackendReceiptError, "seven"):
                validate(path, "a" * 40, "1.1.0-rc.0", candidate, ROOT)


if __name__ == "__main__":
    unittest.main()
