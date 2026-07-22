import hashlib
from pathlib import Path
import runpy
import tempfile
import unittest


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "verify-onnx-inference-receipt.py"
)
canonical = MODULE["canonical"]
validate = MODULE["validate"]
OnnxReceiptError = MODULE["OnnxReceiptError"]


def fixture(root: Path):
    artifacts = []
    records = {}
    for artifact_id, kind, name, payload in (
        ("wheel", "python-wheel", "tritium.whl", b"wheel"),
        ("model", "model-bundle", "qwen.salt", b"model"),
        ("onnx", "onnx-bundle", "qwen-onnx.tar.zst", b"onnx"),
    ):
        path = root / name
        path.write_bytes(payload)
        record = {
            "id": artifact_id, "kind": kind, "name": name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        records[artifact_id] = record
        artifacts.append(
            {
                "id": artifact_id, "kind": kind, "path": name,
                "identity": {"bytes": record["bytes"], "sha256": record["sha256"]},
            }
        )
    candidate = root / "manifest.json"
    candidate.write_bytes(canonical({"artifacts": artifacts}))
    trace = root / "ort-trace.json"
    trace.write_bytes(b'{"custom_ops_executed":true}\n')
    receipt = {
        "schema": MODULE["SCHEMA"], "result": "pass",
        "release": "1.1.0-rc.0", "source_revision": "a" * 40,
        "run_id": "onnx-run-1",
        "candidate_manifest_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "wheel": records["wheel"], "artifact": records["onnx"],
        "model_artifact_id": "model",
        "environment": {
            "python": "3.13", "torch": "2.11.0", "onnx": "1.22.0",
            "onnxruntime": "1.27.0", "tritium_distribution": "1.1.0rc0",
            "repository_absent": True, "compiler_absent": True,
        },
        "model": {
            "model_id": MODULE["MODEL_ID"], "revision": MODULE["MODEL_REVISION"],
            "scope": "language+mtp", "profile": "near-lossless-v1",
            "conversion_mode": "refined", "package_id": "sha256:" + "1" * 64,
        },
        "runtime": {
            "provider": "CPUExecutionProvider", "physical_cpu": True,
            "bundle_schema": "tritium-qwen35-onnx-bundle-v2",
            "sequence_mode": "dynamic-cache-v1", "standard_opset": 21,
            "tritium_opsets": [1, 2], "custom_domain_executed": True,
            "external_data_authenticated": True, "dense_weight_initializers": 0,
            "persistent_dense_shadows": 0,
        },
        "parity": {
            "prompt_cases": 2, "cached_decode_cases": 2, "generation_cases": 2,
            "mtp_cases": 2, "max_abs_error": 0.0001, "tolerance": 0.001,
            "tokens_exact": True, "states_exact": True,
            "generation_exact": True, "mtp_exact": True,
        },
        "faults": {
            "graph_corruption_rejected": True, "weights_corruption_rejected": True,
            "path_traversal_rejected": True, "unknown_operator_rejected": True,
            "trainable_export_rejected": True, "trainable_import_rejected": True,
        },
        "trace": {
            "file": trace.name, "bytes": trace.stat().st_size,
            "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        },
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    path = root / "receipt.json"
    path.write_bytes(canonical(receipt))
    return candidate, path, receipt


class OnnxInferenceReceiptTests(unittest.TestCase):
    def test_accepts_real_whole_model_ort_evidence(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, path, receipt = fixture(Path(raw))
            self.assertEqual(
                validate(path, "a" * 40, "1.1.0-rc.0", candidate), receipt
            )

    def test_rejects_structural_custom_op_substitution(self):
        with tempfile.TemporaryDirectory() as raw:
            candidate, path, receipt = fixture(Path(raw))
            receipt["runtime"]["custom_domain_executed"] = False
            unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
            receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
            path.write_bytes(canonical(receipt))
            with self.assertRaisesRegex(OnnxReceiptError, "real ORT"):
                validate(path, "a" * 40, "1.1.0-rc.0", candidate)


if __name__ == "__main__":
    unittest.main()
