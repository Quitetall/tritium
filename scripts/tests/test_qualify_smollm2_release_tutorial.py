import importlib.util
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/qualify-smollm2-release-tutorial.py"
SPEC = importlib.util.spec_from_file_location("qualify_smollm2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def passing_receipt():
    return {
        "schema": "tritium.smollm2-five-minute.v2",
        "passed": True,
        "model_id": MODULE.SMOLLM2_MODEL_ID,
        "source_revision": MODULE.SMOLLM2_REVISION,
        "device": "cpu",
        "elapsed_seconds_excluding_download": 120.0,
        "coverage": {"selected_parameters": 100},
        "storage": {"selected_dense_bytes": 1000, "compact_checkpoint_bytes": 500},
        "onnx_artifact_id": "sha256:onnx",
        "onnx_graph_optimization_level": "ORT_DISABLE_ALL",
        "onnx_parity_rtol": 1e-4,
        "onnx_parity_atol": 1e-4,
        "onnx_replay_max_abs_error": 5e-5,
        "onnx_replay_max_tolerance_ratio": 0.5,
        "qat_optimizer_state_entries": 1,
    }


class SmolLM2ReleaseTutorialTests(unittest.TestCase):
    def test_receipt_requires_pinned_model_and_all_tutorial_phases(self):
        MODULE.validate_receipt(passing_receipt(), max_seconds=300.0)

    def test_receipt_rejects_synthetic_or_wrong_revision_model(self):
        receipt = passing_receipt()
        receipt["model_id"] = "synthetic-tied-model"
        with self.assertRaisesRegex(ValueError, "model identity"):
            MODULE.validate_receipt(receipt, max_seconds=300.0)

        receipt = passing_receipt()
        receipt["source_revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "revision"):
            MODULE.validate_receipt(receipt, max_seconds=300.0)

    def test_receipt_rejects_slow_or_uncompressed_result(self):
        receipt = passing_receipt()
        receipt["elapsed_seconds_excluding_download"] = 300.0
        with self.assertRaisesRegex(ValueError, "wall-time"):
            MODULE.validate_receipt(receipt, max_seconds=300.0)

        receipt = passing_receipt()
        receipt["storage"]["compact_checkpoint_bytes"] = 1000
        with self.assertRaisesRegex(ValueError, "did not reduce"):
            MODULE.validate_receipt(receipt, max_seconds=300.0)

    def test_receipt_rejects_boolean_values_in_numeric_claims(self):
        for field, mutate in (
            ("elapsed_seconds_excluding_download", lambda value: value.update(
                elapsed_seconds_excluding_download=True
            )),
            ("coverage.selected_parameters", lambda value: value["coverage"].update(
                selected_parameters=True
            )),
            ("storage.selected_dense_bytes", lambda value: value["storage"].update(
                selected_dense_bytes=True
            )),
            ("qat_optimizer_state_entries", lambda value: value.update(
                qat_optimizer_state_entries=True
            )),
        ):
            with self.subTest(field=field):
                receipt = passing_receipt()
                mutate(receipt)
                with self.assertRaises(ValueError):
                    MODULE.validate_receipt(receipt, max_seconds=300.0)

    def test_receipt_rejects_unsafe_onnx_memory_or_parity_claim(self):
        receipt = passing_receipt()
        receipt["onnx_graph_optimization_level"] = "ORT_ENABLE_ALL"
        with self.assertRaisesRegex(ValueError, "expanded packed weights"):
            MODULE.validate_receipt(receipt, max_seconds=300.0)

        receipt = passing_receipt()
        receipt["onnx_replay_max_tolerance_ratio"] = 1.01
        with self.assertRaisesRegex(ValueError, "measured parity"):
            MODULE.validate_receipt(receipt, max_seconds=300.0)

    def test_wheel_workflow_runs_pinned_smollm2_from_candidate_cpu_wheel(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text()
        self.assertIn('  smollm2-cpu-tutorial:\n', workflow)
        start = workflow.index("  smollm2-cpu-tutorial:")
        match = re.search(r"(?m)^  [a-z0-9_-]+:\s*$", workflow[start + 3 :])
        self.assertIsNotNone(match)
        assert match is not None
        job = workflow[start : start + 3 + match.start()]
        self.assertIn("needs: wheels", job)
        self.assertIn('python-version: "3.13"', job)
        self.assertIn("torch==2.11.0", job)
        self.assertIn("transformers==5.5.3", job)
        self.assertIn("onnxruntime==1.27.0", job)
        self.assertIn('          OMP_NUM_THREADS: "1"', job)
        self.assertIn('          MKL_NUM_THREADS: "1"', job)
        self.assertIn("--wheel dist/*.whl", job)
        self.assertIn('device="cpu"', SCRIPT.read_text())
        self.assertIn("SMOLLM2_MODEL_ID", SCRIPT.read_text())
        self.assertIn("SMOLLM2_REVISION", SCRIPT.read_text())
        self.assertIn("evidence/smollm2-cpu/**", job)


if __name__ == "__main__":
    unittest.main()
