import json
from pathlib import Path
import runpy
import tempfile
import unittest

from scripts.tests.test_verify_onnx_inference_receipt import fixture


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-onnx-inference.py")
assemble = MODULE["assemble"]


class QualifyOnnxInferenceTests(unittest.TestCase):
    def test_aggregates_raw_execution_and_self_validates(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, _, source = fixture(root)
            trace = root / source["trace"]["file"]
            receipt = assemble(
                root / "qualification",
                candidate=candidate,
                trace_path=trace,
                wheel_artifact_id="wheel",
                onnx_artifact_id="onnx",
                model_artifact_id="model",
                source_revision="a" * 40,
                release="1.1.0-rc.0",
                run_id="onnx-run-1",
            )
            self.assertTrue(receipt["runtime"]["custom_domain_executed"])
            self.assertEqual(receipt["parity"]["generation_cases"], 2)
            self.assertTrue((root / "qualification/onnx-execution.json").is_file())

    def test_rejects_missing_custom_operator_execution(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, _, source = fixture(root)
            trace_path = root / source["trace"]["file"]
            trace = json.loads(trace_path.read_bytes())
            trace["session"]["custom_operator_calls"].pop()
            trace_path.write_bytes(MODULE["canonical"](trace) + b"\n")
            with self.assertRaisesRegex(
                MODULE["VERIFIER"]["OnnxReceiptError"], "call inventory"
            ):
                assemble(
                    root / "qualification",
                    candidate=candidate,
                    trace_path=trace_path,
                    wheel_artifact_id="wheel",
                    onnx_artifact_id="onnx",
                    model_artifact_id="model",
                    source_revision="a" * 40,
                    release="1.1.0-rc.0",
                    run_id="onnx-run-1",
                )

    def test_rejects_symlinked_raw_trace(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, _, source = fixture(root)
            trace = root / source["trace"]["file"]
            link = root / "linked-trace.json"
            link.symlink_to(trace)
            with self.assertRaisesRegex(MODULE["QualificationError"], "ordinary"):
                assemble(
                    root / "qualification",
                    candidate=candidate,
                    trace_path=link,
                    wheel_artifact_id="wheel",
                    onnx_artifact_id="onnx",
                    model_artifact_id="model",
                    source_revision="a" * 40,
                    release="1.1.0-rc.0",
                    run_id="onnx-run-1",
                )


if __name__ == "__main__":
    unittest.main()
