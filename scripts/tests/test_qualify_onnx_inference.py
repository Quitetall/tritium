import json
from pathlib import Path
import runpy
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from scripts.tests.test_verify_onnx_inference_receipt import fixture


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-onnx-inference.py")
assemble = MODULE["assemble"]
run_installed_worker = MODULE["run_installed_worker"]


class QualifyOnnxInferenceTests(unittest.TestCase):
    def test_installed_worker_isolated_offline_and_candidate_bound(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, _, source = fixture(root)
            wheel = root / "tritium.whl"
            onnx_bundle = root / "onnx-unpacked"
            native_bundle = root / "native-unpacked"
            onnx_bundle.mkdir()
            native_bundle.mkdir()
            target = root / "worker/trace.json"
            observed = {}

            def execute(command, **kwargs):
                observed.update(command=command, kwargs=kwargs)
                output = Path(command[command.index("--output") + 1])
                shutil.copyfile(root / source["trace"]["file"], output)
                return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

            with mock.patch.object(MODULE["subprocess"], "run", execute):
                run_installed_worker(
                    target,
                    candidate=candidate,
                    wheel=wheel,
                    python=Path("/usr/bin/python3"),
                    onnx_bundle=onnx_bundle,
                    native_bundle=native_bundle,
                    wheel_artifact_id="wheel",
                    onnx_artifact_id="onnx",
                    model_artifact_id="model",
                    profile="near-lossless-v1",
                    conversion_mode="refined",
                    source_revision="a" * 40,
                    release="1.1.0-rc.0",
                    run_id="onnx-run-1",
                )
            self.assertEqual(observed["kwargs"]["cwd"], target.parent)
            self.assertEqual(
                observed["kwargs"]["env"]["PATH"],
                str(Path("/usr/bin/python3").resolve().parent),
            )
            self.assertEqual(observed["kwargs"]["env"]["HF_HUB_OFFLINE"], "1")
            self.assertIn("tritium.torch.qualify_onnx", observed["command"])
            self.assertTrue(target.is_file())

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
