import json
from pathlib import Path
import runpy
import tempfile
import unittest

from scripts.tests.test_verify_estimator_refinement_receipt import (
    ablation_trace,
    fixture,
)


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-baseline-ablation.py")
assemble = MODULE["assemble"]


class QualifyBaselineAblationTests(unittest.TestCase):
    def test_rejects_symlinked_raw_trace(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common, _ = fixture(root)
            trace = ablation_trace(root, common)
            link = root / "linked-trace.json"
            link.symlink_to(trace)
            with self.assertRaisesRegex(MODULE["QualificationError"], "ordinary"):
                assemble(
                    root / "qualification",
                    candidate=candidate,
                    trace_path=link,
                    model_artifact_id="s34",
                    source_revision="a" * 40,
                    release="1.1.0-rc.0",
                    run_id="ablation-1",
                )

    def test_aggregates_samples_and_self_validates(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common, _ = fixture(root)
            trace = ablation_trace(root, common)
            receipt = assemble(
                root / "qualification",
                candidate=candidate,
                trace_path=trace,
                model_artifact_id="s34",
                source_revision="a" * 40,
                release="1.1.0-rc.0",
                run_id="ablation-1",
            )
            self.assertEqual(receipt["anchor_artifact"]["id"], "s34")
            self.assertEqual(len(receipt["baselines"]), 7)
            self.assertEqual(receipt["baselines"][1]["runtime_ms"], 2.0)
            self.assertTrue(
                (root / "qualification/baseline-ablation-execution.json").is_file()
            )

    def test_rejects_mixed_physical_devices(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common, _ = fixture(root)
            trace_path = ablation_trace(root, common)
            trace = json.loads(trace_path.read_bytes())
            trace["baselines"][3]["physical_device"] = "cuda:1:GPU-2"
            trace_path.write_bytes(MODULE["canonical"](trace) + b"\n")
            with self.assertRaisesRegex(
                MODULE["VERIFIER"]["EstimatorRefinementError"], "frozen device"
            ):
                assemble(
                    root / "qualification",
                    candidate=candidate,
                    trace_path=trace_path,
                    model_artifact_id="s34",
                    source_revision="a" * 40,
                    release="1.1.0-rc.0",
                    run_id="ablation-1",
                )

    def test_rejects_different_parameter_inventory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common, _ = fixture(root)
            trace_path = ablation_trace(root, common)
            trace = json.loads(trace_path.read_bytes())
            trace["baselines"][1]["parameter_count"] = 401
            trace_path.write_bytes(MODULE["canonical"](trace) + b"\n")
            with self.assertRaisesRegex(
                MODULE["VERIFIER"]["EstimatorRefinementError"],
                "parameter inventories",
            ):
                assemble(
                    root / "qualification",
                    candidate=candidate,
                    trace_path=trace_path,
                    model_artifact_id="s34",
                    source_revision="a" * 40,
                    release="1.1.0-rc.0",
                    run_id="ablation-1",
                )

    def test_rejects_recipe_body_that_does_not_match_identity(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common, _ = fixture(root)
            trace_path = ablation_trace(root, common)
            trace = json.loads(trace_path.read_bytes())
            trace["baselines"][0]["recipe"]["arguments"]["target_bytes"] = 99
            trace_path.write_bytes(MODULE["canonical"](trace) + b"\n")
            with self.assertRaisesRegex(
                MODULE["VERIFIER"]["EstimatorRefinementError"], "recipe identity"
            ):
                assemble(
                    root / "qualification",
                    candidate=candidate,
                    trace_path=trace_path,
                    model_artifact_id="s34",
                    source_revision="a" * 40,
                    release="1.1.0-rc.0",
                    run_id="ablation-1",
                )

    def test_rejects_more_than_thirty_measurement_samples(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common, _ = fixture(root)
            trace_path = ablation_trace(root, common)
            trace = json.loads(trace_path.read_bytes())
            for row in trace["baselines"]:
                row["elapsed_samples_ms"].append(1.0)
                row["resident_samples_bytes"].append(100)
            trace_path.write_bytes(MODULE["canonical"](trace) + b"\n")
            with self.assertRaisesRegex(
                MODULE["VERIFIER"]["EstimatorRefinementError"], "exactly thirty"
            ):
                assemble(
                    root / "qualification",
                    candidate=candidate,
                    trace_path=trace_path,
                    model_artifact_id="s34",
                    source_revision="a" * 40,
                    release="1.1.0-rc.0",
                    run_id="ablation-1",
                )


if __name__ == "__main__":
    unittest.main()
