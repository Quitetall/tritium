import json
from pathlib import Path
import runpy
import tempfile
import unittest

from scripts.tests.test_verify_estimator_refinement_receipt import (
    fixture,
    refinement_trace,
)


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-refinement-campaign.py")
assemble = MODULE["assemble"]


class QualifyRefinementCampaignTests(unittest.TestCase):
    def test_rejects_symlinked_raw_trace(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common, _ = fixture(root)
            trace = refinement_trace(root, common)
            link = root / "linked-trace.json"
            link.symlink_to(trace)
            with self.assertRaisesRegex(MODULE["QualificationError"], "ordinary"):
                assemble(
                    root / "qualification",
                    candidate=candidate,
                    trace_path=link,
                    parent_artifact_id="parent",
                    source_revision="a" * 40,
                    release="1.1.0-rc.0",
                    run_id="refinement-1",
                )

    def test_aggregates_raw_trace_and_self_validates(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common, _ = fixture(root)
            trace = refinement_trace(root, common)
            receipt = assemble(
                root / "qualification",
                candidate=candidate,
                trace_path=trace,
                parent_artifact_id="parent",
                source_revision="a" * 40,
                release="1.1.0-rc.0",
                run_id="refinement-1",
            )
            self.assertEqual(receipt["anchor_artifact"]["id"], "s34")
            self.assertEqual(
                [child["mode"] for child in receipt["children"]],
                list(MODULE["VERIFIER"]["CHILDREN"]),
            )
            self.assertTrue(
                (root / "qualification/refinement-execution.json").is_file()
            )

    def test_rejects_under_sampled_reload_trace(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, common, _ = fixture(root)
            trace_path = refinement_trace(root, common)
            trace = json.loads(trace_path.read_bytes())
            trace["children"][0]["reload_samples"] = 31
            trace_path.write_bytes(MODULE["canonical"](trace) + b"\n")
            with self.assertRaisesRegex(
                MODULE["VERIFIER"]["EstimatorRefinementError"], "reload samples"
            ):
                assemble(
                    root / "qualification",
                    candidate=candidate,
                    trace_path=trace_path,
                    parent_artifact_id="parent",
                    source_revision="a" * 40,
                    release="1.1.0-rc.0",
                    run_id="refinement-1",
                )


if __name__ == "__main__":
    unittest.main()
