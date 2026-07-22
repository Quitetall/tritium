from pathlib import Path
import json
import runpy
import tempfile
import unittest

from scripts.tests.test_verify_training_backend_receipt import fixture


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-training-backends.py")
assemble = MODULE["assemble"]
parse_bindings = MODULE["parse_bindings"]
QualificationError = MODULE["QualificationError"]


def artifact_ids(candidate: Path):
    document = json.loads(candidate.read_bytes())
    return {
        family: document["artifacts"][ordinal]["id"]
        for ordinal, family in enumerate(MODULE["FAMILIES"])
    }


class QualifyTrainingBackendsTests(unittest.TestCase):
    def test_assembles_and_self_validates_all_seven_bundles(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, _, _ = fixture(root)
            output = root / "qualification"
            receipt = assemble(
                output, repo=ROOT, candidate=candidate,
                artifact_ids=artifact_ids(candidate), source_revision="a" * 40,
                release="1.1.0-rc.0", run_id="backend-physical-1",
            )
            self.assertEqual(receipt["result"], "pass")
            self.assertEqual(
                [bundle["family"] for bundle in receipt["bundles"]],
                list(MODULE["FAMILIES"]),
            )

    def test_rejects_swapped_family_artifacts(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidate, _, _ = fixture(root)
            bindings = artifact_ids(candidate)
            bindings["cpu"], bindings["cuda"] = bindings["cuda"], bindings["cpu"]
            with self.assertRaisesRegex(
                MODULE["TrainingBackendReceiptError"], "backend identity"
            ):
                assemble(
                    root / "qualification", repo=ROOT, candidate=candidate,
                    artifact_ids=bindings, source_revision="a" * 40,
                    release="1.1.0-rc.0", run_id="backend-physical-1",
                )

    def test_cli_bindings_are_complete_ordered_and_unique(self):
        values = [
            f"{family}=artifact-{family}" for family in MODULE["FAMILIES"]
        ]
        self.assertEqual(tuple(parse_bindings(values)), MODULE["FAMILIES"])
        with self.assertRaisesRegex(QualificationError, "all seven"):
            parse_bindings(values[:-1])
        with self.assertRaisesRegex(QualificationError, "unique"):
            parse_bindings([
                f"{family}=same" for family in MODULE["FAMILIES"]
            ])


if __name__ == "__main__":
    unittest.main()
