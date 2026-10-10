import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class HuggingFaceLifecycleWorkflowTests(unittest.TestCase):
    def test_clean_wheel_job_runs_and_retains_hf_lifecycle(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
        start = workflow.index("  tutorial-clean-wheel:")
        match = re.search(r"(?m)^  [a-z0-9_-]+:\s*$", workflow[start + 3 :])
        self.assertIsNotNone(match)
        assert match is not None
        job = workflow[start : start + 3 + match.start()]
        self.assertNotIn("actions/checkout", job)
        self.assertIn("transformers==5.5.3", job)
        self.assertEqual(job.count("python -I -m tritium.torch.hf_lifecycle"), 2)
        self.assertIn("evidence/hf-lifecycle-clean/receipt.json", job)
        self.assertIn("evidence/hf-lifecycle-clean/**", job)
    def test_installed_wheel_job_runs_candidate_provenance_regressions(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
        job = workflow.split("  torch-functional-smoke:", 1)[1].split(
            "  tutorial-clean-wheel:", 1
        )[0]
        self.assertIn('TRITIUM_TEST_INSTALLED_WHEEL: "1"', job)
        for test in (
            "test_hf_candidate_provenance.py",
            "test_hf_lifecycle_receipt.py",
            "test_hf_export_lifecycle.py",
        ):
            self.assertIn("crates/tritium-py/tests/" + test, job)

    def test_hf_receipt_has_no_source_checkout_escape_hatch(self):
        source = (
            ROOT / "crates/tritium-py/python/tritium/torch/hf_lifecycle.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("sys.path", source)
        self.assertNotIn("PYTHONPATH", source)
        self.assertIn("verify_installed_candidate", source)
        candidate_source = (
            ROOT / "crates/tritium-py/python/tritium/torch/_installed_candidate.py"
        ).read_text(encoding="utf-8")
        self.assertIn('distribution("pytritium")', candidate_source)
        self.assertNotIn("import transformers", candidate_source)
        self.assertIn("AutoModelForCausalLM.from_pretrained", source)
        self.assertIn("safe_serialization=True", source)

    def test_clean_wheel_job_runs_whole_model_hard_export(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
        start = workflow.index("  tutorial-clean-wheel:")
        match = re.search(r"(?m)^  [a-z0-9_-]+:\s*$", workflow[start + 3 :])
        self.assertIsNotNone(match)
        assert match is not None
        job = workflow[start : start + 3 + match.start()]
        self.assertNotIn("actions/checkout", job)
        self.assertEqual(job.count("python -I -m tritium.torch.hf_export_lifecycle"), 2)
        self.assertIn("evidence/hf-export-clean/receipt.json", job)
        self.assertIn("evidence/hf-export-clean/**", job)


if __name__ == "__main__":
    unittest.main()
