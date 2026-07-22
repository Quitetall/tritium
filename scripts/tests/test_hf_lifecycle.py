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

    def test_hf_receipt_has_no_source_checkout_escape_hatch(self):
        source = (
            ROOT / "crates/tritium-py/python/tritium/torch/hf_lifecycle.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("sys.path", source)
        self.assertNotIn("PYTHONPATH", source)
        self.assertIn('distribution("tritium-torch")', source)
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
