"""Candidate-wheel observability workflow contract."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ObservabilityQualificationTests(unittest.TestCase):
    def test_source_free_job_executes_and_retains_observability_receipt(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
        start = workflow.index("  tutorial-clean-wheel:")
        match = re.search(r"(?m)^  [a-z0-9_-]+:\s*$", workflow[start + 3 :])
        self.assertIsNotNone(match)
        assert match is not None
        job = workflow[start : start + 3 + match.start()]

        self.assertEqual(
            job.count("python -I -m tritium.torch.qualify_observability"), 2
        )
        for package in (
            "tensorboard==2.21.0",
            "wandb==0.28.1",
            "opentelemetry-api==1.44.0",
            "opentelemetry-sdk==1.44.0",
        ):
            self.assertIn(package, job)
        self.assertRegex(
            job,
            r"python -m pip install --isolated[^\n]*\\\n"
            r"\s+--no-index --no-deps --only-binary=:all: --no-compile dist/\*\.whl",
        )
        self.assertIn("--check-receipt evidence/observability-clean/receipt.json", job)
        self.assertIn("evidence/observability-clean/**", job)
        self.assertNotIn("actions/checkout", job)

    def test_worker_has_no_source_escape_or_online_wandb_mode(self):
        source = (
            ROOT
            / "crates/tritium-py/python/tritium/torch/qualify_observability.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("sys.path", source)
        self.assertNotIn("PYTHONPATH", source)
        self.assertNotIn("allow-source", source)
        self.assertIn('distribution("tritium-torch")', source)
        self.assertIn('mode="offline"', source)
        self.assertIn("compiler_absent", source)
        self.assertIn("repository_absent", source)


if __name__ == "__main__":
    unittest.main()
