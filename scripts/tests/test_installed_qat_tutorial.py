import unittest
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = ROOT / "examples/python/installed_qat.py"


class InstalledQatTutorialTests(unittest.TestCase):
    def test_wheel_workflow_executes_example_and_validates_installed_result(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
        example = "python -I ${{ github.workspace }}/examples/python/installed_qat.py"
        validator = "python -I -m tritium.torch.tutorial_qat"
        self.assertEqual(workflow.count(example), 2)
        self.assertEqual(workflow.count(validator), 4)
        self.assertEqual(workflow.count("--check-receipt"), 4)
        self.assertIn("--device cpu", workflow)
        self.assertIn("--device cuda:0", workflow)
        self.assertIn("evidence/tutorial-cpu/receipt.json", workflow)
        self.assertIn("evidence/tutorial-cuda/receipt.json", workflow)
        self.assertIn('"examples/python/installed_qat.py"', workflow)
        self.assertIn('"scripts/tests/test_installed_qat_tutorial.py"', workflow)

    def test_clean_tutorial_job_has_no_checkout_or_compiler(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
        start = workflow.index("  tutorial-clean-wheel:")
        match = re.search(r"(?m)^  [a-z0-9_-]+:\s*$", workflow[start + 3 :])
        self.assertIsNotNone(match)
        assert match is not None
        end = start + 3 + match.start()
        job = workflow[start:end]
        self.assertIn("container: python:3.13-slim", job)
        self.assertNotIn("actions/checkout", job)
        self.assertIn('test ! -e "$GITHUB_WORKSPACE/.git"', job)
        for compiler in (
            "cargo",
            "rustc",
            "cc",
            "c++",
            "gcc",
            "g++",
            "clang",
            "clang++",
        ):
            self.assertIn(f"! command -v {compiler}", job)
        self.assertIn("python -I -m tritium.torch.tutorial_qat", job)
        self.assertIn("--check-receipt", job)
        self.assertIn("test_tutorial_qat.py", workflow)

    def test_installed_qat_tutorial_has_no_source_checkout_escape_hatch(self):
        source = (
            ROOT / "crates/tritium-py/python/tritium/torch/tutorial_qat.py"
        ).read_text(encoding="utf-8")
        receipt_source = (
            ROOT / "crates/tritium-py/python/tritium/torch/tutorial_receipt.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("sys.path", source)
        self.assertNotIn("PYTHONPATH", source)
        self.assertNotIn("allow-source", source)
        self.assertIn("tritium.installed-qat-tutorial.v3", receipt_source)
        self.assertIn("--wheel-artifact", source)
        self.assertIn("--source-revision", source)
        self.assertIn('distribution("tritium-torch")', source)
        self.assertIn("not owned by tritium-torch", source)
        self.assertIn("export_qat_hard", source)
        self.assertIn("load_qat_hard", source)
        wrapper = TUTORIAL.read_text(encoding="utf-8")
        self.assertIn("tritium.torch.tutorial_qat import main", wrapper)


if __name__ == "__main__":
    unittest.main()
