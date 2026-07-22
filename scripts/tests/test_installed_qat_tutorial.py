import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = ROOT / "examples/python/installed_qat.py"


class InstalledQatTutorialTests(unittest.TestCase):
    def test_wheel_workflow_executes_example_and_validates_installed_result(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
        example = "python -I ${{ github.workspace }}/examples/python/installed_qat.py"
        validator = "python -I -m tritium.torch.tutorial_qat"
        self.assertEqual(workflow.count(example), 2)
        self.assertEqual(workflow.count(validator), 2)
        self.assertEqual(workflow.count("--check-receipt"), 2)
        self.assertIn("--device cpu", workflow)
        self.assertIn("--device cuda:0", workflow)
        self.assertIn("evidence/tutorial-cpu/receipt.json", workflow)
        self.assertIn("evidence/tutorial-cuda/receipt.json", workflow)
        self.assertIn('"examples/python/installed_qat.py"', workflow)
        self.assertIn('"scripts/tests/test_installed_qat_tutorial.py"', workflow)

    def test_installed_qat_tutorial_has_no_source_checkout_escape_hatch(self):
        source = (
            ROOT / "crates/tritium-py/python/tritium/torch/tutorial_qat.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("sys.path", source)
        self.assertNotIn("PYTHONPATH", source)
        self.assertNotIn("allow-source", source)
        self.assertIn("tritium.installed-qat-tutorial.v1", source)
        self.assertIn('distribution("tritium-torch")', source)
        self.assertIn("not owned by tritium-torch", source)
        self.assertIn("export_qat_hard", source)
        self.assertIn("load_qat_hard", source)
        wrapper = TUTORIAL.read_text(encoding="utf-8")
        self.assertIn("tritium.torch.tutorial_qat import main", wrapper)


if __name__ == "__main__":
    unittest.main()
