"""Release wheels must exercise PTQ/refinement regressions as installed code."""
from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
TESTS = (
    "test_model_conversion.py",
    "test_ptq_hashing.py", "test_calibration_memory.py", "test_refinement.py",
    "test_refinement_core.py", "test_refinement_lifecycle.py",
    "test_refinement_teacher_state.py", "test_refinement_containers.py",
    "test_module_packing.py", "test_torch_lifecycle.py",
)


class WheelRegressionWorkflowTests(unittest.TestCase):
    def test_installed_candidate_step_runs_ptq_and_refinement_regressions(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
        step = workflow.split("      - name: test exact-wheel public PTQ artifacts and ONNX custody\n", 1)[1].split("      - name:", 1)[0]
        self.assertIn('TRITIUM_TEST_INSTALLED_WHEEL: "1"', step)
        self.assertIn("TRITIUM_TEST_CANDIDATE_WHEEL_DIR: dist", step)
        self.assertIn("python -I -m pytest -q", step)
        for filename in TESTS:
            with self.subTest(filename=filename):
                self.assertTrue((ROOT / "crates/tritium-py/tests" / filename).is_file())
                self.assertIn(f"crates/tritium-py/tests/{filename}", step)

    def test_new_workflow_contract_test_triggers_wheels(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
        paths = workflow.split("  pull_request:\n", 1)[1].split("  workflow_dispatch:", 1)[0]
        self.assertIn('      - "scripts/tests/test_wheel_regression_workflow.py"', paths)


if __name__ == "__main__":
    unittest.main()
