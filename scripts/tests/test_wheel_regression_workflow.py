"""Release wheels must exercise PTQ/refinement regressions as installed code."""
from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
TESTS = (
    "test_model_conversion.py",
    "test_native_module_source.py",
    "test_native_module_load.py",
    "test_native_scale_training.py",
    "test_native_pv_training.py",
    "test_native_pv_resume.py",
    "test_native_refinement.py",
    "test_native_resume.py",
    "test_native_refine.py",
    "test_native_chaining.py",
    "test_native_module_calibration.py",
    "test_native_calibration_selection.py",
    "test_owner_row_score.py",
    "test_owner_capture.py",
    "test_owner_cache.py",
    "test_owner_phases.py",
    "test_owner_conversion.py",
    "test_owner_export.py",
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

    def test_real_rust_fixture_probe_runs_against_installed_candidate(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
        job = workflow.split("  torch-functional-smoke:\n", 1)[1].split("\n  tutorial-clean-wheel:", 1)[0]
        marker = "      - name: test exact-wheel Rust/core/CLI/Python conversion parity\n"
        self.assertIn(marker, job)
        step = job.split(marker, 1)[1].split("      - name:", 1)[0]
        for required in (
            'TRITIUM_TEST_INSTALLED_WHEEL: "1"',
            "TRITIUM_TEST_CANDIDATE_WHEEL_DIR: dist",
            'RAYON_NUM_THREADS: "1"',
            'CARGO_BUILD_JOBS: "2"',
            'TRITIUM_NATIVE_TEST_PYTHON="$(python -c',
            "sys.executable",
            "timeout 900 cargo test --locked -p tritium-cli --test convert_plan_cli",
            "python::actual_python_file_conversion_matches_core_and_cli",
            "-- --ignored --exact --nocapture",
        ):
            with self.subTest(required=required):
                self.assertIn(required, step)
        before = job.split(marker, 1)[0]
        self.assertIn("dtolnay/rust-toolchain@4360b52568e2003a75bf9bc1d59f33a8e3fc893c", before)
        self.assertIn('toolchain: "1.98.0"', before)
        self.assertLess(job.index("install exact Tritium wheel without dependency resolution"), job.index(marker))
        self.assertNotIn("maturin", step)

    def test_fixture_producer_changes_trigger_wheel_probe(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
        paths = workflow.split("  pull_request:\n", 1)[1].split("  workflow_dispatch:", 1)[0]
        for path in (
            "crates/tritium-cli/tests/convert_plan_cli.rs",
            "crates/tritium-cli/tests/convert_plan_cli/**",
        ):
            with self.subTest(path=path):
                self.assertIn(f'      - "{path}"', paths)

    def test_real_fixture_receiver_uses_isolated_python(self):
        producer = (ROOT / "crates/tritium-cli/tests/convert_plan_cli/python.rs").read_text(encoding="utf-8")
        self.assertIn("fn actual_python_file_conversion_matches_core_and_cli()", producer)
        self.assertIn('            "-I",\n            "-m",\n            "pytest",', producer)
        for precision in ("F16", "Bf16", "F32"):
            self.assertIn(f"ScalePrecision::{precision}", producer)
        self.assertIn('for mode in ["direct", "gram", "calibrated"]', producer)

    def test_native_pv_corpus_probe_uses_the_installed_candidate_and_is_triggered(self):
        workflow = (ROOT / ".github/workflows/wheels.yml").read_text(encoding="utf-8")
        marker = "      - name: test exact-wheel native PV child loading and export\n"
        step = workflow.split(marker, 1)[1].split("      - name:", 1)[0]
        for value in ('TRITIUM_TEST_INSTALLED_WHEEL: "1"', "TRITIUM_TEST_CANDIDATE_WHEEL_DIR: dist",
                      'TRITIUM_PV_TEST_PYTHON="$(python -c', "timeout 900 cargo test --locked -p tritium-train --test native_python_pv",
                      "actual_pv_publisher_installed_python_descriptor_round_trip", "-- --ignored --exact --nocapture"):
            self.assertIn(value, step)
        paths = workflow.split("  pull_request:\n", 1)[1].split("  workflow_dispatch:", 1)[0]
        self.assertIn('      - "crates/tritium-train/tests/native_python_pv.rs"', paths)
        producer = (ROOT / "crates/tritium-train/tests/native_python_pv.rs").read_text(encoding="utf-8")
        self.assertIn('"-I",', producer)
        self.assertIn("publish_native_pv_child", producer)
        self.assertIn("test_native_pv_loading.py", producer)


if __name__ == "__main__":
    unittest.main()
