import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "generate-api-diff.py"
SPEC = importlib.util.spec_from_file_location("generate_api_diff", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GenerateApiDiffTests(unittest.TestCase):
    def test_pyo3_v1_surface_is_extracted_from_registration(self):
        source = """
#[pymodule]
fn tritium(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_function(wrap_pyfunction!(ternary_matmul, m)?)?;
    Ok(())
}
"""
        self.assertEqual(
            MODULE.pyo3_exports(source), {"Model", "ternary_matmul"}
        )

    def test_pyo3_extraction_ignores_comments_and_rejects_unsupported_forms(self):
        commented = """
// m.add_class::<Ghost>()?;
#[pymodule]
fn tritium(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(phantom, m)?)?;
    m.add_class::<Model>()?;
    Ok(())
}
"""
        self.assertEqual(MODULE.pyo3_exports(commented), {"Model"})
        renamed = commented.replace(
            "#[pymodule]", '#[pyclass(name = "Alias")]\nstruct Model;\n#[pymodule]'
        )
        with self.assertRaisesRegex(MODULE.ApiDiffError, "renamed"):
            MODULE.pyo3_exports(renamed)
        unsupported = commented.replace(
            "m.add_class::<Model>()?;", 'm.add("Model", py.get_type::<Model>())?;'
        )
        with self.assertRaisesRegex(MODULE.ApiDiffError, "unsupported"):
            MODULE.pyo3_exports(unsupported)
        conditional = commented.replace(
            "m.add_class::<Model>()?;",
            "if enabled() {\n        m.add_class::<Model>()?;\n    }",
        )
        with self.assertRaisesRegex(MODULE.ApiDiffError, "conditional"):
            MODULE.pyo3_exports(conditional)
        attributed = commented.replace(
            "m.add_class::<Model>()?;",
            '#[cfg(feature = "optional")]\n    m.add_class::<Model>()?;',
        )
        with self.assertRaisesRegex(MODULE.ApiDiffError, "attributed"):
            MODULE.pyo3_exports(attributed)
        helper = commented.replace(
            "m.add_class::<Model>()?;",
            "m.add_class::<Model>()?;\n    register_more(m)?;",
        )
        with self.assertRaisesRegex(MODULE.ApiDiffError, "helper"):
            MODULE.pyo3_exports(helper)

    def test_python_all_includes_guarded_optional_extensions(self):
        source = """
__all__ = ["Model", "ternary_matmul"]
try:
    __all__.extend(["torch", "nn"])
except ImportError:
    pass
"""
        self.assertEqual(
            MODULE.python_all(source), {"Model", "ternary_matmul", "torch", "nn"}
        )

    def test_python_all_reassignment_replaces_and_dynamic_forms_fail_closed(self):
        self.assertEqual(
            MODULE.python_all('__all__ = ["old"]\n__all__ = ["new"]\n'),
            {"new"},
        )
        for source in (
            "__all__ = make_exports()",
            '__all__ = ["Model"]\n__all__.extend(dynamic)',
            '__all__ = ["Model"]\n__all__.remove("Model")',
            '__all__ = ["Model"]\ndef hidden():\n    __all__ = ["Ghost"]',
            'if enabled:\n    __all__ = ["Model"]',
        ):
            with self.subTest(source=source), self.assertRaises(MODULE.ApiDiffError):
                MODULE.python_all(source)

    def test_report_rejects_removed_v1_names(self):
        with self.assertRaisesRegex(MODULE.ApiDiffError, "removed"):
            MODULE.build_report(
                baseline="v1.0.0",
                candidate_version="1.1.0-rc.0",
                baseline_exports={"Model", "ternary_matmul"},
                current_exports={"Model"},
            )

    def test_semver_scope_is_extracted_from_literal_checked_crates_array(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            scripts = root / "scripts"
            scripts.mkdir()
            source = """CHECKED_CRATES=(
  tritium-core tritium-spec
  # Explicitly frozen.
  \"tritium-runtime\"
)
"""
            (scripts / "check-semver.sh").write_text(source, encoding="utf-8")
            self.assertEqual(
                MODULE.semver_crates(root),
                ("tritium-core", "tritium-spec", "tritium-runtime"),
            )

    def test_semver_scope_rejects_dynamic_duplicate_or_ambiguous_arrays(self):
        sources = (
            "CHECKED_CRATES=(tritium-core $EXTRA)\n",
            "CHECKED_CRATES=(tritium-core tritium-core)\n",
            "CHECKED_CRATES=(tritium-core)\nCHECKED_CRATES=(tritium-spec)\n",
            "package_args=(-p tritium-core)\n",
        )
        for source in sources:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                scripts = root / "scripts"
                scripts.mkdir()
                (scripts / "check-semver.sh").write_text(source, encoding="utf-8")
                with self.assertRaises(MODULE.ApiDiffError):
                    MODULE.semver_crates(root)

    def test_repository_report_retains_v1_and_is_canonical(self):
        report = MODULE.repository_report(MODULE.ROOT, "v1.0.0")
        self.assertEqual(report["schema"], "tritium.api-diff.v1")
        self.assertEqual(report["python"]["removed"], [])
        self.assertEqual(
            report["python"]["retained"], ["Model", "ternary_matmul"]
        )
        self.assertEqual(
            report["rust"]["frozen_crates"],
            [
                "tritium-core", "tritium-spec", "tritium-format",
                "tritium-runtime", "tritium-cpu", "tritium-quantize",
                "tritium-testkit", "tritium-ffi", "tritium-nn",
                "tritium-salt", "tritium-train", "tritium-serve",
                "tritium-burn", "tritium-candle", "tritium-onnx",
                "tritium-mcu", "tritium-wasm", "tritium-metal",
                "tritium-rocm", "tritium-wgpu", "tritium-build-info",
            ],
        )
        self.assertEqual(json.loads(MODULE.render_json(report)), report)

    def test_report_identity_covers_every_emitted_semantic_field(self):
        report = MODULE.repository_report(MODULE.ROOT, "v1.0.0")
        original = report["report_id"]
        report["scope"]["runtime_receipt"] = "changed"
        self.assertNotEqual(MODULE.report_identity(report), original)

    def test_checked_in_outputs_are_current(self):
        report = MODULE.repository_report(MODULE.ROOT, "v1.0.0")
        MODULE.check_outputs(
            report, MODULE.DEFAULT_JSON, MODULE.DEFAULT_MARKDOWN
        )

    def test_check_detects_stale_outputs(self):
        report = MODULE.repository_report(MODULE.ROOT, "v1.0.0")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            json_path = root / "api.json"
            markdown_path = root / "api.md"
            MODULE.write_outputs(report, json_path, markdown_path)
            MODULE.check_outputs(report, json_path, markdown_path)
            markdown_path.write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.ApiDiffError, "stale"):
                MODULE.check_outputs(report, json_path, markdown_path)


if __name__ == "__main__":
    unittest.main()
