import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nbformat


SCRIPT = Path(__file__).parents[1] / "generate-colab-notebook.py"
SPEC = importlib.util.spec_from_file_location("generate_colab_notebook", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GenerateColabNotebookTests(unittest.TestCase):
    def test_notebook_is_deterministic_and_has_tutorial_sections(self):
        first = MODULE.rendered()
        second = MODULE.rendered()
        self.assertEqual(first, second)
        notebook = nbformat.reads(first, as_version=4)
        markdown = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "markdown"
        )
        for section in ("## Goal", "## Setup", "## Steps", "## Checks", "## Next steps"):
            self.assertIn(section, markdown)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        self.assertIn("run_smollm2_release_demo", source)
        self.assertIn("SMOLLM2_REVISION", source)
        self.assertIn(f'pytritium=={MODULE.pypi_candidate()}', source)
        self.assertNotIn("pytritium==1.1.0rc1", source)
        self.assertIn('"cuda" in tritium.compiled_backends()', source)
        self.assertIn('else "cpu"', source)
        self.assertNotIn("/home/", source)
        self.assertNotIn("drive.mount", source)

    def test_check_mode_reports_missing_and_stale_notebooks_without_tracebacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "tutorial.ipynb"
            command = [sys.executable, str(SCRIPT), "--output", str(output), "--check"]

            missing = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(missing.returncode, 2)
            self.assertIn("is stale; regenerate it", missing.stderr)
            self.assertNotIn("Traceback", missing.stderr)

            output.write_text("{}\n", encoding="utf-8")
            stale = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(stale.returncode, 2)
            self.assertIn("is stale; regenerate it", stale.stderr)
            self.assertNotIn("Traceback", stale.stderr)


if __name__ == "__main__":
    unittest.main()
