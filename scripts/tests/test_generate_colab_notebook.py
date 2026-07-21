import importlib.util
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
        self.assertNotIn("/home/", source)
        self.assertNotIn("drive.mount", source)


if __name__ == "__main__":
    unittest.main()
