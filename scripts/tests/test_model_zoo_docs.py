from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODEL_ZOO = ROOT / "docs" / "book" / "src" / "model-zoo.md"


class ModelZooDocumentationTests(unittest.TestCase):
    def test_release_status_does_not_revert_to_pre_v1_claims(self):
        document = MODEL_ZOO.read_text(encoding="utf-8")

        self.assertIn("## Current v1.1 qualification status", document)
        self.assertNotIn("Caveats and pre-1.0 status", document)
        self.assertNotIn("v1.0 exit gate", document)
        self.assertNotIn("Pre-1.0.", document)


if __name__ == "__main__":
    unittest.main()
