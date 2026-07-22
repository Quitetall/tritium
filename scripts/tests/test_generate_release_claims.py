from pathlib import Path
import runpy
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "generate-release-claims.py")
ClaimGenerationError = MODULE["ClaimGenerationError"]


class GenerateReleaseClaimsTests(unittest.TestCase):
    def test_repository_claim_projections_are_current(self):
        MODULE["check"](ROOT)

    def test_generated_blocks_cover_exact_frozen_ladder(self):
        rendered = MODULE["blocks"]()
        for relative in MODULE["DOCUMENTS"]:
            block = rendered[relative]
            self.assertEqual(block.count(MODULE["BEGIN"]), 1)
            self.assertEqual(block.count(MODULE["END"]), 1)
            for _, _, model_id in MODULE["EXPECTED_MODELS"]:
                self.assertIn(f"`{model_id}`", block)

    def test_rejects_missing_or_duplicate_markers(self):
        with self.assertRaisesRegex(ClaimGenerationError, "exactly one"):
            MODULE["replace_block"]("plain text", "replacement", "doc.md")
        duplicate = (
            f"{MODULE['BEGIN']}x{MODULE['END']}\n"
            f"{MODULE['BEGIN']}y{MODULE['END']}"
        )
        with self.assertRaisesRegex(ClaimGenerationError, "exactly one"):
            MODULE["replace_block"](duplicate, "replacement", "doc.md")


if __name__ == "__main__":
    unittest.main()
