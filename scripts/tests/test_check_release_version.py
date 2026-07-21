import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "check-release-version.py"
SPEC = importlib.util.spec_from_file_location("check_release_version", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReleaseVersionTests(unittest.TestCase):
    def test_candidate_requires_canonical_v11_rc(self):
        self.assertEqual(MODULE.candidate_version("1.1.0-rc.0"), "1.1.0-rc.0")
        self.assertEqual(MODULE.candidate_version("1.1.0-rc.12"), "1.1.0-rc.12")
        for invalid in ("1.0.0", "1.1.0", "1.1.0-rc.01", "1.2.0-rc.0", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                MODULE.candidate_version(invalid)

    def test_mirror_mismatch_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "npm package version"):
            MODULE.require_equal("1.0.0", "1.1.0-rc.0", "npm package version")


if __name__ == "__main__":
    unittest.main()
