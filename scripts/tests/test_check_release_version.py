import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "check-release-version.py"
SPEC = importlib.util.spec_from_file_location("check_release_version", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReleaseVersionTests(unittest.TestCase):
    def test_candidate_accepts_any_canonical_release_or_rc(self):
        """Format is the contract here, not one particular version.

        This test used to require the workspace version to be literally
        `1.1.0-rc.N`, rejecting `1.1.0` and `1.2.0-rc.0`. That is not what the
        script is for -- it exists to catch a version MIRROR drifting from
        Cargo.toml (pyproject, npm, Cargo.lock, the compatibility receipt), and it
        uses the workspace version only as the reference to compare against.

        The pin was actively harmful: this runs inside check-publish.sh, which is
        the REQUIRED `publish-check` CI job, so tagging the 1.1.0 final release
        would have failed ci-required on the release commit itself. A gate that
        forbids shipping the next version is not protecting anything.
        """
        for valid in ("1.1.0-rc.0", "1.1.0-rc.12", "1.1.0", "1.2.0-rc.0", "2.0.0"):
            with self.subTest(valid=valid):
                self.assertEqual(MODULE.candidate_version(valid), valid)

    def test_candidate_still_rejects_non_canonical_spellings(self):
        """Loosening the version must not loosen the SPELLING it accepts."""
        for invalid in (
            "1.1.0-rc.01",  # leading zero
            "1.1.0rc1",     # PEP 440 spelling; Cargo uses -rc.N
            "1.1",          # not three components
            "v1.1.0",       # tag spelling, not a version
            "1.1.0-rc",     # rc without an ordinal
            "1.1.0-alpha.1",
            "",
            None,
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                MODULE.candidate_version(invalid)

    def test_mirror_mismatch_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "npm package version"):
            MODULE.require_equal("1.0.0", "1.1.0-rc.0", "npm package version")


if __name__ == "__main__":
    unittest.main()
