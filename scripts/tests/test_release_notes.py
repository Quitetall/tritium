"""The release workflow generates notes unattended; a crash there fails a release."""

import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(str(ROOT / "scripts" / "release-notes.py"))

SAMPLE = """# Changelog

## [Unreleased] — 1.x dev

pending work

## [1.0.0] — 2026-06-28 — v1.0 Release 🎉

first stable

## [0.9.0] — 2026-06-24 — hardening

older
"""


class ReleaseNotesTests(unittest.TestCase):
    def test_exact_section_wins_and_stops_at_the_next_heading(self):
        body, exact = MODULE["find"](SAMPLE, "1.0.0")
        self.assertTrue(exact)
        self.assertEqual(body, "first stable")
        self.assertNotIn("older", body)

    def test_missing_version_falls_back_to_unreleased_and_says_so(self):
        body, exact = MODULE["find"](SAMPLE, "1.1.0-rc.2")
        self.assertFalse(exact)
        self.assertEqual(body, "pending work")
        out = MODULE["render"]("1.1.0-rc.2", body, exact)
        self.assertIn("No CHANGELOG section for 1.1.0-rc.2", out)

    def test_version_is_matched_on_the_bracket_not_the_dated_title(self):
        """`## [1.0.0] — 2026-06-28 — ...` must match `1.0.0`, and 1.0 must not."""
        self.assertTrue(MODULE["find"](SAMPLE, "1.0.0")[1])
        self.assertFalse(MODULE["find"](SAMPLE, "1.0")[1])

    def test_a_version_that_is_a_regex_is_not_a_wildcard(self):
        body, exact = MODULE["find"](SAMPLE, "1.0.0")
        self.assertTrue(exact)
        # `.` must not match `1x0y0`; the version is escaped before use.
        self.assertFalse(MODULE["find"](SAMPLE.replace("1.0.0", "1x0y0"), "1.0.0")[1])

    def test_real_changelog_renders_for_the_current_workspace_version(self):
        cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
        import re

        version = re.search(r'(?m)^version = "([^"]+)"', cargo).group(1)
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        out = MODULE["render"](version, *MODULE["find"](text, version))
        self.assertTrue(out.strip())
        self.assertIn("CHANGELOG.md", out)


if __name__ == "__main__":
    unittest.main()
