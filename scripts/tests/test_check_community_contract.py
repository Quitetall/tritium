from __future__ import annotations

import shutil
from pathlib import Path
import runpy
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "check-community-contract.py")
check = MODULE["check"]
CommunityContractError = MODULE["CommunityContractError"]
check_local_link = MODULE["_check_local_link"]
github_anchors = MODULE["_github_anchors"]


class CommunityContractTests(unittest.TestCase):
    def test_github_heading_ids_handle_duplicates_and_fenced_code(self):
        anchors = github_anchors(
            "# Release sign-off\n\n# Release sign-off\n\n"
            "```md\n# not-a-heading\n```\n\n<a id=\"explicit-route\"></a>\n"
        )
        self.assertIn("release-sign-off", anchors)
        self.assertIn("release-sign-off-1", anchors)
        self.assertNotIn("not-a-heading", anchors)
        self.assertIn("explicit-route", anchors)

    def test_local_markdown_fragments_must_resolve(self):
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            source = repo / "guide.md"
            source.write_text("# Ready\n", encoding="utf-8")
            self.assertTrue(check_local_link(repo, source, "guide.md", "#ready"))
            self.assertTrue(
                check_local_link(repo, source, "guide.md", "guide.md#ready")
            )
            with self.assertRaisesRegex(CommunityContractError, "missing-anchor"):
                check_local_link(
                    repo, source, "guide.md", "guide.md#missing-anchor"
                )

    def test_repository_governance_contract_passes(self):
        report = check(ROOT)
        self.assertEqual(report["result"], "pass")
        self.assertGreater(report["local_links"], 0)
        self.assertEqual(report["unstaffed_channels"], 0)

    def test_broken_local_link_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git", "target"))
            path = repo / "SUPPORT.md"
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "[SECURITY.md](SECURITY.md)",
                    "[SECURITY.md](missing-security.md)",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CommunityContractError, "missing-security.md"):
                check(repo)

    def test_contact_route_cannot_drift(self):
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git", "target"))
            path = repo / "SECURITY.md"
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "briankhanglam@gmail.com", "security@example.invalid"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CommunityContractError, "private security"):
                check(repo)

    def test_public_unstaffed_channel_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git", "target"))
            path = repo / "COMMUNITY.md"
            path.write_text(
                path.read_text(encoding="utf-8")
                + "\nOfficial Discord: https://discord.gg/example\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CommunityContractError, "unstaffed"):
                check(repo)


if __name__ == "__main__":
    unittest.main()
