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


class CommunityContractTests(unittest.TestCase):
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
