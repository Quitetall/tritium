from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "resolve-release-ref.py")
ReleaseRefError = MODULE["ReleaseRefError"]
resolve_release_ref = MODULE["resolve_release_ref"]


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


class ResolveReleaseRefTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        git(self.repo, "init", "--initial-branch=main")
        git(self.repo, "config", "user.name", "Tritium Test")
        git(self.repo, "config", "user.email", "test@example.invalid")
        (self.repo / "trusted.txt").write_text("trusted\n", encoding="utf-8")
        git(self.repo, "add", "trusted.txt")
        git(self.repo, "commit", "-m", "trusted release commit")
        self.trusted_revision = git(self.repo, "rev-parse", "HEAD")
        git(self.repo, "update-ref", "refs/remotes/origin/main", "HEAD")
        git(self.repo, "tag", "v1.1.0-rc.2")

    def test_accepts_canonical_tag_on_default_branch(self):
        self.assertEqual(
            resolve_release_ref(
                self.repo, "v1.1.0-rc.2", "main", fetch_remote=None
            ),
            {"tag": "v1.1.0-rc.2", "revision": self.trusted_revision},
        )

    def test_rejects_noncanonical_tag_before_ref_lookup(self):
        for tag in ("main", "v1.1.0-rc.2;echo unsafe", "v01.1.0", "v1.1.0-beta.1"):
            with self.subTest(tag=tag), self.assertRaises(ReleaseRefError):
                resolve_release_ref(self.repo, tag, "main", fetch_remote=None)

    def test_rejects_tag_pointing_outside_default_branch(self):
        git(self.repo, "checkout", "--orphan", "unreviewed")
        (self.repo / "untrusted.txt").write_text("untrusted\n", encoding="utf-8")
        git(self.repo, "add", "untrusted.txt")
        git(self.repo, "commit", "-m", "unreviewed release code")
        git(self.repo, "tag", "v1.2.0")

        with self.assertRaisesRegex(ReleaseRefError, "not reachable"):
            resolve_release_ref(self.repo, "v1.2.0", "main", fetch_remote=None)

    def test_rejects_missing_tag_and_unsafe_default_branch(self):
        with self.assertRaises(ReleaseRefError):
            resolve_release_ref(self.repo, "v1.2.0", "main", fetch_remote=None)
        with self.assertRaises(ReleaseRefError):
            resolve_release_ref(self.repo, "v1.1.0-rc.2", "../main", fetch_remote=None)

    def test_release_workflow_only_checks_out_verified_commit(self):
        workflow = (ROOT / ".github/workflows/release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "ref: ${{ github.event.repository.default_branch }}", workflow
        )
        self.assertIn("scripts/resolve-release-ref.py", workflow)
        self.assertIn("needs.resolve.outputs.revision", workflow)
        self.assertIn("needs.resolve.outputs.tag", workflow)
        self.assertIn("github.ref == format('refs/heads/{0}'", workflow)
        self.assertNotIn("needs.resolve.outputs.ref", workflow)
        self.assertNotIn("ref: ${{ github.event.inputs.tag", workflow)


if __name__ == "__main__":
    unittest.main()
