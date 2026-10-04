from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify-workflow-source.py"
WORKFLOW = ROOT / ".github" / "workflows" / "wheels.yml"


class VerifyWorkflowSourceTests(unittest.TestCase):
    def test_workflow_checks_every_checkout_before_building_or_admitting(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("TRITIUM_SOURCE_REVISION: ${{ inputs.source_revision || github.sha }}", workflow)
        self.assertNotIn("--source-revision ${{ inputs.source_revision", workflow)
        self.assertIn("test exact source identity contract", workflow)
        blocks = workflow.split("      - uses: actions/checkout@")
        self.assertGreaterEqual(len(blocks), 2)
        for block in blocks[1:]:
            steps = block.split("      - ", 1)
            checkout = steps[0]
            remainder = steps[1] if len(steps) == 2 else ""
            self.assertIn("ref: ${{ inputs.source_ref || github.ref }}", checkout)
            self.assertTrue(
                remainder.startswith("name: verify checked-out source revision"),
                "every checkout must be followed immediately by source verification",
            )
            self.assertIn("verify-workflow-source.py", remainder.split("      - ", 1)[0])

    def test_accepts_exact_head_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
                   "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.invalid"}
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c",
                            "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "test"],
                           check=True, env=env)
            revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                      check=True, capture_output=True, text=True).stdout.strip()

            passed = subprocess.run([sys.executable, str(SCRIPT), "--expected-revision", revision],
                                    cwd=repo, capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertIn("PASS:", passed.stdout)

            failed = subprocess.run([sys.executable, str(SCRIPT), "--expected-revision", "0" * 40],
                                    cwd=repo, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1)
            self.assertIn("differs from receipt revision", failed.stderr)


if __name__ == "__main__":
    unittest.main()
