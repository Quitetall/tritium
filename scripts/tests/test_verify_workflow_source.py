from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify-workflow-source.py"
WORKFLOW = ROOT / ".github" / "workflows" / "wheels.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


class VerifyWorkflowSourceTests(unittest.TestCase):
    def test_shared_source_helper_and_regressions_trigger_wheels(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        paths = workflow.split("  pull_request:", 1)[1].split("  workflow_dispatch:", 1)[0]
        for path in (
            "scripts/_qualification_git.py",
            "scripts/tests/test_qualification_git_context.py",
            "scripts/tests/test_release_source_git_context.py",
        ):
            with self.subTest(path=path):
                self.assertIn(f'      - "{path}"', paths)

    def test_supply_chain_gate_avoids_docker_hub_without_weakening_checks(self):
        workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        start = workflow.index("  cargo-deny:")
        end = workflow.index("  workflow-lint:", start)
        job = workflow[start:end]
        self.assertNotIn("EmbarkStudios/cargo-deny-action@", job)
        self.assertIn(
            "uses: taiki-e/install-action@742a3317eac7bd62f91cd888b4eead5e784ba833",
            job,
        )
        self.assertIn("tool: cargo-deny@0.20.2", job)
        self.assertIn('checksum: "true"', job)
        self.assertIn("fallback: none", job)
        self.assertIn(
            "cargo deny --locked --all-features check licenses bans sources advisories",
            job,
        )
        self.assertNotIn("continue-on-error", job)

    def test_workflow_checks_every_checkout_before_building_or_admitting(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            "TRITIUM_SOURCE_REF: ${{ inputs.source_ref || github.event.pull_request.head.sha || github.ref }}",
            workflow,
        )
        self.assertIn(
            "TRITIUM_SOURCE_REVISION: ${{ inputs.source_revision || github.event.pull_request.head.sha || github.sha }}",
            workflow,
        )
        self.assertNotIn("--source-revision ${{ inputs.source_revision", workflow)
        self.assertIn("test exact source identity contract", workflow)
        blocks = workflow.split("      - uses: actions/checkout@")
        self.assertGreaterEqual(len(blocks), 2)
        for block in blocks[1:]:
            steps = block.split("      - ", 1)
            checkout = steps[0]
            remainder = steps[1] if len(steps) == 2 else ""
            self.assertIn("ref: ${{ env.TRITIUM_SOURCE_REF }}", checkout)
            self.assertTrue(
                remainder.startswith("name: verify checked-out source revision"),
                "every checkout must be followed immediately by source verification",
            )
            self.assertIn("verify-workflow-source.py", remainder.split("      - ", 1)[0])

    def test_ci_artifact_producers_bind_receipts_to_the_branch_source(self):
        workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            "TRITIUM_SOURCE_REF: ${{ github.event.pull_request.head.sha || github.ref }}",
            workflow,
        )
        self.assertIn(
            "TRITIUM_SOURCE_REVISION: ${{ github.event.pull_request.head.sha || github.sha }}",
            workflow,
        )

        for job_name in ("web-package", "wasm", "publish-check"):
            start = workflow.index(f"  {job_name}:")
            next_job = re.search(r"\n  [a-zA-Z][\w-]*:\n", workflow[start + 1 :])
            end = start + 1 + next_job.start() if next_job else len(workflow)
            block = workflow[start:end]
            self.assertIn("ref: ${{ env.TRITIUM_SOURCE_REF }}", block, job_name)
            self.assertIn("name: verify checked-out source revision", block, job_name)
            self.assertIn("verify-workflow-source.py", block, job_name)
            self.assertIn("$TRITIUM_SOURCE_REVISION", block, job_name)
            self.assertNotIn("${{ github.sha }}", block, job_name)

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
