"""Actual Git context at release/capture gates; no empirical receipts emitted."""

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts.tests.test_qualification_git_context import ROOT, _fixture, _git, _selectors


WORKFLOW = ROOT / "scripts/verify-workflow-source.py"
REPRO = runpy.run_path(ROOT / "scripts/qualify-release-reproduction.py")
BROWSER = runpy.run_path(ROOT / "scripts/produce-browser-native-reference.py")
RELEASE = runpy.run_path(ROOT / "scripts/release-status")
REF = runpy.run_path(ROOT / "scripts/resolve-release-ref.py")
API = runpy.run_path(ROOT / "scripts/generate-api-diff.py")
CUDA = runpy.run_path(ROOT / "scripts/verify-torch-dispatch-cuda-receipt.py")
STAGE_NAMES = ("run-stage7-recipe-freeze.py", "rebind-stage7-campaign.py")
STAGES = {name: runpy.run_path(ROOT / "scripts" / name) for name in STAGE_NAMES}
CAPTURE = runpy.run_path(ROOT / "scripts/capture-qwen36-from-pack.py")
NATIVE = runpy.run_path(ROOT / "scripts/run-stage7-native-matrix.py")


def _stage_qualifier(target):
    (target / "scripts").mkdir(exist_ok=True)
    for name in ("qualify-stage7-recipe-freeze.py", "_qualification_git.py"):
        destination = target / "scripts" / name
        destination.write_bytes((ROOT / "scripts" / name).read_bytes())
    _git(target, "add", "scripts")
    _git(target, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "-qm", "source-bound qualifier fixture")
    return runpy.run_path(target / "scripts/qualify-stage7-recipe-freeze.py")


class ReleaseSourceGitContextTests(unittest.TestCase):
    def test_stage_qualifier_uses_exact_requested_source_and_helper(self):
        with tempfile.TemporaryDirectory() as raw:
            origin, target, _ = _fixture(Path(raw))
            module = _stage_qualifier(target)
            revision = _git(target, "rev-parse", "HEAD")
            with mock.patch.dict(os.environ, _selectors(origin, "selectors")):
                self.assertEqual(module["_git_source"](target), revision)

    def test_stage_qualifier_rejects_replaced_dependency_even_if_clean(self):
        with tempfile.TemporaryDirectory() as raw:
            origin, target, _ = _fixture(Path(raw))
            module = _stage_qualifier(target)
            revision = _git(target, "rev-parse", "HEAD")
            helper = target / "scripts/_qualification_git.py"
            helper.write_bytes(helper.read_bytes() + b"\n# different replacement source\n")
            _git(target, "add", "scripts")
            _git(target, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "replacement dependency fixture")
            replacement = _git(target, "rev-parse", "HEAD")
            _git(target, "update-ref", "HEAD", revision)
            _git(target, "replace", revision, replacement)
            # Git's replacement view is clean, but not the original source.
            self.assertEqual(_git(target, "status", "--porcelain"), "")
            with self.assertRaises(module["Stage7Error"]):
                module["_git_source"](target)

    def test_capture_rejects_dirty_checkout_before_receipt_validation(self):
        with tempfile.TemporaryDirectory() as raw:
            origin, target, revision = _fixture(Path(raw))
            candidate = Path(raw) / "candidate.json"
            candidate.write_text(json.dumps({
                "schema": "tritium.release-candidate.v1", "release": "1.1.0-rc.9",
                "source_revision": revision,
            }))
            (target / "tracked").write_text("dirty capture source\n")
            validate = mock.Mock(return_value={"synthetic": "no qualification credit"})
            globals_ = CAPTURE["_validate_stage7_qualification"].__globals__
            with (
                mock.patch.dict(globals_, {"__file__": str(target / "scripts/capture.py")}),
                mock.patch.dict(CAPTURE["STAGE7"], {"validate": validate}),
                mock.patch.dict(os.environ, _selectors(origin, "selectors")),
            ):
                with self.assertRaisesRegex(ValueError, "clean checkout"):
                    CAPTURE["_validate_stage7_qualification"](argparse.Namespace(
                        release_candidate_manifest=candidate,
                        stage7_qualification_receipt=Path(raw) / "unused.json",
                    ))
            validate.assert_not_called()

    def test_native_matrix_rejects_dirty_checkout_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as raw:
            origin, target, _ = _fixture(Path(raw))
            (target / "tracked").write_text("dirty native source\n")
            globals_ = NATIVE["run"].__globals__
            actual_run = subprocess.run

            def only_git(command, **kwargs):
                if command[0] != "git":
                    self.fail("dirty source reached hardware/build execution")
                return actual_run(command, **kwargs)

            with (
                mock.patch.dict(globals_, {"ROOT": target}),
                mock.patch.dict(os.environ, _selectors(origin, "selectors")),
                mock.patch.object(subprocess, "run", side_effect=only_git),
            ):
                with self.assertRaisesRegex(RuntimeError, "clean source worktree"):
                    NATIVE["run"](Path(raw) / "output/receipt.json")
            self.assertFalse((Path(raw) / "output/receipt.json").exists())

    def test_workflow_cli_uses_current_checkout_not_foreign_head(self):
        with tempfile.TemporaryDirectory() as raw:
            origin, target, revision = _fixture(Path(raw))
            (origin / "tracked").write_text("different foreign revision\n")
            _git(origin, "add", "tracked")
            _git(origin, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "foreign fixture")
            foreign = _git(origin, "rev-parse", "HEAD")
            for expected, code in ((revision, 0), (foreign, 1)):
                result = subprocess.run(
                    [sys.executable, "-B", str(WORKFLOW), "--expected-revision", expected],
                    cwd=target, env=dict(os.environ, **_selectors(origin, "selectors")),
                    text=True, capture_output=True, timeout=30,
                )
                with self.subTest(expected=expected):
                    self.assertEqual(result.returncode, code, result.stderr)

    def test_browser_and_release_reject_dirty_requested_checkout(self):
        with tempfile.TemporaryDirectory() as raw:
            origin, target, revision = _fixture(Path(raw))
            (target / "tracked").write_text("dirty requested source\n")
            with mock.patch.dict(os.environ, _selectors(origin, "selectors")):
                with self.assertRaises(BROWSER["NativeReferenceError"]):
                    BROWSER["source_admission"](revision, target)
                with self.assertRaises(RELEASE["ReleaseError"]):
                    RELEASE["_git_gate"](target, revision)

    def test_stage_execution_and_rebind_accept_exact_requested_root(self):
        with tempfile.TemporaryDirectory() as raw:
            origin, target, revision = _fixture(Path(raw))
            with mock.patch.dict(os.environ, _selectors(origin, "selectors")):
                for name, module in STAGES.items():
                    with self.subTest(gate=name):
                        self.assertEqual(module["_source_identity"](target), revision)

    def test_repository_absence_is_not_inferred_from_git_error(self):
        with tempfile.TemporaryDirectory() as raw:
            origin, target, _ = _fixture(Path(raw))
            env = dict(os.environ, GIT_DIR=str(Path(raw) / "missing-git-directory"))
            # Compiler availability is deliberately excluded from this Git-only
            # admission test; this is NOT second-machine reproduction evidence.
            with mock.patch.object(REPRO["shutil"], "which", return_value=None):
                with self.assertRaises(REPRO["QualificationError"]):
                    REPRO["require_clean_environment"](target, env)
                outside = Path(raw) / "outside"
                outside.mkdir()
                REPRO["require_clean_environment"](
                    outside, dict(os.environ, **_selectors(origin, "selectors")),
                )

    def test_reproduction_rejects_unrelated_git_failures(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            globals_ = REPRO["require_clean_environment"].__globals__
            for code, message in (
                (129, "error: invalid command"),
                (128, "fatal: detected dubious ownership in repository"),
                (1, ""),
            ):
                result = subprocess.CompletedProcess(["git"], code, "", message)
                with (
                    self.subTest(code=code, message=message),
                    mock.patch.dict(globals_, {"_git_result": mock.Mock(return_value=result)}),
                ):
                    with self.assertRaisesRegex(REPRO["QualificationError"], "cannot prove"):
                        REPRO["require_clean_environment"](directory, dict(os.environ))

    def test_release_ref_resolution_does_not_read_foreign_refs(self):
        with tempfile.TemporaryDirectory() as raw:
            origin, target, revision = _fixture(Path(raw))
            tag = "v1.1.0-rc.2"
            _git(origin, "tag", tag)
            _git(origin, "update-ref", "refs/remotes/origin/main", revision)
            # Use an independent repository so the target does not share refs.
            independent = Path(raw) / "independent"
            independent.mkdir()
            _git(independent, "init", "-q")
            with mock.patch.dict(os.environ, _selectors(origin, "selectors")):
                with self.assertRaises(REF["ReleaseRefError"]):
                    REF["resolve_release_ref"](independent, tag, "main", fetch_remote=None)

    def test_api_and_cuda_git_reads_ignore_replacement_views(self):
        with tempfile.TemporaryDirectory() as raw:
            origin, target, revision = _fixture(Path(raw))
            path = CUDA["SOURCE_PATH"]
            original = _git(origin, "rev-parse", f"{revision}:{path}")
            replacement = origin / path
            replacement.write_text("replacement dispatcher source\n")
            _git(origin, "add", path)
            _git(origin, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "replacement fixture")
            _git(origin, "replace", revision, _git(origin, "rev-parse", "HEAD"))
            self.assertEqual(CUDA["git_blob_at"](target, revision), original)
            self.assertEqual(API["_git"](target, "show", f"{revision}:{path}"),
                             "# synthetic frozen-source fixture only\n")


if __name__ == "__main__":
    unittest.main()
