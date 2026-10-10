"""Real disposable Git repositories, not empirical qualification fixtures."""

import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
NAMES = (
    "qualify-estimator-catalog.py", "qualify-refinement-campaign.py",
    "qualify-baseline-ablation.py", "qualify-onnx-inference.py",
    "qualify-training-backends.py", "qualify-training-performance.py",
    "qualify-browser-training.py", "qualify-zoo-community.py",
    "qualify-torch-dispatch-overhead.py", "qualify-torch-dispatch-cuda.py",
    "qualify-api-signature.py", "qualify-hf-distributed.py",
)
MODULES = {name: runpy.run_path(ROOT / "scripts" / name) for name in NAMES}


def _git(repo, *args):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgSign=false",
         "-c", "core.fsmonitor=false", *args], cwd=repo,
        env=environment, capture_output=True, text=True, timeout=30, check=True,
    ).stdout.strip()


def _fixture(root):
    origin = root / "origin"
    origin.mkdir()
    _git(origin, "init", "-q")
    (origin / "tracked").write_text("original\n")
    source = origin / MODULES["qualify-torch-dispatch-cuda.py"]["SOURCE_PATH"]
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic frozen-source fixture only\n")
    _git(origin, "add", ".")
    _git(origin, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "-qm", "initial fixture")
    revision = _git(origin, "rev-parse", "HEAD")
    target = root / "linked-target"
    _git(origin, "worktree", "add", "--detach", "-q", str(target), revision)
    return origin, target, revision


def _check(module, target, revision):
    function = module.get("require_clean_revision", module.get("_require_clean"))
    return function(target, revision)


def _selectors(origin, mode):
    if mode == "configuration":
        return {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.worktree",
                "GIT_CONFIG_VALUE_0": str(origin)}
    return {"GIT_DIR": str(origin / ".git"), "GIT_WORK_TREE": str(origin),
            "GIT_INDEX_FILE": str(origin / ".git/index"),
            "GIT_COMMON_DIR": str(origin / ".git"),
            "GIT_OBJECT_DIRECTORY": str(origin / ".git/objects")}


class QualificationGitContextTests(unittest.TestCase):
    def test_ptq_frontend_clis_reject_dirty_requested_checkout_before_output(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            origin, target, revision = _fixture(root)
            (target / "tracked").write_text("dirty requested source\n")
            specifications = (
                ("qualify-estimator-catalog.py", [
                    "--wheel", str(root / "missing.whl"), "--python", sys.executable,
                ]),
                ("qualify-refinement-campaign.py", [
                    "--trace", str(root / "missing-trace.json"), "--parent-artifact-id", "parent",
                ]),
                ("qualify-baseline-ablation.py", [
                    "--trace", str(root / "missing-trace.json"), "--model-artifact-id", "model",
                ]),
            )
            for name, extra in specifications:
                with self.subTest(qualifier=name):
                    output = root / "never-published" / name
                    result = subprocess.run(
                        [sys.executable, "-B", str(ROOT / "scripts" / name),
                         "--repo", str(target), "--source-revision", revision,
                         "--candidate", str(root / "missing-candidate.json"),
                         "--release", "1.1.0-rc.2", "--run-id", "negative-source-context",
                         "--output-dir", str(output), *extra],
                        env=dict(os.environ, **_selectors(origin, "selectors")),
                        text=True, capture_output=True, timeout=30,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("requires clean tracked source", result.stderr)
                    self.assertFalse(output.parent.exists())

    def test_helper_timeouts_fail_closed(self):
        helper = runpy.run_path(ROOT / "scripts/_qualification_git.py")["run_git"]
        for call in (1, 2):
            calls = []

            def run(*args, **kwargs):
                calls.append(kwargs)
                self.assertEqual(kwargs["timeout"], 30)
                if len(calls) == call:
                    raise subprocess.TimeoutExpired(args[0], 30)
                return subprocess.CompletedProcess(
                    args[0], 0, stdout="GIT_DIR\nGIT_WORK_TREE\nGIT_INDEX_FILE\n", stderr="",
                )

            with self.subTest(call=call), mock.patch.object(subprocess, "run", run):
                with self.assertRaisesRegex(ValueError, "requested Git checkout"):
                    helper(ROOT, "rev-parse", "HEAD")
                self.assertEqual(len(calls), call)

    def test_malformed_selector_discovery_fails_closed(self):
        helper = runpy.run_path(ROOT / "scripts/_qualification_git.py")["run_git"]
        for names in ("", "GIT_DIR\n", "GIT_DIR\nGIT_WORK_TREE\nGIT_INDEX_FILE\nforeign"):
            with self.subTest(names=names), mock.patch.object(
                subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=names),
            ) as run:
                with self.assertRaisesRegex(ValueError, "discover Git local"):
                    helper(ROOT, "rev-parse", "HEAD")
                self.assertEqual(run.call_count, 1)

    def test_dirty_linked_checkout_is_not_hidden_by_foreign_context(self):
        with tempfile.TemporaryDirectory() as raw:
            origin, target, revision = _fixture(Path(raw))
            (target / "tracked").write_text("dirty target\n")
            for staged in (False, True):
                if staged:
                    _git(target, "add", "tracked")
                for mode in ("selectors", "configuration"):
                    with mock.patch.dict(os.environ, _selectors(origin, mode)):
                        for name, module in MODULES.items():
                            with self.subTest(qualifier=name, staged=staged, context=mode):
                                with self.assertRaises(module["QualificationError"]):
                                    _check(module, target, revision)
            self.assertEqual(_git(origin, "status", "--porcelain"), "")
            self.assertEqual(_git(origin, "rev-parse", "HEAD"), revision)

    def test_requested_clean_checkout_and_revision_are_authoritative(self):
        with tempfile.TemporaryDirectory() as raw:
            origin, target, revision = _fixture(Path(raw))
            (origin / "tracked").write_text("different origin commit\n")
            _git(origin, "add", "tracked")
            _git(origin, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "new origin fixture")
            foreign_revision = _git(origin, "rev-parse", "HEAD")
            self.assertNotEqual(revision, foreign_revision)
            for mode in ("selectors", "configuration"):
                with mock.patch.dict(os.environ, _selectors(origin, mode)):
                    before = dict(os.environ)
                    for name, module in MODULES.items():
                        with self.subTest(qualifier=name, context=mode, expected="accept target"):
                            _check(module, target, revision)
                        with self.subTest(qualifier=name, context=mode, expected="reject origin"):
                            with self.assertRaises(module["QualificationError"]):
                                _check(module, target, foreign_revision)
                    self.assertEqual(dict(os.environ), before)
            self.assertEqual(_git(target, "status", "--porcelain"), "")
            self.assertEqual(_git(target, "rev-parse", "HEAD"), revision)


if __name__ == "__main__":
    unittest.main()
