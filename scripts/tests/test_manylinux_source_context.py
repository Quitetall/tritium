"""Real Git and shell builders, stubbed build tools; no wheel qualification."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def git(repo, *args):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("GIT_")}
    return subprocess.run(
        ["/usr/bin/git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null",
         "-c", "commit.gpgSign=false", "-c", "core.fsmonitor=false", *args],
        cwd=repo, env=environment, check=True, capture_output=True, text=True,
        timeout=30,
    ).stdout.strip()


def fixture(root):
    origin = root / "origin"
    origin.mkdir()
    git(origin, "init", "-q")
    (origin / "tracked").write_text("original\n")
    git(origin, "add", ".")
    git(origin, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "original fixture")
    revision = git(origin, "rev-parse", "HEAD")
    target = root / "linked-target"
    git(origin, "worktree", "add", "--detach", "-q", str(target), revision)
    (origin / "tracked").write_text("different origin commit\n")
    git(origin, "add", "tracked")
    git(origin, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "foreign fixture")
    return origin, target, revision


def selectors(origin, mode):
    if mode == "configuration":
        return {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.worktree",
                "GIT_CONFIG_VALUE_0": str(origin)}
    return {"GIT_DIR": str(origin / ".git"), "GIT_WORK_TREE": str(origin),
            "GIT_INDEX_FILE": str(origin / ".git/index"),
            "GIT_COMMON_DIR": str(origin / ".git"),
            "GIT_OBJECT_DIRECTORY": str(origin / ".git/objects")}


def invoke(root, target, backend, extra=None, status_failure=False):
    # Real Git reaches the builder's preflight. All expensive/external build
    # tools are stubs, including the post-build wheel check. Python executes
    # only the shared Git helper, not actual package installation/verification.
    binary = root / f"tools-{backend}"
    binary.mkdir()
    sysroot = root / f"rust-{backend}"
    (sysroot / "bin").mkdir(parents=True)
    for name in ("rustc", "cargo"):
        executable = sysroot / "bin" / name
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    tools = {
        "rustup": '#!/bin/sh\nprintf "%s\\n" "$FIXTURE_SYSROOT"\n',
        "nvcc": '#!/bin/sh\nprintf "Cuda compilation tools, release 13.4\\n"\n',
        "python": """#!/bin/sh
case "$1" in
  */_qualification_git.py) exec /usr/bin/python3 "$@" ;;
  *) exit 0 ;;
esac
""",
        "docker": """#!/usr/bin/python3
import json, os, pathlib, sys
pathlib.Path(os.environ["FIXTURE_DOCKER_ARGS"]).write_text(json.dumps(sys.argv[1:]))
""",
    }
    if status_failure:
        tools["git"] = """#!/bin/sh
case "$*" in
  *status*) printf 'fixture status failure\\n' >&2; exit 7 ;;
  *) exec /usr/bin/git "$@" ;;
esac
"""
    for name, source in tools.items():
        executable = binary / name
        executable.write_text(source)
        executable.chmod(0o755)
    output = root / f"dist-{backend}"
    docker_args = root / f"docker-{backend}.json"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("GIT_")}
    environment.update({
        "PATH": f"{binary}:/usr/bin:/bin",
        "FIXTURE_SYSROOT": str(sysroot), "FIXTURE_DOCKER_ARGS": str(docker_args),
        f"TRITIUM_{backend.upper()}_MANYLINUX_CACHE": str(root / f"cache-{backend}"),
    })
    environment.update(extra or {})
    result = subprocess.run(
        ["bash", str(ROOT / f"scripts/build-{backend}-manylinux-wheel.sh"), str(output)],
        cwd=target, env=environment, text=True, capture_output=True, timeout=30,
    )
    return result, output, docker_args


class ManylinuxSourceContextTests(unittest.TestCase):
    def test_clean_requested_root_and_source_are_not_replaced_by_foreign_context(self):
        for backend in ("cpu", "cuda"):
            for mode in ("selectors", "configuration"):
                with self.subTest(backend=backend, context=mode), tempfile.TemporaryDirectory() as raw:
                    root = Path(raw)
                    origin, target, revision = fixture(root)
                    result, _, docker_args = invoke(root, target, backend, selectors(origin, mode))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    arguments = json.loads(docker_args.read_text())
                    self.assertIn(f"{target}:/io:ro", arguments)
                    self.assertIn(f"TRITIUM_SOURCE_ID=source-git:{revision}", arguments)

    def test_dirty_requested_source_rejects_before_output_or_build(self):
        for backend in ("cpu", "cuda"):
            for mode in ("selectors", "configuration"):
                for state in ("modified", "staged", "untracked"):
                    with self.subTest(backend=backend, context=mode, state=state), tempfile.TemporaryDirectory() as raw:
                        root = Path(raw)
                        origin, target, _ = fixture(root)
                        if state == "untracked":
                            (target / "untracked").write_text("new source\n")
                        else:
                            (target / "tracked").write_text("dirty source\n")
                            if state == "staged":
                                git(target, "add", "tracked")
                        result, output, docker_args = invoke(root, target, backend, selectors(origin, mode))
                        self.assertNotEqual(result.returncode, 0, result.stderr)
                        self.assertIn("require a clean Git worktree", result.stderr)
                        self.assertFalse(output.exists())
                        self.assertFalse(docker_args.exists())

    def test_replacement_view_cannot_admit_different_original_source_bytes(self):
        for backend in ("cpu", "cuda"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                origin, target, revision = fixture(root)
                replacement = git(origin, "rev-parse", "HEAD")
                git(origin, "replace", revision, replacement)
                (target / "tracked").write_text("different origin commit\n")
                git(target, "add", "tracked")
                # The default replacement view is clean; the original commit
                # named by HEAD still has different bytes.
                observed = subprocess.run(
                    ["/usr/bin/git", "status", "--porcelain"], cwd=target,
                    env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
                    check=True, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(observed.stdout, "")
                result, output, docker_args = invoke(root, target, backend)
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn("require a clean Git worktree", result.stderr)
                self.assertFalse(output.exists())
                self.assertFalse(docker_args.exists())

    def test_status_command_failure_is_not_empty_clean_status(self):
        for backend in ("cpu", "cuda"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                _, target, _ = fixture(root)
                result, output, docker_args = invoke(root, target, backend, status_failure=True)
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertIn("fixture status failure", result.stderr)
                self.assertFalse(output.exists())
                self.assertFalse(docker_args.exists())

    def test_shared_helper_cli_preserves_original_blob_bytes_and_failure_status(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            origin, target, revision = fixture(root)
            binary_payload = b"\x00\xfffixture\n\n"
            (origin / "binary").write_bytes(binary_payload)
            git(origin, "add", "binary")
            git(origin, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                "commit", "-qm", "binary fixture")
            binary_revision = git(origin, "rev-parse", "HEAD")
            environment = dict(os.environ, **selectors(origin, "selectors"))
            command = [sys.executable, "-B", str(ROOT / "scripts/_qualification_git.py"),
                       "--repo", str(target), "--"]
            result = subprocess.run(
                [*command, "show", f"{revision}:tracked"], env=environment,
                capture_output=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, b"original\n")
            binary = subprocess.run(
                [*command, "show", f"{binary_revision}:binary"], env=environment,
                capture_output=True, timeout=30,
            )
            self.assertEqual(binary.returncode, 0, binary.stderr)
            self.assertEqual(binary.stdout, binary_payload)
            invalid = subprocess.run(
                [*command, "rev-parse", "--verify", "refs/heads/missing-fixture-ref"],
                env=environment, capture_output=True, timeout=30,
            )
            self.assertNotEqual(invalid.returncode, 0)
            self.assertTrue(invalid.stderr)

    def test_shared_helper_cli_rejects_missing_checkout_or_command(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            command = [sys.executable, "-B", str(ROOT / "scripts/_qualification_git.py"),
                       "--repo", str(root / "absent")]
            missing_repo = subprocess.run(
                [*command, "--", "rev-parse", "HEAD"], capture_output=True, timeout=30,
            )
            self.assertNotEqual(missing_repo.returncode, 0)
            self.assertIn(b"cannot verify requested Git checkout context", missing_repo.stderr)
            missing_command = subprocess.run(command, capture_output=True, timeout=30)
            self.assertEqual(missing_command.returncode, 2)
            self.assertIn(b"a Git command is required", missing_command.stderr)


if __name__ == "__main__":
    unittest.main()
