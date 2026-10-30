"""Real Git fixture checks; Cargo is stubbed, not a compilation gate."""

import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class GitHookIsolationTests(unittest.TestCase):
    def test_prepush_preserves_originating_head_and_index(self):
        cases = [(dirty, selectors, stale)
                 for dirty, selectors in ((False, False), (True, False), (False, True), (True, True))
                 for stale in (False, True)]
        for dirty, explicit_selectors, stale in cases:
            with self.subTest(dirty=dirty, explicit_selectors=explicit_selectors, stale=stale), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                source = root / "source"
                source.mkdir()
                home = root / "home"
                home.mkdir()
                tools = root / "bin"
                tools.mkdir()
                cargo = tools / "cargo"
                cargo.write_text("#!/bin/sh\nexit 0\n")
                cargo.chmod(0o755)
                env = {
                    **os.environ, "HOME": str(home),
                    "XDG_CONFIG_HOME": str(home / ".config"),
                    "XDG_CACHE_HOME": str(root / "cache"),
                    "PATH": str(tools) + os.pathsep + os.environ["PATH"],
                }
                for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
                    env.pop(name, None)

                def git(*args):
                    return subprocess.check_output(
                        ["git", *args], cwd=source, env=env, timeout=30,
                    )

                git("init", "--quiet", "--initial-branch=owner")
                git("config", "user.name", "hook fixture")
                git("config", "user.email", "fixture@invalid")
                git("config", "commit.gpgsign", "false")
                # Exercise the real public-API generator from the pushed tree.
                (source / "scripts").mkdir()
                for name in ("generate-api-diff.py", "_qualification_git.py", "check-semver.sh"):
                    shutil.copyfile(ROOT / "scripts" / name, source / "scripts" / name)
                native = source / "crates/tritium-py/src"
                native.mkdir(parents=True)
                (native / "lib.rs").write_text(
                    '#[pymodule]\nfn _tritium(m: &PyModule) {\n'
                    'm.add_class::<Model>()?;\n'
                    'm.add_function(wrap_pyfunction!(ternary_matmul, m)?)?;\n}\n'
                )
                package = source / "crates/tritium-py/python/tritium"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text('__all__ = ["Model", "ternary_matmul"]\n')
                (source / "Cargo.toml").write_text('[workspace.package]\nversion = "1.1.0-rc.2"\n')
                tracked = source / "model.txt"
                tracked.write_text("owner commit\n")
                git("add", ".")
                git("commit", "--quiet", "-m", "owner")
                git("tag", "v1.0.0")
                subprocess.run(["python", "scripts/generate-api-diff.py"], cwd=source,
                               env=env, check=True, timeout=30)
                git("add", "docs")
                git("commit", "--quiet", "-m", "generated API report")
                initial = git("rev-parse", "HEAD").decode().strip()
                git("switch", "--quiet", "-c", "candidate")
                tracked.write_text("candidate commit\n")
                if stale:
                    (package / "__init__.py").write_text('__all__ = ["Model", "ternary_matmul", "NewApi"]\n')
                git("commit", "--quiet", "-am", "candidate")
                candidate = git("rev-parse", "HEAD").decode().strip()
                git("switch", "--quiet", "owner")
                scratch = root / "cache/tritium-prepush/worktree"
                git("worktree", "add", "--quiet", "--detach", str(scratch), initial)
                if dirty:
                    tracked.write_text("foreign staged work\n")
                    git("add", "model.txt")
                before_index = git("diff", "--cached", "--binary")
                before_bytes = tracked.read_bytes()
                draft = source / "foreign-draft.txt"
                draft.write_text("foreign untracked work\n")
                hook_env = {**env, "GIT_DIR": str(source / ".git")}
                if explicit_selectors:
                    hook_env.update({
                        "GIT_WORK_TREE": str(source),
                        "GIT_INDEX_FILE": str(source / ".git/index"),
                        "GIT_COMMON_DIR": str(source / ".git"),
                    })
                completed = subprocess.run(
                    ["sh", str(ROOT / ".githooks/pre-push")],
                    input=f"refs/heads/candidate {candidate} refs/heads/candidate {initial}\n",
                    cwd=source, env=hook_env, capture_output=True, text=True, timeout=60,
                )
                if stale:
                    self.assertNotEqual(completed.returncode, 0, "stale pushed API projection was admitted")
                    self.assertIn("public API projection", completed.stderr)
                else:
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertEqual(git("branch", "--show-current").decode().strip(), "owner")
                self.assertEqual(git("rev-parse", "HEAD").decode().strip(), initial)
                self.assertEqual(git("diff", "--cached", "--binary"), before_index)
                self.assertEqual(tracked.read_bytes(), before_bytes)
                self.assertEqual(draft.read_text(), "foreign untracked work\n")


if __name__ == "__main__":
    unittest.main()
