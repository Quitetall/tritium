"""Real Git regression for nested web verification during commit --only."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


VERIFY_GATES = Path(__file__).resolve().parents[1] / "verify-gates.sh"


class PrecommitWebGitIsolationTests(unittest.TestCase):
    def test_commit_only_preserves_parent_index_and_isolates_nested_git(self):
        self.assert_isolated_commit()

    def test_explicit_repository_environment_cannot_redirect_snapshot_patch(self):
        self.assert_isolated_commit(explicit_repository=True)

    def assert_isolated_commit(self, *, explicit_repository=False):
        with tempfile.TemporaryDirectory(prefix="tritium-hook-git-", dir=os.environ.get("TMPDIR")) as raw:
            root = Path(raw)
            repo = root / "repo"
            repo.mkdir()
            tools = root / "tools"
            tools.mkdir()
            scratch = root / "scratch"
            scratch.mkdir()
            environment = dict(os.environ, TMPDIR=str(scratch))
            for name in subprocess.check_output(
                ["git", "rev-parse", "--local-env-vars"], text=True
            ).splitlines():
                environment.pop(name, None)
            environment["PATH"] = f"{tools}{os.pathsep}{environment['PATH']}"

            def git(*arguments):
                return subprocess.run(
                    ["git", *arguments], cwd=repo, env=environment,
                    text=True, capture_output=True, timeout=60, check=True,
                ).stdout

            git("init", "-q")
            git("config", "user.name", "Tritium Test")
            git("config", "user.email", "test@tritium.invalid")
            package = repo / "packages/tritium-web"
            package.mkdir(parents=True)
            (package / "package.json").write_text('{"version":"before"}\n')
            (repo / "unrelated.txt").write_text("before\n")
            git("add", ".")
            git("commit", "-qm", "fixture")
            hooks = repo / ".git/hooks"
            hook = hooks / "pre-commit"
            hook.write_text(f'#!/bin/sh\nexec /bin/sh "{VERIFY_GATES}" precommit\n')
            hook.chmod(0o755)
            (tools / "cargo").write_text("#!/bin/sh\nexit 0\n")
            (tools / "cargo").chmod(0o755)
            # The stand-in npm command performs real nested Git operations.
            # Its purpose is hook repository isolation, not package qualification.
            (tools / "npm").write_text(
                '#!/bin/sh\nset -eu\n'
                'test -z "$(git status --porcelain)"\n'
                'mkdir nested\ngit -C nested init -q\n'
                'git -C nested config user.name "Nested Test"\n'
                'git -C nested config user.email test@tritium.invalid\n'
                'printf "nested\\n" > nested/owned.txt\n'
                'git -C nested add owned.txt\ngit -C nested commit -qm nested\n'
            )
            (tools / "npm").chmod(0o755)
            (package / "package.json").write_text('{"version":"after"}\n')
            (repo / "unrelated.txt").write_text("keep staged\n")
            git("add", "unrelated.txt")
            if explicit_repository:
                environment["GIT_DIR"] = str(repo / ".git")
                environment["GIT_WORK_TREE"] = str(repo)
            result = subprocess.run(
                ["git", "commit", "--only", "-m", "web", "--",
                 "packages/tritium-web/package.json"],
                cwd=repo, env=environment, text=True, capture_output=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(git("diff", "--cached", "--name-only").strip(), "unrelated.txt")
            self.assertEqual(git("show", "HEAD:unrelated.txt"), "before\n")
            self.assertEqual(git("show", "HEAD:packages/tritium-web/package.json"),
                             '{"version":"after"}\n')
            self.assertEqual(list(scratch.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
