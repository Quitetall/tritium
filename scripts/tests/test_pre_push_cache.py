"""Exercise the real pre-push script with controlled Git/Cargo adapters."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


HOOK = Path(__file__).resolve().parents[2] / ".githooks" / "pre-push"


class PrePushCacheTests(unittest.TestCase):
    def run_hook(self, target):
        with tempfile.TemporaryDirectory(prefix="tritium-prepush-cache-test-") as root:
            root = Path(root)
            bins = root / "bin"
            bins.mkdir()
            cache = root / "cache"
            (cache / "tritium-prepush" / "worktree").mkdir(parents=True)
            log = root / "cargo-calls"
            git = bins / "git"
            git.write_text("#!/bin/sh\nexit 0\n")
            cargo = bins / "cargo"
            cargo.write_text(
                '#!/bin/sh\nprintf "%s|%s\\n" "$CARGO_TARGET_DIR" "$*" >> "$CALL_LOG"\n'
            )
            git.chmod(0o700)
            cargo.chmod(0o700)
            env = dict(os.environ)
            env.update(PATH=f"{bins}:{env['PATH']}", XDG_CACHE_HOME=str(cache), CALL_LOG=str(log))
            env.pop("TRITIUM_PREPUSH_FMT_ONLY", None)
            env.pop("CARGO_TARGET_DIR", None)
            if target is not None:
                env["CARGO_TARGET_DIR"] = target
            result = subprocess.run(
                ["sh", str(HOOK)],
                input="refs/heads/test " + "1" * 40 + " refs/heads/test " + "2" * 40 + "\n",
                text=True, capture_output=True, env=env, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [line.split("|", 1) for line in log.read_text().splitlines()]
            self.assertEqual([call[1] for call in calls], [
                "fmt --all --check",
                "clippy --locked --workspace --all-targets -- -D warnings",
            ])
            expected = target or str(cache / "tritium-prepush" / "target")
            self.assertEqual([call[0] for call in calls], [expected, expected])
            self.assertFalse((cache / "tritium-prepush" / "lock").exists())

    def test_explicit_project_cache_is_used_without_skipping_checks(self):
        self.run_hook("/mnt/4tb/tmp/tritium-research-target")

    def test_unset_cache_preserves_legacy_default(self):
        self.run_hook(None)

    def test_empty_cache_preserves_legacy_default(self):
        self.run_hook("")


if __name__ == "__main__":
    unittest.main()
