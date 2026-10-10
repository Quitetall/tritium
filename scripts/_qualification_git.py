"""Read the explicitly requested Git checkout without inherited local context."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


def git_result(repo: Path, *args: str, error_type=ValueError,
               environment=None, text: bool = True):
    """Isolated, bounded Git result; nonzero is data, not proof of absence."""
    environment = dict(os.environ if environment is None else environment)
    try:
        # Ask Git itself for the supported local selectors. Discovery must not
        # inherit a malformed/foreign selector or config injection either.
        local_names = subprocess.run(
            ["git", "rev-parse", "--local-env-vars"], cwd=repo,
            env={k: v for k, v in environment.items() if not k.startswith("GIT_")},
            text=True, capture_output=True, check=True, timeout=30,
        ).stdout.splitlines()
        local_names = set(local_names)
        if (
            not {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}.issubset(local_names)
            or any(not name.startswith("GIT_") for name in local_names)
        ):
            raise error_type("cannot discover Git local context selectors")
        environment = {
            key: value for key, value in environment.items()
            if key not in local_names
            and not key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
        }
        # Source admission refers to original commit objects, not an optional
        # local replacement-ref view of the same advertised revision.
        environment["GIT_NO_REPLACE_OBJECTS"] = "1"
        result = subprocess.run(
            ["git", *args], cwd=repo, env=environment,
            text=text, capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise error_type("cannot verify requested Git checkout context") from error
    return result


def run_git(repo: Path, *args: str, error_type=ValueError) -> str:
    """Read text or reject, leaving the parent environment unchanged."""
    result = git_result(repo, *args, error_type=error_type)
    if result.returncode:
        raise error_type(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def main() -> int:
    """Bounded binary-preserving bridge for shell source-admission callers."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("git_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.git_args
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a Git command is required after --")
    try:
        result = git_result(args.repo, *command, text=False)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
