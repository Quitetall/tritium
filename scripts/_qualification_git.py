"""Read the explicitly requested Git checkout without inherited local context."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


def run_git(repo: Path, *args: str, error_type=ValueError) -> str:
    """Isolate each child, preserve parent/global context, and bound execution."""
    environment = dict(os.environ)
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
            text=True, capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise error_type("cannot verify requested Git checkout context") from error
    if result.returncode:
        raise error_type(result.stderr.strip() or "git command failed")
    return result.stdout.strip()
