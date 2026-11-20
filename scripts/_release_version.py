"""Canonical Cargo release identity; syntax is not qualification or authorization."""

from pathlib import Path
import re
from typing import Any


_NUMBER = r"(?:0|[1-9][0-9]*)"
RELEASE_PATTERN = re.compile(
    rf"(?P<base>{_NUMBER}\.{_NUMBER}\.{_NUMBER})(?:-rc\.(?P<rc>{_NUMBER}))?"
)


def candidate_version(value: Any) -> str:
    if not isinstance(value, str) or RELEASE_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"workspace version {value!r} is not a canonical X.Y.Z or X.Y.Z-rc.N"
        )
    return value


def pypi_version(value: str) -> str:
    match = RELEASE_PATTERN.fullmatch(candidate_version(value))
    assert match is not None
    return match["base"] + (f"rc{match['rc']}" if match["rc"] is not None else "")


def workspace_version(root: Path) -> str:
    # Pure version parsing is also used by the CPython3.9 wheel-smoke lane.
    import tomllib

    cargo = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    try:
        value = cargo["workspace"]["package"]["version"]
    except (KeyError, TypeError) as error:
        raise ValueError("Cargo.toml must declare workspace.package.version") from error
    return candidate_version(value)
