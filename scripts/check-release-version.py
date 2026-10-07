#!/usr/bin/env python3
"""Fail when a Tritium local-RC version mirror drifts from Cargo.toml."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


# A release version: X.Y.Z, optionally -rc.N. This used to be pinned literally to
# `1.1.0-rc.N`, which meant the workspace could not hold any other version -- and
# because this runs inside check-publish.sh, which is in the REQUIRED `publish-check`
# CI job, tagging the 1.1.0 FINAL release would have turned ci-required red on the
# release commit itself. A gate that forbids shipping is not a gate.
RELEASE_PATTERN = re.compile(
    r"(?P<base>0|[1-9][0-9]*(?:\.(?:0|[1-9][0-9]*)){2})"
    r"(?:-rc\.(?P<rc>0|[1-9][0-9]*))?"
)


def require_equal(actual: Any, expected: str, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} is {actual!r}, expected {expected!r}")


def require_contains(text: str, expected: str, label: str) -> None:
    if expected not in text:
        raise ValueError(f"{label} is missing current release text {expected!r}")


def candidate_version(value: Any) -> str:
    if not isinstance(value, str) or RELEASE_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"workspace version {value!r} is not a canonical X.Y.Z or X.Y.Z-rc.N"
        )
    return value


def pypi_version(value: str) -> str:
    """Convert the accepted Cargo prerelease spelling to the wheel spelling."""

    return re.sub(r"-rc\.(\d+)$", r"rc\1", value)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def check_document_versions(root: Path, version: str) -> None:
    readme = (root / "README.md").read_text(encoding="utf-8")
    require_contains(
        readme,
        f"cargo install tritium-cli --version {version}",
        "README CLI installation command",
    )
    require_contains(
        readme,
        f"cargo install tritium-serve --version {version}",
        "README serving installation command",
    )
    quantization = (root / "docs/book/src/quantization.md").read_text(
        encoding="utf-8"
    )
    require_contains(
        quantization,
        f"--release {version}",
        "Stage-7 quantization command release version",
    )


def check(root: Path) -> str:
    cargo_path = root / "Cargo.toml"
    cargo = tomllib.loads(cargo_path.read_text(encoding="utf-8"))
    version = candidate_version(cargo["workspace"]["package"]["version"])

    for name, dependency in cargo["workspace"]["dependencies"].items():
        if name.startswith("tritium-"):
            require_equal(dependency.get("version"), version, f"workspace dependency {name}")

    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
            cwd=root,
            text=True,
            timeout=120,
        )
    )
    for package in metadata["packages"]:
        require_equal(package["version"], version, f"Cargo package {package['name']}")
        for dependency in package["dependencies"]:
            if dependency["name"].startswith("tritium-") and dependency["path"] is not None:
                require_equal(
                    dependency["req"],
                    f"^{version}",
                    f"Cargo dependency {package['name']} -> {dependency['name']}",
                )

    cargo_lock = tomllib.loads((root / "Cargo.lock").read_text(encoding="utf-8"))
    for package in cargo_lock["package"]:
        if package["name"].startswith("tritium-"):
            require_equal(package["version"], version, f"Cargo.lock {package['name']}")

    pyproject = tomllib.loads(
        (root / "crates/tritium-py/pyproject.toml").read_text(encoding="utf-8")
    )
    dynamic = pyproject["project"].get("dynamic", [])
    if dynamic != ["version"]:
        raise ValueError("Python package version must be sourced only from Cargo metadata")
    require_equal(pyproject["project"].get("name"), "pytritium", "Python distribution name")
    require_equal(
        pyproject["tool"]["maturin"].get("manifest-path"),
        "Cargo.toml",
        "Python maturin manifest-path",
    )

    package = read_json(root / "packages/tritium-web/package.json")
    package_lock = read_json(root / "packages/tritium-web/package-lock.json")
    require_equal(package.get("version"), version, "npm package version")
    require_equal(package_lock.get("version"), version, "npm lockfile version")
    require_equal(
        package_lock.get("packages", {}).get("", {}).get("version"),
        version,
        "npm lockfile root package version",
    )

    compatibility = read_json(root / "release/compatibility-v1.1.json")
    require_equal(compatibility.get("release"), version, "compatibility release")

    wheel_version = pypi_version(version)
    tutorial = (root / "docs/book/src/tutorial-pytorch-qat.md").read_text(
        encoding="utf-8"
    )
    require_equal(
        f"TRITIUM_WHEEL=./dist/pytritium-{wheel_version}-cp39-abi3-PLATFORM.whl"
        in tutorial,
        True,
        "installed-artifact QAT tutorial wheel version",
    )
    candidate_guide = (root / "docs/release-candidate.md").read_text(
        encoding="utf-8"
    )
    require_equal(
        f'"path": "pytritium-{wheel_version}-cp39-abi3-manylinux_2_28_x86_64.whl"'
        in candidate_guide,
        True,
        "release-candidate example wheel version",
    )
    check_document_versions(root, version)
    return version


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    try:
        version = check(root)
    except (
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"release-version: FAIL: {error}", file=sys.stderr)
        return 1
    print(f"release-version: OK: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
