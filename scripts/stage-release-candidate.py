#!/usr/bin/env python3
"""Split downloaded release artifacts from qualification receipts safely."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any


class StageError(ValueError):
    """Downloaded artifacts do not match the frozen workflow layout."""


WHEEL_RECEIPTS = {
    "linux-x86_64-cpu.json",
    "macos-arm64-cpu.json",
    "windows-x86_64-cpu.json",
}
PLATFORMS = (
    "manylinux_2_28_x86_64",
    "macosx_11_0_arm64",
    "win_amd64",
)
COMPATIBILITY_RECEIPT = "python-abi3-39-plus.json"
CRATE_RECEIPT = "crate-archive-qualification.json"
NPM_RECEIPT = "npm-archive-receipt.json"


def _ordinary_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise StageError(f"{label} must be an ordinary directory: {path}")
    return path


def _inventory(directory: Path, label: str) -> dict[str, Path]:
    directory = _ordinary_directory(directory, label)
    result: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise StageError(f"{label} contains a non-regular file: {path.name}")
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise StageError(f"cannot inspect {label} file: {path.name}") from error
        if not stat.S_ISREG(mode):
            raise StageError(f"{label} contains a non-regular file: {path.name}")
        if path.name in result:
            raise StageError(f"{label} contains duplicate filename: {path.name}")
        result[path.name] = path
    return result


def _child_directories(directory: Path, expected: set[str], label: str) -> None:
    directory = _ordinary_directory(directory, label)
    names: set[str] = set()
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise StageError(f"{label} contains an unexpected file: {path.name}")
        names.add(path.name)
    if names != expected:
        raise StageError(f"{label} directory inventory differs")


def _require_names(files: dict[str, Path], expected: set[str], label: str) -> None:
    if set(files) != expected:
        missing = sorted(expected - set(files))
        unknown = sorted(set(files) - expected)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unexpected " + ", ".join(unknown))
        raise StageError(f"{label} inventory differs: {'; '.join(details)}")


def stage(downloads: Path, payload: Path, evidence: Path) -> dict[str, Any]:
    """Create fresh, disjoint output directories with closed file inventories."""
    downloads = _ordinary_directory(downloads, "downloads")
    _child_directories(
        downloads, {"wheels", "package-evidence", "compatibility-evidence"}, "downloads"
    )
    _child_directories(downloads / "package-evidence", {"crates", "npm"}, "package evidence")
    payload = Path(os.path.abspath(payload))
    evidence = Path(os.path.abspath(evidence))
    if payload == evidence or payload in evidence.parents or evidence in payload.parents:
        raise StageError("payload and evidence output directories must be disjoint")
    if (
        payload == downloads
        or downloads in payload.parents
        or evidence == downloads
        or downloads in evidence.parents
    ):
        raise StageError("output directories must be outside the downloads tree")
    if payload.exists() or payload.is_symlink() or evidence.exists() or evidence.is_symlink():
        raise StageError("payload and evidence output paths must not already exist")
    if (
        not payload.parent.is_dir()
        or payload.parent.resolve() != payload.parent
        or payload.parent.is_symlink()
    ):
        raise StageError("payload output parent must be an ordinary directory")
    if (
        not evidence.parent.is_dir()
        or evidence.parent.resolve() != evidence.parent
        or evidence.parent.is_symlink()
    ):
        raise StageError("evidence output parent must be an ordinary directory")

    wheels = _inventory(downloads / "wheels", "wheel artifacts")
    crates = _inventory(downloads / "package-evidence" / "crates", "crate artifacts")
    npm = _inventory(downloads / "package-evidence" / "npm", "npm artifacts")
    compatibility = _inventory(
        downloads / "compatibility-evidence", "compatibility evidence"
    )

    wheel_assets = {
        name for name in wheels if name.endswith((".whl", ".cdx.json"))
    }
    wheel_receipts = set(wheels) - wheel_assets
    if len([name for name in wheel_assets if name.endswith(".whl")]) != 3:
        raise StageError("wheel inventory must contain exactly three platform wheels")
    if len([name for name in wheel_assets if name.endswith(".cdx.json")]) != 3:
        raise StageError("wheel inventory must contain exactly three SBOMs")
    wheel_names = [name for name in wheel_assets if name.endswith(".whl")]
    for platform in PLATFORMS:
        if sum(name.endswith(f"-{platform}.whl") for name in wheel_names) != 1:
            raise StageError(f"wheel inventory must contain exactly one {platform} wheel")
    if wheel_receipts != WHEEL_RECEIPTS:
        _require_names(wheels, wheel_assets | WHEEL_RECEIPTS, "wheel artifacts")

    crate_assets = {name for name in crates if name.endswith((".crate", ".cdx.json"))}
    crate_receipts = set(crates) - crate_assets
    if not any(name.endswith(".crate") for name in crate_assets):
        raise StageError("crate inventory contains no .crate archives")
    if len([name for name in crate_assets if name.endswith(".cdx.json")]) != len(
        [name for name in crate_assets if name.endswith(".crate")]
    ):
        raise StageError("every crate archive must have one SBOM")
    if crate_receipts != {CRATE_RECEIPT}:
        _require_names(crates, crate_assets | {CRATE_RECEIPT}, "crate artifacts")

    npm_assets = {name for name in npm if name.endswith((".tgz", ".cdx.json"))}
    npm_receipts = set(npm) - npm_assets
    if len([name for name in npm_assets if name.endswith(".tgz")]) != 1:
        raise StageError("npm inventory must contain exactly one archive")
    if len([name for name in npm_assets if name.endswith(".cdx.json")]) != 1:
        raise StageError("npm archive must have exactly one SBOM")
    if npm_receipts != {NPM_RECEIPT}:
        _require_names(npm, npm_assets | {NPM_RECEIPT}, "npm artifacts")

    _require_names(compatibility, {COMPATIBILITY_RECEIPT}, "compatibility evidence")

    payload_names = wheel_assets | crate_assets | npm_assets
    payload_count = len(wheel_assets) + len(crate_assets) + len(npm_assets)
    if len(payload_names) != payload_count:
        raise StageError("candidate payload contains colliding artifact or SBOM filenames")

    created_payload = created_evidence = False
    try:
        payload.mkdir()
        created_payload = True
        evidence.mkdir()
        created_evidence = True
        evidence_dirs = {
            "wheels": evidence / "wheels",
            "crates": evidence / "crates",
            "npm": evidence / "npm",
            "compatibility": evidence / "compatibility",
        }
        for directory in evidence_dirs.values():
            directory.mkdir()
        for name in sorted(wheel_assets):
            shutil.copyfile(wheels[name], payload / name)
        for name in sorted(crate_assets):
            shutil.copyfile(crates[name], payload / name)
        for name in sorted(npm_assets):
            shutil.copyfile(npm[name], payload / name)
        for name in sorted(wheel_receipts):
            shutil.copyfile(wheels[name], evidence_dirs["wheels"] / name)
        shutil.copyfile(crates[CRATE_RECEIPT], evidence_dirs["crates"] / CRATE_RECEIPT)
        shutil.copyfile(npm[NPM_RECEIPT], evidence_dirs["npm"] / NPM_RECEIPT)
        shutil.copyfile(
            compatibility[COMPATIBILITY_RECEIPT],
            evidence_dirs["compatibility"] / COMPATIBILITY_RECEIPT,
        )
    except Exception:
        if created_evidence:
            shutil.rmtree(evidence, ignore_errors=True)
        if created_payload:
            shutil.rmtree(payload, ignore_errors=True)
        raise

    return {
        "schema": "tritium.release-candidate-staging.v1",
        "payload_files": sum(len(names) for names in (wheel_assets, crate_assets, npm_assets)),
        "evidence_files": len(wheel_receipts) + 3,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = stage(args.downloads, args.payload, args.evidence)
    except (OSError, StageError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
