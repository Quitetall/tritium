#!/usr/bin/env python3
"""Qualify exact Tritium crate archives through a clean offline consumer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any


CRATE_SBOM = runpy.run_path(Path(__file__).with_name("generate-crate-sboms.py"))
inspect_archive = CRATE_SBOM["inspect_archive"]
CrateSbomError = CRATE_SBOM["CrateSbomError"]

SCHEMA = "tritium.crate-archive-qualification.v1"
TOP_FIELDS = {
    "schema", "receipt_id", "release", "source_revision", "run_id",
    "started_at_utc", "duration_ms", "machine", "toolchain", "commands",
    "dependency_lock_sha256", "offline", "isolated_cargo_home", "packages",
    "compiled_packages", "result",
}
MACHINE_FIELDS = {"machine_id", "system", "architecture"}
TOOLCHAIN_FIELDS = {"cargo", "rustc"}
PACKAGE_FIELDS = {"artifact_id", "name", "version", "archive", "bytes", "sha256"}
MAX_RECEIPT_BYTES = 4 * 1024 * 1024


class ArchiveError(ValueError):
    """Crate archive set is incomplete, unsafe, stale, or not offline-buildable."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(root: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(
            subprocess.check_output(
                ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
                cwd=root,
                text=True,
                timeout=120,
            )
        )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ArchiveError(f"cannot read locked workspace metadata: {error}") from error
    packages = value.get("packages") if isinstance(value, dict) else None
    if not isinstance(packages, list) or not packages:
        raise ArchiveError("workspace metadata contains no packages")
    return sorted(packages, key=lambda package: package["name"])


def _extract(archive_path: Path, destination: Path) -> Path:
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            prefix = members[0].name.split("/", 1)[0]
            for member in members:
                target = destination / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise ArchiveError(f"cannot extract {member.name}")
                    with target.open("xb") as sink:
                        shutil.copyfileobj(source, sink, 1024 * 1024)
                else:
                    raise ArchiveError(f"crate member is not regular: {member.name}")
    except (OSError, tarfile.TarError) as error:
        raise ArchiveError(f"cannot extract {archive_path.name}: {error}") from error
    root = destination / prefix
    if not root.is_dir():
        raise ArchiveError(f"crate {archive_path.name} has no package root")
    return root


def _consumer_manifest(
    packages: list[dict[str, Any]], sources: dict[str, Path]
) -> tuple[str, list[str]]:
    dependencies = []
    patches = []
    compiled = []
    for ordinal, package in enumerate(packages):
        name = package["name"]
        source = sources[name].as_posix()
        patches.append(f'"{name}" = {{ path = "{source}" }}')
        kinds = {
            kind
            for target in package.get("targets", [])
            for kind in target.get("kind", [])
        }
        if kinds.intersection({"lib", "rlib"}):
            dependencies.append(
                f'archive_{ordinal} = {{ package = "{name}", path = "{source}", '
                "default-features = false }"
            )
            compiled.append(name)
    if not dependencies:
        raise ArchiveError("archive inventory contains no library packages")
    return (
        '[package]\nname = "tritium-archive-smoke"\nversion = "0.0.0"\n'
        'edition = "2024"\npublish = false\n\n[dependencies]\n'
        + "\n".join(dependencies)
        + "\n\n[patch.crates-io]\n"
        + "\n".join(patches)
        + "\n"
    ), compiled


def qualify(
    root: Path, archives: Path, revision: str, release: str, run_id: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ArchiveError("source revision must be a full Git object ID")
    if re.fullmatch(r"1\.1\.0-rc\.(0|[1-9][0-9]*)", release) is None:
        raise ArchiveError("release must be a canonical v1.1 candidate")
    if not run_id:
        raise ArchiveError("run id must be non-empty")
    if archives.is_symlink() or not archives.is_dir():
        raise ArchiveError("archives must be an ordinary directory")
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.monotonic()
    packages = _metadata(root)
    expected_version = release
    if any(package.get("version") != expected_version for package in packages):
        raise ArchiveError("workspace package versions do not match release")
    expected_names = {
        f"{package['name']}-{expected_version}.crate" for package in packages
    }
    actual = {path.name for path in archives.glob("*.crate") if path.is_file()}
    if actual != expected_names:
        raise ArchiveError(
            f"crate archive inventory mismatch; missing={sorted(expected_names - actual)}, "
            f"extra={sorted(actual - expected_names)}"
        )
    inventory = []
    with tempfile.TemporaryDirectory(prefix="tritium-crate-qualification-") as raw:
        work = Path(raw)
        lockfile = root / "Cargo.lock"
        if lockfile.is_symlink() or not lockfile.is_file():
            raise ArchiveError("workspace Cargo.lock is unavailable")
        lock_digest = _sha256(lockfile)
        vendor = work / "vendor"
        try:
            vendor_config = subprocess.check_output(
                [
                    "cargo", "vendor", "--locked", "--versioned-dirs",
                    str(vendor),
                ],
                cwd=root,
                text=True,
                timeout=3600,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ArchiveError(f"cannot stage locked dependencies: {error}") from error
        cargo_home = work / "cargo-home"
        cargo_home.mkdir()
        (cargo_home / "config.toml").write_text(vendor_config, encoding="utf-8")
        source_parent = work / "sources"
        source_parent.mkdir()
        sources: dict[str, Path] = {}
        for package in packages:
            name = package["name"]
            archive_path = archives / f"{name}-{expected_version}.crate"
            try:
                identity = inspect_archive(
                    archive_path, name, expected_version, revision
                )
            except (OSError, CrateSbomError) as error:
                raise ArchiveError(f"crate {name} failed admission: {error}") from error
            sources[name] = _extract(archive_path, source_parent)
            inventory.append(
                {
                    "artifact_id": f"crate-{name}",
                    "name": name,
                    "version": expected_version,
                    "archive": archive_path.name,
                    "bytes": identity["bytes"],
                    "sha256": identity["sha256"],
                }
            )
        consumer = work / "consumer"
        (consumer / "src").mkdir(parents=True)
        manifest, compiled = _consumer_manifest(packages, sources)
        (consumer / "Cargo.toml").write_text(manifest, encoding="utf-8")
        (consumer / "src/lib.rs").write_text("pub fn admitted() -> bool { true }\n", encoding="utf-8")
        environment = {
            **os.environ,
            "CARGO_HOME": str(cargo_home),
            "CARGO_NET_OFFLINE": "true",
        }
        commands = (
            ["cargo", "generate-lockfile", "--offline", "--manifest-path", str(consumer / "Cargo.toml")],
            [
                "cargo", "check", "--offline", "--locked", "--all-targets",
                "--manifest-path", str(consumer / "Cargo.toml"),
                "--target-dir", str(work / "target"),
            ],
        )
        try:
            for command in commands:
                subprocess.run(
                    command, cwd=consumer, env=environment, check=True, timeout=3600
                )
        except (OSError, subprocess.SubprocessError) as error:
            raise ArchiveError(f"offline archive consumer failed: {error}") from error
    machine_material = {
        "node": platform.node(), "system": platform.system(),
        "architecture": platform.machine(),
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "release": release,
        "source_revision": revision,
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "duration_ms": (time.monotonic() - started) * 1000.0,
        "machine": {
            "machine_id": "sha256:" + hashlib.sha256(_canonical(machine_material)).hexdigest(),
            "system": platform.system(),
            "architecture": platform.machine(),
        },
        "toolchain": {
            "cargo": subprocess.check_output(["cargo", "--version"], text=True, timeout=30).strip(),
            "rustc": subprocess.check_output(["rustc", "--version"], text=True, timeout=30).strip(),
        },
        "commands": [
            ["cargo", "vendor", "--locked", "--versioned-dirs"],
            ["cargo", "generate-lockfile", "--offline"],
            ["cargo", "check", "--offline", "--locked", "--all-targets"],
        ],
        "dependency_lock_sha256": lock_digest,
        "offline": True,
        "isolated_cargo_home": True,
        "packages": inventory,
        "compiled_packages": compiled,
        "result": "pass",
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(_canonical(receipt)).hexdigest()
    return receipt


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def validate_receipt(
    path: Path, archives: Path, lockfile: Path, revision: str, release: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RECEIPT_BYTES:
        raise ArchiveError("crate receipt must be a bounded ordinary file")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArchiveError("crate receipt must contain UTF-8 JSON") from error
    if not isinstance(value, dict) or set(value) != TOP_FIELDS or value["schema"] != SCHEMA:
        raise ArchiveError("crate receipt fields or schema mismatch")
    if value["release"] != release or value["source_revision"] != revision:
        raise ArchiveError("crate receipt release identity mismatch")
    if (
        value["result"] != "pass"
        or value["offline"] is not True
        or value["isolated_cargo_home"] is not True
    ):
        raise ArchiveError("crate receipt did not pass offline qualification")
    if not isinstance(value["run_id"], str) or not value["run_id"]:
        raise ArchiveError("crate receipt run id is invalid")
    duration = value["duration_ms"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or duration <= 0:
        raise ArchiveError("crate receipt duration is invalid")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", str(value["started_at_utc"])) is None:
        raise ArchiveError("crate receipt timestamp is invalid")
    machine = value["machine"]
    if not isinstance(machine, dict) or set(machine) != MACHINE_FIELDS or re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(machine.get("machine_id", ""))
    ) is None or any(
        not isinstance(machine[field], str) or not machine[field]
        for field in ("system", "architecture")
    ):
        raise ArchiveError("crate receipt machine identity is invalid")
    toolchain = value["toolchain"]
    if not isinstance(toolchain, dict) or set(toolchain) != TOOLCHAIN_FIELDS or any(
        not isinstance(toolchain[field], str) or not toolchain[field] for field in TOOLCHAIN_FIELDS
    ):
        raise ArchiveError("crate receipt toolchain is invalid")
    packages = value["packages"]
    if not isinstance(packages, list) or not packages:
        raise ArchiveError("crate receipt package inventory is empty")
    names: set[str] = set()
    for ordinal, package in enumerate(packages):
        if not isinstance(package, dict) or set(package) != PACKAGE_FIELDS:
            raise ArchiveError(f"crate receipt package {ordinal} fields mismatch")
        name = package["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ArchiveError("crate receipt package names are invalid or duplicate")
        names.add(name)
        archive = archives / str(package["archive"])
        if archive.name != package["archive"] or archive.is_symlink() or not archive.is_file():
            raise ArchiveError("crate receipt archive path is invalid")
        if package["artifact_id"] != f"crate-{name}" or package["version"] != release:
            raise ArchiveError("crate receipt package identity mismatch")
        if type(package["bytes"]) is not int or package["bytes"] != archive.stat().st_size:
            raise ArchiveError("crate receipt archive byte count mismatch")
        if package["sha256"] != _sha256(archive):
            raise ArchiveError("crate receipt archive digest mismatch")
    compiled = value["compiled_packages"]
    if not isinstance(compiled, list) or any(
        not isinstance(name, str) or name not in names for name in compiled
    ) or len(set(compiled)) != len(compiled):
        raise ArchiveError("crate receipt compiled package inventory is invalid")
    if value["commands"] != [
        ["cargo", "vendor", "--locked", "--versioned-dirs"],
        ["cargo", "generate-lockfile", "--offline"],
        ["cargo", "check", "--offline", "--locked", "--all-targets"],
    ]:
        raise ArchiveError("crate receipt commands are not frozen offline workflow")
    if (
        lockfile.is_symlink()
        or not lockfile.is_file()
        or value["dependency_lock_sha256"] != _sha256(lockfile)
    ):
        raise ArchiveError("crate receipt dependency lock digest mismatch")
    unsigned = dict(value)
    receipt_id = unsigned.pop("receipt_id")
    expected_id = "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()
    if receipt_id != expected_id:
        raise ArchiveError("crate receipt identity mismatch")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    try:
        receipt = qualify(
            root, args.archives.resolve(strict=True), args.source_revision,
            args.release, args.run_id,
        )
        _atomic_write(args.output, receipt)
        validate_receipt(
            args.output, args.archives.resolve(strict=True),
            root / "Cargo.lock", args.source_revision, args.release,
        )
    except (OSError, subprocess.SubprocessError, ArchiveError) as error:
        print(f"qualify-crate-archives: FAIL: {error}", file=sys.stderr)
        return 1
    print(f"qualify-crate-archives: PASS: {len(receipt['packages'])} crates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
