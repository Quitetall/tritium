#!/usr/bin/env python3
"""Verify and optionally clean-install one Tritium abi3 wheel.

The verifier is intentionally dependency-free so it can run before installing
the candidate artifact.  It validates wheel structure and RECORD integrity,
then can install the exact local wheel into an isolated virtual environment and
run a native-kernel smoke test from outside the source checkout.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import venv
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath


DIST_NAME = "tritium-torch"
WHEEL_DIST_NAME = "tritium_torch"
FORBIDDEN_SUFFIXES = (".c", ".cc", ".cpp", ".h", ".hpp", ".pyc", ".rs")
FORBIDDEN_PARTS = {".git", "__pycache__", "target"}
NATIVE_RE = re.compile(r"^tritium/_tritium(?:\.abi3\.(?:so|dylib|pyd)|\.pyd)$")
FILENAME_RE = re.compile(
    r"^tritium_torch-(?P<version>[^-]+)-cp39-abi3-(?P<platform>[^-]+)\.whl$"
)


class WheelError(ValueError):
    """The candidate wheel is malformed or violates the release contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pep440_version(cargo_version: str) -> str:
    match = re.fullmatch(r"(\d+\.\d+\.\d+)-rc\.(\d+)", cargo_version)
    return f"{match.group(1)}rc{match.group(2)}" if match else cargo_version


def _workspace_version(root: Path) -> str:
    document = (root / "Cargo.toml").read_text(encoding="utf-8")
    section = re.search(
        r"(?ms)^\[workspace\.package\]\s*$\n(?P<body>.*?)(?=^\[|\Z)", document
    )
    if section is None:
        raise WheelError("Cargo.toml is missing [workspace.package]")
    version = re.search(
        r'(?m)^version\s*=\s*"(?P<value>[^"]+)"', section.group("body")
    )
    if version is None:
        raise WheelError("workspace package version must be a non-empty string")
    return _pep440_version(version.group("value"))


def _safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        name = info.filename
        logical = PurePosixPath(name)
        if (
            not name
            or logical.is_absolute()
            or ".." in logical.parts
            or "\\" in name
            or any(part in FORBIDDEN_PARTS for part in logical.parts)
        ):
            raise WheelError(f"unsafe wheel member {name!r}")
        if name in members:
            raise WheelError(f"duplicate wheel member {name!r}")
        # Unix symlinks have file type 0120000 in the high mode bits.
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise WheelError(f"wheel member must not be a symlink: {name!r}")
        if not info.is_dir() and name.lower().endswith(FORBIDDEN_SUFFIXES):
            raise WheelError(f"source/build residue is forbidden in wheel: {name!r}")
        members[name] = info
    return members


def _metadata(
    archive: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo], name: str
):
    if name not in members or members[name].is_dir():
        raise WheelError(f"wheel is missing canonical metadata file {name!r}")
    return Parser().parsestr(archive.read(name).decode("utf-8"))


def _urlsafe_digest(algorithm: str, payload: bytes) -> str:
    if algorithm not in {"sha256", "sha384", "sha512"}:
        raise WheelError(f"RECORD hash algorithm must be SHA-256 or stronger, got {algorithm!r}")
    try:
        digest = hashlib.new(algorithm, payload).digest()
    except ValueError as error:
        raise WheelError(f"unsupported RECORD hash algorithm {algorithm!r}") from error
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _verify_record(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    record_name: str,
) -> None:
    if record_name not in members or members[record_name].is_dir():
        raise WheelError(f"wheel is missing canonical RECORD {record_name!r}")
    rows = list(csv.reader(archive.read(record_name).decode("utf-8").splitlines()))
    recorded: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise WheelError("every RECORD row must contain path, hash and size")
        name, encoded_hash, encoded_size = row
        if name in recorded:
            raise WheelError(f"duplicate RECORD path {name!r}")
        recorded.add(name)
        if name not in members or members[name].is_dir():
            raise WheelError(f"RECORD names missing/non-file member {name!r}")
        if name == record_name:
            if encoded_hash or encoded_size:
                raise WheelError("RECORD must leave its own hash and size empty")
            continue
        if not encoded_hash or not encoded_size:
            raise WheelError(f"RECORD entry lacks integrity data: {name!r}")
        try:
            algorithm, expected = encoded_hash.split("=", 1)
            size = int(encoded_size)
        except (ValueError, TypeError) as error:
            raise WheelError(f"invalid RECORD integrity data for {name!r}") from error
        payload = archive.read(name)
        if size != len(payload):
            raise WheelError(f"RECORD size mismatch for {name!r}")
        if _urlsafe_digest(algorithm, payload) != expected:
            raise WheelError(f"RECORD hash mismatch for {name!r}")
    files = {name for name, info in members.items() if not info.is_dir()}
    if recorded != files:
        missing = sorted(files - recorded)
        extra = sorted(recorded - files)
        raise WheelError(f"RECORD coverage mismatch; missing={missing}, extra={extra}")


def inspect_wheel(path: Path, expected_version: str) -> dict[str, object]:
    match = FILENAME_RE.fullmatch(path.name)
    if match is None:
        raise WheelError("wheel filename must be tritium_torch-VERSION-cp39-abi3-PLATFORM.whl")
    if match.group("version") != expected_version:
        raise WheelError(
            f"wheel filename version {match.group('version')!r} != {expected_version!r}"
        )
    try:
        with zipfile.ZipFile(path) as archive:
            members = _safe_members(archive)
            dist_info = f"{WHEEL_DIST_NAME}-{expected_version}.dist-info"
            dist_info_dirs = {
                PurePosixPath(name).parts[0]
                for name in members
                if PurePosixPath(name).parts
                and PurePosixPath(name).parts[0].endswith(".dist-info")
            }
            if dist_info_dirs != {dist_info}:
                raise WheelError(
                    f"wheel must contain only canonical dist-info {dist_info!r}; "
                    f"found {sorted(dist_info_dirs)}"
                )
            metadata = _metadata(archive, members, f"{dist_info}/METADATA")
            wheel = _metadata(archive, members, f"{dist_info}/WHEEL")
            if metadata.get("Name") != DIST_NAME:
                raise WheelError(f"METADATA Name must equal {DIST_NAME!r}")
            if metadata.get("Version") != expected_version:
                raise WheelError("METADATA Version does not match candidate version")
            if wheel.get("Root-Is-Purelib", "").lower() != "false":
                raise WheelError("native wheel must declare Root-Is-Purelib: false")
            tags = wheel.get_all("Tag", [])
            expected_tag = f"cp39-abi3-{match.group('platform')}"
            if expected_tag not in tags:
                raise WheelError(f"WHEEL is missing filename tag {expected_tag!r}")
            native = [name for name in members if NATIVE_RE.fullmatch(name)]
            if len(native) != 1:
                raise WheelError(f"wheel must contain one abi3 native extension; found {native}")
            if "tritium/__init__.py" not in members:
                raise WheelError("wheel is missing tritium/__init__.py")
            _verify_record(archive, members, f"{dist_info}/RECORD")
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as error:
        raise WheelError(f"cannot read wheel: {error}") from error
    return {
        "wheel": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "version": expected_version,
        "platform_tag": match.group("platform"),
    }


def resolve_wheel(path: Path) -> Path:
    if not path.is_dir():
        return path
    candidates = sorted(path.glob("*.whl"))
    if len(candidates) != 1:
        raise WheelError(
            f"wheel directory must contain exactly one wheel; found {len(candidates)}"
        )
    return candidates[0]


def qualify_target(target_id: str, platform_tag: str) -> dict[str, str]:
    host_os = sys.platform
    host_arch = platform.machine().lower()
    contracts = {
        "linux-x86_64-cpu": (
            host_os.startswith("linux") and host_arch in {"amd64", "x86_64"},
            re.fullmatch(r"(?:manylinux|musllinux).*_x86_64", platform_tag) is not None,
        ),
        "macos-arm64-cpu": (
            host_os == "darwin" and host_arch in {"aarch64", "arm64"},
            platform_tag.endswith(("_arm64", "_universal2")),
        ),
        "windows-x86_64-cpu": (
            host_os == "win32" and host_arch in {"amd64", "x86_64"},
            platform_tag == "win_amd64",
        ),
    }
    if target_id not in contracts:
        raise WheelError(f"unsupported compatibility target id {target_id!r}")
    host_matches, wheel_matches = contracts[target_id]
    if not host_matches:
        raise WheelError(
            f"host {host_os}/{host_arch} cannot qualify compatibility target {target_id!r}"
        )
    if not wheel_matches:
        raise WheelError(
            f"wheel platform {platform_tag!r} does not match compatibility target {target_id!r}"
        )
    return {"host_os": host_os, "host_arch": host_arch}


def _qualified_identity(args: argparse.Namespace, result: dict[str, object]) -> dict[str, str]:
    if not args.target_id or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.target_id):
        raise WheelError("evidence requires a canonical --target-id")
    if not args.source_revision or not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        raise WheelError("evidence requires a full lowercase --source-revision")
    return qualify_target(args.target_id, str(result["platform_tag"]))


def runtime_cell_id(
    target_id: str,
    implementation: str,
    version_info: tuple[int, int],
) -> str:
    if implementation != "CPython":
        raise WheelError(f"abi3 evidence requires CPython, got {implementation!r}")
    major, minor = version_info
    if major != 3 or minor < 9:
        raise WheelError(f"abi3 evidence requires CPython 3.9+, got {major}.{minor}")
    return f"{target_id}-cp{major}.{minor}"


def clean_install_smoke(path: Path, forbidden_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="tritium-wheel-smoke-") as raw:
        root = Path(raw)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "--isolated",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--only-binary=:all:",
                str(path.resolve()),
            ],
            cwd=root,
            check=True,
            timeout=180,
        )
        smoke = """
import pathlib
import os
import tritium
module = pathlib.Path(tritium.__file__).resolve()
source = pathlib.Path(os.environ['TRITIUM_FORBIDDEN_ROOT']).resolve()
assert source != module and source not in module.parents, (source, module)
native_name = pathlib.Path(tritium._tritium.__file__).name
assert native_name == '_tritium.pyd' or native_name.startswith('_tritium.abi3.'), native_name
out = tritium.ternary_matmul([[1.0, 2.0]], [[1, -1]], 1.0)
assert len(out) == 1 and len(out[0]) == 1 and out[0][0] < 0.0, out
"""
        smoke_path = root / "smoke.py"
        smoke_path.write_text(smoke, encoding="utf-8")
        environment_vars = os.environ.copy()
        environment_vars.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": "",
                "TRITIUM_FORBIDDEN_ROOT": str(forbidden_root.resolve()),
            }
        )
        subprocess.run(
            [str(python), "-I", str(smoke_path)],
            cwd=root,
            env=environment_vars,
            check=True,
            timeout=60,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--install-smoke", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--smoke-evidence", type=Path)
    parser.add_argument("--target-id")
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    try:
        if args.receipt and args.smoke_evidence:
            raise WheelError("choose either --receipt or --smoke-evidence")
        wheel = resolve_wheel(args.wheel)
        result = inspect_wheel(wheel, _workspace_version(args.workspace))
        if args.install_smoke:
            clean_install_smoke(wheel, args.workspace)
        if args.receipt:
            if not args.install_smoke:
                raise WheelError("compatibility receipts require --install-smoke")
            host = _qualified_identity(args, result)
            receipt = {
                "schema": "tritium.compatibility-receipt.v1",
                "target_id": args.target_id,
                "source_revision": args.source_revision,
                "passed": True,
                "install_smoke": args.install_smoke,
                **host,
                **result,
            }
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        if args.smoke_evidence:
            if not args.install_smoke:
                raise WheelError("smoke evidence requires --install-smoke")
            host = _qualified_identity(args, result)
            implementation = platform.python_implementation()
            cell_id = runtime_cell_id(
                args.target_id, implementation, (sys.version_info.major, sys.version_info.minor)
            )
            evidence = {
                "schema": "tritium.wheel-smoke.v1",
                "cell_id": cell_id,
                "target_id": args.target_id,
                "source_revision": args.source_revision,
                "passed": True,
                "python_implementation": implementation,
                "python_version": platform.python_version(),
                **host,
                **result,
            }
            args.smoke_evidence.parent.mkdir(parents=True, exist_ok=True)
            args.smoke_evidence.write_text(
                json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
            )
        print(json.dumps({"passed": True, "install_smoke": args.install_smoke, **result}))
    except (KeyError, OSError, subprocess.SubprocessError, WheelError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
