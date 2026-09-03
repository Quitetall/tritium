#!/usr/bin/env python3
"""Run an installed-wheel API probe and seal candidate-bound evidence."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile
from typing import Any
from urllib.parse import unquote, urlparse


VERIFY = runpy.run_path(Path(__file__).with_name("verify-api-signature-receipt.py"))
SCHEMA = VERIFY["SCHEMA"]
ApiSignatureError = VERIFY["ApiSignatureError"]
canonical = VERIFY["canonical"]
sha256 = VERIFY["sha256"]
validate = VERIFY["validate"]


class QualificationError(ValueError):
    """The installed API probe cannot be admitted."""


def _publish_noreplace(stage: Path, output: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise QualificationError("renameat2 is required for no-clobber publication")
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(stage), -100, os.fsencode(output), 1) == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise QualificationError(f"output directory already exists: {output}")
    raise QualificationError(f"no-clobber publication failed: {os.strerror(error)}")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise QualificationError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _require_clean(repo: Path, revision: str) -> None:
    if _git(repo, "rev-parse", "HEAD") != revision:
        raise QualificationError("API qualification source revision is not HEAD")
    if _git(repo, "status", "--short", "--untracked-files=no"):
        raise QualificationError("API qualification requires clean tracked source")


PROBE = r'''
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import shutil
import sys
from urllib.parse import unquote, urlparse

wheel, revision, release, run_id, forbidden = sys.argv[1:]
wheel = Path(wheel).resolve(strict=True)
forbidden = Path(forbidden).resolve(strict=True)
dist = importlib.metadata.distribution("pytritium")
files = dist.files
if files is None:
    raise RuntimeError("installed distribution has no file inventory")
installed = sorted(dist.locate_file(item).resolve() for item in files)
module = __import__("tritium")
native = __import__("tritium._tritium", fromlist=["source_identity"])
module_path = Path(module.__file__).resolve(strict=True)
native_path = Path(native.__file__).resolve(strict=True)
if module_path not in installed or native_path not in installed:
    raise RuntimeError("imported module is not owned by installed wheel")
direct = dist.read_text("direct_url.json")
if direct is None:
    raise RuntimeError("installed distribution lacks direct_url.json")
document = json.loads(direct)
parsed = urlparse(document.get("url", ""))
if parsed.scheme != "file" or Path(unquote(parsed.path)).resolve() != wheel:
    raise RuntimeError("installed distribution does not reference candidate wheel")
source_identity = native.source_identity()
if source_identity != "source-git:" + revision:
    raise RuntimeError("native source identity differs")
root_exports = sorted(getattr(module, "__all__", []))
if not root_exports or len(root_exports) != len(set(root_exports)):
    raise RuntimeError("installed root namespace is not canonical")
callable_signatures = {}
opaque_callables = []
for name in root_exports:
    value = getattr(module, name)
    if callable(value):
        try:
            signature = str(inspect.signature(value))
        except (TypeError, ValueError):
            opaque_callables.append(name)
            continue
        if "0x" in signature:
            raise RuntimeError("callable signature contains an unstable address")
        callable_signatures[name] = signature
entries = []
root = dist.locate_file(".").resolve()
for path in installed:
    payload = path.read_bytes()
    entries.append({
        "path": path.relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    })
entries.sort(key=lambda item: item["path"])
compiler_absent = all(shutil.which(name) is None for name in (
    "cc", "c++", "gcc", "g++", "clang", "clang++", "cargo", "rustc"
))
source_tree_absent = all(
    not str(Path(item).resolve()).startswith(str(forbidden) + "/")
    for item in sys.path if item
) and not str(module_path).startswith(str(forbidden) + "/")
print(json.dumps({
    "schema": "tritium.installed-api-signature-trace.v1",
    "result": "complete",
    "release": release,
    "source_revision": revision,
    "run_id": run_id,
    "wheel": {
        "name": wheel.name,
        "bytes": wheel.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest(),
    },
    "runtime": {
        "python_version": platform.python_version(),
        "distribution_version": dist.version,
        "source_identity": source_identity,
        "module_path": str(module_path),
        "native_module_path": str(native_path),
        "wheel_file_count": len(entries),
        "installed_file_count": len(entries),
        "installed_tree_sha256": "sha256:" + hashlib.sha256(
            json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    },
    "environment": {
        "source_tree_absent": source_tree_absent,
        "compiler_absent": compiler_absent,
        "network_mode": "offline",
    },
    "signature": {
        "root_exports": root_exports,
        "callable_signatures": callable_signatures,
        "opaque_callables": sorted(opaque_callables),
    },
}, sort_keys=True))
'''


def _probe(
    *, python: Path, wheel: Path, source_revision: str, release: str,
    run_id: str, forbidden_root: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tritium-api-probe-") as raw:
        result = subprocess.run(
            [
                str(python), "-I", "-c", PROBE, str(wheel), source_revision,
                release, run_id, str(forbidden_root),
            ],
            cwd=raw, text=True, capture_output=True, check=False, timeout=120,
            # Preserve a venv launcher symlink. Resolving it to the system
            # interpreter drops that venv's site-packages under ``-I`` and
            # makes an installed-wheel probe report a false missing
            # distribution.
            env={**os.environ, "PATH": str(python.absolute().parent)},
        )
    if result.returncode:
        raise QualificationError(
            "installed API probe failed:\n" + (result.stderr.strip() or result.stdout.strip())
        )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise QualificationError("installed API probe did not emit one JSON object") from error
    if not isinstance(value, dict):
        raise QualificationError("installed API probe emitted a non-object")
    return value


def assemble(
    stage: Path, *, python: Path, wheel: Path, api_report: Path,
    source_revision: str, release: str, run_id: str, repo: Path,
) -> dict[str, Any]:
    wheel = wheel.resolve(strict=True)
    api_report = api_report.resolve(strict=True)
    report = VERIFY["_report"](api_report, release)
    trace = _probe(
        python=python.absolute(), wheel=wheel,
        source_revision=source_revision, release=release, run_id=run_id,
        forbidden_root=repo.resolve(strict=True),
    )
    trace["api_report"] = report
    trace_path = stage / "trace.json"
    trace_path.write_bytes(canonical(trace) + b"\n")
    receipt: dict[str, Any] = {
        "schema": SCHEMA, "receipt_id": "", "result": "pass",
        "release": release, "source_revision": source_revision, "run_id": run_id,
        "wheel": trace["wheel"], "api_report": report,
        "runtime": trace["runtime"], "environment": trace["environment"],
        "signature": trace["signature"],
        "trace": {
            "path": trace_path.name, "bytes": trace_path.stat().st_size,
            "sha256": "sha256:" + sha256(trace_path),
        },
    }
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    receipt_path = stage / "receipt.json"
    receipt_path.write_bytes(canonical(receipt) + b"\n")
    validate(
        receipt_path, expected_revision=source_revision, expected_release=release,
        expected_wheel=wheel, expected_api_report=api_report,
    )
    return receipt


def qualify(
    output_dir: Path, *, repo: Path, python: Path, wheel: Path, api_report: Path,
    source_revision: str, release: str, run_id: str,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise QualificationError(f"output directory already exists: {output_dir}")
    repo = repo.resolve(strict=True)
    _require_clean(repo, source_revision)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        receipt = assemble(
            stage, python=python, wheel=wheel, api_report=api_report,
            source_revision=source_revision, release=release, run_id=run_id, repo=repo,
        )
        _publish_noreplace(stage, output_dir)
        published = validate(
            output_dir / "receipt.json", expected_revision=source_revision,
            expected_release=release, expected_wheel=wheel, expected_api_report=api_report,
        )
        if published != receipt:
            raise QualificationError("published API receipt changed after publication")
        return receipt
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--api-report", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify(
        args.output_dir.absolute(), repo=args.repo.absolute(), python=args.python.absolute(),
        wheel=args.wheel.absolute(), api_report=args.api_report.absolute(),
        source_revision=args.source_revision, release=args.release,
        run_id=args.run_id,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
