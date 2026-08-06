#!/usr/bin/env python3
"""Run and seal exact-wheel physical CUDA dispatcher qualification."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import tempfile
from typing import Any


VERIFY = runpy.run_path(
    Path(__file__).with_name("verify-torch-dispatch-cuda-receipt.py")
)
SCHEMA = VERIFY["SCHEMA"]
SOURCE_PATH = VERIFY["SOURCE_PATH"]
CUDA_TESTS = VERIFY["CUDA_TESTS"]
MEMCHECK_TESTS = VERIFY["MEMCHECK_TESTS"]
canonical = VERIFY["canonical"]
sha256 = VERIFY["sha256"]
validate_receipt = VERIFY["validate"]
validate_junit = VERIFY["_junit"]


class QualificationError(ValueError):
    """Physical CUDA qualification could not produce admissible evidence."""


def file_record(path: Path) -> dict[str, object]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise QualificationError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def require_clean_revision(repo: Path, revision: str) -> Path:
    repo = repo.resolve(strict=True)
    if git_output(repo, "rev-parse", "HEAD") != revision:
        raise QualificationError("CUDA qualification revision is not checkout HEAD")
    if git_output(repo, "status", "--short", "--untracked-files=no"):
        raise QualificationError("CUDA qualification requires clean tracked source")
    source = repo / SOURCE_PATH
    if source.is_symlink() or not source.is_file():
        raise QualificationError("frozen dispatcher test source is absent")
    if git_output(repo, "hash-object", SOURCE_PATH) != git_output(
        repo, "rev-parse", f"HEAD:{SOURCE_PATH}"
    ):
        raise QualificationError("dispatcher test source differs from committed revision")
    return source


def parse_sanitizer_version(output: str) -> str:
    matches = re.findall(r"^Version ([0-9]+(?:\.[0-9]+){3})(?:\s|$)", output, re.MULTILINE)
    if len(matches) != 1:
        raise QualificationError("compute-sanitizer version output is not canonical")
    return matches[0]


def require_zero_sanitizer_errors(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise QualificationError("compute-sanitizer log is absent")
    summaries = [
        int(value)
        for value in re.findall(rb"ERROR SUMMARY:\s*([0-9]+) errors", path.read_bytes())
    ]
    if summaries != [0]:
        raise QualificationError("compute-sanitizer did not report exactly one zero-error summary")


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )
    if result.returncode != 0:
        detail = (result.stdout + "\n" + result.stderr).strip()
        raise QualificationError(
            f"qualification command exited {result.returncode}: {detail[-4000:]}"
        )
    return result


def probe_environment(
    python: Path, wheel: Path, smoke_script: Path, work: Path, environment: dict[str, str]
) -> dict[str, Any]:
    code = r'''
import importlib.metadata
import json
from pathlib import Path
import platform
import runpy
import subprocess
import sys

wheel = Path(sys.argv[1]).resolve(strict=True)
smoke = runpy.run_path(sys.argv[2])
digest = smoke["_sha256"](wheel)
version, files = smoke["installed_distribution_identity"](wheel, digest)
import torch
import tritium
from tritium import _tritium as native
for path in (Path(tritium.__file__), Path(native.__file__)):
    smoke["require_distribution_file"](path, files)
if not torch.cuda.is_available() or "cuda" not in native.compiled_backends():
    raise RuntimeError("physical CUDA and CUDA-enabled Tritium wheel are required")
source_identity = native.source_identity()
index = torch.cuda.current_device()
properties = torch.cuda.get_device_properties(index)
driver = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", f"--id={index}"],
    text=True,
    timeout=30,
).strip()
print(json.dumps({
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "tritium_version": version,
    "cuda_runtime": torch.version.cuda,
    "cuda_driver": driver,
    "source_identity": source_identity,
    "device": {
        "index": index,
        "uuid": str(properties.uuid),
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
    },
}, sort_keys=True))
'''
    result = _run(
        [str(python), "-c", code, str(wheel), str(smoke_script)],
        cwd=work,
        environment=environment,
    )
    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise QualificationError("installed-wheel probe did not return one JSON object") from error
    if not isinstance(probe, dict):
        raise QualificationError("installed-wheel probe result is not an object")
    return probe


def run_pytest(
    python: Path,
    source: Path,
    selector: str,
    junit: Path,
    work: Path,
    environment: dict[str, str],
) -> None:
    _run(
        [
            str(python), "-m", "pytest", "-q", "--import-mode=importlib",
            str(source), "-k", selector, f"--junitxml={junit}",
        ],
        cwd=work,
        environment=environment,
    )


def run_memcheck(
    sanitizer: Path,
    python: Path,
    source: Path,
    junit: Path,
    log: Path,
    work: Path,
    environment: dict[str, str],
) -> None:
    selector = " or ".join(MEMCHECK_TESTS)
    _run(
        [
            str(sanitizer), "--tool", "memcheck", "--target-processes", "all",
            "--error-exitcode", "86", "--log-file", str(log),
            str(python), "-m", "pytest", "-q", "--import-mode=importlib",
            str(source), "-k", selector, f"--junitxml={junit}",
        ],
        cwd=work,
        environment=environment,
    )


def assemble(
    stage: Path,
    *,
    wheel: Path,
    source: Path,
    source_revision: str,
    release: str,
    run_id: str,
    probe: dict[str, Any],
    suite_junit: Path,
    memcheck_junit: Path,
    sanitizer_log: Path,
    sanitizer_version: str,
) -> dict[str, Any]:
    if not run_id:
        raise QualificationError("run id must be non-empty")
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise QualificationError("source revision must be a full lowercase Git ID")
    if re.fullmatch(r"1\.1\.0-rc\.(0|[1-9][0-9]*)", release) is None:
        raise QualificationError("release must be a canonical v1.1 candidate")
    if probe.get("source_identity") != f"source-git:{source_revision}":
        raise QualificationError("installed extension source identity differs from candidate")
    require_zero_sanitizer_errors(sanitizer_log)
    validate_junit(suite_junit, CUDA_TESTS, "suite.junit")
    validate_junit(memcheck_junit, MEMCHECK_TESTS, "sanitizer.junit")

    retained = {
        source: stage / Path(SOURCE_PATH).name,
        suite_junit: stage / "suite-junit.xml",
        memcheck_junit: stage / "memcheck-junit.xml",
        sanitizer_log: stage / "compute-sanitizer.log",
    }
    for origin, destination in retained.items():
        shutil.copyfile(origin, destination)
    retained_source = retained[source]
    source_bytes = retained_source.read_bytes()
    source_blob = hashlib.sha1(
        f"blob {len(source_bytes)}\0".encode() + source_bytes
    ).hexdigest()
    environment_fields = {
        field: probe[field]
        for field in (
            "python_version", "torch_version", "tritium_version", "cuda_runtime",
            "cuda_driver", "source_identity",
        )
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "receipt_id": "",
        "result": "pass",
        "release": release,
        "source_revision": source_revision,
        "run_id": run_id,
        "artifact": {"kind": "python-wheel", **file_record(wheel)},
        "environment": environment_fields,
        "device": probe["device"],
        "source": {
            "path": SOURCE_PATH,
            "git_blob": source_blob,
            **file_record(retained_source),
        },
        "suite": {
            "selector": "native_cuda",
            "tests": list(CUDA_TESTS),
            "passed": len(CUDA_TESTS),
            "junit": file_record(retained[suite_junit]),
        },
        "sanitizer": {
            "tool": "compute-sanitizer",
            "version": sanitizer_version,
            "error_summary": 0,
            "tests": list(MEMCHECK_TESTS),
            "passed": len(MEMCHECK_TESTS),
            "junit": file_record(retained[memcheck_junit]),
            "log": file_record(retained[sanitizer_log]),
        },
    }
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    (stage / "receipt.json").write_bytes(canonical(receipt) + b"\n")
    validate_receipt(stage / "receipt.json", source_revision, release, wheel)
    return receipt


def publish_directory_noreplace(stage: Path, output: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise QualificationError("renameat2 is required for no-clobber publication")
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(stage), -100, os.fsencode(output), 1) == 0:
        return
    code = ctypes.get_errno()
    if code == errno.EEXIST:
        raise QualificationError(f"output directory already exists: {output}")
    raise QualificationError(f"no-clobber publication failed: {os.strerror(code)}")


def qualify(
    output: Path,
    *,
    repo: Path,
    python: Path,
    wheel: Path,
    sanitizer: Path,
    source_revision: str,
    release: str,
    run_id: str,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise QualificationError(f"output directory already exists: {output}")
    source = require_clean_revision(repo, source_revision)
    if wheel.is_symlink() or not wheel.is_file():
        raise QualificationError("qualified wheel must be an ordinary file")
    python = python.resolve(strict=True)
    wheel = wheel.resolve(strict=True)
    sanitizer = sanitizer.resolve(strict=True)
    smoke_script = repo.resolve(strict=True) / "scripts" / "wheel-functional-smoke.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    work = Path(tempfile.mkdtemp(prefix="tritium-dispatch-cuda-"))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    try:
        wheel_snapshot = work / wheel.name
        shutil.copyfile(wheel, wheel_snapshot)
        if (
            wheel_snapshot.stat().st_size != wheel.stat().st_size
            or sha256(wheel_snapshot) != sha256(wheel)
        ):
            raise QualificationError("qualified wheel changed while snapshotting")
        probe = probe_environment(
            python, wheel_snapshot, smoke_script, work, environment
        )
        version_result = _run(
            [str(sanitizer), "--version"], cwd=work, environment=environment
        )
        version = parse_sanitizer_version(
            version_result.stdout + "\n" + version_result.stderr
        )
        suite_junit = work / "suite-junit.xml"
        memcheck_junit = work / "memcheck-junit.xml"
        sanitizer_log = work / "compute-sanitizer.log"
        run_pytest(python, source, "native_cuda", suite_junit, work, environment)
        run_memcheck(
            sanitizer, python, source, memcheck_junit, sanitizer_log, work, environment
        )
        if require_clean_revision(repo, source_revision) != source:
            raise QualificationError("dispatcher source path changed during qualification")
        receipt = assemble(
            stage,
            wheel=wheel_snapshot,
            source=source,
            source_revision=source_revision,
            release=release,
            run_id=run_id,
            probe=probe,
            suite_junit=suite_junit,
            memcheck_junit=memcheck_junit,
            sanitizer_log=sanitizer_log,
            sanitizer_version=version,
        )
        publish_directory_noreplace(stage, output)
        if validate_receipt(output / "receipt.json", source_revision, release, wheel) != receipt:
            raise QualificationError("published receipt differs after no-clobber publication")
        return receipt
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--compute-sanitizer", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = qualify(
            args.output_dir.absolute(),
            repo=args.repo,
            python=args.python,
            wheel=args.wheel,
            sanitizer=args.compute_sanitizer,
            source_revision=args.source_revision,
            release=args.release,
            run_id=args.run_id,
        )
    except (OSError, QualificationError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
