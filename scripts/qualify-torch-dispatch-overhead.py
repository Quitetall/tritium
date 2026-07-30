#!/usr/bin/env python3
"""Seal raw installed-wheel dispatch timings into retained release evidence."""

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


VERIFY = runpy.run_path(
    Path(__file__).with_name("verify-torch-dispatch-overhead-receipt.py")
)
SCHEMA = VERIFY["SCHEMA"]
canonical = VERIFY["canonical"]
sha256 = VERIFY["sha256"]
aggregate_case = VERIFY["aggregate_case"]
validate_trace = VERIFY["validate_trace"]
validate_receipt = VERIFY["validate"]
DispatchOverheadError = VERIFY["DispatchOverheadError"]


class QualificationError(ValueError):
    """Raw dispatch timings cannot be sealed as release evidence."""


def publish_directory_noreplace(stage: Path, output: Path) -> None:
    """Atomically publish one directory without replacing any existing target."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise QualificationError("renameat2 is required for no-clobber publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(stage),
        -100,
        os.fsencode(output),
        1,
    )
    if result == 0:
        return
    code = ctypes.get_errno()
    if code == errno.EEXIST:
        raise QualificationError(f"output directory already exists: {output}")
    raise QualificationError(f"no-clobber publication failed: {os.strerror(code)}")


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise QualificationError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def require_clean_revision(repo: Path, revision: str) -> None:
    if git_output(repo, "rev-parse", "HEAD") != revision:
        raise QualificationError("dispatch qualification source revision is not HEAD")
    if git_output(repo, "status", "--short", "--untracked-files=no"):
        raise QualificationError(
            "dispatch qualification requires clean tracked source"
        )


def assemble(
    stage: Path,
    *,
    wheel: Path,
    trace_path: Path,
    source_revision: str,
    release: str,
    run_id: str,
    create_stage: bool = True,
) -> dict[str, Any]:
    """Copy one valid raw trace and write a self-validating receipt."""

    wheel = wheel.resolve(strict=True)
    trace_path = trace_path.resolve(strict=True)
    trace = validate_trace(
        trace_path,
        expected_revision=source_revision,
        expected_release=release,
        expected_wheel=wheel,
    )
    if trace["run_id"] != run_id:
        raise QualificationError("trace run id differs from requested run")
    if create_stage:
        stage.mkdir()
    elif stage.is_symlink() or not stage.is_dir() or any(stage.iterdir()):
        raise QualificationError("reserved stage directory changed before assembly")
    retained_trace = stage / "trace.json"
    shutil.copyfile(trace_path, retained_trace)
    measurements = [
        aggregate_case(case, ordinal)
        for ordinal, case in enumerate(trace["cases"])
    ]
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "receipt_id": "",
        "result": "pass",
        "release": release,
        "source_revision": source_revision,
        "run_id": run_id,
        "wheel": trace["wheel"],
        "policy": trace["policy"],
        "environment": trace["environment"],
        "measurements": measurements,
        "trace": {
            "path": retained_trace.name,
            "bytes": retained_trace.stat().st_size,
            "sha256": sha256(retained_trace),
        },
    }
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    receipt["receipt_id"] = (
        "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    )
    receipt_path = stage / "receipt.json"
    receipt_path.write_bytes(canonical(receipt) + b"\n")
    validate_receipt(
        receipt_path,
        expected_revision=source_revision,
        expected_release=release,
        expected_wheel=wheel,
    )
    return receipt


def qualify(
    output_dir: Path,
    *,
    repo: Path,
    wheel: Path,
    trace_path: Path,
    source_revision: str,
    release: str,
    run_id: str,
) -> dict[str, Any]:
    """Atomically qualify a trace from the exact clean source revision."""

    if output_dir.exists() or output_dir.is_symlink():
        raise QualificationError(f"output directory already exists: {output_dir}")
    repo = repo.resolve(strict=True)
    require_clean_revision(repo, source_revision)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        receipt = assemble(
            stage,
            wheel=wheel,
            trace_path=trace_path,
            source_revision=source_revision,
            release=release,
            run_id=run_id,
            create_stage=False,
        )
        publish_directory_noreplace(stage, output_dir)
        published = validate_receipt(
            output_dir / "receipt.json",
            expected_revision=source_revision,
            expected_release=release,
            expected_wheel=wheel,
        )
        if published != receipt:
            raise QualificationError("published receipt differs after no-clobber rename")
        return receipt
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify(
        args.output_dir.absolute(),
        repo=args.repo,
        wheel=args.wheel.absolute(),
        trace_path=args.trace.absolute(),
        source_revision=args.source_revision,
        release=args.release,
        run_id=args.run_id,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
