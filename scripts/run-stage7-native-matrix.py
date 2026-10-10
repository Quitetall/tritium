#!/usr/bin/env python3
"""Run, sanitize, and seal the source-bound Stage-7 native CUDA matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "effd688a12921b4cc83e3312b6feb579f70f9c71"
RELEASE = "1.1.0-rc.1"
QUALIFIER_PATH = ROOT / "scripts/qualify-stage7-recipe-freeze.py"
_run_git = runpy.run_path(ROOT / "scripts/_qualification_git.py")["run_git"]


def _load_qualifier():
    spec = importlib.util.spec_from_file_location("stage7_qualifier", QUALIFIER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Stage-7 qualifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(*args: str) -> str:
    return _run_git(ROOT, *args, error_type=RuntimeError)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_immutable(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(output: Path, *, device: int = 0, target_dir: Path | None = None) -> dict:
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_name(f"{output.stem}.sanitizer.log")
    measurements_path = output.with_name(f"{output.stem}.measurements.json")
    for path in (output, log_path, measurements_path):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to replace existing evidence: {path}")

    revision = _git("rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("HEAD is not a full lowercase Git revision")
    if _git("status", "--porcelain"):
        raise RuntimeError("native qualification requires a clean source worktree")

    version_result = subprocess.run(
        ["compute-sanitizer", "--version"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    version_text = version_result.stdout + version_result.stderr
    match = re.search(r"Version\s+([^\s]+)", version_text)
    if match is None:
        raise RuntimeError("could not parse Compute Sanitizer version")
    sanitizer_version = f"Compute Sanitizer {match.group(1)}"

    env = os.environ.copy()
    env["RUSTC_WRAPPER"] = ""
    env.setdefault("CARGO_BUILD_JOBS", "1")
    if target_dir is not None:
        target_dir = target_dir.absolute()
        env["CARGO_TARGET_DIR"] = str(target_dir)
    elif env.get("CARGO_TARGET_DIR"):
        target_dir = Path(env["CARGO_TARGET_DIR"])
        if not target_dir.is_absolute():
            target_dir = ROOT / target_dir
    else:
        target_dir = ROOT / "target"
        env["CARGO_TARGET_DIR"] = str(target_dir)

    subprocess.run(
        [
            "cargo", "build", "--locked", "-p", "tritium-cuda", "--features",
            "cuda", "--example", "stage7_native_matrix",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    executable = target_dir / "debug/examples/stage7_native_matrix"
    if not executable.is_file():
        raise RuntimeError(f"built native matrix executable is missing: {executable}")

    subprocess.run(
        [
            "compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99",
            "--log-file", str(log_path), str(executable),
            "--release", RELEASE,
            "--source-revision", revision,
            "--model-revision", MODEL_REVISION,
            "--sanitizer-version", sanitizer_version,
            "--sanitizer-log", log_path.name,
            "--output", str(measurements_path),
            "--device", str(device),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

    log_text = log_path.read_text(encoding="utf-8")
    summaries = re.findall(r"ERROR SUMMARY: ([0-9]+) errors", log_text)
    if len(summaries) != 1 or summaries[0] != "0":
        raise RuntimeError("Compute Sanitizer log is missing or reports errors")

    raw = json.loads(measurements_path.read_text(encoding="utf-8"))
    expected = {
        "schema": "tritium.stage7-native-kernels.v1",
        "result": "pass",
        "release": RELEASE,
        "source_revision": revision,
        "model_revision": MODEL_REVISION,
        "sanitizer_version": sanitizer_version,
        "sanitizer_log": log_path.name,
    }
    if any(raw.get(key) != value for key, value in expected.items()):
        raise RuntimeError("native measurements do not bind this run and source")
    raw["sanitizer_log"] = {
        "path": log_path.name,
        "bytes": log_path.stat().st_size,
        "sha256": _sha256(log_path),
    }
    campaign = {
        "release": RELEASE,
        "source_revision": revision,
        "model": {"revision": MODEL_REVISION},
    }
    qualifier = _load_qualifier()
    payload = (json.dumps(raw, indent=2, sort_keys=True) + "\n").encode()
    validation_path = output.with_name(f".{output.name}.{os.getpid()}.validation")
    try:
        with validation_path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if not qualifier._validate_native(validation_path, campaign):
            raise RuntimeError("native matrix qualified negative")
    finally:
        validation_path.unlink(missing_ok=True)
    _publish_immutable(output, payload)
    print(f"PASS: sealed 144 native cases to {output}")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--target-dir", type=Path)
    args = parser.parse_args()
    try:
        run(args.output, device=args.device, target_dir=args.target_dir)
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
