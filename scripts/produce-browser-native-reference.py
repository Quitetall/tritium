#!/usr/bin/env python3
"""Seal source-bound native CPU reference for browser training qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIGEST = "9093a1a7f9a3422c399943782aadf4df6b11833cf2253db0db56ff2d9dedb098"
VECTOR_DIGEST = "38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7"
SCENARIO_ID = "salt-ste-sgd-256-v1"
RELEASE = "1.1.0-rc.0"
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
HEX64 = re.compile(r"[0-9a-f]{64}")
METADATA_FIELDS = {
    "backend_id",
    "backend_build_hex",
    "physical_device_hex",
    "manifest_digest",
    "export_operation",
    "export_input_digest",
    "export_output_digest",
    "export_peak_resident_bytes",
    "export_scratch_bytes",
    "export_host_transfers",
    "export_device_resident",
    "reload_operation",
    "reload_input_digest",
    "reload_output_digest",
    "reload_peak_resident_bytes",
    "reload_scratch_bytes",
    "reload_host_transfers",
    "reload_device_resident",
}


class NativeReferenceError(ValueError):
    """Native reference is stale, partial, non-resident, or byte-drifted."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def integer(value: str, label: str, *, positive: bool = False) -> int:
    if not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise NativeReferenceError(f"native {label} is not canonical")
    parsed = int(value)
    if positive and parsed <= 0:
        raise NativeReferenceError(f"native {label} must be positive")
    return parsed


def lifecycle_receipt(
    metadata: dict[str, str], prefix: str, artifact_digest: str
) -> dict[str, Any]:
    operation = metadata[f"{prefix}_operation"]
    if operation != f"lifecycle.{prefix}":
        raise NativeReferenceError("native lifecycle operation identity differs")
    input_digest = metadata[f"{prefix}_input_digest"]
    output_digest = metadata[f"{prefix}_output_digest"]
    if HEX64.fullmatch(input_digest) is None or HEX64.fullmatch(output_digest) is None:
        raise NativeReferenceError("native lifecycle digest is malformed")
    peak = integer(metadata[f"{prefix}_peak_resident_bytes"], f"{prefix} peak", positive=True)
    scratch = integer(metadata[f"{prefix}_scratch_bytes"], f"{prefix} scratch")
    transfers = integer(metadata[f"{prefix}_host_transfers"], f"{prefix} host transfers")
    if metadata[f"{prefix}_device_resident"] != "true" or transfers != 0:
        raise NativeReferenceError("native lifecycle did not remain device resident")
    return {
        "result": "pass",
        "operation": operation,
        "artifact_sha256": artifact_digest,
        "input_digest": input_digest,
        "output_digest": output_digest,
        "peak_resident_bytes": peak,
        "scratch_bytes": scratch,
        "host_transfers": transfers,
        "device_resident": True,
    }


def build_receipt(
    artifact: bytes,
    reloaded: bytes,
    metadata: dict[str, str],
    revision: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise NativeReferenceError("native source revision is invalid")
    if not artifact or len(artifact) > MAX_ARTIFACT_BYTES or not artifact.startswith(b"TSLT2PKG"):
        raise NativeReferenceError("native artifact is not a bounded SALT V2 package")
    if artifact != reloaded:
        raise NativeReferenceError("native reload changed artifact bytes")
    if set(metadata) != {
        field.replace("_hex", "") if field.endswith("_hex") else field
        for field in METADATA_FIELDS
    }:
        raise NativeReferenceError("native lifecycle metadata fields differ")
    if metadata["backend_id"] != "cpu.reference.v1":
        raise NativeReferenceError("native backend identity differs")
    backend_build = metadata["backend_build"]
    physical_device = metadata["physical_device"]
    if (
        backend_build != f"tritium-train@{RELEASE}+source-git:{revision}"
        or not physical_device.startswith("cpu:")
        or metadata["manifest_digest"] != MANIFEST_DIGEST
    ):
        raise NativeReferenceError("native build, device, or manifest identity differs")
    artifact_digest = sha256_bytes(artifact)
    export = lifecycle_receipt(metadata, "export", artifact_digest)
    reload = {
        **lifecycle_receipt(metadata, "reload", artifact_digest),
        "reloaded_sha256": sha256_bytes(reloaded),
    }
    unsigned = {
        "schema": "tritium.browser-native-reference.v1",
        "result": "pass",
        "scenario_id": SCENARIO_ID,
        "source_revision": revision,
        "backend": "cpu",
        "backend_id": metadata["backend_id"],
        "backend_build": backend_build,
        "physical_device": physical_device,
        "manifest_digest": MANIFEST_DIGEST,
        "vector_digest": VECTOR_DIGEST,
        "artifact": {
            "name": "native.salt",
            "bytes": len(artifact),
            "sha256": artifact_digest,
        },
        "export": export,
        "reload": reload,
    }
    return {**unsigned, "receipt_id": "sha256:" + sha256_bytes(canonical(unsigned))}


def parse_metadata(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if not line or "=" not in line:
            raise NativeReferenceError("native producer emitted malformed metadata")
        key, value = line.split("=", 1)
        if key in values or not value:
            raise NativeReferenceError("native producer emitted duplicate or empty metadata")
        values[key] = value
    if set(values) != METADATA_FIELDS:
        raise NativeReferenceError("native producer metadata fields differ")
    for encoded in ("backend_build_hex", "physical_device_hex"):
        try:
            decoded = bytes.fromhex(values.pop(encoded)).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise NativeReferenceError(f"native {encoded} is malformed") from error
        values[encoded.removesuffix("_hex")] = decoded
    return values


def git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def source_admission(revision: str, root: Path = ROOT) -> None:
    if (
        re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or git_output(root, "rev-parse", "HEAD") != revision
    ):
        raise NativeReferenceError("native source revision differs from checkout")
    if git_output(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise NativeReferenceError("native reference requires a clean source checkout")


def write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def publish(output_dir: Path, artifact: bytes, receipt: dict[str, Any]) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise NativeReferenceError("native reference output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    stage.mkdir(mode=0o755)
    try:
        write_new(stage / "native.salt", artifact)
        write_new(stage / "receipt.json", canonical(receipt) + b"\n")
        directory_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(stage, output_dir)
        parent_fd = os.open(output_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def produce(revision: str, output_dir: Path) -> dict[str, Any]:
    source_admission(revision)
    with tempfile.TemporaryDirectory(prefix="tritium-browser-native-") as raw:
        root = Path(raw)
        artifact_path = root / "native.salt"
        reloaded_path = root / "reloaded.salt"
        completed = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--locked",
                "--offline",
                "-p",
                "tritium-train",
                "--example",
                "browser_native_reference",
                "--",
                str(artifact_path),
                str(reloaded_path),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10 * 60,
        )
        if len(completed.stdout.encode()) > 64 * 1024 or len(completed.stderr.encode()) > 1024 * 1024:
            raise NativeReferenceError("native producer output exceeded bounds")
        metadata = parse_metadata(completed.stdout)
        artifact = artifact_path.read_bytes()
        reloaded = reloaded_path.read_bytes()
        receipt = build_receipt(artifact, reloaded, metadata, revision)
    publish(output_dir, artifact, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = produce(args.source_revision, args.output_dir)
    print(f"PASS {receipt['receipt_id']} {args.output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (NativeReferenceError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"FAIL {error}") from error
