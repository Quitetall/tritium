#!/usr/bin/env python3
"""Fail-closed verifier for retained physical CUDA dispatcher evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET


SCHEMA = "tritium.torch-dispatch-cuda-qualification.v1"
SOURCE_PATH = "crates/tritium-py/tests/test_torch_dispatch.py"
CUDA_TESTS = (
    "test_native_cuda_warm_forward_backward_avoids_composite_tensor_ops",
    "test_native_cuda_autocast_warm_forward_backward_uses_fp16_kernels",
    "test_native_cuda_fp16_tail_paths_for_memcheck",
    "test_native_cuda_tail_forward_backward_parity_for_memcheck",
    "test_native_cuda_preserves_nonfinite_dense_semantics",
    "test_native_cuda_cache_invalidates_on_mutation_and_storage_replacement",
    "test_native_cuda_cache_orders_cross_stream_pack_and_survives_owner_drop",
)
MEMCHECK_TESTS = (
    "test_native_cuda_fp16_tail_paths_for_memcheck",
    "test_native_cuda_tail_forward_backward_parity_for_memcheck",
)
TOP_FIELDS = {
    "schema", "receipt_id", "result", "release", "source_revision", "run_id",
    "artifact", "environment", "device", "source", "suite", "sanitizer",
}
ARTIFACT_FIELDS = {"kind", "name", "bytes", "sha256"}
ENVIRONMENT_FIELDS = {
    "python_version", "torch_version", "tritium_version", "cuda_runtime",
    "cuda_driver", "source_identity",
}
DEVICE_FIELDS = {
    "index", "uuid", "name", "compute_capability", "total_memory_bytes",
}
SOURCE_FIELDS = {"path", "name", "bytes", "sha256", "git_blob"}
SUITE_FIELDS = {"selector", "tests", "passed", "junit"}
SANITIZER_FIELDS = {
    "tool", "version", "error_summary", "tests", "passed", "junit", "log",
}
FILE_FIELDS = {"name", "bytes", "sha256"}
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_RETAINED_BYTES = 32 * 1024 * 1024


class ReceiptError(ValueError):
    """Receipt is malformed, stale, contradictory, or did not pass."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ReceiptError(f"{label} fields do not match frozen schema")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReceiptError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ReceiptError(f"{label} must be a positive integer")
    return value


def _digest(value: Any, size: int, label: str) -> str:
    text = _string(value, label)
    if re.fullmatch(rf"[0-9a-f]{{{size}}}", text) is None:
        raise ReceiptError(f"{label} must be {size} lowercase hexadecimal characters")
    return text


def _ordinary(path: Path, label: str, maximum: int) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ReceiptError(f"{label} must be an ordinary file")
    if path.stat().st_size <= 0 or path.stat().st_size > maximum:
        raise ReceiptError(f"{label} size is outside qualification bounds")
    return path


def _retained(root: Path, value: Any, label: str) -> tuple[Path, dict[str, Any]]:
    record = _object(value, FILE_FIELDS, label)
    name = _string(record["name"], f"{label}.name")
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ReceiptError(f"{label}.name must be a basename")
    path = _ordinary(root / name, label, MAX_RETAINED_BYTES)
    if type(record["bytes"]) is not int or record["bytes"] != path.stat().st_size:
        raise ReceiptError(f"{label}.bytes does not match retained file")
    if _digest(record["sha256"], 64, f"{label}.sha256") != sha256(path):
        raise ReceiptError(f"{label}.sha256 does not match retained file")
    return path, record


def _artifact(value: Any, wheel: Path) -> dict[str, Any]:
    record = _object(value, ARTIFACT_FIELDS, "receipt.artifact")
    if record["kind"] != "python-wheel" or record["name"] != wheel.name:
        raise ReceiptError("receipt artifact does not identify qualified wheel")
    match = re.fullmatch(
        r"tritium_torch-([^-]+)-cp39-abi3-[^-]+\.whl", wheel.name
    )
    if match is None:
        raise ReceiptError("qualified wheel filename is not canonical Tritium abi3")
    if type(record["bytes"]) is not int or record["bytes"] != wheel.stat().st_size:
        raise ReceiptError("receipt artifact byte count differs from qualified wheel")
    if _digest(record["sha256"], 64, "receipt.artifact.sha256") != sha256(wheel):
        raise ReceiptError("receipt artifact digest differs from qualified wheel")
    return record


def _junit(path: Path, expected: tuple[str, ...], label: str) -> None:
    try:
        root = ET.fromstring(path.read_bytes())
    except ET.ParseError as error:
        raise ReceiptError(f"{label} must contain XML") from error
    if root.tag not in {"testsuite", "testsuites"}:
        raise ReceiptError(f"{label} root must be testsuite or testsuites")
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ReceiptError(f"{label} contains no test suite")
    for field, expected_value in (
        ("tests", len(expected)), ("failures", 0), ("errors", 0), ("skipped", 0)
    ):
        try:
            observed = sum(int(suite.attrib.get(field, "")) for suite in suites)
        except ValueError as error:
            raise ReceiptError(f"{label}.{field} is invalid") from error
        if observed != expected_value:
            raise ReceiptError(f"{label}.{field} contradicts qualification policy")
    names = tuple(case.attrib.get("name", "") for case in root.iter("testcase"))
    if len(names) != len(expected) or set(names) != set(expected):
        raise ReceiptError(f"{label} test cases do not match frozen suite")
    if any(
        next(root.iter(outcome), None) is not None
        for outcome in ("failure", "error", "skipped")
    ):
        raise ReceiptError(f"{label} contains a failing or skipped test case")


def _git_blob(path: Path) -> str:
    content = path.read_bytes()
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def git_blob_at(repo: Path, revision: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{revision}:{SOURCE_PATH}"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    blob = result.stdout.strip()
    if result.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", blob) is None:
        raise ReceiptError("candidate revision lacks frozen dispatcher test source")
    return blob


def validate(
    path: Path,
    expected_revision: str,
    expected_release: str,
    expected_wheel: Path,
    expected_source_blob: str,
) -> dict[str, Any]:
    path = _ordinary(path, "receipt", MAX_RECEIPT_BYTES)
    expected_wheel = _ordinary(expected_wheel, "qualified wheel", 2 * 1024**3)
    try:
        receipt = _object(json.loads(path.read_bytes()), TOP_FIELDS, "receipt")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError("receipt must contain UTF-8 JSON") from error
    revision = _digest(expected_revision, 40, "expected source revision")
    if receipt["schema"] != SCHEMA or receipt["result"] != "pass":
        raise ReceiptError("receipt is not passed CUDA dispatcher evidence")
    if receipt["source_revision"] != revision:
        raise ReceiptError("receipt source revision differs from candidate")
    if receipt["release"] != expected_release or re.fullmatch(
        r"1\.1\.0-rc\.(0|[1-9][0-9]*)", expected_release
    ) is None:
        raise ReceiptError("receipt release differs from canonical v1.1 candidate")
    _string(receipt["run_id"], "receipt.run_id")
    _artifact(receipt["artifact"], expected_wheel)

    environment = _object(receipt["environment"], ENVIRONMENT_FIELDS, "environment")
    for field in ENVIRONMENT_FIELDS - {"source_identity"}:
        _string(environment[field], f"environment.{field}")
    if environment["source_identity"] != f"source-git:{revision}":
        raise ReceiptError("installed extension source identity differs from candidate")
    expected_distribution = re.sub(r"-rc\.([0-9]+)$", r"rc\1", expected_release)
    if environment["tritium_version"] != expected_distribution:
        raise ReceiptError("installed distribution version differs from candidate")

    device = _object(receipt["device"], DEVICE_FIELDS, "device")
    if type(device["index"]) is not int or device["index"] < 0:
        raise ReceiptError("device.index must be nonnegative")
    for field in ("uuid", "name"):
        _string(device[field], f"device.{field}")
    if re.fullmatch(r"[1-9][0-9]*\.[0-9]+", str(device["compute_capability"])) is None:
        raise ReceiptError("device.compute_capability is invalid")
    _positive_int(device["total_memory_bytes"], "device.total_memory_bytes")

    source = _object(receipt["source"], SOURCE_FIELDS, "source")
    if source["path"] != SOURCE_PATH or source["name"] != Path(SOURCE_PATH).name:
        raise ReceiptError("receipt does not bind frozen dispatcher test source")
    source_path, _ = _retained(
        path.parent,
        {field: source[field] for field in FILE_FIELDS},
        "source",
    )
    source_blob = _digest(source["git_blob"], 40, "source.git_blob")
    if source_blob != _git_blob(source_path):
        raise ReceiptError("source.git_blob differs from retained test source")
    if source_blob != _digest(expected_source_blob, 40, "expected source blob"):
        raise ReceiptError("retained test source is not candidate revision source")

    suite = _object(receipt["suite"], SUITE_FIELDS, "suite")
    if suite["selector"] != "native_cuda" or suite["tests"] != list(CUDA_TESTS):
        raise ReceiptError("suite does not match frozen native CUDA cases")
    if suite["passed"] != len(CUDA_TESTS):
        raise ReceiptError("suite pass count does not match frozen cases")
    suite_junit, _ = _retained(path.parent, suite["junit"], "suite.junit")
    _junit(suite_junit, CUDA_TESTS, "suite.junit")

    sanitizer = _object(receipt["sanitizer"], SANITIZER_FIELDS, "sanitizer")
    if sanitizer["tool"] != "compute-sanitizer" or re.fullmatch(
        r"[0-9]+(?:\.[0-9]+){2,3}", str(sanitizer["version"])
    ) is None:
        raise ReceiptError("sanitizer tool or version is invalid")
    if sanitizer["error_summary"] != 0:
        raise ReceiptError("compute-sanitizer reported errors")
    if sanitizer["tests"] != list(MEMCHECK_TESTS) or sanitizer["passed"] != len(MEMCHECK_TESTS):
        raise ReceiptError("sanitizer suite does not match frozen memcheck cases")
    memcheck_junit, _ = _retained(path.parent, sanitizer["junit"], "sanitizer.junit")
    _junit(memcheck_junit, MEMCHECK_TESTS, "sanitizer.junit")
    log, _ = _retained(path.parent, sanitizer["log"], "sanitizer.log")
    summaries = [int(value) for value in re.findall(rb"ERROR SUMMARY:\s*([0-9]+) errors", log.read_bytes())]
    if summaries != [0]:
        raise ReceiptError("sanitizer log lacks one zero-error summary")

    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    expected_id = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if receipt["receipt_id"] != expected_id:
        raise ReceiptError("receipt_id does not match canonical receipt bytes")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        source_blob = git_blob_at(args.repo.resolve(strict=True), args.source_revision)
        receipt = validate(
            args.receipt,
            args.source_revision,
            args.release,
            args.wheel,
            source_blob,
        )
    except (OSError, ReceiptError) as error:
        print(f"verify-torch-dispatch-cuda-receipt: FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "verify-torch-dispatch-cuda-receipt: PASS: "
        f"{receipt['device']['name']} {receipt['sanitizer']['version']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
