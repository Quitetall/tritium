#!/usr/bin/env python3
"""Validate installed-wheel Python API signature evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


SCHEMA = "tritium.installed-api-signature.v1"
TRACE_SCHEMA = "tritium.installed-api-signature-trace.v1"
MAX_RECEIPT_BYTES = 8 * 1024 * 1024
MAX_TRACE_BYTES = 8 * 1024 * 1024
HEX = frozenset("0123456789abcdef")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
RELEASE_RE = re.compile(
    r"^(?:[0-9]+\.){2}[0-9]+(?:-(?:alpha|beta|rc)\.[0-9]+)?$"
)

TOP_FIELDS = {
    "schema", "receipt_id", "result", "release", "source_revision", "run_id",
    "wheel", "api_report", "runtime", "environment", "signature", "trace",
}
TRACE_FIELDS = {
    "schema", "result", "release", "source_revision", "run_id", "wheel",
    "api_report", "runtime", "environment", "signature",
}
WHEEL_FIELDS = {"name", "bytes", "sha256"}
API_REPORT_FIELDS = {"baseline", "candidate_version", "report_id", "root_exports"}
RUNTIME_FIELDS = {
    "python_version", "distribution_version", "source_identity", "module_path",
    "native_module_path", "wheel_file_count", "installed_file_count",
    "installed_tree_sha256",
}
ENVIRONMENT_FIELDS = {"source_tree_absent", "compiler_absent", "network_mode"}
SIGNATURE_FIELDS = {"root_exports", "callable_signatures", "opaque_callables"}
TRACE_FILE_FIELDS = {"path", "bytes", "sha256"}


class ApiSignatureError(ValueError):
    """Installed API evidence is malformed, stale, or not wheel-bound."""


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
        raise ApiSignatureError(f"{label} fields do not match frozen schema")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ApiSignatureError(f"{label} must be a non-empty string")
    return value


def _digest(value: Any, label: str) -> str:
    text = _string(value, label)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", text):
        raise ApiSignatureError(f"{label} must be a SHA-256 identity")
    return text


def _revision(value: Any, label: str) -> str:
    text = _string(value, label)
    if REVISION_RE.fullmatch(text) is None:
        raise ApiSignatureError(f"{label} must be a full lowercase Git object ID")
    return text


def _release(value: Any, label: str) -> str:
    text = _string(value, label)
    if RELEASE_RE.fullmatch(text) is None:
        raise ApiSignatureError(f"{label} has invalid Tritium release syntax")
    return text


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ApiSignatureError(f"{label} must be a positive integer")
    return value


def _bounded_file(path: Path, label: str, maximum: int) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ApiSignatureError(f"{label} must be an ordinary file")
    if path.stat().st_size <= 0 or path.stat().st_size > maximum:
        raise ApiSignatureError(f"{label} exceeds bounded size")
    return path.resolve(strict=True)


def _load(path: Path, label: str, maximum: int) -> dict[str, Any]:
    path = _bounded_file(path, label, maximum)
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ApiSignatureError(f"{label} must contain UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ApiSignatureError(f"{label} must contain a JSON object")
    return value


def _validate_wheel(value: Any, wheel: Path) -> dict[str, Any]:
    record = _object(value, WHEEL_FIELDS, "wheel")
    wheel = _bounded_file(wheel, "candidate wheel", 4 * 1024**3)
    if (
        record["name"] != wheel.name
        or record["bytes"] != wheel.stat().st_size
        or record["sha256"] != "sha256:" + sha256(wheel)
    ):
        raise ApiSignatureError("API receipt does not bind exact candidate wheel bytes")
    return record


def _report(path: Path, expected_release: str) -> dict[str, Any]:
    report = _load(path, "API diff report", 4 * 1024 * 1024)
    if report.get("schema") != "tritium.api-diff.v1":
        raise ApiSignatureError("API diff report schema differs")
    if report.get("candidate_version") != expected_release:
        raise ApiSignatureError("API diff report release differs")
    report_id = report.get("report_id")
    unsigned = {key: value for key, value in report.items() if key != "report_id"}
    if report_id != "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest():
        raise ApiSignatureError("API diff report identity differs")
    python = report.get("python")
    if not isinstance(python, dict) or set(python) != {"added", "removed", "retained"}:
        raise ApiSignatureError("API diff report Python section differs")
    if any(not isinstance(python[field], list) for field in ("added", "removed", "retained")):
        raise ApiSignatureError("API diff report Python lists differ")
    if python["removed"] != []:
        raise ApiSignatureError("API diff report contains removed v1 names")
    names = [*python["retained"], *python["added"]]
    if (
        any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
        or python["retained"] != sorted(python["retained"])
        or python["added"] != sorted(python["added"])
    ):
        raise ApiSignatureError("API diff report exports are not canonical")
    names = sorted(names)
    return {
        "baseline": report.get("baseline"),
        "candidate_version": report["candidate_version"],
        "report_id": report_id,
        "root_exports": names,
    }


def _validate_api_report(value: Any, expected: dict[str, Any]) -> dict[str, Any]:
    record = _object(value, API_REPORT_FIELDS, "receipt.api_report")
    if record != expected:
        raise ApiSignatureError("receipt API report identity differs")
    _string(record["baseline"], "api_report.baseline")
    _digest(record["report_id"], "api_report.report_id")
    return record


def _validate_runtime(value: Any, *, expected_revision: str, expected_release: str) -> dict[str, Any]:
    runtime = _object(value, RUNTIME_FIELDS, "runtime")
    if runtime["distribution_version"] != expected_release.replace("-rc.", "rc"):
        raise ApiSignatureError("installed distribution version differs")
    if runtime["source_identity"] != f"source-git:{expected_revision}":
        raise ApiSignatureError("installed native source identity differs")
    for field in ("python_version", "module_path", "native_module_path"):
        _string(runtime[field], f"runtime.{field}")
    _positive_int(runtime["wheel_file_count"], "runtime.wheel_file_count")
    _positive_int(runtime["installed_file_count"], "runtime.installed_file_count")
    if runtime["installed_file_count"] != runtime["wheel_file_count"]:
        raise ApiSignatureError("installed wheel inventory is incomplete")
    _digest(runtime["installed_tree_sha256"], "runtime.installed_tree_sha256")
    return runtime


def _validate_environment(value: Any) -> dict[str, Any]:
    environment = _object(value, ENVIRONMENT_FIELDS, "environment")
    if environment["source_tree_absent"] is not True:
        raise ApiSignatureError("source checkout was visible during API probe")
    if environment["compiler_absent"] is not True:
        raise ApiSignatureError("compiler was visible during API probe")
    if environment["network_mode"] != "offline":
        raise ApiSignatureError("API probe was not declared offline")
    return environment


def _validate_signature(value: Any, expected: dict[str, Any]) -> dict[str, Any]:
    signature = _object(value, SIGNATURE_FIELDS, "signature")
    exports = signature["root_exports"]
    if exports != expected["root_exports"]:
        raise ApiSignatureError("installed root namespace differs from API report")
    callables = signature["callable_signatures"]
    if not isinstance(callables, dict):
        raise ApiSignatureError("callable signatures must be an object")
    opaque = signature["opaque_callables"]
    if (
        not isinstance(opaque, list)
        or opaque != sorted(opaque)
        or len(opaque) != len(set(opaque))
        or any(not isinstance(name, str) or name not in exports for name in opaque)
        or set(opaque) & set(callables)
    ):
        raise ApiSignatureError("opaque callable inventory is malformed")
    if any(
        not isinstance(name, str)
        or name not in exports
        or not isinstance(value, str)
        or not value
        or "0x" in value
        for name, value in callables.items()
    ):
        raise ApiSignatureError("callable signature inventory is malformed")
    return signature


def _retained_file(root: Path, value: Any) -> Path:
    record = _object(value, TRACE_FILE_FIELDS, "trace")
    logical = PurePosixPath(_string(record["path"], "trace.path"))
    if logical.is_absolute() or ".." in logical.parts or "\\" in str(logical):
        raise ApiSignatureError("trace path is unsafe")
    path = (root / logical).resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ApiSignatureError("trace escapes receipt directory") from error
    _bounded_file(path, "retained API trace", MAX_TRACE_BYTES)
    if (
        record["bytes"] != path.stat().st_size
        or record["sha256"] != "sha256:" + sha256(path)
    ):
        raise ApiSignatureError("retained API trace bytes drifted")
    return path


def _validate_trace(
    path: Path,
    *,
    expected_revision: str,
    expected_release: str,
    expected_wheel: Path,
    expected_report: dict[str, Any],
) -> dict[str, Any]:
    trace = _load(path, "API trace", MAX_TRACE_BYTES)
    if trace.get("schema") != TRACE_SCHEMA or trace.get("result") != "complete":
        raise ApiSignatureError("API trace schema or result differs")
    if trace["release"] != expected_release or trace["source_revision"] != expected_revision:
        raise ApiSignatureError("API trace source or release is stale")
    _string(trace["run_id"], "trace.run_id")
    _validate_wheel(trace["wheel"], expected_wheel)
    _validate_api_report(trace["api_report"], expected_report)
    _validate_runtime(
        trace["runtime"], expected_revision=expected_revision, expected_release=expected_release
    )
    _validate_environment(trace["environment"])
    _validate_signature(trace["signature"], expected_report)
    return trace


def validate(
    path: Path,
    *,
    expected_revision: str,
    expected_release: str,
    expected_wheel: Path,
    expected_api_report: Path,
) -> dict[str, Any]:
    """Validate one retained installed-wheel API signature receipt."""

    _revision(expected_revision, "expected source revision")
    _release(expected_release, "expected release")
    expected = _report(expected_api_report, expected_release)
    receipt = _load(path, "API signature receipt", MAX_RECEIPT_BYTES)
    _object(receipt, TOP_FIELDS, "receipt")
    if receipt["schema"] != SCHEMA or receipt["result"] != "pass":
        raise ApiSignatureError("API signature receipt schema or result differs")
    if receipt["release"] != expected_release or receipt["source_revision"] != expected_revision:
        raise ApiSignatureError("receipt source or release is stale")
    _string(receipt["run_id"], "receipt.run_id")
    _validate_wheel(receipt["wheel"], expected_wheel)
    _validate_api_report(receipt["api_report"], expected)
    _validate_runtime(
        receipt["runtime"], expected_revision=expected_revision, expected_release=expected_release
    )
    _validate_environment(receipt["environment"])
    _validate_signature(receipt["signature"], expected)
    trace_path = _retained_file(path.parent, receipt["trace"])
    trace = _validate_trace(
        trace_path,
        expected_revision=expected_revision,
        expected_release=expected_release,
        expected_wheel=expected_wheel,
        expected_report=expected,
    )
    for field in ("run_id", "wheel", "api_report", "runtime", "environment", "signature"):
        if receipt[field] != trace[field]:
            raise ApiSignatureError(f"receipt does not bind retained trace field {field}")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    expected_id = "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()
    if receipt["receipt_id"] != expected_id:
        raise ApiSignatureError("API signature receipt identity differs")
    return receipt


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--api-report", type=Path, required=True)
    args = parser.parse_args()
    validate(
        args.receipt,
        expected_revision=args.source_revision,
        expected_release=args.release,
        expected_wheel=args.wheel,
        expected_api_report=args.api_report,
    )
    print("installed API signature receipt: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
