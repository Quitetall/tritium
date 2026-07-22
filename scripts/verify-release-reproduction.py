#!/usr/bin/env python3
"""Validate second-machine reproduction and independent-review receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any


SECOND_SCHEMA = "tritium.second-machine-reproduction.v1"
REVIEW_SCHEMA = "tritium.independent-release-review.v1"
SECOND_FIELDS = {
    "schema",
    "receipt_id",
    "result",
    "release",
    "source_revision",
    "run_id",
    "operator",
    "machine",
    "primary_machine_id",
    "candidate_manifest_sha256",
    "anchor_artifact",
    "artifacts",
    "commands",
    "checks",
    "outputs",
    "divergences",
    "wall_time_seconds",
}
REVIEW_FIELDS = {
    "schema",
    "receipt_id",
    "result",
    "release",
    "source_revision",
    "run_id",
    "reviewer",
    "candidate_manifest_sha256",
    "anchor_artifact",
    "reviewed_receipt_ids",
    "scopes",
    "findings",
    "verdict",
}
OPERATOR_FIELDS = {"id", "organization", "independent"}
MACHINE_FIELDS = {"machine_id", "system", "version", "architecture", "cpu", "gpus"}
ARTIFACT_FIELDS = {"id", "kind", "name", "bytes", "sha256"}
COMMAND_FIELDS = {
    "id",
    "argv",
    "exit_code",
    "duration_seconds",
    "stdout_sha256",
    "stderr_sha256",
}
CHECK_FIELDS = {
    "source_verified",
    "artifacts_verified",
    "repository_absent",
    "compiler_absent",
    "tutorial",
    "bitnet_native",
    "qwen_flagship",
    "bounded_validation",
    "native_backend",
    "onnx",
    "serving",
    "browser",
    "generated_tables_exact",
}
OUTPUT_FIELDS = {"name", "expected_sha256", "observed_sha256", "bytes"}
REVIEWER_FIELDS = {"id", "organization", "independent", "tool", "model"}
FINDING_FIELDS = {"total", "verified", "fixed", "false_positive", "open"}
REQUIRED_COMMANDS = frozenset(
    {
        "verify-source",
        "verify-artifacts",
        "clean-install",
        "tutorial",
        "bitnet-native",
        "qwen-flagship",
        "bounded-validation",
        "native-backend",
        "onnx",
        "serving",
        "generate-model-card",
        "generate-compatibility",
        "generate-release-status",
    }
)
REQUIRED_OUTPUTS = frozenset({"model-card", "compatibility", "release-status"})
REVIEW_SCOPES = ["code", "security", "evidence"]
HEX = frozenset("0123456789abcdef")
MAX_RECEIPT_BYTES = 32 * 1024 * 1024


class ReproductionError(ValueError):
    """Reproduction or independent-review evidence fails frozen policy."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def object_(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ReproductionError(f"{label} fields do not match frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReproductionError(f"{label} must be non-empty")
    return value


def hex_(value: Any, length: int, label: str) -> str:
    text = string(value, label)
    if len(text) != length or any(character not in HEX for character in text):
        raise ReproductionError(
            f"{label} must be {length} lowercase hexadecimal characters"
        )
    return text


def digest(value: Any, label: str) -> str:
    text = string(value, label)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", text):
        raise ReproductionError(f"{label} must be a canonical SHA-256 digest")
    return text


def positive(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ReproductionError(f"{label} must be finite and positive")
    return float(value)


def load(path: Path, fields: set[str], label: str) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > MAX_RECEIPT_BYTES
    ):
        raise ReproductionError(f"{label} must be a bounded ordinary file")
    try:
        return object_(json.loads(path.read_bytes()), fields, label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReproductionError(f"{label} must contain UTF-8 JSON") from error


def validate_common(
    receipt: dict[str, Any],
    schema: str,
    revision: str,
    release: str,
    candidate: Path,
    anchor: Path,
) -> dict[str, Any]:
    if receipt["schema"] != schema or receipt["result"] != "pass":
        raise ReproductionError("receipt schema or result mismatch")
    if receipt["source_revision"] != revision or receipt["release"] != release:
        raise ReproductionError("receipt source or release is stale")
    hex_(revision, 40, "expected source revision")
    string(release, "expected release")
    string(receipt["run_id"], "receipt.run_id")
    if candidate.is_symlink() or not candidate.is_file():
        raise ReproductionError("candidate manifest must be an ordinary file")
    if receipt["candidate_manifest_sha256"] != sha256(candidate):
        raise ReproductionError("receipt does not bind candidate manifest bytes")
    anchor_record = object_(
        receipt["anchor_artifact"], ARTIFACT_FIELDS, "anchor artifact"
    )
    if anchor.is_symlink() or not anchor.is_file():
        raise ReproductionError("anchor artifact must be an ordinary file")
    if (
        anchor_record["kind"] != "python-wheel"
        or anchor_record["name"] != anchor.name
        or anchor_record["bytes"] != anchor.stat().st_size
        or anchor_record["sha256"] != sha256(anchor)
    ):
        raise ReproductionError("receipt does not bind candidate anchor wheel")
    anchor_identity = (
        anchor_record["id"],
        anchor_record["kind"],
        anchor_record["name"],
        anchor_record["bytes"],
        anchor_record["sha256"],
    )
    if anchor_identity not in candidate_artifacts(candidate):
        raise ReproductionError("anchor wheel is absent from candidate inventory")
    return receipt


def candidate_artifacts(candidate: Path) -> set[tuple[Any, ...]]:
    try:
        document = json.loads(candidate.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReproductionError("candidate manifest must contain UTF-8 JSON") from error
    artifacts = document.get("artifacts") if isinstance(document, dict) else None
    if not isinstance(artifacts, list) or not artifacts:
        raise ReproductionError("candidate manifest artifact inventory is empty")
    result = set()
    paths = set()
    root = candidate.parent.resolve(strict=True)
    for ordinal, value in enumerate(artifacts):
        if not isinstance(value, dict):
            raise ReproductionError(f"candidate artifact {ordinal} is malformed")
        identity = value.get("identity")
        path = value.get("path")
        if not isinstance(identity, dict) or not isinstance(path, str):
            raise ReproductionError(
                f"candidate artifact {ordinal} identity is malformed"
            )
        logical = PurePosixPath(path)
        if logical.is_absolute() or ".." in logical.parts or "\\" in path:
            raise ReproductionError("candidate artifact path is unsafe")
        cursor = root
        for part in logical.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ReproductionError("candidate artifact path traverses a symlink")
        artifact = cursor.resolve(strict=True)
        try:
            artifact.relative_to(root)
        except ValueError as error:
            raise ReproductionError(
                "candidate artifact escapes candidate directory"
            ) from error
        if artifact.is_symlink() or not artifact.is_file():
            raise ReproductionError("candidate artifact must be an ordinary file")
        actual = (
            value.get("id"),
            value.get("kind"),
            artifact.name,
            artifact.stat().st_size,
            sha256(artifact),
        )
        declared = (
            value.get("id"),
            value.get("kind"),
            Path(path).name,
            identity.get("bytes"),
            identity.get("sha256"),
        )
        if actual != declared or actual in result or artifact in paths:
            raise ReproductionError(
                "candidate artifact identity is inconsistent or duplicate"
            )
        result.add(actual)
        paths.add(artifact)
    return result


def validate_second_machine(
    receipt_path: Path,
    revision: str,
    release: str,
    candidate: Path,
    anchor: Path,
) -> dict[str, Any]:
    receipt = validate_common(
        load(receipt_path, SECOND_FIELDS, "second-machine receipt"),
        SECOND_SCHEMA,
        revision,
        release,
        candidate,
        anchor,
    )
    operator = object_(receipt["operator"], OPERATOR_FIELDS, "receipt.operator")
    if operator["independent"] is not True:
        raise ReproductionError("second-machine operator must be independent")
    string(operator["id"], "receipt.operator.id")
    string(operator["organization"], "receipt.operator.organization")
    machine = object_(receipt["machine"], MACHINE_FIELDS, "receipt.machine")
    machine_id = digest(machine["machine_id"], "receipt.machine.machine_id")
    if (
        digest(receipt["primary_machine_id"], "receipt.primary_machine_id")
        == machine_id
    ):
        raise ReproductionError("reproduction machine must differ from primary machine")
    for field in ("system", "version", "architecture", "cpu"):
        string(machine[field], f"receipt.machine.{field}")
    if not isinstance(machine["gpus"], list) or any(
        not isinstance(item, str) or not item for item in machine["gpus"]
    ):
        raise ReproductionError("receipt.machine.gpus must be a string array")
    observed_artifacts = set()
    if not isinstance(receipt["artifacts"], list):
        raise ReproductionError("receipt.artifacts must be an array")
    for ordinal, value in enumerate(receipt["artifacts"]):
        item = object_(value, ARTIFACT_FIELDS, f"receipt.artifacts[{ordinal}]")
        record = (
            string(item["id"], "artifact.id"),
            string(item["kind"], "artifact.kind"),
            string(item["name"], "artifact.name"),
            item["bytes"],
            hex_(item["sha256"], 64, "artifact.sha256"),
        )
        if type(item["bytes"]) is not int or item["bytes"] <= 0:
            raise ReproductionError("reproduced artifact bytes must be positive")
        observed_artifacts.add(record)
    if len(observed_artifacts) != len(
        receipt["artifacts"]
    ) or observed_artifacts != candidate_artifacts(candidate):
        raise ReproductionError(
            "reproduction does not cover exact candidate artifact inventory"
        )
    commands = receipt["commands"]
    if not isinstance(commands, list):
        raise ReproductionError("receipt.commands must be an array")
    command_ids = set()
    for ordinal, value in enumerate(commands):
        command = object_(value, COMMAND_FIELDS, f"receipt.commands[{ordinal}]")
        command_id = string(command["id"], "command.id")
        if command_id in command_ids or command["exit_code"] != 0:
            raise ReproductionError(
                "reproduction commands must be unique and successful"
            )
        command_ids.add(command_id)
        if (
            not isinstance(command["argv"], list)
            or not command["argv"]
            or any(not isinstance(item, str) or not item for item in command["argv"])
        ):
            raise ReproductionError("reproduction command argv is invalid")
        positive(command["duration_seconds"], "command.duration_seconds")
        digest(command["stdout_sha256"], "command.stdout_sha256")
        digest(command["stderr_sha256"], "command.stderr_sha256")
    if command_ids != REQUIRED_COMMANDS:
        raise ReproductionError("reproduction command inventory is incomplete")
    checks = object_(receipt["checks"], CHECK_FIELDS, "receipt.checks")
    for field in CHECK_FIELDS - {"browser"}:
        if checks[field] is not True:
            raise ReproductionError(f"reproduction check {field} did not pass")
    browser = checks["browser"]
    if browser not in {"pass", "not-applicable"}:
        raise ReproductionError("reproduction browser check has invalid status")
    outputs = receipt["outputs"]
    if not isinstance(outputs, list):
        raise ReproductionError("receipt.outputs must be an array")
    output_names = set()
    for ordinal, value in enumerate(outputs):
        output = object_(value, OUTPUT_FIELDS, f"receipt.outputs[{ordinal}]")
        name = string(output["name"], "output.name")
        expected = digest(output["expected_sha256"], "output.expected_sha256")
        if digest(output["observed_sha256"], "output.observed_sha256") != expected:
            raise ReproductionError("regenerated output differs from candidate claim")
        if type(output["bytes"]) is not int or output["bytes"] <= 0:
            raise ReproductionError("regenerated output bytes must be positive")
        output_names.add(name)
    if len(output_names) != len(outputs) or output_names != REQUIRED_OUTPUTS:
        raise ReproductionError("regenerated output inventory is incomplete")
    if receipt["divergences"] != []:
        raise ReproductionError("second-machine reproduction contains divergences")
    positive(receipt["wall_time_seconds"], "receipt.wall_time_seconds")
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    if receipt_id != "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest():
        raise ReproductionError("second-machine receipt identity mismatch")
    return receipt


def validate_independent_review(
    receipt_path: Path,
    revision: str,
    release: str,
    candidate: Path,
    anchor: Path,
) -> dict[str, Any]:
    receipt = validate_common(
        load(receipt_path, REVIEW_FIELDS, "independent-review receipt"),
        REVIEW_SCHEMA,
        revision,
        release,
        candidate,
        anchor,
    )
    reviewer = object_(receipt["reviewer"], REVIEWER_FIELDS, "receipt.reviewer")
    if reviewer["independent"] is not True:
        raise ReproductionError("release reviewer must be independent")
    for field in REVIEWER_FIELDS - {"independent"}:
        string(reviewer[field], f"receipt.reviewer.{field}")
    reviewed = receipt["reviewed_receipt_ids"]
    if (
        not isinstance(reviewed, list)
        or not reviewed
        or len(set(reviewed)) != len(reviewed)
    ):
        raise ReproductionError("reviewed receipt IDs must be a non-empty unique array")
    for value in reviewed:
        digest(value, "reviewed receipt id")
    if receipt["scopes"] != REVIEW_SCOPES:
        raise ReproductionError(
            "independent review must cover code, security, and evidence"
        )
    findings = object_(receipt["findings"], FINDING_FIELDS, "receipt.findings")
    if any(
        type(findings[field]) is not int or findings[field] < 0
        for field in FINDING_FIELDS
    ):
        raise ReproductionError("review finding counts must be non-negative integers")
    if (
        findings["total"] != findings["verified"] + findings["false_positive"]
        or findings["fixed"] != findings["verified"]
        or findings["open"] != 0
        or receipt["verdict"] != "pass"
    ):
        raise ReproductionError(
            "independent review has unresolved or inconsistent findings"
        )
    unsigned = dict(receipt)
    receipt_id = unsigned.pop("receipt_id")
    if receipt_id != "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest():
        raise ReproductionError("independent-review receipt identity mismatch")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("second-machine", "independent-review"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--anchor-artifact", type=Path, required=True)
    args = parser.parse_args()
    validator = (
        validate_second_machine
        if args.kind == "second-machine"
        else validate_independent_review
    )
    receipt = validator(
        args.receipt.absolute(),
        args.source_revision,
        args.release,
        args.candidate.absolute(),
        args.anchor_artifact.absolute(),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
