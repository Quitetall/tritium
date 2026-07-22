#!/usr/bin/env python3
"""Run and atomically seal one independent second-machine reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shutil
import subprocess
import tempfile
import time
from typing import Any


VERIFIER = runpy.run_path(
    Path(__file__).with_name("verify-release-reproduction.py")
)
SECOND_SCHEMA = VERIFIER["SECOND_SCHEMA"]
OPERATOR_FIELDS = VERIFIER["OPERATOR_FIELDS"]
MACHINE_FIELDS = VERIFIER["MACHINE_FIELDS"]
CHECK_FIELDS = VERIFIER["CHECK_FIELDS"]
COMMAND_ORDER = VERIFIER["COMMAND_ORDER"]
REQUIRED_OUTPUTS = VERIFIER["REQUIRED_OUTPUTS"]
canonical = VERIFIER["canonical"]
sha256 = VERIFIER["sha256"]
candidate_artifacts = VERIFIER["candidate_artifacts"]
validate_receipt = VERIFIER["validate_second_machine"]

SPEC_SCHEMA = "tritium.second-machine-run-spec.v1"
SPEC_FIELDS = {
    "schema", "release", "source_revision", "run_id", "operator", "machine",
    "primary_machine_id", "commands", "checks_path", "outputs",
}
COMMAND_FIELDS = {"id", "argv", "timeout_seconds"}
OUTPUT_FIELDS = {"name", "expected_path", "observed_path"}
HEX = frozenset("0123456789abcdef")
SECRET_NAME = re.compile(
    r"(?:TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.IGNORECASE
)
COMPILERS = ("cargo", "rustc", "cc", "gcc", "clang", "cl")
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_LOG_BYTES = 32 * 1024 * 1024
MAX_TIMEOUT_SECONDS = 6 * 60 * 60


class QualificationError(ValueError):
    """The reproduction environment or observed result is not admissible."""


def object_(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise QualificationError(f"{label} fields do not match frozen schema")
    return value


def string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationError(f"{label} must be non-empty")
    return value


def digest(value: Any, label: str) -> str:
    text = string(value, label)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", text):
        raise QualificationError(f"{label} must be a canonical SHA-256 digest")
    return text


def ordinary(path: Path, label: str, maximum: int = MAX_INPUT_BYTES) -> Path:
    if (
        path.is_symlink() or not path.is_file() or path.stat().st_size <= 0
        or path.stat().st_size > maximum
    ):
        raise QualificationError(f"{label} must be a bounded ordinary file")
    return path.resolve(strict=True)


def contained(root: Path, logical_value: Any, label: str) -> Path:
    text = string(logical_value, label)
    logical = PurePosixPath(text)
    if logical.is_absolute() or ".." in logical.parts or "\\" in text:
        raise QualificationError(f"{label} path is unsafe")
    cursor = root.resolve(strict=True)
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise QualificationError(f"{label} traverses a symlink")
    resolved = ordinary(cursor, label)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise QualificationError(f"{label} escapes root") from error
    return resolved


def load(path: Path, fields: set[str], label: str) -> dict[str, Any]:
    path = ordinary(path, label)
    try:
        return object_(json.loads(path.read_bytes()), fields, label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"{label} must contain UTF-8 JSON") from error


def anchor_record(candidate: Path, anchor: Path) -> dict[str, Any]:
    anchor = ordinary(anchor, "anchor wheel")
    matches = [
        record for record in candidate_artifacts(candidate)
        if record[1] == "python-wheel" and record[2:] == (
            anchor.name, anchor.stat().st_size, sha256(anchor)
        )
    ]
    if len(matches) != 1:
        raise QualificationError("candidate must bind exactly one matching anchor wheel")
    artifact_id, kind, name, size, checksum = matches[0]
    return {
        "id": artifact_id, "kind": kind, "name": name,
        "bytes": size, "sha256": checksum,
    }


def clean_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if not SECRET_NAME.search(key)
    }
    environment.update({
        "CARGO_NET_OFFLINE": "true", "HF_HUB_OFFLINE": "1",
        "PIP_NO_INDEX": "1", "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline", "PYTHONHASHSEED": "0",
    })
    return environment


def require_clean_environment(work_dir: Path, environment: dict[str, str]) -> None:
    if not work_dir.is_dir() or work_dir.is_symlink():
        raise QualificationError("work directory must be an ordinary directory")
    result = subprocess.run(
        ["git", "-C", str(work_dir), "rev-parse", "--is-inside-work-tree"],
        text=True, capture_output=True, check=False, env=environment,
    )
    if result.returncode == 0:
        raise QualificationError("second-machine work directory is inside a repository")
    present = [name for name in COMPILERS if shutil.which(name, path=environment.get("PATH"))]
    if present:
        raise QualificationError(
            "second-machine environment contains forbidden compilers: " + ", ".join(present)
        )


def validate_spec(
    spec: dict[str, Any], *, revision: str, release: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        spec["schema"] != SPEC_SCHEMA or spec["release"] != release
        or spec["source_revision"] != revision
    ):
        raise QualificationError("reproduction spec identity differs from candidate")
    if len(revision) != 40 or any(character not in HEX for character in revision):
        raise QualificationError("source revision must be 40 lowercase hexadecimal")
    string(spec["run_id"], "run_id")
    operator = object_(spec["operator"], OPERATOR_FIELDS, "operator")
    if operator["independent"] is not True:
        raise QualificationError("reproduction operator must be independent")
    string(operator["id"], "operator.id")
    string(operator["organization"], "operator.organization")
    machine = object_(spec["machine"], MACHINE_FIELDS, "machine")
    machine_id = digest(machine["machine_id"], "machine.machine_id")
    if digest(spec["primary_machine_id"], "primary_machine_id") == machine_id:
        raise QualificationError("reproduction machine must differ from primary machine")
    for field in ("system", "version", "architecture", "cpu"):
        string(machine[field], f"machine.{field}")
    if not isinstance(machine["gpus"], list) or any(
        not isinstance(item, str) or not item for item in machine["gpus"]
    ):
        raise QualificationError("machine.gpus must be a string array")
    commands = spec["commands"]
    if not isinstance(commands, list) or len(commands) != len(COMMAND_ORDER):
        raise QualificationError("reproduction command inventory is incomplete")
    parsed_commands = []
    for ordinal, command_id in enumerate(COMMAND_ORDER):
        command = object_(commands[ordinal], COMMAND_FIELDS, f"commands[{ordinal}]")
        if command["id"] != command_id:
            raise QualificationError("reproduction command order differs from policy")
        argv = command["argv"]
        if (
            not isinstance(argv, list) or not argv
            or any(not isinstance(item, str) or not item for item in argv)
        ):
            raise QualificationError("reproduction command argv is invalid")
        timeout = command["timeout_seconds"]
        if (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not 0 < float(timeout) <= MAX_TIMEOUT_SECONDS
        ):
            raise QualificationError("reproduction command timeout is invalid")
        parsed_commands.append(command)
    outputs = spec["outputs"]
    if not isinstance(outputs, list) or len(outputs) != len(REQUIRED_OUTPUTS):
        raise QualificationError("reproduction output inventory is incomplete")
    parsed_outputs = []
    expected_names = sorted(REQUIRED_OUTPUTS)
    for ordinal, name in enumerate(expected_names):
        output = object_(outputs[ordinal], OUTPUT_FIELDS, f"outputs[{ordinal}]")
        if output["name"] != name:
            raise QualificationError("reproduction output order differs from policy")
        parsed_outputs.append(output)
    string(spec["checks_path"], "checks_path")
    return parsed_commands, parsed_outputs


def run_command(
    command: dict[str, Any], *, work_dir: Path, logs_dir: Path,
    environment: dict[str, str], ordinal: int,
) -> dict[str, Any]:
    stdout_path = logs_dir / f"{ordinal:02d}-{command['id']}.stdout"
    stderr_path = logs_dir / f"{ordinal:02d}-{command['id']}.stderr"
    started = time.monotonic()
    try:
        result = subprocess.run(
            command["argv"], cwd=work_dir, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            timeout=float(command["timeout_seconds"]),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise QualificationError(f"reproduction command {command['id']} failed to run") from error
    duration = max(time.monotonic() - started, 1e-9)
    if len(result.stdout) > MAX_LOG_BYTES or len(result.stderr) > MAX_LOG_BYTES:
        raise QualificationError(f"reproduction command {command['id']} log exceeds limit")
    stdout_path.write_bytes(result.stdout)
    stderr_path.write_bytes(result.stderr)
    if result.returncode != 0:
        raise QualificationError(
            f"reproduction command {command['id']} exited {result.returncode}"
        )
    return {
        "id": command["id"], "argv": command["argv"],
        "exit_code": result.returncode, "duration_seconds": duration,
        "stdout_sha256": "sha256:" + sha256(stdout_path),
        "stderr_sha256": "sha256:" + sha256(stderr_path),
        "stdout_path": stdout_path.relative_to(logs_dir.parent).as_posix(),
        "stderr_path": stderr_path.relative_to(logs_dir.parent).as_posix(),
    }


def observed_checks(spec: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    checks = load(
        contained(work_dir, spec["checks_path"], "checks_path"),
        CHECK_FIELDS, "reproduction checks",
    )
    for field in CHECK_FIELDS - {"browser"}:
        if checks[field] is not True:
            raise QualificationError(f"reproduction check {field} did not pass")
    if checks["browser"] not in {"pass", "not-applicable"}:
        raise QualificationError("reproduction browser check has invalid status")
    return checks


def copy_outputs(
    outputs: list[dict[str, Any]], *, spec_root: Path, work_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    records = []
    for ordinal, output in enumerate(outputs):
        expected = contained(spec_root, output["expected_path"], "expected output")
        observed = contained(work_dir, output["observed_path"], "observed output")
        if sha256(expected) != sha256(observed) or expected.stat().st_size != observed.stat().st_size:
            raise QualificationError(f"regenerated {output['name']} differs from candidate claim")
        destination = output_dir / f"{ordinal:02d}-{observed.name}"
        shutil.copyfile(observed, destination)
        checksum = "sha256:" + sha256(destination)
        records.append({
            "name": output["name"], "expected_sha256": checksum,
            "observed_sha256": checksum, "bytes": destination.stat().st_size,
            "path": destination.relative_to(output_dir.parent).as_posix(),
        })
    return records


def assemble(
    stage: Path, *, candidate: Path, anchor: Path, spec_path: Path,
    work_dir: Path, source_revision: str, release: str,
) -> dict[str, Any]:
    candidate = ordinary(candidate, "candidate manifest")
    anchor = ordinary(anchor, "anchor wheel")
    spec_path = ordinary(spec_path, "reproduction spec")
    spec = load(spec_path, SPEC_FIELDS, "reproduction spec")
    commands, outputs = validate_spec(spec, revision=source_revision, release=release)
    environment = clean_environment()
    require_clean_environment(work_dir, environment)
    stage.mkdir()
    logs_dir = stage / "logs"
    outputs_dir = stage / "outputs"
    logs_dir.mkdir()
    outputs_dir.mkdir()
    started = time.monotonic()
    command_records = [
        run_command(
            command, work_dir=work_dir, logs_dir=logs_dir,
            environment=environment, ordinal=ordinal,
        )
        for ordinal, command in enumerate(commands)
    ]
    checks = observed_checks(spec, work_dir)
    output_records = copy_outputs(
        outputs, spec_root=spec_path.parent.resolve(strict=True), work_dir=work_dir,
        output_dir=outputs_dir,
    )
    artifact_records = [
        {"id": item[0], "kind": item[1], "name": item[2], "bytes": item[3], "sha256": item[4]}
        for item in sorted(candidate_artifacts(candidate))
    ]
    receipt: dict[str, Any] = {
        "schema": SECOND_SCHEMA, "result": "pass", "release": release,
        "source_revision": source_revision, "run_id": spec["run_id"],
        "operator": spec["operator"], "machine": spec["machine"],
        "primary_machine_id": spec["primary_machine_id"],
        "candidate_manifest_sha256": sha256(candidate),
        "anchor_artifact": anchor_record(candidate, anchor),
        "artifacts": artifact_records, "commands": command_records,
        "checks": checks, "outputs": output_records, "divergences": [],
        "wall_time_seconds": max(time.monotonic() - started, 1e-9),
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    receipt_path = stage / "receipt.json"
    receipt_path.write_bytes(canonical(receipt) + b"\n")
    validate_receipt(receipt_path, source_revision, release, candidate, anchor)
    return receipt


def qualify(
    output_dir: Path, *, candidate: Path, anchor: Path, spec_path: Path,
    work_dir: Path, source_revision: str, release: str,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise QualificationError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    stage.rmdir()
    try:
        receipt = assemble(
            stage, candidate=candidate, anchor=anchor, spec_path=spec_path,
            work_dir=work_dir.resolve(strict=True), source_revision=source_revision,
            release=release,
        )
        os.replace(stage, output_dir)
        return receipt
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--anchor-wheel", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify(
        args.output_dir.absolute(), candidate=args.candidate.absolute(),
        anchor=args.anchor_wheel.absolute(), spec_path=args.spec.absolute(),
        work_dir=args.work_dir.absolute(), source_revision=args.source_revision,
        release=args.release,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
