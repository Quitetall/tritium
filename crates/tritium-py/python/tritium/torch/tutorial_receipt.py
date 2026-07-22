"""Portable, dependency-free validator for installed QAT tutorial evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "tritium.installed-qat-tutorial.v3"
FIELDS = {
    "schema",
    "receipt_id",
    "passed",
    "device",
    "seed",
    "torch_version",
    "distribution_version",
    "tritium_module",
    "source_revision",
    "release",
    "run_id",
    "wheel_name",
    "wheel_bytes",
    "wheel_sha256",
    "loss",
    "gradient_norm",
    "converted_parameters",
    "aliases",
    "algorithm_id",
    "planes",
    "artifact_id",
    "hard_state_digest",
    "artifact_dir",
    "hard_artifact_bytes",
    "hard_artifact_file_count",
    "hard_artifact_tree_sha256",
    "checkpoint_model_bytes",
    "checkpoint_model_sha256",
    "checkpoint_optimizer_bytes",
    "checkpoint_optimizer_sha256",
    "optimizer_state_entries",
    "resume_steps",
}
HEX = frozenset("0123456789abcdef")
MAX_RECEIPT_BYTES = 1024 * 1024


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def receipt_id(receipt: dict[str, object]) -> str:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    return "sha256:" + hashlib.sha256(canonical(unsigned)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _ordinary_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an ordinary non-symlink file")
    return path.resolve(strict=True)


def tree_identity(root: Path) -> dict[str, object]:
    """Return a path-sensitive identity for one symlink-free artifact tree."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("tutorial hard artifact must be an ordinary directory")
    root = root.resolve(strict=True)
    entries: list[dict[str, object]] = []

    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if child.is_symlink():
                raise ValueError("tutorial hard artifact must not contain symlinks")
            if child.is_dir():
                visit(child)
            elif child.is_file():
                entries.append(
                    {
                        "path": child.relative_to(root).as_posix(),
                        "bytes": child.stat().st_size,
                        "sha256": _file_sha256(child),
                    }
                )
            else:
                raise ValueError("tutorial hard artifact contains a non-file entry")

    visit(root)
    if not entries:
        raise ValueError("tutorial hard artifact must contain files")
    return {
        "bytes": sum(int(entry["bytes"]) for entry in entries),
        "file_count": len(entries),
        "sha256": "sha256:" + hashlib.sha256(canonical(entries)).hexdigest(),
    }


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in HEX for character in value[7:])
    ):
        raise ValueError(f"tutorial {label} is not a canonical digest")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"tutorial {label} must be a positive integer")
    return value


def validate_receipt(
    receipt_path: Path,
    *,
    expected_device: str,
    expected_wheel: Path | None = None,
    expected_source_revision: str | None = None,
    expected_release: str | None = None,
) -> dict[str, object]:
    """Validate portable receipt fields and every referenced artifact byte."""

    receipt_path = _ordinary_file(receipt_path, "tutorial receipt")
    if receipt_path.stat().st_size > MAX_RECEIPT_BYTES:
        raise ValueError("tutorial receipt exceeds metadata size limit")
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("tutorial receipt must contain UTF-8 JSON") from error
    if not isinstance(receipt, dict) or set(receipt) != FIELDS:
        raise ValueError("tutorial receipt fields do not match schema version 3")
    if receipt["schema"] != SCHEMA:
        raise ValueError("unsupported tutorial receipt schema")
    if receipt["passed"] is not True or receipt["device"] != expected_device:
        raise ValueError("tutorial receipt result or device mismatch")
    if type(receipt["seed"]) is not int:
        raise ValueError("tutorial seed must be an integer")
    for field in (
        "torch_version",
        "distribution_version",
        "tritium_module",
        "release",
        "run_id",
    ):
        if not isinstance(receipt[field], str) or not receipt[field]:
            raise ValueError(f"tutorial {field} must be a non-empty string")
    revision = receipt["source_revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in HEX for character in revision)
    ):
        raise ValueError(
            "tutorial source revision must be 40 lowercase hexadecimal characters"
        )
    if expected_source_revision is not None and revision != expected_source_revision:
        raise ValueError("tutorial source revision mismatch")
    if expected_release is not None and receipt["release"] != expected_release:
        raise ValueError("tutorial release mismatch")
    for field in ("loss", "gradient_norm"):
        value = receipt[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"tutorial {field} must be finite and positive")
    if receipt["converted_parameters"] != 1:
        raise ValueError("tutorial converted-parameter coverage mismatch")
    if receipt["aliases"] != ["embed.weight", "head.weight"]:
        raise ValueError("tutorial tied aliases mismatch")
    if receipt["algorithm_id"] != "tritium.additive-2/tritium.salt-ste@1":
        raise ValueError("tutorial estimator identity mismatch")
    if receipt["planes"] != 2:
        raise ValueError("tutorial plane count mismatch")
    if receipt["artifact_dir"] != "qat-hard":
        raise ValueError("tutorial artifact directory must equal 'qat-hard'")
    for field in (
        "wheel_bytes",
        "hard_artifact_bytes",
        "hard_artifact_file_count",
        "checkpoint_model_bytes",
        "checkpoint_optimizer_bytes",
    ):
        _positive_int(receipt[field], field)
    if receipt["optimizer_state_entries"] != 1:
        raise ValueError("tutorial optimizer state entry count mismatch")
    if receipt["resume_steps"] != 1:
        raise ValueError("tutorial resume step count mismatch")
    for field in (
        "wheel_sha256",
        "artifact_id",
        "hard_state_digest",
        "hard_artifact_tree_sha256",
        "checkpoint_model_sha256",
        "checkpoint_optimizer_sha256",
        "receipt_id",
    ):
        _digest(receipt[field], field)
    wheel_name = receipt["wheel_name"]
    if (
        not isinstance(wheel_name, str)
        or not wheel_name
        or Path(wheel_name).name != wheel_name
        or not wheel_name.endswith(".whl")
    ):
        raise ValueError("tutorial wheel_name must be a wheel basename")
    if receipt["receipt_id"] != receipt_id(receipt):
        raise ValueError("tutorial receipt identity mismatch")

    root = receipt_path.parent.resolve(strict=True)
    artifact = root / "qat-hard"
    observed_tree = tree_identity(artifact)
    declared_tree = {
        "bytes": receipt["hard_artifact_bytes"],
        "file_count": receipt["hard_artifact_file_count"],
        "sha256": receipt["hard_artifact_tree_sha256"],
    }
    if observed_tree != declared_tree:
        raise ValueError("tutorial hard artifact tree identity mismatch")
    checkpoint = root / "latent-checkpoint"
    if (
        checkpoint.is_symlink()
        or not checkpoint.is_dir()
        or checkpoint.resolve().parent != root
    ):
        raise ValueError("tutorial checkpoint must be an ordinary directory contained in result")
    for filename, bytes_field, digest_field in (
        ("model.safetensors", "checkpoint_model_bytes", "checkpoint_model_sha256"),
        ("optimizer.pt", "checkpoint_optimizer_bytes", "checkpoint_optimizer_sha256"),
    ):
        path = _ordinary_file(checkpoint / filename, f"tutorial {filename}")
        if path.parent != checkpoint.resolve():
            raise ValueError(f"tutorial {filename} is outside the checkpoint")
        if (
            path.stat().st_size != receipt[bytes_field]
            or _file_sha256(path) != receipt[digest_field]
        ):
            raise ValueError(f"tutorial {digest_field} file identity mismatch")
    if expected_wheel is not None:
        wheel = _ordinary_file(expected_wheel, "tutorial candidate wheel")
        if (
            wheel.name != wheel_name
            or wheel.stat().st_size != receipt["wheel_bytes"]
            or _file_sha256(wheel) != receipt["wheel_sha256"]
        ):
            raise ValueError("tutorial receipt does not bind the candidate wheel")
    return receipt


__all__ = ["SCHEMA", "receipt_id", "tree_identity", "validate_receipt"]
