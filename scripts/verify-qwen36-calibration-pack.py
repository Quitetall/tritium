#!/usr/bin/env python3
"""Verify the pinned Qwen tokenizer and frozen calibration token pack.

This produces pack-provenance evidence only. It does not prove that model
activations were captured from these tokens or that an S2KF set is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import stat
import struct
from typing import Any


SCHEMA = "tritium.qwen36-calibration-pack-receipt.v1"
PACK_SCHEMA = "tritium.stage7-token-evidence-pack.v1"
REPOSITORY = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
VOCAB_SIZE = 248_320
PARTITIONS = ("calibration", "refinement", "validation", "evaluation")
SEQUENCES_PER_PARTITION = 512
TOKENS_PER_SEQUENCE = 2_048
TOKENIZER_FILES = ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")
DATASETS = {
    "allenai/c4": {
        "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        "config": "en",
        "data_dir": None,
        "split": "train",
        "text_field": "text",
        "sequences": 256,
    },
    "open-web-math/open-web-math": {
        "revision": "fde8ef8de2300f5e778f56261843dab89f230815",
        "config": "default",
        "data_dir": None,
        "split": "train",
        "text_field": "text",
        "sequences": 128,
    },
    "bigcode/starcoderdata": {
        "revision": "9fc30b578cedaec69e47302df72cf00feed7c8c4",
        "config": "default",
        "data_dir": "python",
        "split": "train",
        "text_field": "content",
        "sequences": 128,
    },
}
TOKEN_PAYLOAD_BYTES = (
    len(PARTITIONS) * SEQUENCES_PER_PARTITION * TOKENS_PER_SEQUENCE * 4
)
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
HEX = frozenset("0123456789abcdef")
OFFICIAL_IDENTITY_MODULE = runpy.run_path(
    Path(__file__).with_name("verify-qwen36-official-source-identity.py")
)


class CalibrationPackError(ValueError):
    """Calibration pack or source identity does not match the frozen contract."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate field {key!r}")
        value[key] = item
    return value


def _load_json(path: Path, label: str, limit: int) -> dict[str, Any]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise CalibrationPackError(f"{label} must be an ordinary file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CalibrationPackError(f"{label} must be an ordinary file")
        if before.st_size <= 0 or before.st_size > limit:
            raise CalibrationPackError(f"{label} exceeds its size limit")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or current.st_ino != after.st_ino
            or not stat.S_ISREG(current.st_mode)
        ):
            raise CalibrationPackError(f"{label} changed during verification")
    except OSError as error:
        raise CalibrationPackError(f"cannot read {label}") from error
    finally:
        os.close(descriptor)
    try:
        value = json.loads(
            data,
            object_pairs_hook=_reject_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CalibrationPackError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CalibrationPackError(f"{label} must be a JSON object")
    return value


def _sha256_text(value: Any, label: str, *, prefixed: bool = False) -> str:
    if not isinstance(value, str):
        raise CalibrationPackError(f"{label} must be a SHA-256 digest")
    prefix = "sha256:" if prefixed else ""
    raw = value.removeprefix(prefix)
    if (prefixed and not value.startswith(prefix)) or len(raw) != 64 or any(
        char not in HEX for char in raw
    ):
        raise CalibrationPackError(f"{label} must be a lowercase SHA-256 digest")
    return raw


def _hash_regular_file(path: Path, algorithm: str) -> tuple[int, str, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CalibrationPackError(f"cannot open ordinary file {path.name}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CalibrationPackError(f"{path.name} is not a regular file")
        if algorithm == "sha256":
            digest = hashlib.sha256()
        elif algorithm == "git-sha1":
            digest = hashlib.sha1(usedforsecurity=False)
            digest.update(f"blob {before.st_size}\0".encode("ascii"))
        else:
            raise CalibrationPackError("unsupported official file digest algorithm")
        sha256 = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                sha256.update(chunk)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or current.st_ino != after.st_ino
            or not stat.S_ISREG(current.st_mode)
        ):
            raise CalibrationPackError(f"{path.name} changed during verification")
        return after.st_size, digest.hexdigest(), sha256.hexdigest()
    finally:
        os.close(descriptor)


def _tokenizer_digest(model_dir: Path, identity: dict[str, Any]) -> str:
    if (
        identity.get("repository") != REPOSITORY
        or identity.get("revision") != REVISION
        or identity.get("result") != "pass"
    ):
        raise CalibrationPackError("official source identity does not match pinned Qwen")
    entries = {
        entry["name"]: entry
        for entry in identity.get("files", [])
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise CalibrationPackError("Qwen model directory must be an ordinary directory")
    config_entry = entries.get("config.json")
    if config_entry is None:
        raise CalibrationPackError("official Qwen inventory lacks config.json")
    config_path = model_dir / "config.json"
    config_size, config_digest, _ = _hash_regular_file(
        config_path, config_entry["algorithm"]
    )
    if config_size != config_entry["size"] or config_digest != config_entry["digest"]:
        raise CalibrationPackError("local config differs from official Qwen config")
    config = _load_json(config_path, "Qwen config", 4 * 1024 * 1024)
    text_config = config.get("text_config")
    vocab_size = text_config.get("vocab_size") if isinstance(text_config, dict) else None
    if vocab_size != VOCAB_SIZE:
        raise CalibrationPackError("official Qwen text vocabulary size differs")
    records = []
    for name in TOKENIZER_FILES:
        entry = entries.get(name)
        if entry is None:
            raise CalibrationPackError(f"official Qwen inventory lacks {name}")
        path = model_dir / name
        size, digest, sha256_digest = _hash_regular_file(path, entry["algorithm"])
        if size != entry["size"] or digest != entry["digest"]:
            raise CalibrationPackError(f"local tokenizer asset differs from official {name}")
        records.append({"path": name, "bytes": size, "sha256": sha256_digest})
    records.sort(key=lambda record: record["path"])
    return "sha256:" + hashlib.sha256(canonical(records)).hexdigest()


def _safe_payload_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise CalibrationPackError("token payload path must be a string")
    logical = PurePosixPath(relative)
    if logical.as_posix() != "stage7.u32le":
        raise CalibrationPackError("Qwen token payload must use canonical stage7.u32le path")
    path = root.joinpath(*logical.parts)
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise CalibrationPackError("token payload path escapes the pack directory") from error
    if path.is_symlink() or not path.is_file():
        raise CalibrationPackError("token payload must be an ordinary file")
    return path


def verify_pack(
    manifest_path: Path,
    model_dir: Path,
    official_identity_path: Path,
) -> dict[str, Any]:
    """Validate Qwen tokenizer assets and every token/sample record in a pack."""
    try:
        identity = OFFICIAL_IDENTITY_MODULE["validate_identity_receipt"](
            official_identity_path
        )
    except (ValueError, OSError, KeyError) as error:
        raise CalibrationPackError("official source identity receipt is invalid") from error
    tokenizer_digest = _tokenizer_digest(model_dir, identity)
    if manifest_path.is_symlink() or manifest_path.parent.is_symlink():
        raise CalibrationPackError("token evidence manifest and root must not be symlinks")
    manifest = _load_json(manifest_path, "token evidence manifest", MAX_MANIFEST_BYTES)
    fields = {
        "schema", "pack_id", "tokenizer_digest", "tokenizer_vocab_size",
        "token_encoding", "tokens", "partitions",
    }
    if set(manifest) != fields or manifest["schema"] != PACK_SCHEMA:
        raise CalibrationPackError("token evidence manifest fields or schema differ")
    expected_pack_id = "sha256:" + hashlib.sha256(
        canonical({key: value for key, value in manifest.items() if key != "pack_id"})
    ).hexdigest()
    if manifest["pack_id"] != expected_pack_id:
        raise CalibrationPackError("token evidence pack ID differs")
    if (
        manifest["tokenizer_digest"] != tokenizer_digest
        or manifest["tokenizer_vocab_size"] != VOCAB_SIZE
        or manifest["token_encoding"] != "u32le"
    ):
        raise CalibrationPackError("token pack tokenizer identity differs from official Qwen")
    token_record = manifest["tokens"]
    if not isinstance(token_record, dict) or set(token_record) != {"path", "bytes", "sha256"}:
        raise CalibrationPackError("token payload record fields differ")
    if token_record["bytes"] != TOKEN_PAYLOAD_BYTES:
        raise CalibrationPackError("token payload geometry differs from four frozen partitions")
    _sha256_text(token_record["sha256"], "token payload digest")
    pack_root = manifest_path.parent.resolve(strict=True)
    if not pack_root.is_dir():
        raise CalibrationPackError("token pack root must be an ordinary directory")
    token_path = _safe_payload_path(pack_root, token_record["path"])
    payload_digest = hashlib.sha256()
    payload = bytearray()
    try:
        descriptor = os.open(
            token_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as error:
        raise CalibrationPackError("cannot open ordinary token payload") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != TOKEN_PAYLOAD_BYTES:
            raise CalibrationPackError("token payload file type or size differs")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            remaining = TOKEN_PAYLOAD_BYTES
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise CalibrationPackError("token payload is truncated")
                payload_digest.update(chunk)
                payload.extend(chunk)
                remaining -= len(chunk)
            if stream.read(1):
                raise CalibrationPackError("token payload contains trailing bytes")
        after = os.fstat(descriptor)
        current = token_path.stat(follow_symlinks=False)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or current.st_ino != after.st_ino
            or not stat.S_ISREG(current.st_mode)
        ):
            raise CalibrationPackError("token payload changed during verification")
    finally:
        os.close(descriptor)
    if payload_digest.hexdigest() != token_record["sha256"]:
        raise CalibrationPackError("token payload checksum differs")
    partitions = manifest["partitions"]
    if not isinstance(partitions, dict) or set(partitions) != set(PARTITIONS):
        raise CalibrationPackError("token pack partition inventory differs")
    expected_offset = 0
    seen_sequences: set[str] = set()
    seen_rows: set[tuple[Any, ...]] = set()
    seen_content: set[str] = set()
    sequence_fields = {
        "id", "dataset_repo_id", "dataset_revision", "dataset_config",
        "dataset_data_dir", "dataset_split", "source_rows", "token_offset",
        "token_count", "token_sha256",
    }
    source_row_fields = {"row_index", "text_field", "content_sha256"}
    provenance: dict[str, Any] = {}
    for partition_name in PARTITIONS:
        partition = partitions[partition_name]
        if not isinstance(partition, dict) or set(partition) != {"sampling_seed", "sequences"}:
            raise CalibrationPackError(f"{partition_name} partition fields differ")
        sequences = partition["sequences"]
        if not isinstance(sequences, list) or len(sequences) != SEQUENCES_PER_PARTITION:
            raise CalibrationPackError(f"{partition_name} must contain exactly 512 sequences")
        counts = {name: 0 for name in DATASETS}
        ordered_ids = []
        sequence_payloads = []
        for ordinal, sequence in enumerate(sequences):
            label = f"{partition_name}.sequences[{ordinal}]"
            if not isinstance(sequence, dict) or set(sequence) != sequence_fields:
                raise CalibrationPackError(f"{label} fields differ")
            dataset_name = sequence["dataset_repo_id"]
            if not isinstance(dataset_name, str):
                raise CalibrationPackError(f"{label} dataset name is invalid")
            dataset = DATASETS.get(dataset_name)
            if dataset is None:
                raise CalibrationPackError(f"{label} dataset is outside the frozen mix")
            if any(
                sequence[field] != dataset[expected]
                for field, expected in (
                    ("dataset_revision", "revision"), ("dataset_config", "config"),
                    ("dataset_data_dir", "data_dir"), ("dataset_split", "split"),
                )
            ):
                raise CalibrationPackError(f"{label} dataset provenance differs")
            counts[dataset_name] += 1
            rows = sequence["source_rows"]
            if not isinstance(rows, list) or not rows:
                raise CalibrationPackError(f"{label} source rows are empty")
            for row in rows:
                if not isinstance(row, dict) or set(row) != source_row_fields:
                    raise CalibrationPackError(f"{label} source-row fields differ")
                if type(row["row_index"]) is not int or row["row_index"] < 0:
                    raise CalibrationPackError(f"{label} source row index is invalid")
                if row["text_field"] != dataset["text_field"]:
                    raise CalibrationPackError(f"{label} source text field differs")
                content = _sha256_text(row["content_sha256"], f"{label} source content digest")
                locator = (
                    dataset_name, sequence["dataset_revision"], sequence["dataset_config"],
                    sequence["dataset_data_dir"], sequence["dataset_split"], row["row_index"],
                )
                if locator in seen_rows or content in seen_content:
                    raise CalibrationPackError("token pack reuses a source row or content")
                seen_rows.add(locator)
                seen_content.add(content)
            offset = sequence["token_offset"]
            token_count = sequence["token_count"]
            if type(offset) is not int or offset != expected_offset:
                raise CalibrationPackError(f"{label} token offset is not canonical")
            if type(token_count) is not int or token_count != TOKENS_PER_SEQUENCE:
                raise CalibrationPackError(f"{label} token count differs")
            start = offset * 4
            end = start + token_count * 4
            sequence_bytes = payload[start:end]
            token_digest = _sha256_text(
                sequence["token_sha256"], f"{label} token digest", prefixed=True
            )
            observed_token_digest = hashlib.sha256(sequence_bytes).hexdigest()
            if token_digest != observed_token_digest:
                raise CalibrationPackError(f"{label} token payload digest differs")
            if any(token[0] >= VOCAB_SIZE for token in struct.iter_unpack("<I", sequence_bytes)):
                raise CalibrationPackError(f"{label} token exceeds Qwen vocabulary")
            scope = {key: value for key, value in sequence.items() if key != "id"}
            expected_id = "sha256:" + hashlib.sha256(canonical(scope)).hexdigest()
            if sequence["id"] != expected_id or expected_id in seen_sequences:
                raise CalibrationPackError(f"{label} ID is invalid or duplicated")
            seen_sequences.add(expected_id)
            ordered_ids.append(expected_id)
            sequence_payloads.append(sequence_bytes)
            expected_offset += token_count
        if counts != {name: item["sequences"] for name, item in DATASETS.items()}:
            raise CalibrationPackError(f"{partition_name} dataset proportions differ")
        if type(partition["sampling_seed"]) is not int or partition["sampling_seed"] < 0:
            raise CalibrationPackError(f"{partition_name} sampling seed is invalid")
        if partition_name == "calibration":
            provenance = {
                "sampling_seed": partition["sampling_seed"],
                "sequence_count": len(sequences),
                "tokens_per_sequence": TOKENS_PER_SEQUENCE,
                "token_count": len(sequences) * TOKENS_PER_SEQUENCE,
                "ordered_members_sha256": "sha256:" + hashlib.sha256(
                    canonical(ordered_ids)
                ).hexdigest(),
                "ordered_token_sha256": "sha256:" + hashlib.sha256(
                    b"".join(sequence_payloads)
                ).hexdigest(),
                "sequence_provenance_sha256": "sha256:" + hashlib.sha256(
                    canonical(sequences)
                ).hexdigest(),
                "dataset_counts": counts,
                "dataset_revisions": {name: DATASETS[name]["revision"] for name in DATASETS},
            }
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "result": "pass",
        "source_identity_receipt_id": identity["receipt_id"],
        "source_model_id": identity["source_model_id"],
        "repository": REPOSITORY,
        "revision": REVISION,
        "tokenizer_digest": tokenizer_digest,
        "tokenizer_vocab_size": VOCAB_SIZE,
        "pack_id": manifest["pack_id"],
        "token_payload_sha256": "sha256:" + payload_digest.hexdigest(),
        "token_payload_bytes": TOKEN_PAYLOAD_BYTES,
        "calibration": provenance,
    }
    receipt["receipt_id"] = "sha256:" + hashlib.sha256(canonical(receipt)).hexdigest()
    return receipt


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise CalibrationPackError("receipt parent must be an ordinary existing directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as error:
        raise CalibrationPackError("receipt output must be a new ordinary file") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            encoded = json.dumps(
                value, ensure_ascii=False, indent=2, sort_keys=True
            ).encode()
            stream.write(encoded + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        path.unlink(missing_ok=True)
        raise CalibrationPackError("cannot durably write calibration pack receipt") from error
    if os.name == "posix":
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as error:
            path.unlink(missing_ok=True)
            raise CalibrationPackError(
                "cannot durably sync calibration receipt directory"
            ) from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--official-source-identity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        receipt = verify_pack(args.manifest, args.model_dir, args.official_source_identity)
        _write_new(args.output, receipt)
    except (CalibrationPackError, OSError) as error:
        parser.error(str(error))
    print(f"PASS {receipt['receipt_id']} pack={receipt['pack_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
