"""Dependency-free wheel and installed-file identity primitives."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import zipfile


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _ordinary_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an ordinary non-symlink file")
    return path.resolve(strict=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _portable_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or str(path) != value
    ):
        raise ValueError(f"observability {label} is not a canonical relative path")
    return path


def wheel_identity(wheel_path: Path) -> dict[str, object]:
    """Validate one wheel RECORD and return its immutable installed inventory."""

    wheel_path = _ordinary_file(wheel_path, "observability candidate wheel")
    try:
        archive = zipfile.ZipFile(wheel_path)
    except zipfile.BadZipFile as error:
        raise ValueError("observability candidate wheel is not a valid ZIP archive") from error
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("observability candidate wheel contains duplicate paths")
        files = {}
        for info in infos:
            path = _portable_path(info.filename.rstrip("/"), "wheel member")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError("observability candidate wheel contains a symlink")
            if not info.is_dir():
                files[str(path)] = archive.read(info)
        records = [name for name in files if name.endswith(".dist-info/RECORD")]
        metadata_files = [name for name in files if name.endswith(".dist-info/METADATA")]
        if len(records) != 1 or len(metadata_files) != 1:
            raise ValueError("observability candidate wheel metadata inventory differs")
        dist_info = records[0].removesuffix("RECORD")
        if metadata_files[0] != dist_info + "METADATA":
            raise ValueError("observability candidate wheel dist-info roots differ")
        metadata = BytesParser().parsebytes(files[metadata_files[0]])
        if metadata.get("Name", "").lower().replace("_", "-") != "tritium-torch":
            raise ValueError("observability candidate wheel distribution name differs")
        version = metadata.get("Version")
        if not version:
            raise ValueError("observability candidate wheel version is absent")
        try:
            rows = list(csv.reader(io.StringIO(files[records[0]].decode("utf-8"))))
        except (UnicodeDecodeError, csv.Error) as error:
            raise ValueError("observability candidate wheel RECORD is invalid") from error
        if not rows or any(len(row) != 3 for row in rows):
            raise ValueError("observability candidate wheel RECORD fields differ")
        recorded_paths = [row[0] for row in rows]
        if len(recorded_paths) != len(set(recorded_paths)) or set(recorded_paths) != set(files):
            raise ValueError("observability candidate wheel RECORD coverage differs")
        entries = []
        for logical, encoded_digest, encoded_size in rows:
            _portable_path(logical, "RECORD member")
            if logical == records[0]:
                if encoded_digest or encoded_size:
                    raise ValueError("observability candidate wheel RECORD self-hash differs")
                continue
            if not encoded_digest.startswith("sha256="):
                raise ValueError("observability candidate wheel RECORD digest differs")
            try:
                size = int(encoded_size)
            except ValueError as error:
                raise ValueError("observability candidate wheel RECORD size differs") from error
            payload = files[logical]
            digest = hashlib.sha256(payload).digest()
            expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
            if encoded_digest.removeprefix("sha256=") != expected or size != len(payload):
                raise ValueError("observability candidate wheel RECORD identity differs")
            entries.append(
                {"path": logical, "bytes": size, "sha256": "sha256:" + digest.hex()}
            )
        record_payload = files[records[0]]
        entries.append(
            {
                "path": records[0],
                "bytes": len(record_payload),
                "sha256": "sha256:" + hashlib.sha256(record_payload).hexdigest(),
            }
        )
        entries.sort(key=lambda entry: str(entry["path"]))
        required = {
            "tritium/__init__.py",
            "tritium/torch/_telemetry_binary.py",
            "tritium/torch/_wheel_identity.py",
            "tritium/torch/qualify_observability.py",
            "tritium/torch/observability_receipt.py",
        }
        if not required.issubset({str(entry["path"]) for entry in entries}):
            raise ValueError("observability candidate wheel omits qualification code")
        return {
            "distribution_version": version,
            "file_count": len(entries),
            "tree_sha256": "sha256:" + hashlib.sha256(_canonical(entries)).hexdigest(),
            "entries": entries,
            "record_path": records[0],
        }


__all__ = ["file_sha256", "wheel_identity"]
