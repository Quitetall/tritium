"""Strict, bounded, resumable artifacts for generic module PTQ conversion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import torch

from .config import TernaryConfig
from .coverage import CoverageReport
from .projection import TernaryPlane

Pathish = Union[str, os.PathLike[str]]
_MANIFEST = "conversion.json"
_TOP_FIELDS = {
    "schema_version",
    "artifact_kind",
    "source_model_digest",
    "evidence_id",
    "algorithm_id",
    "recipe_id",
    "artifact_id",
    "config",
    "coverage",
    "weight_receipts",
}
_WEIGHT_FIELDS_V1 = {
    "schema_version",
    "recipe_id",
    "path",
    "aliases",
    "shape",
    "weighted_mse",
    "planes",
}
_WEIGHT_FIELDS_V2 = {
    *_WEIGHT_FIELDS_V1,
    "fit_chunk_rows",
    "max_working_bytes",
}
_RECEIPT_REF_FIELDS = {"file", "digest", "bytes"}
_TRIT_BYTES = frozenset((0, 1, 255))
_PLANE_FIELDS = {
    "trits_file",
    "trits_digest",
    "trits_bytes",
    "scales_file",
    "scales_digest",
    "scales_bytes",
    "scales_shape",
    "group_size",
}


@dataclass(frozen=True)
class FittedPlaneRef:
    trits_path: Path
    trits_digest: str
    trits_bytes: int
    scales_path: Path
    scales_digest: str
    scales_bytes: int
    scales_shape: Tuple[int, ...]
    group_size: int


@dataclass(frozen=True)
class FittedWeightRef:
    path: str
    aliases: Tuple[str, ...]
    shape: Tuple[int, int]
    planes: Tuple[FittedPlaneRef, ...]
    weighted_mse: float
    fit_chunk_rows: int
    max_working_bytes: int | None


@dataclass(frozen=True, eq=False)
class FittedWeight:
    path: str
    aliases: Tuple[str, ...]
    planes: Tuple[TernaryPlane, ...]
    weighted_mse: float


@dataclass(frozen=True)
class ModuleQuantizationResult:
    artifact_dir: Path
    artifact_id: str
    source_model_digest: str
    evidence_id: str
    algorithm_id: str
    recipe_id: str
    config: TernaryConfig
    coverage: CoverageReport
    weights: Tuple[FittedWeightRef, ...]
    schema_version: int = 2

    @property
    def weight_names(self) -> Tuple[str, ...]:
        return tuple(weight.path for weight in self.weights)

    def weight(self, path: str) -> FittedWeight:
        """Load and rehash one fitted weight without retaining other weights."""

        try:
            reference = next(weight for weight in self.weights if weight.path == path)
        except StopIteration as error:
            raise KeyError(path) from error
        planes = []
        rows, columns = reference.shape
        for plane in reference.planes:
            trits_payload = _read_exact(
                plane.trits_path,
                plane.trits_digest,
                plane.trits_bytes,
            )
            scales_payload = _read_exact(
                plane.scales_path,
                plane.scales_digest,
                plane.scales_bytes,
            )
            trits = torch.frombuffer(trits_payload, dtype=torch.int8).reshape(
                rows, columns
            )
            scales = torch.frombuffer(scales_payload, dtype=torch.float16).reshape(
                plane.scales_shape
            )
            if not bool(torch.all((trits >= -1) & (trits <= 1))):
                raise ValueError("conversion plane contains non-ternary values")
            if not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
                raise ValueError("conversion plane scales are invalid")
            planes.append(
                TernaryPlane(
                    trits=trits,
                    scales=scales,
                    group_size=plane.group_size,
                )
            )
        return FittedWeight(
            path=reference.path,
            aliases=reference.aliases,
            planes=tuple(planes),
            weighted_mse=reference.weighted_mse,
        )


def _pairs_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate conversion manifest field {key!r}")
        value[key] = item
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def module_recipe_id(
    source_model_digest: str,
    evidence_id: str,
    algorithm_id: str,
    config: TernaryConfig,
    coverage: CoverageReport,
) -> str:
    identity = {
        "schema_version": 1,
        "source_model_digest": source_model_digest,
        "evidence_id": evidence_id,
        "algorithm_id": algorithm_id,
        "config": config.to_dict(),
        "coverage": coverage.to_dict(),
    }
    return _digest_bytes(_canonical(identity))


def _digest_file(
    path: Path,
    maximum: int,
    allowed_bytes: frozenset[int] | None = None,
) -> Tuple[str, int]:
    metadata = path.lstat()
    if path.is_symlink() or not path.is_file():
        raise ValueError("conversion payload must be an ordinary file")
    if metadata.st_size <= 0 or metadata.st_size > maximum:
        raise ValueError("conversion payload exceeds byte ceiling")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            if allowed_bytes is not None and not set(chunk).issubset(allowed_bytes):
                raise ValueError("conversion plane contains non-ternary values")
            digest.update(chunk)
    return "sha256:" + digest.hexdigest(), metadata.st_size


def _read_exact(path: Path, digest: str, byte_count: int) -> bytearray:
    """Read and hash one payload into one exact-size allocation."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("conversion payload must be an ordinary file") from error
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != byte_count:
            raise ValueError("conversion payload identity mismatch")
        payload = bytearray(byte_count)
        view = memoryview(payload)
        hasher = hashlib.sha256()
        offset = 0
        while offset < byte_count:
            end = min(offset + 1024 * 1024, byte_count)
            count = stream.readinto(view[offset:end])
            if count is None or count <= 0:
                raise ValueError("conversion payload identity mismatch")
            hasher.update(view[offset : offset + count])
            offset += count
        if stream.read(1):
            raise ValueError("conversion payload identity mismatch")
        del view
    if "sha256:" + hasher.hexdigest() != digest:
        raise ValueError("conversion payload identity mismatch")
    return payload


def _read_json(path: Path, maximum: int = 1024 * 1024) -> Dict[str, Any]:
    metadata = path.lstat()
    if path.is_symlink() or not path.is_file() or metadata.st_size > maximum:
        raise ValueError("conversion receipt must be a bounded ordinary file")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_pairs_without_duplicates)
    if not isinstance(value, dict):
        raise ValueError("conversion receipt must contain one JSON object")
    return value


def _validate_digest(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
    ):
        raise ValueError(f"invalid {field}")
    try:
        bytes.fromhex(value[7:])
    except ValueError as error:
        raise ValueError(f"invalid {field}") from error
    return value


def _load_weight_receipt(
    directory: Path,
    receipt_name: str,
    index: int,
    maximum_bytes: int,
    expected_recipe_id: str,
    expected_schema_version: int,
) -> FittedWeightRef:
    if receipt_name != f"weight-{index:05d}.json":
        raise ValueError("conversion weight receipts are out of canonical order")
    value = _read_json(directory / receipt_name)
    schema_version = value.get("schema_version")
    expected_fields = {
        1: _WEIGHT_FIELDS_V1,
        2: _WEIGHT_FIELDS_V2,
    }.get(expected_schema_version)
    if schema_version != expected_schema_version or set(value) != expected_fields:
        raise ValueError("conversion weight receipt fields differ from schema")
    if value["recipe_id"] != expected_recipe_id:
        raise ValueError("conversion weight receipt belongs to another recipe")
    aliases = value["aliases"]
    shape = value["shape"]
    if (
        not isinstance(value["path"], str)
        or not value["path"]
        or not isinstance(aliases, list)
        or not aliases
        or any(not isinstance(alias, str) or not alias for alias in aliases)
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
    ):
        raise ValueError("conversion weight identity or geometry is invalid")
    weighted_mse = value["weighted_mse"]
    if (
        type(weighted_mse) not in {int, float}
        or not math.isfinite(weighted_mse)
        or weighted_mse < 0
    ):
        raise ValueError("conversion weighted_mse must be finite and nonnegative")
    fit_chunk_rows = value.get("fit_chunk_rows", shape[0])
    max_working_bytes = value.get("max_working_bytes")
    if type(fit_chunk_rows) is not int or not 1 <= fit_chunk_rows <= shape[0]:
        raise ValueError("conversion working-set receipt is invalid")
    if max_working_bytes is not None and (
        type(max_working_bytes) is not int or max_working_bytes <= 0
    ):
        raise ValueError("conversion working-set receipt is invalid")
    plane_values = value["planes"]
    if not isinstance(plane_values, list) or not 1 <= len(plane_values) <= 3:
        raise ValueError("conversion weight requires one to three planes")
    planes = []
    elements = shape[0] * shape[1]
    for plane_index, plane in enumerate(plane_values):
        if not isinstance(plane, dict) or set(plane) != _PLANE_FIELDS:
            raise ValueError("conversion plane fields differ from schema")
        trits_name = f"weight-{index:05d}-plane-{plane_index}.trits.i8"
        scales_name = f"weight-{index:05d}-plane-{plane_index}.scales.f16le"
        if plane["trits_file"] != trits_name or plane["scales_file"] != scales_name:
            raise ValueError("conversion plane filenames are noncanonical")
        scales_shape = plane["scales_shape"]
        if (
            plane["trits_bytes"] != elements
            or not isinstance(scales_shape, list)
            or not scales_shape
            or any(
                type(dimension) is not int or dimension <= 0
                for dimension in scales_shape
            )
            or scales_shape != [shape[0], 1]
            or plane["scales_bytes"] != math.prod(scales_shape) * 2
            or type(plane["group_size"]) is not int
            or plane["group_size"] != shape[1]
        ):
            raise ValueError("conversion plane byte ledger or geometry is invalid")
        trits_digest, trits_bytes = _digest_file(
            directory / trits_name, maximum_bytes, _TRIT_BYTES
        )
        scales_digest, scales_bytes = _digest_file(
            directory / scales_name, maximum_bytes
        )
        if (
            trits_digest != _validate_digest(plane["trits_digest"], "trits_digest")
            or scales_digest
            != _validate_digest(plane["scales_digest"], "scales_digest")
            or trits_bytes != plane["trits_bytes"]
            or scales_bytes != plane["scales_bytes"]
        ):
            raise ValueError("conversion plane payload identity mismatch")
        planes.append(
            FittedPlaneRef(
                trits_path=directory / trits_name,
                trits_digest=trits_digest,
                trits_bytes=trits_bytes,
                scales_path=directory / scales_name,
                scales_digest=scales_digest,
                scales_bytes=scales_bytes,
                scales_shape=tuple(scales_shape),
                group_size=plane["group_size"],
            )
        )
    return FittedWeightRef(
        path=value["path"],
        aliases=tuple(aliases),
        shape=(shape[0], shape[1]),
        planes=tuple(planes),
        weighted_mse=float(weighted_mse),
        fit_chunk_rows=fit_chunk_rows,
        max_working_bytes=max_working_bytes,
    )


def load_module_conversion(
    artifact_dir: Pathish,
    *,
    max_payload_bytes: int = 8 * 1024 * 1024 * 1024,
) -> ModuleQuantizationResult:
    requested = Path(artifact_dir)
    if requested.is_symlink():
        raise ValueError("module conversion directory must not be a symlink")
    directory = requested.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("module conversion must be an ordinary directory")
    value = _read_json(directory / _MANIFEST)
    schema_version = value.get("schema_version")
    if set(value) != _TOP_FIELDS or schema_version not in {1, 2}:
        raise ValueError("module conversion manifest fields differ from schema")
    if value["artifact_kind"] != f"tritium.module-additive-ptq-v{schema_version}":
        raise ValueError("unsupported module conversion artifact kind")
    for field in (
        "source_model_digest",
        "evidence_id",
        "recipe_id",
        "artifact_id",
    ):
        _validate_digest(value[field], field)
    if not isinstance(value["algorithm_id"], str) or not value["algorithm_id"]:
        raise ValueError("invalid conversion algorithm_id")
    receipt_values = value["weight_receipts"]
    if not isinstance(receipt_values, list) or not receipt_values:
        raise ValueError("module conversion has no weight receipts")
    receipt_names = []
    weights = []
    for index, reference in enumerate(receipt_values):
        if not isinstance(reference, dict) or set(reference) != _RECEIPT_REF_FIELDS:
            raise ValueError("conversion receipt reference fields differ from schema")
        name = f"weight-{index:05d}.json"
        if reference["file"] != name or type(reference["bytes"]) is not int:
            raise ValueError("conversion receipt reference is noncanonical")
        digest, byte_count = _digest_file(directory / name, 1024 * 1024)
        if (
            digest != _validate_digest(reference["digest"], "receipt digest")
            or byte_count != reference["bytes"]
        ):
            raise ValueError("conversion weight receipt identity mismatch")
        receipt_names.append(name)
        weights.append(
            _load_weight_receipt(
                directory,
                name,
                index,
                max_payload_bytes,
                value["recipe_id"],
                schema_version,
            )
        )
    weights = tuple(weights)
    expected_files = {_MANIFEST, *receipt_names}
    for weight in weights:
        for plane in weight.planes:
            expected_files.add(plane.trits_path.name)
            expected_files.add(plane.scales_path.name)
    if {child.name for child in directory.iterdir()} != expected_files:
        raise ValueError("module conversion directory contains unknown files")
    identity = dict(value)
    artifact_id = identity.pop("artifact_id")
    if artifact_id != _digest_bytes(_canonical(identity)):
        raise ValueError("module conversion artifact identity mismatch")
    config = TernaryConfig.from_dict(value["config"])
    coverage = CoverageReport.from_dict(value["coverage"])
    if value["recipe_id"] != module_recipe_id(
        value["source_model_digest"],
        value["evidence_id"],
        value["algorithm_id"],
        config,
        coverage,
    ):
        raise ValueError("module conversion recipe identity mismatch")
    if any(len(weight.planes) != config.planes for weight in weights):
        raise ValueError("module conversion plane count differs from recipe")
    return ModuleQuantizationResult(
        artifact_dir=directory,
        artifact_id=artifact_id,
        source_model_digest=value["source_model_digest"],
        evidence_id=value["evidence_id"],
        algorithm_id=value["algorithm_id"],
        recipe_id=value["recipe_id"],
        config=config,
        coverage=coverage,
        weights=weights,
        schema_version=schema_version,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sync_directory(directory: Path) -> None:
    """Persist completed renames where directory fsync is supported."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


class WeightCheckpointWriter:
    """Append row chunks directly into one weight's final plane files."""

    def __init__(
        self,
        directory: Path,
        index: int,
        path: str,
        aliases: Tuple[str, ...],
        shape: Tuple[int, int],
        plane_count: int,
        fit_chunk_rows: int,
        max_working_bytes: int,
    ) -> None:
        self.directory = directory
        self.index = index
        self.path = path
        self.aliases = aliases
        self.shape = shape
        self.plane_count = plane_count
        self.fit_chunk_rows = fit_chunk_rows
        self.max_working_bytes = max_working_bytes
        self.rows_written = 0
        self._files = []
        self._digests = []
        self._bytes = []
        self._temporary_paths = []
        self._final_paths = []
        for plane_index in range(plane_count):
            for suffix in ("trits.i8", "scales.f16le"):
                final = directory / f"weight-{index:05d}-plane-{plane_index}.{suffix}"
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".tmp-plane-", dir=str(directory)
                )
                self._files.append(os.fdopen(descriptor, "wb"))
                self._digests.append(hashlib.sha256())
                self._bytes.append(0)
                self._temporary_paths.append(Path(temporary_name))
                self._final_paths.append(final)

    def append(self, planes: Tuple[TernaryPlane, ...]) -> None:
        if len(planes) != self.plane_count:
            raise ValueError("fit chunk plane count differs from conversion recipe")
        chunk_rows = planes[0].trits.shape[0]
        if chunk_rows <= 0 or self.rows_written + chunk_rows > self.shape[0]:
            raise ValueError("fit chunk rows exceed weight geometry")
        for plane_index, plane in enumerate(planes):
            if (
                tuple(plane.trits.shape) != (chunk_rows, self.shape[1])
                or tuple(plane.scales.shape) != (chunk_rows, 1)
                or plane.group_size != self.shape[1]
            ):
                raise ValueError("fit chunk plane geometry differs from weight")
            trits = plane.trits.detach().to(torch.int8).cpu().contiguous()
            scales = plane.scales.detach().to(torch.float16).cpu().contiguous()
            if not bool(torch.all((trits >= -1) & (trits <= 1))):
                raise ValueError("fit chunk contains non-ternary values")
            if not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
                raise ValueError("fit chunk contains invalid scales")
            for slot, payload in (
                (plane_index * 2, trits.numpy().tobytes()),
                (plane_index * 2 + 1, scales.numpy().tobytes()),
            ):
                self._files[slot].write(payload)
                self._digests[slot].update(payload)
                self._bytes[slot] += len(payload)
        self.rows_written += chunk_rows

    def finish(self, weighted_mse: float) -> FittedWeightRef:
        if self.rows_written != self.shape[0]:
            raise ValueError("fit did not emit every output row")
        if not math.isfinite(weighted_mse) or weighted_mse < 0:
            raise ValueError("fit weighted_mse must be finite and nonnegative")
        for stream in self._files:
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        for temporary, final in zip(self._temporary_paths, self._final_paths):
            os.replace(temporary, final)
        _sync_directory(self.directory)
        planes = []
        for plane_index in range(self.plane_count):
            trits_slot = plane_index * 2
            scales_slot = trits_slot + 1
            planes.append(
                FittedPlaneRef(
                    trits_path=self._final_paths[trits_slot],
                    trits_digest="sha256:" + self._digests[trits_slot].hexdigest(),
                    trits_bytes=self._bytes[trits_slot],
                    scales_path=self._final_paths[scales_slot],
                    scales_digest="sha256:" + self._digests[scales_slot].hexdigest(),
                    scales_bytes=self._bytes[scales_slot],
                    scales_shape=(self.shape[0], 1),
                    group_size=self.shape[1],
                )
            )
        return FittedWeightRef(
            path=self.path,
            aliases=self.aliases,
            shape=self.shape,
            planes=tuple(planes),
            weighted_mse=weighted_mse,
            fit_chunk_rows=self.fit_chunk_rows,
            max_working_bytes=self.max_working_bytes,
        )

    def abort(self) -> None:
        for stream in self._files:
            if not stream.closed:
                stream.close()
        for path in self._temporary_paths:
            path.unlink(missing_ok=True)


def seal_module_conversion(
    artifact_dir: Pathish,
    *,
    source_model_digest: str,
    evidence_id: str,
    algorithm_id: str,
    recipe_id: str,
    config: TernaryConfig,
    coverage: CoverageReport,
    records: Iterable[Any],
    fit_weight: Callable[[Any, WeightCheckpointWriter], float],
    fit_chunk_rows: Callable[[Any], int],
    max_working_bytes: int,
) -> ModuleQuantizationResult:
    """Resume per-weight fitting, then atomically seal one conversion manifest."""

    requested = Path(artifact_dir)
    if requested.is_symlink():
        raise ValueError("module conversion directory must not be a symlink")
    directory = requested.resolve()
    if (directory / _MANIFEST).exists():
        result = load_module_conversion(directory)
        expected = (
            source_model_digest,
            evidence_id,
            algorithm_id,
            recipe_id,
            config,
            coverage,
        )
        observed = (
            result.source_model_digest,
            result.evidence_id,
            result.algorithm_id,
            result.recipe_id,
            result.config,
            result.coverage,
        )
        if observed != expected:
            raise ValueError("sealed module conversion belongs to another recipe")
        return result
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if (
            child.name.startswith(".tmp-")
            and child.is_file()
            and not child.is_symlink()
        ):
            child.unlink()
    receipt_references = []
    for index, record in enumerate(records):
        receipt_name = f"weight-{index:05d}.json"
        receipt_path = directory / receipt_name
        if receipt_path.exists():
            reference = _load_weight_receipt(
                directory,
                receipt_name,
                index,
                8 * 1024**3,
                recipe_id,
                2,
            )
            if (
                reference.path != record.weight_aliases[0]
                or reference.aliases != record.weight_aliases
                or reference.shape != (record.outputs, record.features)
                or len(reference.planes) != config.planes
            ):
                raise ValueError(
                    "resumed conversion weight identity differs from calibration"
                )
            receipt_digest, receipt_bytes = _digest_file(receipt_path, 1024 * 1024)
            receipt_references.append(
                {"file": receipt_name, "digest": receipt_digest, "bytes": receipt_bytes}
            )
            continue
        writer = WeightCheckpointWriter(
            directory,
            index,
            record.weight_aliases[0],
            record.weight_aliases,
            (record.outputs, record.features),
            config.planes,
            fit_chunk_rows(record),
            max_working_bytes,
        )
        try:
            weighted_mse = fit_weight(record, writer)
            fitted = writer.finish(weighted_mse)
        except BaseException:
            writer.abort()
            raise
        plane_values = [
            {
                "trits_file": plane.trits_path.name,
                "trits_digest": plane.trits_digest,
                "trits_bytes": plane.trits_bytes,
                "scales_file": plane.scales_path.name,
                "scales_digest": plane.scales_digest,
                "scales_bytes": plane.scales_bytes,
                "scales_shape": list(plane.scales_shape),
                "group_size": plane.group_size,
            }
            for plane in fitted.planes
        ]
        receipt = {
            "schema_version": 2,
            "recipe_id": recipe_id,
            "path": fitted.path,
            "aliases": list(fitted.aliases),
            "shape": list(fitted.shape),
            "weighted_mse": fitted.weighted_mse,
            "fit_chunk_rows": fitted.fit_chunk_rows,
            "max_working_bytes": fitted.max_working_bytes,
            "planes": plane_values,
        }
        _atomic_write(receipt_path, _canonical(receipt))
        _load_weight_receipt(
            directory,
            receipt_name,
            index,
            8 * 1024**3,
            recipe_id,
            2,
        )
        receipt_digest, receipt_bytes = _digest_file(receipt_path, 1024 * 1024)
        receipt_references.append(
            {"file": receipt_name, "digest": receipt_digest, "bytes": receipt_bytes}
        )
    manifest = {
        "schema_version": 2,
        "artifact_kind": "tritium.module-additive-ptq-v2",
        "source_model_digest": source_model_digest,
        "evidence_id": evidence_id,
        "algorithm_id": algorithm_id,
        "recipe_id": recipe_id,
        "config": config.to_dict(),
        "coverage": coverage.to_dict(),
        "weight_receipts": receipt_references,
    }
    expected_files = {reference["file"] for reference in receipt_references}
    for index, reference in enumerate(receipt_references):
        receipt = _read_json(directory / reference["file"])
        for plane in receipt["planes"]:
            expected_files.add(plane["trits_file"])
            expected_files.add(plane["scales_file"])
    unknown = {
        child.name
        for child in directory.iterdir()
        if child.name not in expected_files and not child.name.startswith(".tmp-")
    }
    if unknown:
        raise ValueError("module conversion work directory contains unknown files")
    manifest["artifact_id"] = _digest_bytes(_canonical(manifest))
    _atomic_write(directory / _MANIFEST, _canonical(manifest))
    return load_module_conversion(directory)


__all__ = [
    "FittedPlaneRef",
    "FittedWeight",
    "FittedWeightRef",
    "ModuleQuantizationResult",
    "WeightCheckpointWriter",
    "load_module_conversion",
    "module_recipe_id",
    "seal_module_conversion",
]
