"""Strict parser for the frozen portable-training manifest v1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Dict, Iterable, Tuple, Union


_SCHEMA_ID = "tritium.training_op_manifest"
_SCHEMA_VERSION = 1
_DTYPE = "f32"
_TOP_LEVEL_FIELDS = {"schema_id", "schema_version", "dtype", "operations"}
_OPERATION_FIELDS = {
    "id",
    "category",
    "forward",
    "vjp",
    "mutates",
    "checkpoint_planes",
}
_CATEGORIES = {"graph", "loss", "optimizer", "lifecycle"}
_VJPS = {"none", "first_order"}


class TrainingManifestError(ValueError):
    """The supplied bytes are not the exact frozen v1 training registry."""


@dataclass(frozen=True)
class TrainingOpDescriptorV1:
    """One canonically ordered operation descriptor."""

    id: str
    category: str
    forward: bool
    vjp: str
    mutates: bool
    checkpoint_planes: Tuple[str, ...]


@dataclass(frozen=True)
class TrainingOpManifestV1:
    """Validated immutable view of ``TrainingOpManifestV1``."""

    schema_id: str
    schema_version: int
    dtype: str
    operations: Tuple[TrainingOpDescriptorV1, ...]

    def canonical_json(self) -> bytes:
        """Return the packaged canonical bytes, including the terminal newline."""

        return canonical_training_manifest_json()


def canonical_training_manifest_json() -> bytes:
    """Return the byte-identical manifest copy shipped in the Python package."""

    return (
        resources.files(__package__)
        .joinpath("training_manifest_v1.json")
        .read_bytes()
    )


def _object_without_duplicates(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise TrainingManifestError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _reject_nonstandard_constant(value: str) -> None:
    raise TrainingManifestError(f"invalid JSON numeric constant {value!r}")


def _decode_json(data: bytes) -> Dict[str, Any]:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise TrainingManifestError("training manifest is not valid UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_nonstandard_constant,
        )
    except TrainingManifestError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise TrainingManifestError(f"invalid training manifest JSON: {error}") from error
    if not isinstance(value, dict):
        raise TrainingManifestError("training manifest JSON root must be an object")
    return value


def _validate_wire(value: Dict[str, Any]) -> None:
    if set(value) != _TOP_LEVEL_FIELDS:
        raise TrainingManifestError("training manifest top-level fields differ from v1")
    if type(value["schema_id"]) is not str or value["schema_id"] != _SCHEMA_ID:
        raise TrainingManifestError("unsupported training manifest schema_id")
    if type(value["schema_version"]) is not int:
        raise TrainingManifestError("training manifest schema_version must be an integer")
    if value["schema_version"] != _SCHEMA_VERSION:
        raise TrainingManifestError("unsupported training manifest schema_version")
    if type(value["dtype"]) is not str or value["dtype"] != _DTYPE:
        raise TrainingManifestError("unsupported training manifest dtype")
    operations = value["operations"]
    if type(operations) is not list:
        raise TrainingManifestError("training manifest operations must be an array")
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict) or set(operation) != _OPERATION_FIELDS:
            raise TrainingManifestError(
                f"training operation {index} fields differ from v1"
            )
        if type(operation["id"]) is not str or not operation["id"]:
            raise TrainingManifestError(f"training operation {index} has invalid id")
        if (
            type(operation["category"]) is not str
            or operation["category"] not in _CATEGORIES
        ):
            raise TrainingManifestError(
                f"training operation {index} has invalid category"
            )
        if type(operation["forward"]) is not bool:
            raise TrainingManifestError(
                f"training operation {index} forward must be boolean"
            )
        if type(operation["vjp"]) is not str or operation["vjp"] not in _VJPS:
            raise TrainingManifestError(f"training operation {index} has invalid vjp")
        if type(operation["mutates"]) is not bool:
            raise TrainingManifestError(
                f"training operation {index} mutates must be boolean"
            )
        planes = operation["checkpoint_planes"]
        if type(planes) is not list or any(type(plane) is not str for plane in planes):
            raise TrainingManifestError(
                f"training operation {index} checkpoint_planes must be strings"
            )


@lru_cache(maxsize=1)
def _canonical_wire() -> Dict[str, Any]:
    value = _decode_json(canonical_training_manifest_json())
    _validate_wire(value)
    return value


def parse_training_manifest(
    data: Union[str, bytes, bytearray, memoryview],
) -> TrainingOpManifestV1:
    """Parse text/bytes and require exact v1 fields, types, operations and order."""

    if isinstance(data, str):
        encoded = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray, memoryview)):
        encoded = bytes(data)
    else:
        raise TypeError("training manifest must be str or bytes-like")
    value = _decode_json(encoded)
    _validate_wire(value)
    expected = _canonical_wire()
    if value != expected:
        raise TrainingManifestError(
            "training manifest operations differ from the frozen v1 registry"
        )
    return TrainingOpManifestV1(
        schema_id=value["schema_id"],
        schema_version=value["schema_version"],
        dtype=value["dtype"],
        operations=tuple(
            TrainingOpDescriptorV1(
                id=operation["id"],
                category=operation["category"],
                forward=operation["forward"],
                vjp=operation["vjp"],
                mutates=operation["mutates"],
                checkpoint_planes=tuple(operation["checkpoint_planes"]),
            )
            for operation in value["operations"]
        ),
    )
