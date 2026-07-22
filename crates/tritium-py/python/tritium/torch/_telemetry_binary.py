"""Dependency-free parsers for retained TensorBoard and W&B telemetry."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib


def _ordinary_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an ordinary non-symlink file")
    return path.resolve(strict=True)


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("telemetry protobuf contains a truncated varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
    raise ValueError("telemetry protobuf varint exceeds 64 bits")


def _protobuf_fields(data: bytes) -> dict[int, list[tuple[int, object]]]:
    fields: dict[int, list[tuple[int, object]]] = {}
    offset = 0
    while offset < len(data):
        key, offset = _varint(data, offset)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise ValueError("telemetry protobuf contains field zero")
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire == 1:
            if offset + 8 > len(data):
                raise ValueError("telemetry protobuf fixed64 field is truncated")
            value, offset = data[offset : offset + 8], offset + 8
        elif wire == 2:
            length, offset = _varint(data, offset)
            if offset + length > len(data):
                raise ValueError("telemetry protobuf bytes field is truncated")
            value, offset = data[offset : offset + length], offset + length
        elif wire == 5:
            if offset + 4 > len(data):
                raise ValueError("telemetry protobuf fixed32 field is truncated")
            value, offset = data[offset : offset + 4], offset + 4
        else:
            raise ValueError("telemetry protobuf uses an unsupported wire type")
        fields.setdefault(number, []).append((wire, value))
    return fields


def _one_field(
    fields: dict[int, list[tuple[int, object]]],
    number: int,
    wire: int,
    label: str,
) -> object:
    values = fields.get(number, [])
    if len(values) != 1 or values[0][0] != wire:
        raise ValueError(f"{label} protobuf field differs")
    return values[0][1]


def _tfrecord_payloads(path: Path) -> tuple[bytes, ...]:
    data = _ordinary_file(path, "TensorBoard event file").read_bytes()
    payloads = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < 16:
            raise ValueError("TensorBoard event file has a truncated TFRecord")
        length = struct.unpack_from("<Q", data, offset)[0]
        length_bytes = data[offset : offset + 8]
        length_crc = struct.unpack_from("<I", data, offset + 8)[0]
        end = offset + 12 + length + 4
        if length == 0 or end > len(data):
            raise ValueError("TensorBoard event file has invalid TFRecord framing")
        payload = data[offset + 12 : offset + 12 + length]
        payload_crc = struct.unpack_from("<I", data, offset + 12 + length)[0]
        if length_crc != masked_crc32c(length_bytes) or payload_crc != masked_crc32c(payload):
            raise ValueError("TensorBoard event file CRC differs")
        payloads.append(payload)
        offset = end
    if not payloads or b"brain.Event:2" not in payloads[0]:
        raise ValueError("TensorBoard event stream header differs")
    return tuple(payloads)


def tensorboard_values(
    paths: tuple[Path, ...],
    *,
    expected_scalar_tags: frozenset[str],
    expected_step: int,
) -> tuple[dict[str, float], dict[str, object]]:
    scalars: dict[str, float] = {}
    histogram = None
    for path in paths:
        for event in _tfrecord_payloads(path):
            event_fields = _protobuf_fields(event)
            summaries = event_fields.get(5, [])
            if not summaries:
                continue
            step = _one_field(event_fields, 2, 0, "TensorBoard step")
            if step != expected_step:
                raise ValueError("TensorBoard event step differs")
            for wire, summary_bytes in summaries:
                if wire != 2 or not isinstance(summary_bytes, bytes):
                    raise ValueError("TensorBoard summary field differs")
                summary_fields = _protobuf_fields(summary_bytes)
                for value_wire, value_bytes in summary_fields.get(1, []):
                    if value_wire != 2 or not isinstance(value_bytes, bytes):
                        raise ValueError("TensorBoard value field differs")
                    value_fields = _protobuf_fields(value_bytes)
                    tag_bytes = _one_field(value_fields, 1, 2, "TensorBoard tag")
                    assert isinstance(tag_bytes, bytes)
                    try:
                        tag = tag_bytes.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise ValueError("TensorBoard tag is not UTF-8") from error
                    if tag in scalars or (histogram is not None and tag == histogram["tag"]):
                        raise ValueError("TensorBoard tag is duplicated")
                    if 2 in value_fields:
                        raw = _one_field(value_fields, 2, 5, "TensorBoard scalar")
                        assert isinstance(raw, bytes)
                        scalars[tag] = struct.unpack("<f", raw)[0]
                    elif 5 in value_fields:
                        raw = _one_field(value_fields, 5, 2, "TensorBoard histogram")
                        assert isinstance(raw, bytes)
                        hist = _protobuf_fields(raw)
                        fixed = {}
                        for number, name in (
                            (1, "minimum"),
                            (2, "maximum"),
                            (3, "count"),
                            (4, "sum"),
                            (5, "sum_squares"),
                        ):
                            values = hist.get(number, [])
                            if not values:
                                fixed[name] = 0.0
                            else:
                                encoded = _one_field(hist, number, 1, f"TensorBoard histogram {name}")
                                assert isinstance(encoded, bytes)
                                fixed[name] = struct.unpack("<d", encoded)[0]
                        packed = {}
                        for number, name in ((6, "bucket_limits"), (7, "bucket_counts")):
                            encoded = _one_field(hist, number, 2, f"TensorBoard histogram {name}")
                            assert isinstance(encoded, bytes)
                            if len(encoded) % 8:
                                raise ValueError("TensorBoard histogram packed doubles differ")
                            packed[name] = list(struct.unpack(f"<{len(encoded) // 8}d", encoded))
                        histogram = {"tag": tag, **fixed, **packed}
                    else:
                        raise ValueError("TensorBoard value payload differs")
    if set(scalars) != expected_scalar_tags or histogram is None:
        raise ValueError("TensorBoard metric inventory differs")
    return scalars, histogram


def _wandb_records(path: Path) -> tuple[bytes, ...]:
    data = _ordinary_file(path, "W&B offline run").read_bytes()
    if not data.startswith(b":W&B\xe1\xbe\x00") or len(data) < 7:
        raise ValueError("observability W&B offline run header differs")
    records = []
    fragments: list[bytes] | None = None
    offset = 7
    block_size = 32768
    while offset < len(data):
        block_remaining = block_size - (offset % block_size)
        if block_remaining < 7:
            if any(data[offset : offset + block_remaining]):
                raise ValueError("W&B offline run block padding differs")
            offset += block_remaining
            continue
        if offset + 7 > len(data):
            raise ValueError("W&B offline run record header is truncated")
        checksum, length, kind = struct.unpack_from("<IHB", data, offset)
        offset += 7
        if length == 0 and kind == 0:
            padding = block_size - (offset % block_size)
            if any(data[offset - 7 : offset + padding]):
                raise ValueError("W&B offline run zero record differs")
            offset += padding
            continue
        if kind not in {1, 2, 3, 4} or length > block_size - (offset % block_size):
            raise ValueError("W&B offline run fragment header differs")
        if offset + length > len(data):
            raise ValueError("W&B offline run fragment is truncated")
        payload = data[offset : offset + length]
        offset += length
        expected_checksum = zlib.crc32(payload, zlib.crc32(bytes([kind]))) & 0xFFFFFFFF
        if checksum != expected_checksum:
            raise ValueError("W&B offline run fragment CRC differs")
        if kind == 1:
            if fragments is not None:
                raise ValueError("W&B offline run fragment sequence differs")
            records.append(payload)
        elif kind == 2:
            if fragments is not None:
                raise ValueError("W&B offline run fragment sequence differs")
            fragments = [payload]
        elif kind == 3:
            if fragments is None:
                raise ValueError("W&B offline run fragment sequence differs")
            fragments.append(payload)
        else:
            if fragments is None:
                raise ValueError("W&B offline run fragment sequence differs")
            records.append(b"".join((*fragments, payload)))
            fragments = None
    if fragments is not None or not records:
        raise ValueError("W&B offline run record sequence differs")
    return tuple(records)


def wandb_values(
    path: Path,
    *,
    expected_scalar_tags: frozenset[str],
    expected_histogram_tag: str,
    expected_step: int,
) -> tuple[dict[str, float], dict[str, object]]:
    histories = []
    for record in _wandb_records(path):
        fields = _protobuf_fields(record)
        for wire, history_bytes in fields.get(2, []):
            if wire != 2 or not isinstance(history_bytes, bytes):
                raise ValueError("W&B history field differs")
            histories.append(_protobuf_fields(history_bytes))
    if len(histories) != 1:
        raise ValueError("W&B offline run history count differs")
    history = histories[0]
    step_bytes = _one_field(history, 2, 2, "W&B history step")
    assert isinstance(step_bytes, bytes)
    step = _one_field(_protobuf_fields(step_bytes), 1, 0, "W&B step number")
    if step != expected_step:
        raise ValueError("W&B history step differs")
    scalars = {}
    histogram = {}
    system = {}
    for wire, item_bytes in history.get(1, []):
        if wire != 2 or not isinstance(item_bytes, bytes):
            raise ValueError("W&B history item differs")
        item = _protobuf_fields(item_bytes)
        key_parts = []
        for key_wire, key_bytes in item.get(2, []):
            if key_wire != 2 or not isinstance(key_bytes, bytes):
                raise ValueError("W&B nested key differs")
            try:
                key_parts.append(key_bytes.decode("utf-8"))
            except UnicodeDecodeError as error:
                raise ValueError("W&B nested key is not UTF-8") from error
        if not key_parts and 1 in item:
            key_bytes = _one_field(item, 1, 2, "W&B key")
            assert isinstance(key_bytes, bytes)
            key_parts.append(key_bytes.decode("utf-8"))
        value_bytes = _one_field(item, 16, 2, "W&B JSON value")
        assert isinstance(value_bytes, bytes)
        try:
            value = json.loads(value_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("W&B history value is not JSON") from error
        if len(key_parts) == 1 and key_parts[0] in expected_scalar_tags:
            if key_parts[0] in scalars:
                raise ValueError("W&B scalar metric is duplicated")
            scalars[key_parts[0]] = value
        elif key_parts and key_parts[0] == expected_histogram_tag:
            if len(key_parts) != 2 or key_parts[1] in histogram:
                raise ValueError("W&B histogram field differs")
            histogram[key_parts[1]] = value
        elif len(key_parts) == 1 and key_parts[0] in {"_runtime", "_step", "_timestamp"}:
            system[key_parts[0]] = value
        else:
            raise ValueError("W&B offline run contains an unexpected history key")
    if set(scalars) != expected_scalar_tags or system.get("_step") != expected_step:
        raise ValueError("W&B metric inventory differs")
    return scalars, histogram


__all__ = ["masked_crc32c", "tensorboard_values", "wandb_values"]
