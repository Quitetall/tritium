#!/usr/bin/env python3
"""Fetch the pinned Qwen3.6-27B SafeTensors headers into a metadata fixture.

Only the eight-byte SafeTensors header length and JSON header ranges are read.
Tensor payload ranges are never requested. The output is the canonical metadata
record stream used by the coverage-manifest integration test:

    name<TAB>dtype<TAB>comma-separated-shape<LF>
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import urllib.request
from pathlib import Path
from typing import Any


REPOSITORY = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
BASE_URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}"
SHARDS = tuple(f"model-{index:05d}-of-00015.safetensors" for index in range(1, 16))
MAX_HEADER_BYTES = 1 << 20
MAX_INDEX_BYTES = 1 << 20
NETWORK_TIMEOUT_SECONDS = 30
EXPECTED_TOTAL_SIZE = 55_562_855_904


def fetch_range(filename: str, start: int, end: int) -> tuple[bytes, int]:
    """Fetch one inclusive byte range and reject servers ignoring Range."""

    request = urllib.request.Request(
        f"{BASE_URL}/{filename}", headers={"Range": f"bytes={start}-{end}"}
    )
    with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
        if (
            response.status != 206
            or match is None
            or (int(match[1]), int(match[2])) != (start, end)
        ):
            raise RuntimeError(
                f"server did not honor range for {filename}: "
                f"status={response.status}, content-range={content_range!r}"
            )
        data = response.read(end - start + 1)
        if len(data) != end - start + 1 or response.read(1):
            raise RuntimeError(f"unexpected range length for {filename}")
        return data, int(match[3])


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys while decoding official headers."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def fetch_header(filename: str) -> dict[str, Any]:
    """Read and decode one SafeTensors JSON header by byte range."""

    length_bytes, file_size = fetch_range(filename, 0, 7)
    header_length = struct.unpack("<Q", length_bytes)[0]
    if header_length == 0 or header_length > MAX_HEADER_BYTES:
        raise RuntimeError(f"invalid header length {header_length} for {filename}")
    header, repeated_file_size = fetch_range(filename, 8, 8 + header_length - 1)
    if repeated_file_size != file_size:
        raise RuntimeError(f"file size changed while reading {filename}")
    decoded = json.loads(header, object_pairs_hook=unique_object)

    spans: list[tuple[int, int, str]] = []
    for name, metadata in decoded.items():
        if name == "__metadata__":
            continue
        start, end = metadata["data_offsets"]
        shape = metadata["shape"]
        coefficients = 1
        for dimension in shape:
            coefficients *= dimension
        dtype_bytes = {"BF16": 2}.get(metadata["dtype"])
        if dtype_bytes is None or end - start != coefficients * dtype_bytes:
            raise RuntimeError(f"payload length disagrees with metadata for {name!r}")
        spans.append((start, end, name))

    cursor = 0
    for start, end, name in sorted(spans):
        if start != cursor or end < start:
            raise RuntimeError(
                f"non-contiguous payload before {name!r} in {filename}: "
                f"expected {cursor}, got {start}..{end}"
            )
        cursor = end
    expected_payload = file_size - 8 - header_length
    if cursor != expected_payload:
        raise RuntimeError(
            f"payload extent mismatch for {filename}: {cursor} != {expected_payload}"
        )
    return decoded


def fetch_index() -> dict[str, Any]:
    """Fetch the pinned shard index and reject duplicate JSON object keys."""

    with urllib.request.urlopen(
        f"{BASE_URL}/model.safetensors.index.json", timeout=NETWORK_TIMEOUT_SECONDS
    ) as response:
        encoded = response.read(MAX_INDEX_BYTES + 1)
    if len(encoded) > MAX_INDEX_BYTES:
        raise RuntimeError("pinned shard index exceeds byte bound")
    return json.loads(encoded, object_pairs_hook=unique_object)


def canonical_records() -> bytes:
    """Return sorted canonical metadata records from all pinned shard headers."""

    tensors: dict[str, tuple[str, tuple[int, ...], str]] = {}
    for shard in SHARDS:
        header = fetch_header(shard)
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise RuntimeError(f"duplicate tensor {name!r}")
            dtype = metadata["dtype"]
            shape = tuple(metadata["shape"])
            tensors[name] = (dtype, shape, shard)

    index = fetch_index()
    if index.get("metadata", {}).get("total_size") != EXPECTED_TOTAL_SIZE:
        raise RuntimeError("pinned index total_size changed")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("pinned index has no weight_map object")
    actual_weight_map = {name: shard for name, (_, _, shard) in tensors.items()}
    if weight_map != actual_weight_map:
        raise RuntimeError("pinned index membership disagrees with shard headers")

    total_size = 0
    for dtype, shape, _ in tensors.values():
        if dtype != "BF16":
            raise RuntimeError(f"unexpected dtype {dtype!r}")
        coefficients = 1
        for dimension in shape:
            coefficients *= dimension
        total_size += coefficients * 2
    if total_size != EXPECTED_TOTAL_SIZE:
        raise RuntimeError(f"header total_size changed: {total_size}")

    lines = (
        f"{name}\t{dtype}\t{','.join(str(dimension) for dimension in shape)}\n"
        for name, (dtype, shape, _) in sorted(tensors.items())
    )
    return "".join(lines).encode("utf-8")


def main() -> None:
    """Generate the fixture at the requested path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    records = canonical_records()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(records)
    print(f"wrote {len(records)} bytes to {args.output}")


if __name__ == "__main__":
    main()
