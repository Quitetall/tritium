#!/usr/bin/env python3
"""Prepare (but do not execute) the frozen eight-probe Qwen36 GDN study.

The result is a local inventory preflight, not an official-source receipt or a
measurement receipt. It reads only config/index JSON and Safetensors headers;
tensor payloads and the model are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tempfile
from typing import Any


REPOSITORY = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
CLASSES = ("qkv", "output", "gate_up", "down")
FAMILIES = ("deltanet", "full_attention")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024 * 1024
LAYER_NAME = re.compile(r"^model\.language_model\.layers\.(0|[1-9][0-9]*)\.(.+)$")


class PreflightError(ValueError):
    """Pinned-model probe selection is malformed or inconsistent."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreflightError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _read_regular(path: Path, limit: int) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise PreflightError(f"cannot safely open {path.name}") from error
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > limit:
            raise PreflightError(f"{path.name} is not a bounded ordinary file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        after = os.fstat(fd)
        current = path.stat(follow_symlinks=False)
        if (
            len(data) != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise PreflightError(f"{path.name} changed while being read")
        return data, after
    finally:
        os.close(fd)


def _json_file(path: Path) -> tuple[dict[str, Any], str]:
    data, _ = _read_regular(path, MAX_JSON_BYTES)
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(PreflightError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"{path.name} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise PreflightError(f"{path.name} must contain a JSON object")
    return value, hashlib.sha256(data).hexdigest()


def _safetensors_header(model_dir: Path, shard_name: str) -> dict[str, Any]:
    logical = PurePosixPath(shard_name)
    if (
        not shard_name
        or "\\" in shard_name
        or "\0" in shard_name
        or logical.is_absolute()
        or len(logical.parts) != 1
        or ".." in logical.parts
    ):
        raise PreflightError("index contains an unsafe shard path")
    path = model_dir / shard_name
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise PreflightError(f"cannot safely open Safetensors shard {shard_name}") from error
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 10:
            raise PreflightError(f"Safetensors shard {shard_name} is not an ordinary data file")
        prefix = os.read(fd, 8)
        if len(prefix) != 8:
            raise PreflightError(f"Safetensors shard {shard_name} has a truncated header")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 1 or header_size > MAX_HEADER_BYTES or header_size + 8 > before.st_size:
            raise PreflightError(f"Safetensors shard {shard_name} has an invalid header length")
        chunks = bytearray()
        while len(chunks) < header_size:
            chunk = os.read(fd, min(1024 * 1024, header_size - len(chunks)))
            if not chunk:
                raise PreflightError(f"Safetensors shard {shard_name} has a truncated header")
            chunks.extend(chunk)
        after = os.fstat(fd)
        current = path.stat(follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise PreflightError(f"Safetensors shard {shard_name} changed while reading its header")
    finally:
        os.close(fd)
    try:
        header = json.loads(chunks, object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"Safetensors shard {shard_name} has an invalid header") from error
    if not isinstance(header, dict):
        raise PreflightError(f"Safetensors shard {shard_name} header must be an object")
    return header


def _family_and_class(name: str, config: dict[str, Any]) -> tuple[str, str, int]:
    match = LAYER_NAME.fullmatch(name)
    if match is None or not name.endswith(".weight"):
        raise PreflightError(f"unsupported probe tensor name {name!r}")
    layer = int(match.group(1))
    text = config.get("text_config")
    layer_types = text.get("layer_types") if isinstance(text, dict) else None
    if not isinstance(layer_types, list) or layer >= len(layer_types):
        raise PreflightError("config lacks the selected language layer type")
    kind = layer_types[layer]
    family = {"linear_attention": "deltanet", "full_attention": "full_attention"}.get(kind)
    if family is None:
        raise PreflightError(f"unsupported Qwen layer type {kind!r}")
    projection = match.group(2)
    if family == "deltanet":
        suffixes = {
            "linear_attn.in_proj_qkv.weight": "qkv",
            "linear_attn.out_proj.weight": "output",
            "mlp.gate_proj.weight": "gate_up",
            "mlp.down_proj.weight": "down",
        }
    else:
        suffixes = {
            "self_attn.q_proj.weight": "qkv",
            "self_attn.o_proj.weight": "output",
            "mlp.gate_proj.weight": "gate_up",
            "mlp.down_proj.weight": "down",
        }
    tensor_class = suffixes.get(projection)
    if tensor_class is None:
        raise PreflightError(f"tensor {name!r} does not match a frozen probe class")
    return family, tensor_class, layer


def prepare(model_dir: Path, selections: list[str]) -> dict[str, Any]:
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise PreflightError("model directory must be an ordinary directory")
    model_dir = model_dir.resolve(strict=True)
    config, config_digest = _json_file(model_dir / "config.json")
    index, index_digest = _json_file(model_dir / "model.safetensors.index.json")
    text = config.get("text_config")
    if (
        not isinstance(text, dict)
        or text.get("model_type") != "qwen3_5_text"
        or text.get("num_hidden_layers") != 64
        or not isinstance(text.get("layer_types"), list)
        or len(text["layer_types"]) != 64
    ):
        raise PreflightError("config does not match the frozen 64-layer Qwen36 text model")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != 1199:
        raise PreflightError("weight index differs from the frozen 1,199-tensor inventory")
    if not all(isinstance(shard, str) for shard in weight_map.values()):
        raise PreflightError("weight index contains an invalid shard inventory")
    shards = sorted(set(weight_map.values()))
    if not shards:
        raise PreflightError("weight index contains an empty shard inventory")
    headers = {shard: _safetensors_header(model_dir, shard) for shard in shards}
    tensor_metadata: dict[str, dict[str, Any]] = {}
    for shard, header in headers.items():
        for name, metadata in header.items():
            if name not in weight_map:
                continue
            if weight_map[name] != shard or name in tensor_metadata:
                raise PreflightError(f"weight index/header disagreement for tensor {name!r}")
            tensor_metadata[name] = metadata
    if set(tensor_metadata) != set(weight_map):
        raise PreflightError("one or more indexed tensors are absent from Safetensors headers")
    matrix_names = []
    for name, metadata in tensor_metadata.items():
        if not (
            name.startswith("model.language_model.")
            or name.startswith("lm_head.")
            or name.startswith("mtp.")
        ):
            continue
        shape = metadata.get("shape") if isinstance(metadata, dict) else None
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(size) is not int or size <= 0 for size in shape)
        ):
            raise PreflightError(f"language/MTP matrix {name!r} has invalid geometry")
        if len(shape) == 2:
            matrix_names.append(name)
    matrix_names.sort()
    if len(matrix_names) != 506:
        raise PreflightError("source headers do not yield the frozen 506 language/MTP matrices")
    matrix_ordinals = {name: ordinal for ordinal, name in enumerate(matrix_names)}
    if len(selections) != 8:
        raise PreflightError("provide exactly eight FAMILY/CLASS=TENSOR selections")

    requested: dict[tuple[str, str], str] = {}
    for selection in selections:
        if "=" not in selection or "/" not in selection.split("=", 1)[0]:
            raise PreflightError("selection syntax is FAMILY/CLASS=TENSOR_NAME")
        label, tensor_name = selection.split("=", 1)
        family, tensor_class = label.split("/", 1)
        key = (family, tensor_class)
        if family not in FAMILIES or tensor_class not in CLASSES or key in requested:
            raise PreflightError("selections must have one unique entry per frozen family/class")
        requested[key] = tensor_name
    if set(requested) != {(family, cls) for family in FAMILIES for cls in CLASSES}:
        raise PreflightError("selection set must cover all four classes in both families")

    probes = []
    seen_names: set[str] = set()
    seen_layers: set[tuple[str, int]] = set()
    for family in FAMILIES:
        for tensor_class in CLASSES:
            name = requested[(family, tensor_class)]
            if name in seen_names:
                raise PreflightError("probe tensor names must be unique")
            seen_names.add(name)
            derived_family, derived_class, layer = _family_and_class(name, config)
            if (derived_family, derived_class) != (family, tensor_class):
                raise PreflightError(f"{name!r} does not match its declared family/class")
            if (family, layer) in seen_layers:
                raise PreflightError("each family must use four distinct block layers")
            seen_layers.add((family, layer))
            shard = weight_map.get(name)
            if not isinstance(shard, str):
                raise PreflightError(f"selected tensor {name!r} is absent from the pinned index")
            metadata = tensor_metadata[name]
            shape = metadata.get("shape") if isinstance(metadata, dict) else None
            if not isinstance(shape, list) or len(shape) != 2 or any(
                type(size) is not int or size <= 0 for size in shape
            ):
                raise PreflightError(f"selected tensor {name} is not a rank-2 matrix")
            probes.append({
                "family": family,
                "tensor_class": tensor_class,
                "tensor_name": name,
                "tensor_index": matrix_ordinals[name],
                "layer": layer,
                "shape": shape,
                "source_shard": shard,
            })
    prepared = {
        "schema": "tritium.qwen36-gdn-probe-preflight.v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "state": "prepared-not-measured",
        "evidence_scope": "local-config-index-and-safetensors-header-only",
        "config_sha256": config_digest,
        "weight_index_sha256": index_digest,
        "probes": probes,
        "limitations": [
            "does not authenticate the checkpoint against Hugging Face",
            "does not verify calibration-pack provenance",
            "reads headers for all indexed shards but does not load tensor payloads or execute the model",
            "does not produce a measurement or release receipt",
        ],
    }
    prepared["preflight_id"] = "sha256:" + hashlib.sha256(canonical(prepared)).hexdigest()
    return prepared


def write_preflight(output: Path, result: dict[str, Any]) -> None:
    """Atomically publish one local preflight without replacing existing evidence."""

    requested = Path(output).absolute()
    parent = requested.parent.resolve(strict=True)
    if not parent.is_dir():
        raise PreflightError("output parent must be an ordinary directory")
    target = parent / requested.name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"preflight output already exists: {target}")

    payload = canonical(result) + b"\n"
    fd, temporary_name = tempfile.mkstemp(prefix=".tritium-gdn-preflight-", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-filesystem hard link publishes atomically and fails rather
        # than replacing a file or symlink created after the initial check.
        os.link(temporary, target, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="atomically save the preflight JSON; an existing path is never replaced",
    )
    parser.add_argument(
        "--probe", action="append", default=[], metavar="FAMILY/CLASS=TENSOR_NAME",
        help="repeat exactly eight times, once for each frozen family/class",
    )
    args = parser.parse_args()
    try:
        result = prepare(args.model_dir, args.probe)
        if args.output is not None:
            write_preflight(args.output, result)
    except (OSError, PreflightError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
