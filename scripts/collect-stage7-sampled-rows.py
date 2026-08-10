#!/usr/bin/env python3
"""Collect frozen, source-bound rows for Plan-0043 Stage-7 evidence."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, BinaryIO, Iterator, Mapping, NamedTuple, Protocol


SAMPLED_ROWS_SCHEMA = "tritium.stage7-sampled-rows.v1"
ACQUISITION_SCHEMA = "tritium.stage7-row-acquisition.v1"
TOKENS_PER_SEQUENCE = 2_048
DEFAULT_CHARS_PER_TOKEN = 8
MAX_ROW_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024
AT_FDCWD = -100
LINUX_RENAME_NOREPLACE = 1
DARWIN_RENAME_EXCL = 0x4
PARTITIONS = ("calibration", "refinement", "validation", "evaluation")
PARTITION_SEEDS = {
    "calibration": 0x5341_4C54_0007_0001,
    "refinement": 0x5341_4C54_0007_0002,
    "validation": 0x5341_4C54_0007_0003,
    "evaluation": 0x5341_4C54_0007_0004,
}


class DatasetContract(NamedTuple):
    repo_id: str
    revision: str
    config: str
    data_dir: str | None
    split: str
    text_field: str
    sequence_count: int
    source_prefix: str
    source_suffix: str


class SourceShard(NamedTuple):
    path: str
    bytes: int
    sha256: str
    hub_oid: str


DATASETS = (
    DatasetContract(
        "allenai/c4",
        "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        "en",
        None,
        "train",
        "text",
        256,
        "en/c4-train.",
        ".json.gz",
    ),
    DatasetContract(
        "open-web-math/open-web-math",
        "fde8ef8de2300f5e778f56261843dab89f230815",
        "default",
        None,
        "train",
        "text",
        128,
        "data/train-",
        ".parquet",
    ),
    DatasetContract(
        "bigcode/starcoderdata",
        "9fc30b578cedaec69e47302df72cf00feed7c8c4",
        "default",
        "python",
        "train",
        "content",
        128,
        "python/train-",
        ".parquet",
    ),
)


class DatasetSource(Protocol):
    def list_shards(self, contract: DatasetContract) -> tuple[SourceShard, ...]: ...

    def preflight(self, contract: DatasetContract, shard: SourceShard) -> None: ...

    def materialize(self, contract: DatasetContract, shard: SourceShard) -> Path: ...


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(stream: BinaryIO) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_noreplace(staging: Path, output: Path) -> None:
    """Atomically publish one directory without replacing a racing target."""

    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOSYS, "atomic no-replace rename is unavailable")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            AT_FDCWD,
            os.fsencode(staging),
            AT_FDCWD,
            os.fsencode(output),
            LINUX_RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        renamex_np = getattr(library, "renamex_np", None)
        if renamex_np is None:
            raise OSError(errno.ENOSYS, "atomic no-replace rename is unavailable")
        renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex_np.restype = ctypes.c_int
        result = renamex_np(
            os.fsencode(staging), os.fsencode(output), DARWIN_RENAME_EXCL
        )
    elif os.name == "nt":
        try:
            os.rename(staging, output)
        except FileExistsError as error:
            raise FileExistsError(
                errno.EEXIST, "Stage-7 sampled-row output already exists", output
            ) from error
        return
    else:
        raise OSError(errno.ENOSYS, "atomic no-replace rename is unavailable")
    if result == 0:
        return
    code = ctypes.get_errno()
    if code in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            errno.EEXIST, "Stage-7 sampled-row output already exists", output
        )
    raise OSError(code, os.strerror(code), output)


def _iter_gzip_json(stream: BinaryIO, field: str) -> Iterator[tuple[int, str]]:
    with gzip.GzipFile(fileobj=stream, mode="rb") as decoded:
        for row_index, line in enumerate(decoded):
            if len(line) > MAX_ROW_BYTES:
                raise ValueError(f"source row {row_index} exceeds byte ceiling")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"source row {row_index} is invalid JSON") from error
            text = value.get(field) if isinstance(value, dict) else None
            if not isinstance(text, str):
                raise ValueError(f"source row {row_index} lacks string field {field}")
            yield row_index, text


def _iter_parquet(stream: BinaryIO, field: str) -> Iterator[tuple[int, str]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("Stage-7 collection requires pyarrow") from error
    source_row = 0
    source = parquet.ParquetFile(stream)
    if field not in source.schema.names:
        raise ValueError(f"Parquet source lacks field {field}")
    for batch in source.iter_batches(batch_size=128, columns=[field]):
        for text in batch.column(0).to_pylist():
            if not isinstance(text, str):
                raise ValueError(f"source row {source_row} lacks string field {field}")
            if len(text.encode("utf-8")) > MAX_ROW_BYTES:
                raise ValueError(f"source row {source_row} exceeds byte ceiling")
            yield source_row, text
            source_row += 1


def _iter_source_rows(
    stream: BinaryIO, source_name: str, field: str
) -> Iterator[tuple[int, str]]:
    if source_name.endswith(".json.gz"):
        yield from _iter_gzip_json(stream, field)
        return
    if source_name.endswith(".parquet"):
        yield from _iter_parquet(stream, field)
        return
    raise ValueError(f"unsupported Stage-7 source format: {source_name}")


@contextmanager
def _verified_source_stream(path: Path, shard: SourceShard) -> Iterator[BinaryIO]:
    resolved = path.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    stream = os.fdopen(descriptor, "rb", buffering=0)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != shard.bytes:
            raise ValueError(f"materialized shard metadata differs: {shard.path}")
        if _sha256_stream(stream) != shard.sha256:
            raise ValueError(f"materialized shard differs from declared identity: {shard.path}")
        yield stream
        if os.fstat(descriptor).st_size != shard.bytes or _sha256_stream(stream) != shard.sha256:
            raise ValueError(f"materialized shard changed during collection: {shard.path}")
    finally:
        stream.close()


class HubDatasetSource:
    """Immutable-revision Hugging Face Hub source with verified LFS payloads."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        token: bool | str | None = None,
    ) -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:
            raise RuntimeError("Stage-7 collection requires huggingface_hub") from error
        self._api = HfApi()
        self._cache_dir = cache_dir
        self._token = token

    def list_shards(self, contract: DatasetContract) -> tuple[SourceShard, ...]:
        root = contract.source_prefix.split("/", 1)[0]
        entries = self._api.list_repo_tree(
            contract.repo_id,
            path_in_repo=root,
            recursive=False,
            expand=True,
            revision=contract.revision,
            repo_type="dataset",
            token=self._token,
        )
        shards = []
        for entry in entries:
            path = getattr(entry, "path", None)
            if (
                not isinstance(path, str)
                or not path.startswith(contract.source_prefix)
                or not path.endswith(contract.source_suffix)
            ):
                continue
            size = getattr(entry, "size", None)
            lfs = getattr(entry, "lfs", None)
            sha256 = getattr(lfs, "sha256", None)
            if (
                not isinstance(size, int)
                or size <= 0
                or not isinstance(sha256, str)
                or len(sha256) != 64
            ):
                raise ValueError(f"Hub shard lacks immutable LFS identity: {path}")
            shards.append(SourceShard(path, size, sha256, f"lfs-sha256:{sha256}"))
        shards.sort(key=lambda shard: shard.path)
        if not shards:
            raise ValueError(f"frozen dataset has no admitted source shards: {contract.repo_id}")
        return tuple(shards)

    def materialize(self, contract: DatasetContract, shard: SourceShard) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError("Stage-7 collection requires huggingface_hub") from error
        path = Path(
            hf_hub_download(
                contract.repo_id,
                shard.path,
                repo_type="dataset",
                revision=contract.revision,
                cache_dir=self._cache_dir,
                token=self._token,
            )
        )
        if path.stat().st_size != shard.bytes or _sha256_file(path) != shard.sha256:
            raise ValueError(f"downloaded Hub shard differs from frozen identity: {shard.path}")
        return path

    def preflight(self, contract: DatasetContract, shard: SourceShard) -> None:
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import GatedRepoError
        except ImportError as error:
            raise RuntimeError("Stage-7 collection requires huggingface_hub") from error
        try:
            info = hf_hub_download(
                contract.repo_id,
                shard.path,
                repo_type="dataset",
                revision=contract.revision,
                cache_dir=self._cache_dir,
                token=self._token,
                dry_run=True,
            )
        except GatedRepoError as error:
            raise PermissionError(
                f"authorized Hugging Face access required for {contract.repo_id}; "
                "request dataset access and authenticate with `hf auth login`"
            ) from error
        if info.file_size != shard.bytes:
            raise ValueError(f"Hub preflight size differs for frozen shard: {shard.path}")


def _default_minimum_utf8_bytes() -> dict[str, int]:
    return {
        contract.repo_id: contract.sequence_count
        * TOKENS_PER_SEQUENCE
        * DEFAULT_CHARS_PER_TOKEN
        for contract in DATASETS
    }


def _default_minimum_rows() -> dict[str, int]:
    return {contract.repo_id: contract.sequence_count for contract in DATASETS}


def _partition_score(
    partition: str,
    contract: DatasetContract,
    row_index: int,
    content_sha256: str,
) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"tritium-stage7-partition-ranking-v1\0")
    digest.update(PARTITION_SEEDS[partition].to_bytes(8, "little"))
    digest.update(contract.repo_id.encode())
    digest.update(b"\0")
    digest.update(contract.revision.encode())
    digest.update(b"\0")
    digest.update(row_index.to_bytes(8, "little"))
    digest.update(bytes.fromhex(content_sha256))
    return digest.digest()


def _collect_dataset(
    contract: DatasetContract,
    source: DatasetSource,
    shards: tuple[SourceShard, ...],
    minimum_utf8_bytes: int,
    minimum_rows: int,
    remaining_download_bytes: int,
    seen_content: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[SourceShard], int]:
    if minimum_utf8_bytes <= 0 or minimum_rows <= 0:
        raise ValueError("minimum UTF-8 byte and row targets must be positive")
    selected = {partition: [] for partition in PARTITIONS}
    selected_bytes = {partition: 0 for partition in PARTITIONS}
    global_row_index = 0
    downloaded = 0
    used_shards = []
    for shard in shards:
        if all(
            selected_bytes[partition] >= minimum_utf8_bytes
            and len(selected[partition]) >= minimum_rows
            for partition in PARTITIONS
        ):
            break
        if shard.bytes <= 0 or len(shard.sha256) != 64:
            raise ValueError("source shard identity is invalid")
        if downloaded + shard.bytes > remaining_download_bytes:
            raise ValueError(
                f"campaign download ceiling exceeded while collecting {contract.repo_id}"
            )
        downloaded += shard.bytes
        local = source.materialize(contract, shard)
        with _verified_source_stream(local, shard) as stream:
            rows = _iter_source_rows(stream, local.name, contract.text_field)
            try:
                for shard_row_index, text in rows:
                    pending = [
                        partition
                        for partition in PARTITIONS
                        if selected_bytes[partition] < minimum_utf8_bytes
                        or len(selected[partition]) < minimum_rows
                    ]
                    if not pending:
                        break
                    raw = text.encode("utf-8")
                    digest = _sha256_bytes(raw)
                    row_index = global_row_index
                    global_row_index += 1
                    if not raw or digest in seen_content:
                        continue
                    partition = min(
                        pending,
                        key=lambda candidate: _partition_score(
                            candidate, contract, row_index, digest
                        ),
                    )
                    selected[partition].append(
                        {
                            "row_index": row_index,
                            "content_sha256": digest,
                            "text": text,
                            "source_shard": shard.path,
                            "source_shard_row_index": shard_row_index,
                        }
                    )
                    seen_content.add(digest)
                    selected_bytes[partition] += len(raw)
            finally:
                rows.close()
        used_shards.append(shard)
    missing = next(
        (
            partition
            for partition in PARTITIONS
            if selected_bytes[partition] < minimum_utf8_bytes
            or len(selected[partition]) < minimum_rows
        ),
        None,
    )
    if missing is not None:
        raise ValueError(
            f"frozen dataset exhausted before {missing} reached row/byte targets: "
            f"{contract.repo_id}"
        )
    return selected, used_shards, downloaded


def collect_sampled_rows(
    output_dir: Path,
    source: DatasetSource,
    *,
    minimum_utf8_bytes: Mapping[str, int] | None = None,
    minimum_rows: Mapping[str, int] | None = None,
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
) -> dict[str, Any]:
    """Collect and atomically publish current Stage-7 sampled-row schema."""

    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Stage-7 sampled-row output already exists: {output}")
    if max_download_bytes <= 0:
        raise ValueError("download ceiling must be positive")
    requested = dict(minimum_utf8_bytes or _default_minimum_utf8_bytes())
    requested_rows = dict(minimum_rows or _default_minimum_rows())
    expected_repositories = {contract.repo_id for contract in DATASETS}
    if set(requested) != expected_repositories or set(requested_rows) != expected_repositories:
        raise ValueError("minimum row/byte targets differ from frozen dataset inventory")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("Stage-7 sampled-row parent must be an ordinary directory")
    source_shards = {}
    for contract in DATASETS:
        shards = source.list_shards(contract)
        source.preflight(contract, shards[0])
        source_shards[contract.repo_id] = shards
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        rows_dir = staging / "rows"
        rows_dir.mkdir(mode=0o755)
        seen_content: set[str] = set()
        collected = {}
        admitted_shards = {}
        remaining_download_bytes = max_download_bytes
        for contract in DATASETS:
            rows, shards, downloaded = _collect_dataset(
                contract,
                source,
                source_shards[contract.repo_id],
                requested[contract.repo_id],
                requested_rows[contract.repo_id],
                remaining_download_bytes,
                seen_content,
            )
            remaining_download_bytes -= downloaded
            collected[contract.repo_id] = rows
            admitted_shards[contract.repo_id] = shards

        partitions = {}
        acquisition_partitions = {}
        for partition in PARTITIONS:
            datasets = []
            acquisition_datasets = []
            for contract in DATASETS:
                normalized = []
                acquisition_rows = []
                for row in collected[contract.repo_id][partition]:
                    normalized.append(
                        {
                            "row_index": row["row_index"],
                            "content_sha256": row["content_sha256"],
                            "text": row["text"],
                        }
                    )
                    acquisition_rows.append(
                        {
                            "row_index": row["row_index"],
                            "source_shard": row["source_shard"],
                            "source_shard_row_index": row["source_shard_row_index"],
                            "content_sha256": row["content_sha256"],
                        }
                    )
                lane = rows_dir / (
                    f"{partition}-{contract.repo_id.replace('/', '--')}.jsonl"
                )
                payload = b"".join(_canonical(row) + b"\n" for row in normalized)
                _write_new(lane, payload)
                datasets.append(
                    {
                        "repo_id": contract.repo_id,
                        "revision": contract.revision,
                        "config": contract.config,
                        "data_dir": contract.data_dir,
                        "split": contract.split,
                        "text_field": contract.text_field,
                        "rows": _file_record(staging, lane),
                    }
                )
                acquisition_datasets.append(
                    {"repo_id": contract.repo_id, "rows": acquisition_rows}
                )
            partitions[partition] = {
                "sampling_seed": PARTITION_SEEDS[partition],
                "datasets": datasets,
            }
            acquisition_partitions[partition] = {
                "sampling_seed": PARTITION_SEEDS[partition],
                "datasets": acquisition_datasets,
            }

        source_manifest = {
            "schema": SAMPLED_ROWS_SCHEMA,
            "partitions": partitions,
        }
        source_path = staging / "sampled-rows.json"
        _write_new(source_path, json.dumps(source_manifest, indent=2).encode() + b"\n")
        acquisition_scope = {
            "schema": ACQUISITION_SCHEMA,
            "sampled_rows": _file_record(staging, source_path),
            "collection_policy": {
                "order": "seeded-partition-ranking-over-lexicographic-shard-row-v1",
                "partition_score": (
                    "sha256(domain,seed-u64le,repo,nul,revision,nul,row-u64le,content-sha256)"
                ),
                "partition_order": list(PARTITIONS),
                "partition_seeds": dict(PARTITION_SEEDS),
                "minimum_utf8_bytes": {
                    contract.repo_id: requested[contract.repo_id]
                    for contract in DATASETS
                },
                "minimum_rows": {
                    contract.repo_id: requested_rows[contract.repo_id]
                    for contract in DATASETS
                },
                "maximum_download_bytes": max_download_bytes,
                "duplicate_policy": "skip-repeated-sha256-across-all-datasets",
                "row_index": "zero-based-row-ordinal-across-ordered-shards",
            },
            "datasets": [
                {
                    "repo_id": contract.repo_id,
                    "revision": contract.revision,
                    "config": contract.config,
                    "data_dir": contract.data_dir,
                    "split": contract.split,
                    "text_field": contract.text_field,
                    "shards": [shard._asdict() for shard in admitted_shards[contract.repo_id]],
                }
                for contract in DATASETS
            ],
            "partitions": acquisition_partitions,
        }
        acquisition = {
            **acquisition_scope,
            "receipt_id": "sha256:" + _sha256_bytes(_canonical(acquisition_scope)),
        }
        acquisition_path = staging / "acquisition-receipt.json"
        _write_new(acquisition_path, _canonical(acquisition) + b"\n")
        _sync_directory(rows_dir)
        _sync_directory(staging)
        _publish_directory_noreplace(staging, output)
        _sync_directory(parent)
        return acquisition
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--max-download-bytes", type=int, default=DEFAULT_MAX_DOWNLOAD_BYTES
    )
    args = parser.parse_args()
    receipt = collect_sampled_rows(
        args.output_dir,
        HubDatasetSource(
            cache_dir=args.cache_dir,
            token=None,
        ),
        max_download_bytes=args.max_download_bytes,
    )
    print(json.dumps({"receipt_id": receipt["receipt_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
