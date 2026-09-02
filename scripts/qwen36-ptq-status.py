#!/usr/bin/env python3
"""Inspect a resumable Qwen3.6 PTQ campaign without mutating it.

This is an operational probe, not release evidence. It reports the newest
staged tensor stream, process liveness, published master count, and optional
byte-rate sampling. It never opens or rewrites campaign records.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable


TEMP_RE = re.compile(r"^record\.tmp\.(?P<pid>[0-9]+)\.[^.]+\..+$")
SCHEMA = "tritium.qwen36-ptq-status.v1"

# `campaign.tq36p` and its nested master catalog are canonical Rust records.
# Keep these offsets local to this read-only operational probe; malformed files
# are ignored rather than treated as evidence.
CAMPAIGN_PREFIX_BYTES = 313
CAMPAIGN_CHECKSUM_BYTES = 32
CATALOG_HEADER_BYTES = 16
MASTER_METADATA_TRAILER_BYTES = 40
RECORD_FIXED_BYTES = 136
RECORD_FOOTER_BYTES = 32


class StatusError(ValueError):
    """The requested campaign path or probe options are invalid."""


def _ordinary_dir(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise StatusError("work directory must be an ordinary directory")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise StatusError(f"work directory is unavailable: {path}") from error
    if resolved.is_symlink() or not resolved.is_dir():
        raise StatusError("work directory must be an ordinary directory")
    return resolved


def _regular_files(root: Path) -> Iterable[Path]:
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        dirnames[:] = [
            name for name in dirnames if not (base / name).is_symlink()
        ]
        for name in filenames:
            path = base / name
            try:
                if path.is_file() and not path.is_symlink():
                    yield path
            except OSError:
                continue


def _discover(root: Path) -> tuple[list[tuple[Path, int]], int, int]:
    temps: list[tuple[Path, int]] = []
    objects = 0
    seals = 0
    for path in _regular_files(root):
        name = path.name
        if name == "workspace.complete.tq36c":
            seals += 1
        if path.parent.name == "objects" and path.suffix == ".s2kf":
            objects += 1
        match = TEMP_RE.fullmatch(name)
        if match is not None:
            temps.append((path, int(match.group("pid"))))
    return temps, objects, seals


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OverflowError:
        # A filename can contain arbitrary decimal digits; values outside the
        # host PID type cannot name a live process and are stale by definition.
        return False
    except PermissionError:
        return True
    return True


def _record(root: Path, path: Path, pid: int) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as error:
        raise StatusError(f"staged record disappeared during probe: {path}") from error
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "pid": pid,
        "pid_alive": _pid_alive(pid),
    }


def _u32(data: bytes, offset: int) -> int | None:
    end = offset + 4
    if offset < 0 or end > len(data):
        return None
    return int.from_bytes(data[offset:end], "little")


def _u64(data: bytes, offset: int) -> int | None:
    end = offset + 8
    if offset < 0 or end > len(data):
        return None
    return int.from_bytes(data[offset:end], "little")


def _campaign_totals(path: Path) -> tuple[int, int, int] | None:
    """Return (record bytes, payload bytes, tensor count) for one campaign."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < CAMPAIGN_PREFIX_BYTES + CAMPAIGN_CHECKSUM_BYTES:
        return None
    # The checksum is intentionally not recomputed here, but its envelope must
    # be present before parsing bounded catalog fields.
    if data[:8] != b"TSQ36CP\x00":
        return None
    catalog_length = _u32(data, CAMPAIGN_PREFIX_BYTES - 4)
    if catalog_length is None:
        return None
    catalog_start = CAMPAIGN_PREFIX_BYTES
    catalog_end = catalog_start + catalog_length
    if catalog_end + CAMPAIGN_CHECKSUM_BYTES != len(data):
        return None
    catalog = data[catalog_start:catalog_end]
    if len(catalog) < CATALOG_HEADER_BYTES or catalog[:8] != b"TSQ36SC\x00":
        return None
    count = _u32(catalog, 12)
    if count is None:
        return None
    offset = CATALOG_HEADER_BYTES
    total_records = 0
    total_payload = 0
    for _ in range(count):
        metadata_length = _u32(catalog, offset)
        if metadata_length is None:
            return None
        offset += 4
        metadata_end = offset + metadata_length
        if metadata_length < MASTER_METADATA_TRAILER_BYTES or metadata_end > len(catalog):
            return None
        metadata = catalog[offset:metadata_end]
        payload = _u64(metadata, len(metadata) - MASTER_METADATA_TRAILER_BYTES)
        name_length = _u32(metadata, 314)
        if payload is None or name_length is None:
            return None
        rank_offset = 318 + name_length
        rank = _u32(metadata, rank_offset)
        if rank is None:
            return None
        record_bytes = (
            RECORD_FIXED_BYTES
            + name_length
            + (rank * 8)
            + metadata_length
            + payload
            + RECORD_FOOTER_BYTES
        )
        total_records += record_bytes
        total_payload += payload
        offset = metadata_end
    if offset != len(catalog):
        return None
    return total_records, total_payload, count


def _campaign_for(root: Path, staged: dict[str, Any] | None) -> Path | None:
    candidates: list[Path] = []
    if staged is not None:
        staged_path = root / staged["path"]
        candidate = staged_path.parent.parent / "campaign.tq36p"
        if candidate.is_file() and not candidate.is_symlink():
            candidates.append(candidate)
    if not candidates:
        candidates = [
            path
            for path in _regular_files(root)
            if path.name == "campaign.tq36p"
        ]
    return max(candidates, key=lambda path: path.stat().st_mtime_ns, default=None)


def inspect(
    work_dir: Path,
    sample_seconds: float = 0.0,
    target_bytes: int | None = None,
) -> dict[str, Any]:
    """Return a non-authoritative operational snapshot."""
    if sample_seconds < 0 or sample_seconds > 3600:
        raise StatusError("sample seconds must be between 0 and 3600")
    if target_bytes is not None and target_bytes < 0:
        raise StatusError("target bytes must be non-negative")
    root = _ordinary_dir(work_dir)
    temps, objects, seals = _discover(root)
    newest = max(temps, key=lambda item: item[0].stat().st_mtime_ns, default=None)
    staged = _record(root, *newest) if newest is not None else None
    campaign = _campaign_for(root, staged)
    campaign_totals = _campaign_totals(campaign) if campaign is not None else None
    rate: float | None = None
    eta: float | None = None
    campaign_eta: float | None = None
    if staged is not None and sample_seconds:
        time.sleep(sample_seconds)
        try:
            after = staged["bytes"]
            after_stat = (root / staged["path"]).stat()
            after = after_stat.st_size
        except OSError as error:
            raise StatusError("staged record disappeared during sample") from error
        rate = (after - staged["bytes"]) / sample_seconds
        staged["mtime_ns"] = after_stat.st_mtime_ns
        staged["bytes_after_sample"] = after
        staged["sample_seconds"] = sample_seconds
        if target_bytes is not None and rate > 0:
            eta = max(target_bytes - after, 0) / rate
        if campaign_totals is not None and rate > 0:
            campaign_eta = max(campaign_totals[0] - after, 0) / rate
    if seals:
        status = "complete"
    elif not staged:
        status = "idle"
    else:
        # A leftover temp record after its owner exits is recoverable evidence,
        # not proof of active work. Keep it distinct so operators do not wait
        # indefinitely on a dead campaign.
        live_staged = any(_pid_alive(pid) for _, pid in temps)
        status = "running" if live_staged else "stalled"
    return {
        "schema": SCHEMA,
        "work_dir": str(root),
        "status": status,
        "staged_record": staged,
        "staged_record_count": len(temps),
        "published_master_count": objects,
        "seal_count": seals,
        "bytes_per_second": rate,
        "target_bytes": target_bytes,
        "estimated_seconds_remaining": eta,
        "campaign_expected_record_bytes": campaign_totals[0] if campaign_totals else None,
        "campaign_expected_payload_bytes": campaign_totals[1] if campaign_totals else None,
        "campaign_tensor_count": campaign_totals[2] if campaign_totals else None,
        "campaign_estimated_seconds_remaining": campaign_eta,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument(
        "--sample-seconds", type=float, default=0.0,
        help="sample staged-byte growth for this many seconds (default: 0)",
    )
    parser.add_argument(
        "--target-bytes", type=int,
        help="optional staged-record size target for a rate-based ETA",
    )
    parser.add_argument("--json", action="store_true", help="emit canonical JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        snapshot = inspect(args.work_dir, args.sample_seconds, args.target_bytes)
    except StatusError as error:
        print(f"qwen36-ptq-status: ERROR: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(snapshot, sort_keys=True, separators=(",", ":")))
    else:
        staged = snapshot["staged_record"]
        print(f"status={snapshot['status']}")
        print(f"published_masters={snapshot['published_master_count']}")
        print(f"sealed={snapshot['seal_count']}")
        if staged is None:
            print("staged_record=none")
        else:
            print(
                "staged_record="
                f"{staged['path']} bytes={staged['bytes']} "
                f"pid={staged['pid']} alive={staged['pid_alive']}"
            )
        if snapshot["bytes_per_second"] is not None:
            print(f"bytes_per_second={snapshot['bytes_per_second']:.3f}")
        if snapshot["target_bytes"] is not None:
            print(f"target_bytes={snapshot['target_bytes']}")
            eta = snapshot["estimated_seconds_remaining"]
            print(
                "estimated_seconds_remaining="
                + (f"{eta:.3f}" if eta is not None else "unknown")
            )
        if snapshot["campaign_expected_record_bytes"] is not None:
            print(
                "campaign_expected_record_bytes="
                f"{snapshot['campaign_expected_record_bytes']}"
            )
            eta = snapshot["campaign_estimated_seconds_remaining"]
            print(
                "campaign_estimated_seconds_remaining="
                + (f"{eta:.3f}" if eta is not None else "unknown")
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
