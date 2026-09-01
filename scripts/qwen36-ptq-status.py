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


def inspect(work_dir: Path, sample_seconds: float = 0.0) -> dict[str, Any]:
    """Return a non-authoritative operational snapshot."""
    if sample_seconds < 0 or sample_seconds > 3600:
        raise StatusError("sample seconds must be between 0 and 3600")
    root = _ordinary_dir(work_dir)
    temps, objects, seals = _discover(root)
    newest = max(temps, key=lambda item: item[0].stat().st_mtime_ns, default=None)
    staged = _record(root, *newest) if newest is not None else None
    rate: float | None = None
    if staged is not None and sample_seconds:
        time.sleep(sample_seconds)
        try:
            after = staged["bytes"]
            after_stat = (root / staged["path"]).stat()
            after = after_stat.st_size
        except OSError as error:
            raise StatusError("staged record disappeared during sample") from error
        rate = (after - staged["bytes"]) / sample_seconds
        staged["bytes_after_sample"] = after
        staged["sample_seconds"] = sample_seconds
    status = "complete" if seals else "running" if staged else "idle"
    return {
        "schema": SCHEMA,
        "work_dir": str(root),
        "status": status,
        "staged_record": staged,
        "staged_record_count": len(temps),
        "published_master_count": objects,
        "seal_count": seals,
        "bytes_per_second": rate,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument(
        "--sample-seconds", type=float, default=0.0,
        help="sample staged-byte growth for this many seconds (default: 0)",
    )
    parser.add_argument("--json", action="store_true", help="emit canonical JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        snapshot = inspect(args.work_dir, args.sample_seconds)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
