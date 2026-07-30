#!/usr/bin/env python3
"""Synchronize registry-package copies of the canonical training contracts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parent.parent
MAPPINGS = (
    (
        ROOT / "spec/training/v1/manifest.json",
        ROOT / "crates/tritium-spec/data/training/v1/manifest.json",
    ),
    (
        ROOT / "spec/training/v1/vectors/v1.json",
        ROOT / "crates/tritium-spec/data/training/v1/vectors/v1.json",
    ),
    (
        ROOT / "spec/training/v1/manifest.json",
        ROOT / "crates/tritium-wgpu/data/training/v1/manifest.json",
    ),
    (
        ROOT / "spec/training/v1/webgpu-dispatch-v1.json",
        ROOT / "crates/tritium-wgpu/data/training/v1/webgpu-dispatch-v1.json",
    ),
    (
        ROOT / "spec/training/v2/manifest.json",
        ROOT / "crates/tritium-spec/data/training/v2/manifest.json",
    ),
    (
        ROOT / "spec/training/v2/vectors/v2.json",
        ROOT / "crates/tritium-spec/data/training/v2/vectors/v2.json",
    ),
    (
        ROOT / "spec/training/v2/manifest.json",
        ROOT / "crates/tritium-wgpu/data/training/v2/manifest.json",
    ),
    (
        ROOT / "spec/training/v2/webgpu-dispatch-v2.json",
        ROOT / "crates/tritium-wgpu/data/training/v2/webgpu-dispatch-v2.json",
    ),
)


def _ordinary_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"contract is not an ordinary file: {path.relative_to(ROOT)}")
    return path.read_bytes()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale: list[Path] = []
    try:
        for source, destination in MAPPINGS:
            payload = _ordinary_bytes(source)
            if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != payload:
                stale.append(destination)
                if not args.check:
                    _atomic_write(destination, payload)
    except (OSError, ValueError) as error:
        print(f"packaged-training-spec: FAIL: {error}")
        return 1
    if args.check and stale:
        names = ", ".join(str(path.relative_to(ROOT)) for path in stale)
        print(f"packaged-training-spec: FAIL: stale or missing copies: {names}")
        return 1
    action = "checked" if args.check else "synchronized"
    print(f"packaged-training-spec: PASS: {action} {len(MAPPINGS)} copies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
