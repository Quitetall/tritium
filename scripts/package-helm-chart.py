#!/usr/bin/env python3
"""Build Tritium's deterministic, source-closed Helm chart archive."""

from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import stat
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


RELEASE_PATTERN = re.compile(r"1\.1\.0-rc\.(0|[1-9][0-9]*)")
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_CHART_BYTES = 64 * 1024 * 1024
CANONICAL_GZIP_HEADER = bytes.fromhex("1f8b08000000000002ff")
REQUIRED_FILES = {"Chart.yaml", "values.yaml"}
TOP_LEVEL_FIELDS = {
    "apiVersion",
    "name",
    "description",
    "type",
    "version",
    "appVersion",
    "kubeVersion",
    "annotations",
}
ANNOTATIONS = {
    "tritium.ai/artifact-schema": "3",
    "tritium.ai/startup-receipt-schema": "1",
}


class ChartPackageError(ValueError):
    """Chart source cannot produce canonical Tritium release bytes."""


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _scalar(value: str, label: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    if not value or "#" in value:
        raise ChartPackageError(f"Chart.yaml {label} must be one plain scalar")
    return value


def _chart_metadata(payload: bytes, release: str) -> dict[str, str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ChartPackageError("Chart.yaml must be UTF-8") from error
    fields: dict[str, str] = {}
    annotations: dict[str, str] = {}
    in_annotations = False
    for ordinal, line in enumerate(lines, 1):
        if not line or line.lstrip().startswith("#"):
            continue
        if "\t" in line:
            raise ChartPackageError(f"Chart.yaml line {ordinal} contains a tab")
        if line.startswith("  ") and in_annotations:
            key, separator, value = line[2:].partition(":")
            if not separator or not key or key in annotations:
                raise ChartPackageError("Chart.yaml annotations are not canonical")
            annotations[key] = _scalar(value, f"annotation {key}")
            continue
        if line.startswith(" "):
            raise ChartPackageError("Chart.yaml has unsupported nested fields")
        key, separator, value = line.partition(":")
        if not separator or not key or key in fields:
            raise ChartPackageError("Chart.yaml top-level fields are not canonical")
        in_annotations = key == "annotations"
        fields[key] = "" if in_annotations else _scalar(value, key)
    if set(fields) != TOP_LEVEL_FIELDS or annotations != ANNOTATIONS:
        raise ChartPackageError("Chart.yaml fields or annotations are not canonical")
    if (
        fields["apiVersion"] != "v2"
        or fields["name"] != "tritium"
        or fields["type"] != "application"
        or fields["version"] != release
        or fields["appVersion"] != release
        or fields["kubeVersion"] != ">=1.29.0-0"
    ):
        raise ChartPackageError("Chart.yaml version or application contract differs")
    return fields


def _source_files(source: Path, release: str) -> list[tuple[str, bytes]]:
    if source.is_symlink() or not source.is_dir():
        raise ChartPackageError("chart source must be an ordinary directory, not a symlink")
    files: list[tuple[str, bytes]] = []
    portable: set[str] = set()
    total = 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ChartPackageError(f"chart source contains symlink {path.relative_to(source)}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ChartPackageError("chart source contains a non-file entry")
        relative = path.relative_to(source).as_posix()
        logical = PurePosixPath(relative)
        archive_name = f"tritium/{relative}"
        if (
            logical.is_absolute()
            or ".." in logical.parts
            or "\\" in relative
            or len(archive_name.encode("utf-8")) > 100
        ):
            raise ChartPackageError(f"chart path {relative!r} is not canonical ustar")
        folded = archive_name.casefold()
        if folded in portable:
            raise ChartPackageError(f"chart contains duplicate portable path {relative!r}")
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
            raise ChartPackageError(f"chart file {relative!r} exceeds release bounds")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                payload = stream.read(MAX_FILE_BYTES + 1)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current = path.stat(follow_symlinks=False)
        if (
            _stat_signature(before) != _stat_signature(opened)
            or _stat_signature(opened) != _stat_signature(after)
            or _stat_signature(after) != _stat_signature(current)
        ):
            raise ChartPackageError(f"chart file {relative!r} changed while packaging")
        if len(payload) != opened.st_size:
            raise ChartPackageError(f"chart file {relative!r} changed while packaging")
        total += len(payload)
        if total > MAX_CHART_BYTES:
            raise ChartPackageError("chart source exceeds release bounds")
        files.append((archive_name, payload))
        portable.add(folded)
    names = {name.removeprefix("tritium/") for name, _ in files}
    if not REQUIRED_FILES.issubset(names) or not any(
        name.startswith("templates/") for name in names
    ):
        raise ChartPackageError("chart source lacks required metadata, values, or templates")
    chart = dict(files)["tritium/Chart.yaml"]
    _chart_metadata(chart, release)
    return files


def _archive(files: list[tuple[str, bytes]]) -> bytes:
    uncompressed = io.BytesIO()
    with tarfile.open(fileobj=uncompressed, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, payload in files:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.mtime = 0
            info.uname = ""
            info.gname = ""
            tar.addfile(info, io.BytesIO(payload))
    return _gzip(uncompressed.getvalue())


def _gzip(payload: bytes) -> bytes:
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=compressed, compresslevel=9, mtime=0
    ) as stream:
        stream.write(payload)
    encoded = compressed.getvalue()
    if encoded[:10] != CANONICAL_GZIP_HEADER:
        raise ChartPackageError("runtime cannot emit canonical gzip header")
    return encoded


def _publish(output: Path, payload: bytes) -> None:
    if output.exists() or output.is_symlink():
        raise ChartPackageError("chart output already exists")
    parent = output.parent.resolve(strict=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=parent)
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        published = True
        temporary.unlink()
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        if published:
            output.unlink(missing_ok=True)
        raise


def package(source: Path, output: Path, release: str) -> None:
    if RELEASE_PATTERN.fullmatch(release) is None:
        raise ChartPackageError("release must be canonical 1.1.0-rc.N")
    if output.name != f"tritium-{release}.tgz":
        raise ChartPackageError("chart output filename must bind release")
    source = source.absolute()
    output = output.absolute()
    try:
        output.parent.resolve(strict=True).relative_to(source.resolve(strict=True))
    except ValueError:
        pass
    else:
        raise ChartPackageError("chart output must be outside chart source")
    files = _source_files(source, release)
    _publish(output, _archive(files))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("deploy/helm/tritium"))
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        package(args.source, args.output, args.release)
    except (OSError, ChartPackageError) as error:
        print(f"package-helm-chart: FAIL: {error}", file=sys.stderr)
        return 1
    print(f"package-helm-chart: OK: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
