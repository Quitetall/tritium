"""Shared installed-candidate admission without a Transformers dependency."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
from pathlib import Path

import tritium
from tritium import _tritium

from ._wheel_identity import file_sha256, wheel_identity


def verify_installed_candidate(
    *,
    wheel_artifact: Path | None,
    source_revision: str,
    release: str,
    executing_files: tuple[Path, ...] = (),
) -> tuple[str, Path]:
    """Bind native source, release and executing payload to the candidate.

    Entry points supply their installed code origins. Installers may add
    metadata/bytecode and rewrite RECORD, but cannot change wheel payloads.
    Without an external wheel this checks installed RECORD integrity, not
    independent archive-byte identity. It is not a hostile-Python sandbox.
    """
    try:
        distribution = importlib.metadata.distribution("pytritium")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("qualification requires installed pytritium") from error
    identity = getattr(_tritium, "source_identity", None)
    if not callable(identity) or identity() != "source-git:" + source_revision:
        raise ValueError("candidate native source identity differs")
    if distribution.version != release.replace("-rc.", "rc"):
        raise ValueError("candidate installed release version differs")
    module = Path(tritium.__file__).resolve(strict=True)
    if distribution.files is None:
        raise RuntimeError("installed pytritium has no file inventory")
    files = tuple(distribution.files)
    logical = {str(item).replace("\\", "/"): item for item in files}
    if len(logical) != len(files):
        raise ValueError("candidate installed RECORD contains duplicate paths")
    owned = {distribution.locate_file(item).resolve() for item in files}
    origins = (
        Path(tritium.__file__), Path(_tritium.__file__), Path(__file__),
        *executing_files,
    )
    if any(path.is_symlink() or path.resolve(strict=True) not in owned for path in origins):
        raise RuntimeError("candidate executing package is not owned by pytritium")
    if wheel_artifact is not None:
        inventory = wheel_identity(wheel_artifact)
        if inventory["distribution_version"] != distribution.version:
            raise ValueError("candidate wheel distribution version differs")
        for entry in inventory["entries"]:
            name = entry["path"]
            if name not in logical:
                raise ValueError("candidate wheel member is absent from installed RECORD")
            installed = distribution.locate_file(logical[name])
            if installed.is_symlink() or not installed.is_file():
                raise ValueError("candidate installed member is not an ordinary file")
            if name == inventory["record_path"]:
                continue
            if (
                installed.stat().st_size != entry["bytes"]
                or file_sha256(installed) != entry["sha256"]
            ):
                raise ValueError("candidate wheel differs from executing installed files")
        expected = {
            distribution.locate_file(logical[entry["path"]]).resolve()
            for entry in inventory["entries"]
        }
        if any(path.resolve(strict=True) not in expected for path in origins):
            raise ValueError("candidate wheel does not own executing qualification code")
    else:
        for item in files:
            name = str(item).replace("\\", "/")
            if not name.startswith("tritium/") or name.endswith(".pyc"):
                continue
            installed = distribution.locate_file(item)
            digest = item.hash
            if (
                installed.is_symlink()
                or not installed.is_file()
                or digest is None
                or digest.mode not in {"sha256", "sha384", "sha512"}
            ):
                raise ValueError("candidate installed package lacks strong RECORD integrity")
            hasher = hashlib.new(digest.mode)
            with installed.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    hasher.update(chunk)
            encoded = base64.urlsafe_b64encode(hasher.digest()).rstrip(b"=").decode()
            if encoded != digest.value or installed.stat().st_size != item.size:
                raise ValueError("candidate installed package RECORD identity differs")
    return distribution.version, module
