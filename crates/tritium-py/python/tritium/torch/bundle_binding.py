"""Stream candidate archive/bundle equality through retained no-follow handles."""

from __future__ import annotations

from contextlib import ExitStack
import ctypes
import hashlib
import os
from pathlib import Path
import stat
import tarfile
from typing import Any, BinaryIO, Mapping


CHUNK_BYTES = 1024 * 1024
MAX_ZSTD_WINDOW_KIB = 128 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024**3
MAX_MEMBER_BYTES = 256 * 1024**3
MAX_UNCOMPRESSED_BYTES = 1024 * 1024**3
FILES = {
    "onnx-bundle": {
        "language.onnx", "mtp.onnx", "tritium-onnx-manifest.json", "weights.bin",
    },
    "model-bundle": {
        "compact.tsalt2", "near-lossless.tsalt2", "preserved.safetensors",
        "tritium.json", "chat_template.jinja", "config.json", "configuration.json",
        "generation_config.json", "merges.txt", "tokenizer.json",
        "tokenizer_config.json", "vocab.json",
    },
}


class BundleBindingError(ValueError):
    """The executed directory is not the unchanged candidate archive payload."""


def _signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns, value.st_nlink,
    )


def _exact(stream: BinaryIO, size: int) -> bytes:
    value = bytearray()
    while len(value) < size:
        chunk = stream.read(size - len(value))
        if not chunk:
            raise BundleBindingError("candidate tar is truncated")
        value.extend(chunk)
    return bytes(value)


class BoundBundle:
    """Verify once, retain every handle, and recheck custody after execution.

    Candidate archives use the flat regular-file POSIX ustar layout admitted by
    the release SBOM gate. The directory is never extracted or copied here.
    Linux no-follow descriptor traversal and directory move watches are
    required; unsupported hosts fail rather than turning path-only inspection
    into qualification.
    """

    def __init__(self, archive: Path, bundle: Path, identity: Mapping[str, Any]):
        self.archive = Path(os.path.abspath(archive))
        self.bundle = Path(os.path.abspath(bundle))
        self.identity = identity
        self._stack = ExitStack()
        self._anchors: list[tuple[int, int, str, tuple[int, ...]]] = []
        self._root = -1
        self._names: set[str] = set()
        self._root_signature: tuple[int, ...] = ()
        self._watch = -1

    def _start_watch(self) -> None:
        try:
            library = ctypes.CDLL(None, use_errno=True)
            initialize = library.inotify_init1
            self._add_watch = library.inotify_add_watch
        except AttributeError as error:
            raise BundleBindingError("platform lacks directory-move custody watches") from error
        initialize.argtypes = [ctypes.c_int]
        initialize.restype = ctypes.c_int
        self._add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self._add_watch.restype = ctypes.c_int
        self._watch = initialize(os.O_NONBLOCK | os.O_CLOEXEC)
        if self._watch < 0:
            raise OSError(ctypes.get_errno(), "cannot open directory custody watch")
        self._stack.callback(os.close, self._watch)

    def _open(self, parent: int, name: str, *, directory: bool) -> int:
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        if directory:
            flags |= os.O_DIRECTORY
        else:
            # A hostile FIFO must fail promptly instead of hanging before fstat.
            flags |= os.O_NONBLOCK
        descriptor = os.open(name, flags, dir_fd=parent)
        self._stack.callback(os.close, descriptor)
        observed = os.fstat(descriptor)
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected_type(observed.st_mode):
            raise BundleBindingError("bundle custody requires regular files/directories")
        if directory:
            # Watch the opened inode through its retained descriptor. MOVE_SELF,
            # DELETE_SELF and UNMOUNT detect a transient ancestor swap even when
            # the old path is restored; ordinary writes to shared siblings do
            # not invalidate a run. Queue overflow/ignored watches fail closed.
            result = self._add_watch(
                self._watch, os.fsencode(f"/proc/self/fd/{descriptor}"), 0x2C00
            )
            if result < 0:
                raise OSError(ctypes.get_errno(), "cannot watch directory custody")
        # Ancestor directories may serve other concurrent tasks. Retain their
        # identity, while regular payloads and the actual bundle root also bind
        # mutation timestamps below.
        signature = _signature(observed)[:3] if directory else _signature(observed)
        self._anchors.append((descriptor, parent, name, signature))
        return descriptor

    def _path(self, path: Path, *, directory: bool) -> int:
        descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self._stack.callback(os.close, descriptor)
        for part in path.parts[1:-1]:
            descriptor = self._open(descriptor, part, directory=True)
        return self._open(descriptor, path.name, directory=directory)

    def __enter__(self) -> BoundBundle:
        try:
            if (
                not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY")
                or os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd
            ):
                raise BundleBindingError("platform lacks no-follow bundle custody")
            kind = self.identity.get("kind")
            if kind not in FILES:
                raise BundleBindingError("unsupported candidate bundle kind")
            self._start_watch()
            archive_fd = self._path(self.archive, directory=False)
            self._root = self._path(self.bundle, directory=True)
            self._root_signature = _signature(os.fstat(self._root))
            size = os.fstat(archive_fd).st_size
            if (
                self.archive.name != self.identity.get("name")
                or type(self.identity.get("bytes")) is not int
                or not 0 < size <= MAX_ARCHIVE_BYTES
                or size != self.identity["bytes"]
            ):
                raise BundleBindingError("candidate archive name/bytes differ")
            with ExitStack() as verification:
                digest = hashlib.sha256()
                stream = verification.enter_context(os.fdopen(os.dup(archive_fd), "rb"))
                while chunk := stream.read(CHUNK_BYTES):
                    digest.update(chunk)
                if digest.hexdigest() != self.identity.get("sha256"):
                    raise BundleBindingError("candidate archive SHA-256 differs")
                stream.seek(0)
                if self.archive.name.endswith(".tar"):
                    payload = stream
                elif self.archive.name.endswith((".tar.zst", ".tzst")):
                    try:
                        import zstandard
                    except ImportError as error:
                        raise BundleBindingError("zstandard is required for candidate .tar.zst") from error
                    payload = verification.enter_context(
                        zstandard.ZstdDecompressor(max_window_size=MAX_ZSTD_WINDOW_KIB).stream_reader(
                            stream, read_across_frames=True
                        )
                    )
                else:
                    raise BundleBindingError("candidate bundle must use .tar, .tar.zst, or .tzst")
                try:
                    self._compare(payload, FILES[kind])
                except Exception as error:
                    if payload is not stream and isinstance(error, zstandard.ZstdError):
                        raise BundleBindingError("candidate zstd payload is invalid") from error
                    raise
            self.assert_unchanged()
            return self
        except (OSError, tarfile.TarError, UnicodeError) as error:
            self._stack.close()
            raise BundleBindingError("candidate archive/bundle custody failed") from error
        except BaseException:
            self._stack.close()
            raise

    def _compare(self, stream: BinaryIO, allowed: set[str]) -> None:
        total = 0
        while True:
            header = _exact(stream, 512)
            if not any(header):
                if any(_exact(stream, 512)):
                    raise BundleBindingError("candidate tar requires two zero end blocks")
                trailing = 1024
                while chunk := stream.read(CHUNK_BYTES):
                    trailing += len(chunk)
                    if any(chunk) or total + trailing > MAX_UNCOMPRESSED_BYTES:
                        raise BundleBindingError("candidate tar has invalid trailing data")
                if trailing % 512:
                    raise BundleBindingError("candidate tar padding is not block aligned")
                break
            if (
                header[257:265] != b"ustar\x0000"
                or any(header[345:512]) or any(header[157:257])
                or header[156:157] not in {b"0", b"\x00"}
            ):
                raise BundleBindingError("candidate tar requires flat regular POSIX ustar members")
            member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            name = member.name
            if name not in allowed or name.casefold() in {n.casefold() for n in self._names}:
                raise BundleBindingError("candidate tar has an unknown/duplicate member")
            if not 0 <= member.size <= MAX_MEMBER_BYTES:
                raise BundleBindingError("candidate tar member exceeds physical bounds")
            total += 512 + member.size + (-member.size) % 512
            if total > MAX_UNCOMPRESSED_BYTES:
                raise BundleBindingError("candidate tar exceeds uncompressed bounds")
            descriptor = self._open(self._root, name, directory=False)
            if os.fstat(descriptor).st_size != member.size:
                raise BundleBindingError(f"executed bundle file {name!r} bytes differ")
            with os.fdopen(os.dup(descriptor), "rb") as actual:
                remaining = member.size
                while remaining:
                    count = min(remaining, CHUNK_BYTES)
                    if _exact(stream, count) != _exact(actual, count):
                        raise BundleBindingError(f"executed bundle file {name!r} differs from archive")
                    remaining -= count
            padding = (-member.size) % 512
            if padding and any(_exact(stream, padding)):
                raise BundleBindingError("candidate tar member padding is not zero")
            self._names.add(name)
        if not self._names or set(os.listdir(self._root)) != self._names:
            raise BundleBindingError("executed bundle inventory differs from archive")

    def assert_unchanged(self) -> None:
        try:
            try:
                events = os.read(self._watch, 4096)
            except BlockingIOError:
                events = b""
            if events:
                raise BundleBindingError("candidate archive/bundle directory custody changed")
            for descriptor, parent, name, initial in self._anchors:
                current = _signature(os.fstat(descriptor))[:len(initial)]
                named = _signature(
                    os.stat(name, dir_fd=parent, follow_symlinks=False)
                )[:len(initial)]
                if current != initial or named != initial:
                    raise BundleBindingError("candidate archive/bundle changed during execution")
            if (
                _signature(os.fstat(self._root)) != self._root_signature
                or set(os.listdir(self._root)) != self._names
            ):
                raise BundleBindingError("executed bundle inventory changed during execution")
        except OSError as error:
            raise BundleBindingError("candidate archive/bundle custody changed") from error

    def __exit__(self, kind: Any, value: Any, traceback: Any) -> None:
        try:
            if kind is None:
                self.assert_unchanged()
        finally:
            self._stack.close()
