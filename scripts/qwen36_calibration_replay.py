"""Pack-backed replay batches for the pinned Qwen3.6 calibration capture."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import runpy
import stat
import struct
from typing import Any, Callable, Iterable, Mapping, Sequence


_VERIFIER = runpy.run_path(
    Path(__file__).with_name("verify-qwen36-calibration-pack.py")
)
_CalibrationPackError = _VERIFIER["CalibrationPackError"]
_CONSTRUCTOR_KEY = object()


def _read_calibration_tokens(
    manifest_path: Path,
    pack_receipt: dict[str, Any],
) -> bytes:
    load_json = _VERIFIER["_load_json"]
    safe_payload_path = _VERIFIER["_safe_payload_path"]
    manifest = load_json(
        manifest_path,
        "token evidence manifest",
        _VERIFIER["MAX_MANIFEST_BYTES"],
    )
    if manifest.get("pack_id") != pack_receipt["pack_id"]:
        raise _CalibrationPackError("replay source pack ID differs from receipt")
    expected_pack_id = "sha256:" + hashlib.sha256(
        _VERIFIER["canonical"](
            {key: value for key, value in manifest.items() if key != "pack_id"}
        )
    ).hexdigest()
    if expected_pack_id != manifest["pack_id"]:
        raise _CalibrationPackError("replay source manifest ID differs")
    sequences = manifest["partitions"]["calibration"]["sequences"]
    sequence_count = _VERIFIER["SEQUENCES_PER_PARTITION"]
    tokens_per_sequence = _VERIFIER["TOKENS_PER_SEQUENCE"]
    if len(sequences) != sequence_count:
        raise _CalibrationPackError("replay source calibration sequence count differs")
    token_path = safe_payload_path(
        manifest_path.parent.resolve(strict=True), manifest["tokens"]["path"]
    )
    start = sequences[0]["token_offset"] * 4
    expected_bytes = sequence_count * tokens_per_sequence * 4
    descriptor = os.open(token_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise _CalibrationPackError("replay token payload is not a regular file")
        os.lseek(descriptor, start, os.SEEK_SET)
        chunks = bytearray()
        while len(chunks) < expected_bytes:
            chunk = os.read(descriptor, expected_bytes - len(chunks))
            if not chunk:
                raise _CalibrationPackError("replay token window is truncated")
            chunks.extend(chunk)
        after = os.fstat(descriptor)
        current = token_path.stat(follow_symlinks=False)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or current.st_ino != after.st_ino
            or not stat.S_ISREG(current.st_mode)
        ):
            raise _CalibrationPackError("replay token payload changed while reading")
    finally:
        os.close(descriptor)
    token_bytes = bytes(chunks)
    expected_digest = pack_receipt["calibration"]["ordered_token_sha256"]
    observed_digest = "sha256:" + hashlib.sha256(token_bytes).hexdigest()
    if observed_digest != expected_digest:
        raise _CalibrationPackError(
            "replay token window differs from verified pack receipt"
        )
    for index, sequence in enumerate(sequences):
        offset = index * tokens_per_sequence * 4
        body = token_bytes[offset : offset + tokens_per_sequence * 4]
        sequence_digest = hashlib.sha256(body).hexdigest()
        if sequence_digest != sequence["token_sha256"].removeprefix("sha256:"):
            raise _CalibrationPackError("replay sequence tokens differ from manifest")
    return token_bytes


def iter_capture_batches(
    token_bytes: bytes,
    tensor_factory: Callable[[Sequence[int], tuple[int, int]], Any],
) -> Iterable[Mapping[str, Any]]:
    """Yield the one-sequence batches frozen by the replay contract."""
    if not callable(tensor_factory):
        raise TypeError("tensor_factory must be callable")
    tokens_per_sequence = _VERIFIER["TOKENS_PER_SEQUENCE"]
    sequence_count = _VERIFIER["SEQUENCES_PER_PARTITION"]
    if len(token_bytes) != sequence_count * tokens_per_sequence * 4:
        raise ValueError("calibration token bytes differ from frozen replay geometry")
    attention = [1] * tokens_per_sequence
    shape = (1, tokens_per_sequence)
    for index in range(sequence_count):
        start = index * tokens_per_sequence * 4
        sequence = token_bytes[start : start + tokens_per_sequence * 4]
        input_ids = [value[0] for value in struct.iter_unpack("<I", sequence)]
        yield {
            "attention_mask": tensor_factory(attention, shape),
            "input_ids": tensor_factory(input_ids, shape),
        }


def torch_int64_tensor_factory(
    device: Any = "cpu",
) -> Callable[[Sequence[int], tuple[int, int]], Any]:
    """Create the capture tensor factory without importing Torch eagerly."""
    import torch

    def make(values: Sequence[int], shape: tuple[int, int]):
        return torch.tensor(values, dtype=torch.int64, device=device).reshape(shape)

    return make


class Qwen36CalibrationReplay:
    """A verified calibration window and factory for capture API replays."""

    def __init__(
        self,
        token_bytes: bytes,
        receipt: dict[str, Any],
        contract: dict[str, Any],
        *,
        _constructor_key: object = None,
    ):
        if _constructor_key is not _CONSTRUCTOR_KEY:
            raise TypeError("use Qwen36CalibrationReplay.open to verify the pack")
        self._token_bytes = token_bytes
        self.receipt = receipt
        self.contract = contract
        self._token_stream_digest = contract["capture_batch_sha256"]

    @classmethod
    def open(
        cls,
        manifest_path: Path,
        model_dir: Path,
        official_source_identity: Path,
        pack_receipt_path: Path,
    ) -> "Qwen36CalibrationReplay":
        verifier = _VERIFIER
        stored = verifier["validate_receipt"](pack_receipt_path)
        fresh = verifier["verify_pack"](
            manifest_path, model_dir, official_source_identity
        )
        if stored != fresh:
            raise _CalibrationPackError(
                "stored calibration receipt differs from current pack"
            )
        token_bytes = _read_calibration_tokens(manifest_path, fresh)
        contract = verifier["make_replay_contract"](manifest_path, fresh)
        return cls(
            token_bytes,
            fresh,
            contract,
            _constructor_key=_CONSTRUCTOR_KEY,
        )

    @property
    def token_stream_digest(self) -> str:
        """Digest the native capture guard must enforce for every replay."""
        return self._token_stream_digest

    def data_factory(
        self,
        tensor_factory: Callable[[Sequence[int], tuple[int, int]], Any],
    ) -> Callable[[Any], Iterable[Mapping[str, Any]]]:
        """Return a fresh pack-backed iterator for every native capture task."""
        def factory(_task: Any) -> Iterable[Mapping[str, Any]]:
            return iter_capture_batches(self._token_bytes, tensor_factory)

        return factory
