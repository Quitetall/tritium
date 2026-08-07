"""Source-bound, replayable PyTorch batches from frozen Stage-7 evidence."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Union

import torch

from .. import _tritium

Pathish = Union[str, os.PathLike[str]]
_SCHEMA = "tritium.stage7-causal-data.v1"
_TOKENS_PER_SEQUENCE = 2_048


@dataclass(frozen=True)
class Stage7CausalDataReceipt:
    """Exact evidence identity and selected window used by model execution."""

    schema: str
    pack_id: str
    tokenizer_digest: str
    tokenizer_vocab_size: int
    token_payload_sha256: str
    partition: str
    sampling_seed: int
    start_sequence: int
    sequence_count: int
    tokens_per_sequence: int
    token_count: int
    sequence_ids: tuple[str, ...]
    ordered_members_sha256: str
    ordered_token_sha256: str
    batch_sequences: int
    terminal_validated: bool


class Stage7CausalData:
    """Replayable causal-LM batches backed by one terminally validated token window."""

    def __init__(
        self,
        tokens: torch.Tensor,
        receipt: Stage7CausalDataReceipt,
        batch_sequences: int,
    ) -> None:
        self._tokens = tokens
        self._receipt = receipt
        self._batch_sequences = batch_sequences

    @classmethod
    def open(
        cls,
        manifest_path: Pathish,
        *,
        expected_pack_id: str,
        expected_tokenizer_digest: str,
        expected_tokenizer_vocab_size: int,
        partition: str,
        start_sequence: int,
        sequence_count: int,
        batch_sequences: int,
        device: Any = "cpu",
    ) -> "Stage7CausalData":
        """Admit, select and terminally validate one exact Stage-7 sequence window."""

        if sys.byteorder != "little":
            raise RuntimeError("Stage7CausalData currently requires a little-endian host")
        for label, value in (
            ("start_sequence", start_sequence),
            ("sequence_count", sequence_count),
            ("batch_sequences", batch_sequences),
            ("expected_tokenizer_vocab_size", expected_tokenizer_vocab_size),
        ):
            if type(value) is not int:
                raise ValueError(f"{label} must be an integer")
            if label != "start_sequence" and value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if start_sequence < 0:
            raise ValueError("start_sequence must be a nonnegative integer")
        if expected_tokenizer_vocab_size > 2**31 - 1:
            raise ValueError("PyTorch Stage-7 token ids must fit signed int32")
        if batch_sequences > sequence_count:
            raise ValueError("batch_sequences must not exceed sequence_count")
        resolved_device = torch.device(device)
        if resolved_device.type == "meta":
            raise ValueError("Stage7CausalData requires a materializing device; meta is unsupported")
        native = _tritium.Stage7TokenEvidencePack(
            os.fspath(Path(manifest_path)),
            expected_pack_id,
            expected_tokenizer_digest,
            expected_tokenizer_vocab_size,
        )
        opened = native.receipt
        selected = native.read_sequences(partition, start_sequence, sequence_count)
        terminal = native.finish()
        opened_identity = (
            opened.pack_id,
            opened.tokenizer_digest,
            opened.tokenizer_vocab_size,
            opened.token_payload_sha256,
        )
        terminal_identity = (
            terminal.pack_id,
            terminal.tokenizer_digest,
            terminal.tokenizer_vocab_size,
            terminal.token_payload_sha256,
        )
        if opened_identity != terminal_identity:
            raise RuntimeError("Stage-7 token identity changed before terminal validation")
        encoded = bytearray(selected.tokens_u32le)
        expected_bytes = sequence_count * _TOKENS_PER_SEQUENCE * 4
        if len(encoded) != expected_bytes:
            raise RuntimeError("native Stage-7 token window byte geometry differs")
        tokens = torch.frombuffer(encoded, dtype=torch.int32)
        if bool((tokens < 0).any()):
            raise RuntimeError("native Stage-7 token window exceeded signed int32")
        tokens = tokens.to(device=resolved_device, dtype=torch.int64).reshape(
            sequence_count, _TOKENS_PER_SEQUENCE
        )
        sequence_ids = tuple(selected.sequence_ids)
        ordered_members = json.dumps(
            sequence_ids,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        receipt = Stage7CausalDataReceipt(
            schema=_SCHEMA,
            pack_id=terminal.pack_id,
            tokenizer_digest=terminal.tokenizer_digest,
            tokenizer_vocab_size=terminal.tokenizer_vocab_size,
            token_payload_sha256=terminal.token_payload_sha256,
            partition=selected.partition,
            sampling_seed=selected.sampling_seed,
            start_sequence=selected.start_sequence,
            sequence_count=selected.sequence_count,
            tokens_per_sequence=selected.tokens_per_sequence,
            token_count=selected.token_count,
            sequence_ids=sequence_ids,
            ordered_members_sha256="sha256:"
            + hashlib.sha256(ordered_members).hexdigest(),
            ordered_token_sha256=selected.ordered_token_sha256,
            batch_sequences=batch_sequences,
            terminal_validated=True,
        )
        return cls(tokens, receipt, batch_sequences)

    @property
    def receipt(self) -> Stage7CausalDataReceipt:
        """Immutable execution-input receipt."""

        return self._receipt

    def __len__(self) -> int:
        return (
            self._receipt.sequence_count + self._batch_sequences - 1
        ) // self._batch_sequences

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for start in range(0, self._receipt.sequence_count, self._batch_sequences):
            stop = min(start + self._batch_sequences, self._receipt.sequence_count)
            input_ids = self._tokens[start:stop].clone()
            yield {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": input_ids.clone(),
                "use_cache": False,
            }


__all__ = ["Stage7CausalData", "Stage7CausalDataReceipt"]
