# Ported verbatim from blut-lamu python/lamu/kd_cache.py by its copyright holder (ADR 0037 Stage 4).
"""Top-k teacher-probability cache — shared format helpers (T2b).

The cache is produced by `make_teacher_topk.py` (teacher-forced forward of the
spec-decode TARGET over the teacher token streams) and consumed by
`trainer_distill.py` (kd_cache spec key). Layout, on disk in one directory:

  manifest.json            k, n_positions, shard table, per-stream table
  shard_NNNN.ids.npy       uint32 [n, k]  top-k token ids (descending prob)
  shard_NNNN.probs.npy     float16 [n, k] full-vocab softmax mass at those ids

Positions are FLAT across streams in source-file order: a stream of L tokens
contributes L-1 positions (position i holds the teacher's distribution over
token i+1 given tokens[..=i] — the same alignment as the hard CE target
`tgt = window[1:]`). The final token of a stream has no next-token
distribution and is not stored.

ALIGNMENT CONTRACT (how the trainer finds coverage): streams are keyed by
`stream_key(tokens)` — sha256 of the uint32-le token bytes. The training mix
(make_train_mix.py) concatenates corpus + N verbatim repeats of the teacher
jsonl, so every teacher line in the mix hashes to a cache stream; corpus
lines miss and fall back to plain CE. No separate mix manifest is needed and
the mix file/windowing stays byte-identical to the non-KD path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MANIFEST_NAME = "manifest.json"
FORMAT_NAME = "blut-topk-teacher-cache"
FORMAT_VERSION = 1


def stream_key(tokens) -> str:
    """Content hash of one token stream (uint32 little-endian bytes)."""
    return hashlib.sha256(np.asarray(tokens, dtype=np.uint32).tobytes()).hexdigest()


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class TopkCache:
    """In-RAM view of a dumped cache: flat [N, k] ids/probs + stream lookup."""

    def __init__(self, ids: np.ndarray, probs: np.ndarray, k: int,
                 streams: dict[str, tuple[int, int]], manifest: dict):
        self.ids = ids        # uint32 [N, k]
        self.probs = probs    # float16 [N, k]
        self.k = k
        self.streams = streams  # stream_key -> (position offset, n_tokens)
        self.manifest = manifest

    @classmethod
    def load(cls, cache_dir: str | Path) -> "TopkCache":
        cache_dir = Path(cache_dir)
        man = json.loads((cache_dir / MANIFEST_NAME).read_text())
        if man.get("format") != FORMAT_NAME:
            raise ValueError(f"{cache_dir}: not a {FORMAT_NAME} (manifest 'format' key)")
        k = int(man["k"])
        if not man.get("shards"):
            raise ValueError(f"{cache_dir}: manifest lists no shards (empty dump?)")
        ids_parts, prob_parts = [], []
        for sh in man["shards"]:
            ids_parts.append(np.load(cache_dir / sh["ids"]))
            prob_parts.append(np.load(cache_dir / sh["probs"]))
        ids = np.concatenate(ids_parts) if len(ids_parts) > 1 else ids_parts[0]
        probs = np.concatenate(prob_parts) if len(prob_parts) > 1 else prob_parts[0]
        n = int(man["n_positions"])
        if ids.shape != (n, k) or probs.shape != (n, k):
            raise ValueError(
                f"{cache_dir}: shard payload {ids.shape}/{probs.shape} != manifest ({n}, {k})"
            )
        streams = {s["key"]: (int(s["offset"]), int(s["n_tokens"])) for s in man["streams"]}
        return cls(ids, probs, k, streams, man)
