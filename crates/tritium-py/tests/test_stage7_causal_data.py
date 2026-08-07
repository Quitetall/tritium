import hashlib
import json
import struct

import pytest

torch = pytest.importorskip("torch")

from tritium import (  # noqa: E402
    Stage7EvidenceContractError,
    Stage7EvidenceStateError,
    Stage7TokenEvidencePack,
)
from tritium.torch import Stage7CausalData  # noqa: E402


DATASETS = (
    (
        "allenai/c4",
        "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        "en",
        None,
        "text",
        256,
    ),
    (
        "open-web-math/open-web-math",
        "fde8ef8de2300f5e778f56261843dab89f230815",
        "default",
        None,
        "text",
        128,
    ),
    (
        "bigcode/starcoderdata",
        "9fc30b578cedaec69e47302df72cf00feed7c8c4",
        "default",
        "python",
        "content",
        128,
    ),
)


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def prefixed_digest(value):
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def write_pack(root):
    root.mkdir()
    payload = bytearray()
    partitions = {}
    global_ordinal = 0
    for partition_ordinal, (partition, seed) in enumerate(
        (
            ("calibration", 11),
            ("refinement", 12),
            ("validation", 13),
            ("evaluation", 14),
        )
    ):
        sequences = []
        dataset_ordinal = 0
        lane_ordinal = 0
        for sequence_ordinal in range(512):
            while lane_ordinal == DATASETS[dataset_ordinal][5]:
                dataset_ordinal += 1
                lane_ordinal = 0
            repo, revision, config, data_dir, text_field, _ = DATASETS[dataset_ordinal]
            tokens = struct.pack("<I", global_ordinal) + bytes((2_048 - 1) * 4)
            token_digest = "sha256:" + hashlib.sha256(tokens).hexdigest()
            row_index = partition_ordinal * 10_000 + sequence_ordinal
            sequence = {
                "dataset_repo_id": repo,
                "dataset_revision": revision,
                "dataset_config": config,
                "dataset_data_dir": data_dir,
                "dataset_split": "train",
                "source_rows": [
                    {
                        "row_index": row_index,
                        "text_field": text_field,
                        "content_sha256": hashlib.sha256(
                            f"{partition}:{repo}:{row_index}".encode()
                        ).hexdigest(),
                    }
                ],
                "token_offset": global_ordinal * 2_048,
                "token_count": 2_048,
                "token_sha256": token_digest,
            }
            sequence["id"] = prefixed_digest(sequence)
            sequences.append(sequence)
            payload.extend(tokens)
            lane_ordinal += 1
            global_ordinal += 1
        partitions[partition] = {"sampling_seed": seed, "sequences": sequences}
    payload_path = root / "stage7.u32le"
    payload_path.write_bytes(payload)
    tokenizer_digest = "sha256:" + "a5" * 32
    manifest = {
        "schema": "tritium.stage7-token-evidence-pack.v1",
        "tokenizer_digest": tokenizer_digest,
        "tokenizer_vocab_size": 4_096,
        "token_encoding": "u32le",
        "tokens": {
            "path": "stage7.u32le",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "partitions": partitions,
    }
    manifest["pack_id"] = prefixed_digest(manifest)
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(canonical(manifest) + b"\n")
    return manifest_path, manifest


def test_stage7_causal_data_replays_exact_source_bound_batches(tmp_path):
    manifest_path, manifest = write_pack(tmp_path / "pack")

    data = Stage7CausalData.open(
        manifest_path,
        expected_pack_id=manifest["pack_id"],
        expected_tokenizer_digest=manifest["tokenizer_digest"],
        expected_tokenizer_vocab_size=4_096,
        partition="calibration",
        start_sequence=0,
        sequence_count=2,
        batch_sequences=1,
    )

    assert data.receipt.schema == "tritium.stage7-causal-data.v1"
    assert data.receipt.pack_id == manifest["pack_id"]
    assert data.receipt.partition == "calibration"
    assert data.receipt.sampling_seed == 11
    assert data.receipt.start_sequence == 0
    assert data.receipt.sequence_count == 2
    assert data.receipt.tokens_per_sequence == 2_048
    assert data.receipt.token_count == 4_096
    assert data.receipt.sequence_ids == tuple(
        sequence["id"]
        for sequence in manifest["partitions"]["calibration"]["sequences"][:2]
    )
    assert data.receipt.ordered_members_sha256 == prefixed_digest(
        list(data.receipt.sequence_ids)
    )
    assert data.receipt.terminal_validated is True

    first = list(data)
    second = list(data)
    assert len(first) == len(second) == 2
    assert first[0]["input_ids"].shape == (1, 2_048)
    assert first[0]["input_ids"].dtype == torch.int64
    assert first[0]["input_ids"][0, 0].item() == 0
    assert first[1]["input_ids"][0, 0].item() == 1
    assert torch.equal(first[0]["labels"], first[0]["input_ids"])
    assert torch.equal(first[0]["attention_mask"], torch.ones((1, 2_048), dtype=torch.int64))
    assert first[0]["use_cache"] is False
    for left, right in zip(first, second):
        assert torch.equal(left["input_ids"], right["input_ids"])

    with pytest.raises(Stage7EvidenceContractError, match="expected campaign"):
        Stage7TokenEvidencePack(
            str(manifest_path),
            "sha256:" + "00" * 32,
            manifest["tokenizer_digest"],
            4_096,
        )

    native = Stage7TokenEvidencePack(
        str(manifest_path),
        manifest["pack_id"],
        manifest["tokenizer_digest"],
        4_096,
    )
    selected = native.read_sequences("calibration", 0, 2)
    assert len(selected.tokens_u32le) == 2 * 2_048 * 4
    assert selected.sequence_ids == list(data.receipt.sequence_ids)
    native.finish()
    with pytest.raises(Stage7EvidenceStateError, match="already terminal"):
        native.read_sequences("calibration", 0, 1)


def test_stage7_causal_data_rejects_nonmaterializing_device_before_pack_open(tmp_path):
    with pytest.raises(ValueError, match="materializing"):
        Stage7CausalData.open(
            tmp_path / "missing-manifest.json",
            expected_pack_id="sha256:" + "01" * 32,
            expected_tokenizer_digest="sha256:" + "02" * 32,
            expected_tokenizer_vocab_size=4_096,
            partition="calibration",
            start_sequence=0,
            sequence_count=1,
            batch_sequences=1,
            device="meta",
        )
