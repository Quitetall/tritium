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
from tritium.torch import (  # noqa: E402
    Stage7CausalData,
    run_stage7_smoke_model,
    run_stage7_smollm2_smoke,
)


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


class TinyCausalLM(torch.nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4_096, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, 4_096, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        del attention_mask, use_cache
        hidden = torch.tanh(self.proj(self.embed_tokens(input_ids)))
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.shape[-1]),
                labels[..., 1:].reshape(-1),
            )
        return {"logits": logits, "loss": loss}


@pytest.mark.parametrize("hidden_size", [128, 64])
def test_stage7_smoke_model_executes_real_pipeline_and_resumes(tmp_path, hidden_size):
    torch.manual_seed(7)
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
    model = TinyCausalLM(hidden_size).eval()

    first = run_stage7_smoke_model(
        model,
        data,
        tmp_path / "smoke",
        packing="b3",
        max_working_bytes=16 * 1024 * 1024,
    )

    assert first.schema == "tritium.stage7-smoke-model.v1"
    assert first.result == "pass"
    assert first.package_id.startswith("trp1_")
    assert first.packing == "b3"
    assert first.tensor_count == 2
    assert first.evaluated_tokens == 2 * (2_048 - 1)
    assert first.mean_loss > 0
    assert first.terminal_validated is True
    assert first.stage_names == (
        "capture",
        "fit",
        "allocate",
        "package",
        "evaluate",
    )
    assert first.artifact_path.is_file()
    package_version = int.from_bytes(first.artifact_path.read_bytes()[8:10], "little")
    assert package_version == (1 if hidden_size == 128 else 2)
    allocation = json.loads((tmp_path / "smoke" / "allocation.json").read_text())
    assert {tensor["scale_group_size"] for tensor in allocation["tensors"]} == {
        hidden_size
    }

    resumed = run_stage7_smoke_model(
        model,
        data,
        tmp_path / "smoke",
        packing="b3",
        max_working_bytes=16 * 1024 * 1024,
    )
    assert resumed == first


def test_stage7_smoke_model_rejects_invalid_ptq_profile_options(tmp_path):
    torch.manual_seed(8)
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
    model = TinyCausalLM(64).eval()

    with pytest.raises(ValueError, match="profile must be"):
        run_stage7_smoke_model(model, data, tmp_path / "invalid-profile", profile="bad")
    with pytest.raises(ValueError, match="target_bpw must be"):
        run_stage7_smoke_model(model, data, tmp_path / "invalid-bpw", target_bpw=-1)


def test_stage7_smollm2_smoke_rejects_noncanonical_model_before_output(tmp_path):
    campaign = {
        "schema": "tritium.stage7-campaign.v1",
        "release": "1.1.0-rc.0",
        "source_revision": "12" * 20,
        "run_id": "stage7-test",
        "model": {},
        "smoke_model": {
            "repo_id": "HuggingFaceTB/SmolLM2-135M",
            "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
            "files": [],
        },
        "smoke_provenance": {},
        "provenance": {},
        "thresholds": {},
        "recipe_count": 1_404,
        "recipe_grid_id": "sha256:" + "34" * 32,
        "token_evidence_pack": {},
        "evidence": [],
    }
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_bytes(canonical(campaign) + b"\n")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output = tmp_path / "smoke"

    with pytest.raises(ValueError, match="model file inventory"):
        run_stage7_smollm2_smoke(
            campaign_path,
            model_dir,
            output,
            device="cpu",
        )

    assert not output.exists()
