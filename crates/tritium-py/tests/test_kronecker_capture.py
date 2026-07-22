from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tritium import KroneckerPublicationError, KroneckerStateError
from tritium.torch import (
    KroneckerCalibrationWriter,
    KroneckerModuleCaptureReceipt,
    bind_kronecker_activation_cache_digest,
    capture_kronecker_embedding,
    capture_kronecker_module,
    capture_qwen36_kronecker_evidence,
)
from tritium.torch import ptq


class _TinyObjectiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(128, 4, bias=False)

    def forward(self, values, labels=None, attention_mask=None):
        logits = self.proj(values)
        loss = None
        if labels is not None:
            per_token = (logits - labels).square().mean(dim=-1)
            if attention_mask is not None:
                per_token = per_token[attention_mask.bool()]
            loss = per_token.mean()
        return SimpleNamespace(logits=logits, loss=loss)


def _writer(tmp_path, curvature, guided_reduction="mean-attention-mask"):
    objective = {
        "input-hessian": "tritium.input-gram@1",
        "guided-fisher": f"tritium.model-loss-guided-fisher.{guided_reduction}@1",
        "forward-kl-kronecker": "tritium.softmax-fisher-rademacher.single-probe@1",
    }[curvature]
    return KroneckerCalibrationWriter(
        tmp_path / curvature,
        tensor_index=0,
        tensor_name="proj.weight",
        rows=4,
        columns=128,
        curvature=curvature,
        source_model_digest="01" * 32,
        activation_cache_digest="02" * 32,
        token_stream_digest="03" * 32,
        damping=0.01,
        objective_id=objective,
        max_batch_bytes=16 * 1024,
    )


def _embedding_writer(tmp_path, curvature):
    objective = {
        "guided-fisher": "tritium.model-loss-guided-fisher.mean-attention-mask@1",
        "forward-kl-kronecker": "tritium.softmax-fisher-rademacher.single-probe@1",
    }[curvature]
    return KroneckerCalibrationWriter(
        tmp_path / f"embedding-{curvature}",
        tensor_index=1,
        tensor_name="embed.weight",
        rows=8,
        columns=128,
        curvature=curvature,
        source_model_digest="01" * 32,
        activation_cache_digest="02" * 32,
        token_stream_digest="03" * 32,
        damping=0.01,
        objective_id=objective,
        indexed_output=True,
        max_batch_bytes=16 * 1024,
    )


def _batches():
    return [
        {
            "values": torch.arange(3 * 128, dtype=torch.float32).view(1, 3, 128)
            / 128,
            "labels": torch.zeros(1, 3, 4),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        },
        {
            "values": torch.ones(1, 2, 128),
            "labels": torch.full((1, 2, 4), 0.5),
            "attention_mask": torch.tensor([[1, 1]]),
        },
    ]


@pytest.mark.parametrize(
    "curvature", ["input-hessian", "guided-fisher", "forward-kl-kronecker"]
)
def test_model_aware_capture_publishes_all_estimators(tmp_path, curvature):
    model = _TinyObjectiveModel().train()
    writer = _writer(tmp_path, curvature)
    receipt = capture_kronecker_module(
        model,
        _batches(),
        module="proj",
        writer=writer,
        curvature=curvature,
        guided_loss_reduction=(
            "mean-attention-mask" if curvature == "guided-fisher" else None
        ),
    )

    assert isinstance(receipt, KroneckerModuleCaptureReceipt)
    assert receipt.module == "proj"
    assert receipt.curvature == curvature
    assert receipt.batches == 2
    assert receipt.module_calls == 2
    assert receipt.samples == 5
    assert receipt.selected_samples == 4
    assert receipt.record.tensor_index == 0
    assert not writer.active
    assert model.training is True
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("curvature", ["guided-fisher", "forward-kl-kronecker"])
def test_embedding_capture_streams_sparse_vocabulary_factors(tmp_path, curvature):
    class TinyEmbeddingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 128)
            self.head = torch.nn.Linear(128, 8, bias=False)

        def forward(self, input_ids, labels=None, attention_mask=None):
            hidden = self.embed(input_ids)
            logits = self.head(hidden)
            loss = logits.square().mean()
            return SimpleNamespace(logits=logits, loss=loss)

    model = TinyEmbeddingModel().train()
    receipt = capture_kronecker_embedding(
        model,
        [
            {
                "input_ids": torch.tensor([[1, 3, 1]]),
                "attention_mask": torch.tensor([[1, 1, 0]]),
            }
        ],
        module="embed",
        writer=_embedding_writer(tmp_path, curvature),
        curvature=curvature,
        guided_loss_reduction=(
            "mean-attention-mask" if curvature == "guided-fisher" else None
        ),
    )

    assert receipt.module == "embed"
    assert receipt.samples == 3
    assert receipt.selected_samples == 2
    assert receipt.module_calls == 1
    assert model.training is True
    assert all(parameter.grad is None for parameter in model.parameters())


def test_embedding_capture_rejects_input_hessian(tmp_path):
    writer = KroneckerCalibrationWriter(
        tmp_path / "bad-embedding",
        tensor_index=2,
        tensor_name="embed.weight",
        rows=8,
        columns=128,
        curvature="input-hessian",
        source_model_digest="01" * 32,
        activation_cache_digest="02" * 32,
        token_stream_digest="03" * 32,
        damping=0.01,
        objective_id="tritium.input-gram@1",
    )
    with pytest.raises(ValueError, match="does not support input-hessian"):
        capture_kronecker_embedding(
            torch.nn.Sequential(torch.nn.Embedding(8, 128)),
            [torch.tensor([[1]])],
            module="0",
            writer=writer,
            curvature="input-hessian",
        )


def test_qwen_capture_session_dispatches_embedding_and_output_head(tmp_path):
    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 128)

    class TinyQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Body()
            self.lm_head = torch.nn.Linear(128, 8, bias=False)

        def forward(self, input_ids, attention_mask=None):
            logits = self.lm_head(self.model.embed_tokens(input_ids))
            return SimpleNamespace(logits=logits, loss=logits.square().mean())

    class FakeSession:
        def __init__(
            self,
            model_dir,
            revision,
            work_dir,
            evidence_dir,
            curvature,
            activation_cache_digest,
            token_stream_digest,
            damping,
            **kwargs,
        ):
            common = dict(
                rows=8,
                columns=128,
                scope="language",
                source_model_digest="01" * 32,
                activation_cache_digest=activation_cache_digest,
                token_stream_digest=token_stream_digest,
                curvature=curvature,
                damping=damping,
            )
            self.tasks = iter(
                [
                    SimpleNamespace(
                        tensor_index=0,
                        tensor_name="model.language_model.embed_tokens.weight",
                        role="token-embedding",
                        **common,
                    ),
                    SimpleNamespace(
                        tensor_index=1,
                        tensor_name="lm_head.weight",
                        role="output-head",
                        **common,
                    ),
                ]
            )
            self.current = None
            self.accepted = 0

        def next_request(self):
            self.current = next(self.tasks, None)
            return self.current

        def accept_current(self):
            assert self.current is not None
            self.accepted += 1
            self.current = None
            return True

        def finish(self):
            assert self.accepted == 2
            return SimpleNamespace(records=2, produced=2, reused=0)

    model = TinyQwen()
    receipt = capture_qwen36_kronecker_evidence(
        model,
        lambda task: [
            {
                "input_ids": torch.tensor([[1, 3]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }
        ],
        model_dir=tmp_path / "model",
        declared_revision="test-revision",
        work_dir=tmp_path / "work",
        evidence_dir=tmp_path / "evidence",
        curvature="guided-fisher",
        activation_cache_digest="02" * 32,
        token_stream_digest="03" * 32,
        damping=0.01,
        guided_loss_reduction="mean-attention-mask",
        _session_factory=FakeSession,
    )

    assert receipt.records == 2
    assert (tmp_path / "evidence" / "000000.s2kf").is_file()
    assert (tmp_path / "evidence" / "000001.s2kf").is_file()


def test_capture_preserves_reused_module_call_order(tmp_path):
    class Reused(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(128, 4, bias=False)

        def forward(self, values, labels=None, attention_mask=None):
            logits = self.proj(values) + self.proj(values + 1)
            return SimpleNamespace(logits=logits, loss=logits.square().mean())

    receipt = capture_kronecker_module(
        Reused(),
        [{"values": torch.ones(1, 2, 128), "attention_mask": torch.ones(1, 2)}],
        module="proj",
        writer=_writer(tmp_path, "guided-fisher"),
        curvature="guided-fisher",
        guided_loss_reduction="mean-attention-mask",
    )
    assert receipt.module_calls == 2
    assert receipt.samples == 4
    assert receipt.selected_samples == 4


def test_capture_snapshots_inputs_before_in_place_reuse(tmp_path):
    class Reused(torch.nn.Module):
        def __init__(self, safe):
            super().__init__()
            self.safe = safe
            self.proj = torch.nn.Linear(128, 4, bias=False)

        def forward(self, values):
            first = self.proj(values.clone() if self.safe else values)
            values.add_(1)
            return first + self.proj(values)

    unsafe = Reused(False)
    safe = Reused(True)
    safe.load_state_dict(unsafe.state_dict())
    unsafe_receipt = capture_kronecker_module(
        unsafe,
        [torch.zeros(1, 2, 128)],
        module="proj",
        writer=_writer(tmp_path / "unsafe", "input-hessian"),
        curvature="input-hessian",
    )
    safe_receipt = capture_kronecker_module(
        safe,
        [torch.zeros(1, 2, 128)],
        module="proj",
        writer=_writer(tmp_path / "safe", "input-hessian"),
        curvature="input-hessian",
    )
    assert unsafe_receipt.record.record_digest == safe_receipt.record.record_digest


def test_capture_restores_mixed_training_flags(tmp_path):
    model = _TinyObjectiveModel().train()
    model.proj.eval()
    capture_kronecker_module(
        model,
        _batches(),
        module="proj",
        writer=_writer(tmp_path, "input-hessian"),
        curvature="input-hessian",
    )
    assert model.training is True
    assert model.proj.training is False


def test_capture_aborts_partial_or_malformed_runs(tmp_path):
    model = _TinyObjectiveModel()
    writer = _writer(tmp_path, "guided-fisher")
    bad = _batches()
    bad.append({"values": torch.ones(1, 1, 127), "labels": torch.zeros(1, 1, 4)})
    with pytest.raises(ValueError, match="feature"):
        capture_kronecker_module(
            model,
            bad,
            module="proj",
            writer=writer,
            curvature="guided-fisher",
            guided_loss_reduction="mean-attention-mask",
        )
    assert not writer.active
    assert not (tmp_path / "guided-fisher" / "000000.s2kf").exists()
    with pytest.raises(KroneckerStateError):
        writer.finish()

    with pytest.raises(ValueError, match="module"):
        capture_kronecker_module(
            model,
            _batches(),
            module="missing",
            writer=_writer(tmp_path, "input-hessian"),
            curvature="input-hessian",
        )


def test_capture_requires_estimator_writer_agreement(tmp_path):
    with pytest.raises(ValueError, match="curvature"):
        capture_kronecker_module(
            _TinyObjectiveModel(),
            _batches(),
            module="proj",
            writer=_writer(tmp_path, "input-hessian"),
            curvature="guided-fisher",
            guided_loss_reduction="mean-attention-mask",
        )

    wrong_objective = KroneckerCalibrationWriter(
        tmp_path / "wrong-objective",
        tensor_index=0,
        tensor_name="proj.weight",
        rows=4,
        columns=128,
        curvature="guided-fisher",
        source_model_digest="01" * 32,
        activation_cache_digest="02" * 32,
        token_stream_digest="03" * 32,
        damping=0.01,
        objective_id="tritium.model-loss-guided-fisher.sum@1",
    )
    with pytest.raises(ValueError, match="objective_id"):
        capture_kronecker_module(
            _TinyObjectiveModel(),
            _batches(),
            module="proj",
            writer=wrong_objective,
            curvature="guided-fisher",
            guided_loss_reduction="mean-attention-mask",
        )
    assert not wrong_objective.active


def test_guided_fisher_mean_is_rescaled_and_partition_independent(tmp_path):
    model = _TinyObjectiveModel().eval()
    with torch.no_grad():
        model.proj.weight.zero_()
    values = torch.arange(4 * 128, dtype=torch.float32).view(1, 4, 128) / 100
    labels = torch.ones(1, 4, 4)
    one = capture_kronecker_module(
        model,
        [{"values": values.clone(), "labels": labels, "attention_mask": torch.ones(1, 4)}],
        module="proj",
        writer=_writer(tmp_path / "one", "guided-fisher"),
        curvature="guided-fisher",
        guided_loss_reduction="mean-attention-mask",
    )
    two = capture_kronecker_module(
        model,
        [
            {
                "values": values[:, :2].clone(),
                "labels": labels[:, :2],
                "attention_mask": torch.ones(1, 2),
            },
            {
                "values": values[:, 2:].clone(),
                "labels": labels[:, 2:],
                "attention_mask": torch.ones(1, 2),
            },
        ],
        module="proj",
        writer=_writer(tmp_path / "two", "guided-fisher"),
        curvature="guided-fisher",
        guided_loss_reduction="mean-attention-mask",
    )
    assert one.record.record_digest == two.record.record_digest


def test_causal_guided_fisher_uses_shifted_label_selection(tmp_path):
    class CausalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(128, 4, bias=False)
            with torch.no_grad():
                self.proj.weight.zero_()

        def forward(self, values, labels, attention_mask):
            logits = self.proj(values)
            loss = torch.nn.functional.cross_entropy(
                logits[..., :-1, :].reshape(-1, 4),
                labels[..., 1:].reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )
            return SimpleNamespace(logits=logits, loss=loss)

    model = CausalModel().eval()
    values = torch.ones(2, 4, 128)
    labels = torch.tensor([[0, 1, -100, 2], [0, 3, 1, -100]])
    mask = torch.ones(2, 4)
    one = capture_kronecker_module(
        model,
        [{"values": values.clone(), "labels": labels.clone(), "attention_mask": mask.clone()}],
        module="proj",
        writer=_writer(tmp_path / "one", "guided-fisher", "mean-valid-causal-labels"),
        curvature="guided-fisher",
        guided_loss_reduction="mean-valid-causal-labels",
    )
    two = capture_kronecker_module(
        model,
        [
            {
                "values": values[:1].clone(),
                "labels": labels[:1].clone(),
                "attention_mask": mask[:1].clone(),
            },
            {
                "values": values[1:].clone(),
                "labels": labels[1:].clone(),
                "attention_mask": mask[1:].clone(),
            },
        ],
        module="proj",
        writer=_writer(tmp_path / "two", "guided-fisher", "mean-valid-causal-labels"),
        curvature="guided-fisher",
        guided_loss_reduction="mean-valid-causal-labels",
    )
    assert one.selected_samples == 4
    assert two.selected_samples == 4
    assert one.record.record_digest == two.record.record_digest

    left_padded = capture_kronecker_module(
        model,
        [
            {
                "values": torch.ones(1, 4, 128),
                "labels": torch.tensor([[-100, -100, 1, 2]]),
                "attention_mask": torch.tensor([[0, 0, 1, 1]]),
            }
        ],
        module="proj",
        writer=_writer(
            tmp_path / "left-padded", "guided-fisher", "mean-valid-causal-labels"
        ),
        curvature="guided-fisher",
        guided_loss_reduction="mean-valid-causal-labels",
    )
    assert left_padded.selected_samples == 2

    invalid_mask_writer = _writer(
        tmp_path / "invalid-mask", "guided-fisher", "mean-valid-causal-labels"
    )
    with pytest.raises(ValueError, match="boolean or 0/1"):
        capture_kronecker_module(
            model,
            [
                {
                    "values": torch.ones(1, 4, 128),
                    "labels": torch.tensor([[-100, -100, 1, 2]]),
                    "attention_mask": torch.tensor([[0, 2, 1, 1]]),
                }
            ],
            module="proj",
            writer=invalid_mask_writer,
            curvature="guided-fisher",
            guided_loss_reduction="mean-valid-causal-labels",
        )
    assert not invalid_mask_writer.active

    bounded_writer = _writer(
        tmp_path / "bounded-causal", "guided-fisher", "mean-valid-causal-labels"
    )
    with pytest.raises(ValueError, match="capture snapshots require 2084 bytes"):
        capture_kronecker_module(
            model,
            [
                {
                    "values": torch.ones(1, 4, 128),
                    "labels": torch.tensor([[-100, -100, 1, 2]]),
                    "attention_mask": torch.ones(1, 4, dtype=torch.int64),
                }
            ],
            module="proj",
            writer=bounded_writer,
            curvature="guided-fisher",
            guided_loss_reduction="mean-valid-causal-labels",
            max_capture_bytes=2083,
        )
    assert not bounded_writer.active


def test_objective_identity_is_bound_into_cache_provenance(tmp_path):
    writer = _writer(tmp_path, "input-hessian")
    expected = bind_kronecker_activation_cache_digest(
        "02" * 32, "tritium.input-gram@1"
    )
    assert writer.activation_cache_digest == expected
    assert expected != "02" * 32


def test_capture_retains_completed_state_for_publication_retry(tmp_path):
    model = _TinyObjectiveModel()
    writer = _writer(tmp_path, "input-hessian")
    root = tmp_path / "input-hessian"
    away = tmp_path / "evidence-away"

    def interrupted_publication():
        yield _batches()[0]
        root.rename(away)

    with pytest.raises(KroneckerPublicationError):
        capture_kronecker_module(
            model,
            interrupted_publication(),
            module="proj",
            writer=writer,
            curvature="input-hessian",
        )
    assert writer.active
    away.rename(root)
    assert writer.finish().tensor_index == 0


def test_forward_kl_is_independent_of_batch_partition(tmp_path):
    del tmp_path
    logits = torch.arange(4 * 7, dtype=torch.float32).view(1, 4, 7) / 10
    one = ptq._forward_kl_factors(
        logits,
        0,
        torch.ones(1, 4),
        1024 * 1024,
    )
    first = ptq._forward_kl_factors(
        logits[:, :2],
        0,
        torch.ones(1, 2),
        1024 * 1024,
    )
    second = ptq._forward_kl_factors(
        logits[:, 2:],
        2,
        torch.ones(1, 2),
        1024 * 1024,
    )
    assert torch.equal(one, torch.cat((first, second), dim=1))


def test_forward_kl_preflights_objective_memory(tmp_path):
    writer = _writer(tmp_path, "forward-kl-kronecker")
    with pytest.raises(ValueError, match="forward-KL factors require"):
        capture_kronecker_module(
            _TinyObjectiveModel(),
            _batches()[:1],
            module="proj",
            writer=writer,
            curvature="forward-kl-kronecker",
            max_objective_bytes=64,
        )
    assert not writer.active


def test_capture_snapshots_mask_before_model_mutation(tmp_path):
    class MutatesMask(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(128, 4, bias=False)

        def forward(self, values, attention_mask):
            output = self.proj(values)
            attention_mask.zero_()
            return SimpleNamespace(logits=output, loss=output.square().mean())

    receipt = capture_kronecker_module(
        MutatesMask(),
        [{"values": torch.ones(1, 2, 128), "attention_mask": torch.ones(1, 2)}],
        module="proj",
        writer=_writer(tmp_path, "input-hessian"),
        curvature="input-hessian",
    )
    assert receipt.selected_samples == 2
