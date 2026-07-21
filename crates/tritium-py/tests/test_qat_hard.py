"""Distinct hard QAT conversion gates from ADR 0033 and plan 0048."""

import pytest

torch = pytest.importorskip("torch")

from tritium.nn import (  # noqa: E402
    AdditiveTernaryEmbedding,
    AdditiveTernaryLinear,
    TernaryEmbedding,
)
from tritium.torch import TernaryConfig, TritiumError, convert, prepare  # noqa: E402
from tritium.torch.qat import QatHardResult  # noqa: E402


class _TiedLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens):
        return self.head(self.embed(tokens))


def test_convert_qat_freezes_tied_masters_without_ptq_relabeling():
    torch.manual_seed(73)
    source = _TiedLanguageModel()
    original = source.embed.weight.detach().clone()
    prepared = prepare(
        source,
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Linear", "Embedding"),
            planes=2,
        ),
        inplace=False,
    )
    optimizer = torch.optim.AdamW(prepared.model.parameters(), lr=1e-3)
    tokens = torch.tensor([[1, 2, 3, 4]])
    prepared.model(tokens).square().mean().backward()
    optimizer.step()
    prepared.model.eval()
    expected = prepared.model(tokens).detach()

    result = convert(prepared)
    assert isinstance(result, QatHardResult)
    assert result.mode == "qat-hard"
    assert result.config.mode == "qat"
    assert result.source_checkpoint_digest.startswith("sha256:")
    assert result.hard_state_digest.startswith("sha256:")
    assert result.artifact_id.startswith("sha256:")
    assert len(result.weights) == 1
    assert result.weights[0].aliases == ("embed.weight", "head.weight")
    assert isinstance(result.model.embed, AdditiveTernaryEmbedding)
    assert isinstance(result.model.head, AdditiveTernaryLinear)
    assert result.model.embed.packed_weight is result.model.head.packed_weight
    assert not any("estimator" in name for name in result.model.state_dict())
    assert not any(name.endswith(".weight") for name in result.model.state_dict())
    torch.testing.assert_close(result.model(tokens), expected, rtol=0, atol=0)
    assert torch.equal(source.embed.weight, original)

    with pytest.raises(TritiumError) as captured:
        convert(prepared)
    assert captured.value.code == "coverage_mismatch"


def test_convert_qat_rejects_calibration_and_preflights_embedding_options():
    prepared = prepare(
        torch.nn.Embedding(8, 4, max_norm=1.0),
        TernaryConfig.qat(target_modules=("Embedding",)),
        inplace=True,
    )
    with pytest.raises(TypeError, match="does not accept calibration"):
        convert(prepared, object())
    with pytest.raises(TritiumError) as captured:
        convert(prepared)
    assert captured.value.code == "unsupported_module_option"
    assert isinstance(prepared.model, TernaryEmbedding)


def test_ttq_uses_two_exportable_sign_planes():
    torch.manual_seed(79)
    prepared = prepare(
        torch.nn.Linear(8, 4),
        TernaryConfig.qat(estimator="ttq", planes=2),
        inplace=True,
    )
    sample = torch.randn(3, 8)
    prepared.model.eval()
    expected = prepared.model(sample).detach()
    result = convert(prepared)
    assert result.weights[0].algorithm_id == "tritium.ttq"
    assert result.weights[0].planes == 2
    torch.testing.assert_close(result.model(sample), expected, rtol=0, atol=0)

    with pytest.raises(TritiumError) as captured:
        prepare(
            torch.nn.Linear(8, 4),
            TernaryConfig.qat(estimator="ttq", planes=1),
            inplace=True,
        )
    assert captured.value.code == "unsupported_recipe"


def test_qat_hard_hashes_and_replays_bfloat16_state():
    prepared = prepare(
        torch.nn.Linear(8, 4, dtype=torch.bfloat16),
        TernaryConfig.qat(estimator="salt-ste"),
        inplace=True,
    )
    sample = torch.randn(2, 8, dtype=torch.bfloat16)
    prepared.model.eval()
    expected = prepared.model(sample).detach()
    result = convert(prepared)
    assert result.source_checkpoint_digest.startswith("sha256:")
    torch.testing.assert_close(result.model(sample), expected, rtol=0, atol=0)
