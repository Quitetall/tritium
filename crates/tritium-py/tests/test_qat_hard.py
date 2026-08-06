"""Distinct hard QAT conversion gates from ADR 0033 and plan 0048."""

import pytest

torch = pytest.importorskip("torch")

from tritium.nn import (  # noqa: E402
    AdditiveTernaryConv1d,
    AdditiveTernaryConv2d,
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


class _ConvolutionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = torch.nn.Conv1d(
            4,
            6,
            kernel_size=3,
            padding=1,
            groups=2,
            padding_mode="reflect",
        )
        self.spatial = torch.nn.Conv2d(
            4,
            6,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=2,
            bias=False,
        )

    def forward(self, temporal, spatial):
        return self.temporal(temporal), self.spatial(spatial)


class _TiedConvolutionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv1d(2, 3, kernel_size=3, padding=1)
        self.second = torch.nn.Conv1d(2, 3, kernel_size=3, padding=1)
        self.second.weight = self.first.weight
        self.second.bias = self.first.bias

    def forward(self, inputs):
        return self.first(inputs) + self.second(inputs)


class _CrossBoundaryTiedBiasModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(2, 3, kernel_size=3, padding=1)
        self.norm = torch.nn.LayerNorm(3)
        self.norm.bias = self.conv.bias

    def forward(self, inputs):
        hidden = self.conv(inputs).transpose(1, 2)
        return self.norm(hidden).transpose(1, 2)


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


def test_convert_qat_hard_lowers_convolutions_without_dense_masters():
    torch.manual_seed(89)
    source = _ConvolutionModel().eval()
    prepared = prepare(
        source,
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Conv1d", "Conv2d"),
            planes=2,
        ),
        inplace=False,
    )
    temporal = torch.randn(2, 4, 11)
    spatial = torch.randn(2, 4, 7, 5)
    expected = prepared.model(temporal, spatial)

    result = convert(prepared)

    assert isinstance(result.model.temporal, AdditiveTernaryConv1d)
    assert isinstance(result.model.spatial, AdditiveTernaryConv2d)
    assert [weight.consumer_kinds for weight in result.weights] == [
        ("conv1d",),
        ("conv2d",),
    ]
    assert [weight.shape for weight in result.weights] == [(6, 6), (6, 6)]
    assert not any(name.endswith(".weight") for name in result.model.state_dict())
    actual = result.model(temporal, spatial)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_convert_qat_hard_preserves_root_convolution_string_padding():
    torch.manual_seed(101)
    prepared = prepare(
        torch.nn.Conv1d(2, 3, kernel_size=3, padding="same", bias=False).eval(),
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Conv1d",),
        ),
        inplace=True,
    )
    sample = torch.randn(2, 2, 13)
    expected = prepared.model(sample)

    hard = convert(prepared)

    assert isinstance(hard.model, AdditiveTernaryConv1d)
    assert hard.model.padding == "same"
    torch.testing.assert_close(hard.model(sample), expected, rtol=0, atol=0)


def test_convert_qat_hard_preserves_distinct_consumers_tied_bias():
    torch.manual_seed(107)
    prepared = prepare(
        _TiedConvolutionModel().eval(),
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Conv1d",),
        ),
        inplace=True,
    )
    sample = torch.randn(2, 2, 9)
    expected = prepared.model(sample)

    hard = convert(prepared)

    assert hard.model.first is not hard.model.second
    assert hard.model.first.packed_weight is hard.model.second.packed_weight
    assert hard.model.first.bias is hard.model.second.bias
    torch.testing.assert_close(hard.model(sample), expected, rtol=0, atol=0)


def test_convert_qat_hard_preserves_target_to_preserved_bias_tie():
    torch.manual_seed(113)
    prepared = prepare(
        _CrossBoundaryTiedBiasModel().eval(),
        TernaryConfig.qat(
            estimator="salt-ste",
            target_modules=("Conv1d",),
        ),
        inplace=True,
    )
    sample = torch.randn(2, 2, 9)
    expected = prepared.model(sample)

    hard = convert(prepared)

    assert hard.model.conv.bias is hard.model.norm.bias
    torch.testing.assert_close(hard.model(sample), expected, rtol=0, atol=0)
    hard.model.to(dtype=torch.float64)
    assert hard.model.conv.bias is hard.model.norm.bias
