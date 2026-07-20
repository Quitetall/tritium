"""Public PyTorch research API gates for ADR 0033 / plan 0045."""

import pytest

torch = pytest.importorskip("torch")

from tritium.nn import TernaryConv1d, TernaryConv2d, TernaryLinear  # noqa: E402
from tritium.torch import (  # noqa: E402
    CoverageReport,
    Estimator,
    TernaryConfig,
    TernaryProjection,
    TritiumError,
    inspect,
    prepare_qat,
)


def test_ternary_linear_uses_hard_absmean_projection_and_masked_ste():
    layer = TernaryLinear(4, 2, bias=True)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [-2.0, -0.4, 0.3, 2.1],
                    [0.0, 1.0, -1.0, 0.2],
                ]
            )
        )
        layer.bias.copy_(torch.tensor([0.1, -0.2]))

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    output = layer(x)

    # Row AbsMean scales are 1.2 and 0.55. Hard trits are [-1,0,0,1]
    # and [0,1,-1,0], so this expectation is independent of implementation.
    assert torch.allclose(output, torch.tensor([[3.7, -0.75]]), atol=1e-6)

    output.sum().backward()
    assert torch.allclose(
        layer.weight.grad,
        torch.tensor([[0.0, 2.0, 3.0, 0.0], [1.0, 0.0, 0.0, 4.0]]),
        atol=1e-6,
    )
    assert torch.equal(layer.bias.grad, torch.ones(2))
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_prepare_qat_preserves_tied_weights_and_accounts_for_every_parameter():
    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Linear(4, 3, bias=False)
            self.right = torch.nn.Linear(4, 3, bias=False)
            self.right.weight = self.left.weight
            self.norm = torch.nn.LayerNorm(3)
            self.head = torch.nn.Linear(3, 2)

    model = TiedModel()
    shared_weight = model.left.weight
    converted = prepare_qat(
        model,
        TernaryConfig.qat(estimator="salt-ste", target_modules=("Linear",), planes=1),
    )

    assert converted is model
    assert isinstance(model.left, TernaryLinear)
    assert isinstance(model.right, TernaryLinear)
    assert isinstance(model.head, TernaryLinear)
    assert isinstance(model.norm, torch.nn.LayerNorm)
    assert model.left.weight is shared_weight
    assert model.right.weight is shared_weight

    coverage = inspect(model)
    assert coverage.total_parameters == 5
    assert coverage.converted_parameters == 2
    assert coverage.preserved_parameters == 3
    assert coverage.total_numel == 26
    tied = next(entry for entry in coverage.entries if entry.path == "left.weight")
    assert tied.aliases == ("left.weight", "right.weight")
    assert tied.disposition == "converted"


def test_ternary_config_round_trips_without_a_nominal_bits_control():
    config = TernaryConfig.qat(
        estimator="salt-ste", target_modules=("Linear", "Embedding"), planes=2
    )
    encoded = config.to_dict()

    assert "bits" not in encoded
    assert encoded["planes"] == 2
    assert TernaryConfig.from_dict(encoded) == config

    ptq = TernaryConfig.ptq(profile="compact-v1", target_bpw=2.25)
    assert ptq.target_bpw == 2.25
    assert TernaryConfig.from_dict(ptq.to_dict()) == ptq


def test_root_linear_conversion_and_state_dict_round_trip():
    torch.manual_seed(7)
    source = torch.nn.Linear(4, 3)
    source_weight = source.weight
    config = TernaryConfig.qat()
    converted = prepare_qat(source, config)

    assert isinstance(converted, TernaryLinear)
    assert converted.weight is source_weight
    assert inspect(converted).converted_parameters == 1

    restored = prepare_qat(torch.nn.Linear(4, 3), config)
    restored.load_state_dict(converted.state_dict())
    sample = torch.randn(2, 4)
    assert torch.equal(converted(sample), restored(sample))


def test_failed_inspection_leaves_module_graph_unchanged():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU())
    original_linear = model[0]

    with pytest.raises(TritiumError) as caught:
        prepare_qat(model, TernaryConfig.qat(target_modules=("Conv3d",)))

    assert caught.value.code == "unsupported_module"
    assert model[0] is original_linear
    assert isinstance(model[0], torch.nn.Linear)


def test_custom_estimator_contract_fails_closed():
    class BadEstimator(Estimator):
        algorithm_id = "example.bad"
        schema_version = 1

        def project(self, master, *, context):
            del context
            return TernaryProjection(
                dense=master,
                trits=torch.full_like(master, 2, dtype=torch.int8),
                scales=torch.ones(master.shape[0], 1),
                group_size=master.shape[1],
                algorithm_id=self.algorithm_id,
                schema_version=self.schema_version,
            )

    layer = TernaryLinear(3, 2, estimator=BadEstimator())
    with pytest.raises(TritiumError) as caught:
        layer(torch.randn(1, 3))
    assert caught.value.code == "estimator_contract"


def test_coverage_report_json_shape_round_trips():
    model = prepare_qat(torch.nn.Sequential(torch.nn.Linear(3, 2)), TernaryConfig.qat())
    report = inspect(model)
    assert CoverageReport.from_dict(report.to_dict()) == report


def test_prepare_qat_preserves_shared_module_identity():
    class SharedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            shared = torch.nn.Linear(3, 2)
            self.left = shared
            self.right = shared

    model = prepare_qat(SharedModel(), TernaryConfig.qat())
    assert isinstance(model.left, TernaryLinear)
    assert model.left is model.right
    report = inspect(model)
    assert report.converted_parameters == 1
    weight = next(entry for entry in report.entries if entry.disposition == "converted")
    assert weight.aliases == ("left.weight", "right.weight")


def test_estimator_cannot_mislabel_its_projection_identity():
    class MislabelledEstimator(Estimator):
        algorithm_id = "example.expected"
        schema_version = 1

        def project(self, master, *, context):
            del context
            scales = torch.ones(master.shape[0], 1, device=master.device, dtype=master.dtype)
            trits = torch.zeros_like(master, dtype=torch.int8)
            return TernaryProjection(
                dense=trits.to(master.dtype) * scales,
                trits=trits,
                scales=scales,
                group_size=master.shape[1],
                algorithm_id="example.lie",
                schema_version=1,
            )

    with pytest.raises(TritiumError) as caught:
        TernaryLinear(3, 2, estimator=MislabelledEstimator())(torch.randn(1, 3))
    assert caught.value.code == "estimator_contract"


@pytest.mark.parametrize(
    ("source", "sample", "expected_type", "target"),
    [
        (
            torch.nn.Conv1d(4, 6, kernel_size=3, stride=2, padding=2, dilation=2, groups=2),
            torch.randn(2, 4, 11),
            TernaryConv1d,
            "Conv1d",
        ),
        (
            torch.nn.Conv2d(
                4,
                6,
                kernel_size=(3, 2),
                stride=(2, 1),
                padding=(1, 2),
                dilation=(1, 2),
                groups=2,
                padding_mode="reflect",
            ),
            torch.randn(2, 4, 9, 10),
            TernaryConv2d,
            "Conv2d",
        ),
    ],
)
def test_convolution_conversion_preserves_options_identity_and_gradients(
    source, sample, expected_type, target
):
    weight = source.weight
    bias = source.bias
    converted = prepare_qat(
        source,
        TernaryConfig.qat(estimator="twn", target_modules=(target,)),
    )

    assert isinstance(converted, expected_type)
    assert converted.weight is weight
    assert converted.bias is bias
    assert converted.stride == source.stride
    assert converted.padding == source.padding
    assert converted.dilation == source.dilation
    assert converted.groups == source.groups
    assert converted.padding_mode == source.padding_mode

    output = converted(sample.requires_grad_())
    output.square().mean().backward()
    assert output.numel() > 0
    assert converted.weight.grad is not None
    assert sample.grad is not None
    assert inspect(converted).converted_parameters == 1


def test_convolution_state_dict_round_trip_is_exact():
    config = TernaryConfig.qat(estimator="sparse-ternary", target_modules=("Conv2d",))
    source = prepare_qat(torch.nn.Conv2d(3, 5, 3, padding="same"), config)
    restored = prepare_qat(torch.nn.Conv2d(3, 5, 3, padding="same"), config)
    restored.load_state_dict(source.state_dict())
    sample = torch.randn(2, 3, 7, 7)
    assert torch.equal(restored(sample), source(sample))
