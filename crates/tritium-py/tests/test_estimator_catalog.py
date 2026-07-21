"""Differentiable estimator catalog gates for plan 0048."""

import pytest

torch = pytest.importorskip("torch")

from tritium.nn import TernaryLinear  # noqa: E402
from tritium.torch import (  # noqa: E402
    AdditiveEstimator,
    Estimator,
    ProjectionContext,
    TernaryConfig,
    TernaryPlane,
    TernaryProjection,
    TritiumError,
    create_estimator,
    inspect,
    prepare_qat,
    register_estimator,
    registered_estimators,
    validate_projection,
)


@pytest.mark.parametrize("planes", [2, 3])
def test_additive_qat_planes_have_exact_hard_decode_and_finite_backward(planes):
    torch.manual_seed(97 + planes)
    config = TernaryConfig.qat(estimator="salt-ste", planes=planes)
    model = prepare_qat(torch.nn.Sequential(torch.nn.Linear(8, 4)), config)
    layer = model[0]
    assert isinstance(layer.estimator, AdditiveEstimator)

    projection = layer.estimator.project(
        layer.weight,
        context=ProjectionContext(step=0, training=True, role="weight"),
    )
    validate_projection(
        projection,
        layer.weight,
        algorithm_id=layer.estimator.algorithm_id,
        schema_version=layer.estimator.schema_version,
    )
    assert len(projection.planes) == planes
    decoded = sum(
        (
            plane.trits.to(layer.weight.dtype) * plane.scales
            for plane in projection.planes
        ),
        torch.zeros_like(layer.weight),
    )
    assert torch.equal(projection.dense.detach(), decoded)

    sample = torch.randn(3, 8, requires_grad=True)
    model(sample).square().mean().backward()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert sample.grad is not None and torch.isfinite(sample.grad).all()


def test_additive_qat_state_dict_round_trip_is_exact():
    torch.manual_seed(103)
    config = TernaryConfig.qat(estimator="lsq", planes=3)
    source = prepare_qat(torch.nn.Sequential(torch.nn.Linear(8, 4)), config)
    sample = torch.randn(2, 8)
    source(sample).square().mean().backward()
    torch.optim.AdamW(source.parameters(), lr=1e-3).step()
    expected = source(sample).detach()

    restored = prepare_qat(torch.nn.Sequential(torch.nn.Linear(8, 4)), config)
    restored.load_state_dict(source.state_dict())
    assert torch.equal(restored(sample), expected)


def test_additive_qat_compiles_as_one_full_graph():
    torch.manual_seed(107)
    model = prepare_qat(
        torch.nn.Linear(8, 4),
        TernaryConfig.qat(estimator="salt-ste", planes=3),
    )
    sample = torch.randn(2, 8)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    assert torch.equal(compiled(sample), model(sample))


@pytest.mark.parametrize(
    "name",
    ["absmean-ste", "salt-ste", "annealed-ste", "lsq", "twn", "ttq", "sparse-ternary"],
)
def test_builtin_estimator_hard_decode_and_finite_backward(name):
    torch.manual_seed(101)
    estimator = create_estimator(name)
    master = torch.randn(4, 8, requires_grad=True)
    projection = estimator.project(
        master, context=ProjectionContext(step=17, training=True, role="weight")
    )
    validate_projection(
        projection,
        master,
        algorithm_id=estimator.algorithm_id,
        schema_version=estimator.schema_version,
    )

    assert all(
        set(plane.trits.unique().tolist()) <= {-1, 0, 1}
        for plane in projection.planes
    )
    decoded = sum(
        (
            plane.trits.to(master.dtype) * plane.scales.to(master.dtype)
            for plane in projection.planes
        ),
        torch.zeros_like(master),
    )
    assert torch.equal(projection.dense.detach(), decoded)
    projection.dense.square().mean().backward()
    assert master.grad is not None and torch.isfinite(master.grad).all()
    for parameter in estimator.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_config_catalog_conversion_accounts_for_learned_state():
    model = prepare_qat(
        torch.nn.Sequential(torch.nn.Linear(8, 4)),
        TernaryConfig.qat(estimator="ttq", planes=2),
    )
    layer = model[0]
    assert isinstance(layer, TernaryLinear)
    assert any(True for _ in layer.estimator.parameters())
    report = inspect(model)
    estimator_entries = [entry for entry in report.entries if entry.reason == "estimator_state"]
    assert estimator_entries
    assert report.total_numel == sum(parameter.numel() for parameter in model.parameters())


def test_tied_latent_weight_shares_learned_estimator_state():
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Linear(4, 4, bias=False)
            self.right = torch.nn.Linear(4, 4, bias=False)
            self.right.weight = self.left.weight

    model = prepare_qat(Tied(), TernaryConfig.qat(estimator="lsq"))
    assert model.left.weight is model.right.weight
    assert model.left.estimator is model.right.estimator
    estimator_entry = next(
        entry for entry in inspect(model).entries if entry.reason == "estimator_state"
    )
    assert len(estimator_entry.aliases) == 2


def test_learned_estimator_safetensors_state_round_trip(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    torch.manual_seed(131)
    config = TernaryConfig.qat(estimator="ttq", planes=2)
    model = prepare_qat(torch.nn.Sequential(torch.nn.Linear(8, 4)), config)
    sample = torch.randn(3, 8)
    model(sample).square().mean().backward()
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    expected = model(sample).detach()

    path = tmp_path / "ttq.safetensors"
    state = model.state_dict()
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    safetensors.save_file(state, path)

    restored = prepare_qat(torch.nn.Sequential(torch.nn.Linear(8, 4)), config)
    restored.load_state_dict(safetensors.load_file(path))
    assert torch.equal(restored(sample), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("name", ["lsq", "ttq"])
def test_learned_estimator_conversion_is_device_resident_on_cuda(name):
    model = prepare_qat(
        torch.nn.Linear(8, 4, device="cuda"),
        TernaryConfig.qat(estimator=name, planes=2 if name == "ttq" else 1),
    )
    sample = torch.randn(3, 8, device="cuda")
    model(sample).square().mean().backward()
    assert all(parameter.is_cuda for parameter in model.parameters())
    assert all(parameter.grad is not None for parameter in model.estimator.parameters())


def test_external_estimator_factory_registers_without_source_edits():
    class Zeros(Estimator):
        algorithm_id = "test.zeros"
        schema_version = 1

        def project(self, master, *, context):
            del context
            trits = torch.zeros_like(master, dtype=torch.int8)
            scales = torch.zeros((master.shape[0], 1), dtype=master.dtype, device=master.device)
            return TernaryProjection(
                dense=master * 0,
                planes=(TernaryPlane(trits, scales, master.shape[1]),),
                algorithm_id=self.algorithm_id,
                schema_version=self.schema_version,
            )

    register_estimator("test-zeros", Zeros)
    assert "test-zeros" in registered_estimators()
    assert isinstance(create_estimator("test-zeros"), Zeros)
    with pytest.raises(TritiumError) as caught:
        register_estimator("test-zeros", Zeros)
    assert caught.value.code == "estimator_registry"
