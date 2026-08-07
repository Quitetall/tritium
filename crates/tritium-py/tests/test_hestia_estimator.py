"""HESTIA soft-expectation oracle gates (ADR 0035 / plan 0054 WS-C4)."""

import pytest

torch = pytest.importorskip("torch")

from tritium.torch import (  # noqa: E402
    HestiaEstimator,
    ProjectionContext,
    TernaryConfig,
    TritiumError,
    collect_diagnostics,
    convert_qat_hard,
    create_estimator,
    hestia_soft_expectation,
    prepare,
    prepare_qat,
    registered_estimators,
    validate_projection,
)


def _reference_hestia(master, tau):
    scales = (
        master.detach()
        .to(torch.float32)
        .abs()
        .mean(dim=1, keepdim=True)
        .to(master.dtype)
    )
    safe = scales.clamp_min(torch.finfo(master.dtype).tiny)
    grid = master.new_tensor((-1.0, 0.0, 1.0))
    logits = -((master / safe).unsqueeze(-1) - grid).square() / tau
    return scales * (torch.softmax(logits, dim=-1) * grid).sum(dim=-1)


def test_hestia_is_registered_and_matches_soft_expectation_oracle():
    assert "hestia" in registered_estimators()
    estimator = create_estimator("hestia")
    assert isinstance(estimator, HestiaEstimator)

    master = torch.tensor(
        [[0.2, -0.35, 0.75, -0.9], [1.4, -1.8, 0.05, 0.62]],
        dtype=torch.float32,
        requires_grad=True,
    )
    projection = estimator.project(
        master,
        context=ProjectionContext(step=0, training=True, role="weight"),
    )
    validate_projection(
        projection,
        master,
        algorithm_id=estimator.algorithm_id,
        schema_version=estimator.schema_version,
    )

    assert projection.exportable is False
    assert torch.allclose(
        projection.dense, _reference_hestia(master, 1.0), atol=0, rtol=0
    )
    projection.dense.square().mean().backward()
    assert master.grad is not None
    assert torch.isfinite(master.grad).all()
    assert torch.count_nonzero(master.grad) > 0


def test_hestia_floor_switches_to_canonical_hard_forward():
    estimator = HestiaEstimator(
        initial_temperature=1.0,
        temperature_floor=0.01,
        total_steps=10,
    )
    master = torch.tensor([[0.2, -0.8, 1.4, -1.8]], requires_grad=True)
    projection = estimator.project(
        master,
        context=ProjectionContext(step=10, training=True, role="weight"),
    )
    validate_projection(projection, master)

    decoded = projection.trits.to(master.dtype) * projection.scales.to(master.dtype)
    assert projection.exportable is True
    assert torch.equal(projection.dense.detach(), decoded)
    projection.dense.sum().backward()
    assert master.grad is not None and torch.isfinite(master.grad).all()


def test_hestia_oracle_differentiates_weight_and_temperature():
    master = torch.tensor([[0.3, -0.8, 1.2]], requires_grad=True)
    scales = master.detach().abs().mean(dim=1, keepdim=True)
    temperature = torch.tensor(0.7, requires_grad=True)
    output = hestia_soft_expectation(master, scales, temperature)
    output.square().mean().backward()

    assert master.grad is not None and torch.isfinite(master.grad).all()
    assert torch.count_nonzero(master.grad) > 0
    assert temperature.grad is not None and torch.isfinite(temperature.grad)
    assert temperature.grad != 0


def test_hestia_soft_phase_fails_closed_at_hard_export():
    prepared = prepare(
        torch.nn.Linear(4, 2, bias=False),
        TernaryConfig.qat(estimator="hestia"),
        inplace=False,
    )
    with pytest.raises(TritiumError) as caught:
        convert_qat_hard(prepared)
    assert caught.value.code == "invalid_phase"
    assert "temperature floor" in str(caught.value)


def test_hestia_model_progresses_to_floor_and_hard_export_succeeds():
    prepared = prepare(
        torch.nn.Linear(4, 2, bias=False),
        TernaryConfig.qat(estimator="hestia"),
        inplace=False,
    )
    estimator = prepared.model.estimator
    sample = torch.randn(3, 4)
    soft = prepared.model(sample)
    assert estimator.schedule_step == 0

    estimator.set_step(estimator.total_steps)
    hard = prepared.model(sample)
    projection = estimator.project(
        prepared.model.weight,
        context=ProjectionContext(
            step=estimator.schedule_step,
            training=True,
            role="weight",
        ),
    )
    decoded = projection.trits.to(prepared.model.weight.dtype) * projection.scales.to(
        prepared.model.weight.dtype
    )
    assert projection.exportable is True
    assert torch.equal(projection.dense.detach(), decoded)
    assert not torch.equal(soft, hard)

    result = convert_qat_hard(prepared)
    assert result.mode == "qat-hard"
    assert result.weights[0].algorithm_id == "tritium.hestia"


def test_hestia_schedule_step_round_trips_in_state_dict():
    source = HestiaEstimator(total_steps=10).set_step(7)
    restored = HestiaEstimator(total_steps=10)
    restored.load_state_dict(source.state_dict())
    assert restored.schedule_step == 7
    assert restored.temperature(restored.schedule_step) == source.temperature(7)


def test_additive_hestia_propagates_soft_state_and_exports_at_floor():
    prepared = prepare(
        torch.nn.Linear(4, 2, bias=False),
        TernaryConfig.qat(estimator="hestia", planes=2),
        inplace=False,
    )
    estimator = prepared.model.estimator
    projection = estimator.project(
        prepared.model.weight,
        context=ProjectionContext(training=True, role="weight"),
    )
    validate_projection(projection, prepared.model.weight)
    assert projection.exportable is False
    prepared.model(torch.randn(3, 4)).square().mean().backward()

    estimator.set_step(estimator.estimators[0].total_steps)
    floor_projection = estimator.project(
        prepared.model.weight,
        context=ProjectionContext(
            step=estimator.projection_step,
            training=True,
            role="weight",
        ),
    )
    validate_projection(floor_projection, prepared.model.weight)
    assert floor_projection.exportable is True
    result = convert_qat_hard(prepared)
    assert result.weights[0].planes == 2


@pytest.mark.parametrize(
    ("scales", "temperature", "message"),
    [
        (torch.tensor([[float("nan")]]), torch.tensor(0.7), "scales"),
        (torch.tensor([[-1.0]]), torch.tensor(0.7), "scales"),
        (
            torch.tensor([[1.0]], dtype=torch.float32),
            torch.tensor(1e-300, dtype=torch.float64),
            "representable",
        ),
        (
            torch.tensor([[1.0]], dtype=torch.float32),
            torch.tensor(1e-45, dtype=torch.float32),
            "representable",
        ),
    ],
)
def test_hestia_oracle_rejects_invalid_or_unrepresentable_inputs(
    scales, temperature, message
):
    with pytest.raises(ValueError, match=message):
        hestia_soft_expectation(torch.ones(1, 2), scales, temperature)


@pytest.mark.parametrize(
    ("master", "scales", "temperature", "expected"),
    [
        (
            torch.tensor([[1.0, -2.0]]),
            torch.tensor([[0.0]]),
            torch.tensor(0.7),
            torch.zeros(1, 2),
        ),
        (
            torch.tensor([[1e20, -1e20]]),
            torch.tensor([[1.0]]),
            torch.tensor(0.7),
            torch.tensor([[1.0, -1.0]]),
        ),
    ],
)
def test_hestia_oracle_stays_finite_at_extreme_valid_inputs(
    master, scales, temperature, expected
):
    output = hestia_soft_expectation(master, scales, temperature)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    ("master", "scales", "tau"),
    [
        (torch.tensor([[1.0, -2.0]]), torch.tensor([[0.0]]), 0.7),
        (torch.tensor([[1e20, -1e20]]), torch.tensor([[1.0]]), 0.7),
        (torch.tensor([[1e20, -1e20]]), torch.tensor([[1.0]]), 1.01),
        (torch.tensor([[1e20, -1e20]]), torch.tensor([[1.0]]), 10.0),
        (torch.tensor([[1e20, -1e20]]), torch.tensor([[1.0]]), 1e20),
    ],
)
def test_hestia_extreme_fallback_backward_remains_finite(master, scales, tau):
    master = master.requires_grad_()
    temperature = torch.tensor(tau, requires_grad=True)
    hestia_soft_expectation(master, scales, temperature).sum().backward()
    assert master.grad is not None and torch.isfinite(master.grad).all()
    assert temperature.grad is not None and torch.isfinite(temperature.grad)


def test_hestia_f32_max_temperature_matches_f64_reference_and_backpropagates():
    tau_value = torch.finfo(torch.float32).max
    master = torch.tensor([[0.5 * tau_value]], requires_grad=True)
    scales = torch.ones(1, 1)
    temperature = torch.tensor(tau_value, requires_grad=True)
    output = hestia_soft_expectation(master, scales, temperature)
    # z=tau/2 makes squared-distance differences relative to q=+1 exactly
    # [2*tau, tau, 0] at this scale. Avoid absolute squared distances even in
    # f64: their common z^2 term cannot retain a one-tau difference.
    reference_logits = torch.tensor((-2.0, -1.0, 0.0), dtype=torch.float64)
    reference_grid = torch.tensor((-1.0, 0.0, 1.0), dtype=torch.float64)
    reference = (torch.softmax(reference_logits, dim=0) * reference_grid).sum().float()
    assert torch.allclose(output, reference, atol=1e-6, rtol=0)

    output.sum().backward()
    assert master.grad is not None and torch.isfinite(master.grad).all()
    assert temperature.grad is not None and torch.isfinite(temperature.grad)


def test_explicit_projection_step_zero_overrides_stored_product_step():
    estimator = HestiaEstimator(total_steps=10).set_step(10)
    master = torch.tensor([[0.2, -0.8]])
    direct = estimator.project(
        master,
        context=ProjectionContext(step=0, training=True, role="weight"),
    )
    assert direct.exportable is False
    assert estimator.projection_step == 10


def test_hestia_is_admitted_by_builtin_diagnostics():
    model = prepare_qat(
        torch.nn.Linear(4, 2, bias=False),
        TernaryConfig.qat(estimator="hestia"),
    )
    diagnostics = collect_diagnostics(model)
    assert diagnostics.tensors[0].estimator_id == "tritium.hestia"


def test_hestia_qat_compiles_as_one_full_graph():
    model = prepare_qat(
        torch.nn.Linear(4, 2, bias=False),
        TernaryConfig.qat(estimator="hestia"),
    )
    sample = torch.randn(3, 4)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    assert torch.equal(compiled(sample), model(sample))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_temperature": 0.0},
        {"temperature_floor": 0.0},
        {"initial_temperature": 0.01, "temperature_floor": 1.0},
        {"total_steps": 0},
    ],
)
def test_hestia_rejects_invalid_temperature_schedule(kwargs):
    with pytest.raises(ValueError, match="HESTIA temperature schedule"):
        HestiaEstimator(**kwargs)
