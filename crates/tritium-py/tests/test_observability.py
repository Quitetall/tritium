"""Accountable, network-free diagnostics and adapter contracts."""

import math

import pytest

torch = pytest.importorskip("torch")

from tritium.nn import AdditiveTernaryLinear, TernaryLinear  # noqa: E402
from tritium.torch import (  # noqa: E402
    AbsMeanSTE,
    Estimator,
    OpenTelemetryDiagnostics,
    TernaryPlane,
    TernaryProjection,
    SparseTernaryEstimator,
    WandbDiagnostics,
    TernaryConfig,
    collect_diagnostics,
    log_opentelemetry,
    log_tensorboard,
    log_wandb,
    prepare_qat,
)


def _qat_model():
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Linear(4, 2, bias=False)
            self.right = torch.nn.Linear(4, 2, bias=False)
            self.right.weight = self.left.weight

        def forward(self, values):
            return self.left(values) + self.right(values)

    model = prepare_qat(Tied(), TernaryConfig.qat())
    with torch.no_grad():
        model.left.weight.copy_(
            torch.tensor([[-2.0, -0.4, 0.3, 2.1], [0.0, 1.0, -1.0, 0.2]])
        )
    model(torch.ones(1, 4)).sum().backward()
    return model


def test_latent_snapshot_deduplicates_ties_and_reports_exact_metrics():
    model = _qat_model()
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    mode_before = model.training

    snapshot = collect_diagnostics(model, step=7)

    assert snapshot.schema_version == 1
    assert snapshot.step == 7
    assert len(snapshot.tensors) == 1
    tensor = snapshot.tensors[0]
    assert tensor.path == "left.weight"
    assert tensor.aliases == ("left.weight", "right.weight")
    assert tensor.estimator_id == "tritium.salt-ste"
    assert tensor.codec_id is None
    assert tensor.shape == (2, 4)
    assert tensor.planes[0].trits.as_tuple() == (2, 4, 2)
    assert tensor.zero_rate == pytest.approx(0.5)
    assert tensor.saturation_rate == pytest.approx(0.5)
    assert tensor.reconstruction_rmse > 0
    assert tensor.gradient_l2 is not None and tensor.gradient_l2 > 0
    assert tensor.gradient_finite is True
    assert tensor.physical_bytes == 6
    assert all(torch.equal(model.state_dict()[name], value) for name, value in state_before.items())
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model.training is mode_before

    metrics = snapshot.scalar_metrics()
    assert metrics["tritium/tensors"] == 1.0
    assert metrics["tritium/physical_bytes"] == 6.0
    assert metrics["tritium/code_scale_bpw"] == 6.0
    assert metrics["tritium/tensor/left.weight/zero_rate"] == pytest.approx(0.5)


def test_hard_snapshot_reads_packed_counts_without_dense_shadow():
    plane = type(
        "Plane",
        (),
        {
            "trits": torch.tensor([[-1, 0, 1, 0], [1, -1, 0, 0]], dtype=torch.int8),
            "scales": torch.tensor([[0.5], [1.0]], dtype=torch.float16),
            "group_size": 4,
        },
    )()
    model = torch.nn.Sequential(AdditiveTernaryLinear((plane,), bias=None))

    snapshot = collect_diagnostics(model, step=0)

    assert len(snapshot.tensors) == 1
    tensor = snapshot.tensors[0]
    assert tensor.path == "0.weight"
    assert tensor.mode == "hard"
    assert tensor.estimator_id is None
    assert tensor.codec_id == "tritium.b3-additive"
    assert tensor.planes[0].trits.as_tuple() == (2, 4, 2)
    assert tensor.zero_rate == pytest.approx(0.5)
    assert tensor.reconstruction_rmse is None
    assert tensor.gradient_l2 is None
    assert tensor.physical_bytes == model[0].packed_weight.physical_bytes


class _Writer:
    def __init__(self):
        self.scalars = []
        self.histograms = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_histogram_raw(self, **kwargs):
        self.histograms.append(kwargs)


class _Run:
    def __init__(self):
        self.calls = []

    def log(self, payload, step):
        self.calls.append((payload, step))


class _Gauge:
    def __init__(self):
        self.records = []

    def set(self, value, attributes):
        self.records.append((value, attributes))


class _Meter:
    def __init__(self):
        self.gauges = {}

    def create_gauge(self, name, unit="1", description=""):
        del unit, description
        return self.gauges.setdefault(name, _Gauge())


def test_adapters_use_injected_clients_and_never_require_accounts():
    snapshot = collect_diagnostics(
        _qat_model(), step=11, extra_metrics={"runtime/decode_ms": 3.5}
    )
    assert snapshot.scalar_metrics()["tritium/runtime/decode_ms"] == 3.5

    writer = _Writer()
    log_tensorboard(snapshot, writer)
    assert ("tritium/tensors", 1.0, 11) in writer.scalars
    assert writer.histograms[0]["bucket_counts"] == [2, 4, 2]

    run = _Run()
    log_wandb(
        snapshot,
        run,
        histogram_factory=lambda *, np_histogram: {"histogram": np_histogram},
    )
    payload, step = run.calls[0]
    assert step == 11
    assert payload["tritium/tensor/left.weight/plane_0/trits"]["histogram"][0] == (
        2,
        4,
        2,
    )

    meter = _Meter()
    log_opentelemetry(snapshot, meter)
    assert meter.gauges["tritium.snapshot.tensor_count"].records[0][0] == 1

    tensor_meter = _Meter()
    tensor_adapter = OpenTelemetryDiagnostics(tensor_meter, include_tensors=True)
    tensor_adapter.log(snapshot)
    zero_record = tensor_meter.gauges["tritium.tensor.zero_rate"].records[0]
    assert zero_record[0] == pytest.approx(0.5)
    assert zero_record[1]["tensor"] == "left.weight"
    assert math.isfinite(zero_record[0])

    exact_budget_meter = _Meter()
    with pytest.raises(ValueError, match="series budget"):
        OpenTelemetryDiagnostics(
            exact_budget_meter,
            include_tensors=True,
            max_tensor_series=10,
        ).log(snapshot)
    OpenTelemetryDiagnostics(
        _Meter(), include_tensors=True, max_tensor_series=11
    ).log(snapshot)

    adapter = OpenTelemetryDiagnostics(meter)
    instrument_ids = {name: id(gauge) for name, gauge in adapter.instruments.items()}
    adapter.log(snapshot)
    adapter.log(snapshot)
    assert {name: id(gauge) for name, gauge in adapter.instruments.items()} == instrument_ids
    assert adapter.model_instruments["extra"].records[-1] == (
        3.5,
        {"metric": "runtime/decode_ms", "source": "caller"},
    )
    changed_metrics = collect_diagnostics(
        _qat_model(), step=12, extra_metrics={"teacher_kl": 0.25}
    )
    with pytest.raises(ValueError, match="names changed"):
        adapter.log(changed_metrics)

    wandb_adapter = WandbDiagnostics(
        run,
        histogram_factory=lambda *, np_histogram: {"histogram": np_histogram},
    )
    wandb_adapter.log(snapshot)
    with pytest.raises(ValueError, match="must not decrease"):
        wandb_adapter.log(collect_diagnostics(_qat_model(), step=10))


def test_diagnostics_reject_ambiguous_or_nonfinite_external_metrics():
    model = _qat_model()
    with pytest.raises(ValueError, match="collides"):
        collect_diagnostics(model, extra_metrics={"physical_bytes": 1})
    with pytest.raises(ValueError, match="finite"):
        collect_diagnostics(model, extra_metrics={"teacher_kl": float("nan")})
    with pytest.raises(ValueError, match="canonical"):
        collect_diagnostics(model, extra_metrics={"bad metric": 1})


def test_mixed_latent_and_hard_graph_reports_both_unique_owners():
    plane = type(
        "Plane",
        (),
        {
            "trits": torch.tensor([[-1, 0, 1, 0], [1, -1, 0, 0]], dtype=torch.int8),
            "scales": torch.ones(2, 1, dtype=torch.float16),
            "group_size": 4,
        },
    )()
    model = torch.nn.ModuleDict(
        {
            "latent": prepare_qat(torch.nn.Linear(4, 2), TernaryConfig.qat()),
            "hard": AdditiveTernaryLinear((plane,), bias=None),
        }
    )

    snapshot = collect_diagnostics(model)

    assert [(tensor.path, tensor.mode) for tensor in snapshot.tensors] == [
        ("hard.weight", "hard"),
        ("latent.weight", "latent"),
    ]


def test_s34_rate_uses_five_bits_per_valid_quartet_and_rejects_bad_structure():
    class S34Estimator(Estimator):
        algorithm_id = "test.s34"

        def __init__(self, valid):
            super().__init__()
            self.valid = valid

        def project(self, master, *, context):
            del context
            row = [0, 1, -1, 1] if self.valid else [1, 1, -1, 1]
            trits = torch.tensor([row] * 4, dtype=torch.int8, device=master.device)
            scales = torch.ones(4, 1, dtype=torch.float16, device=master.device)
            decoded = trits.to(master.dtype) * scales.to(master.dtype)
            return TernaryProjection(
                dense=decoded + (master - master.detach()),
                planes=(TernaryPlane(trits, scales, 4, structure="s34"),),
                algorithm_id=self.algorithm_id,
                schema_version=1,
            )

    valid = TernaryLinear(4, 4, estimator=S34Estimator(True))
    tensor = collect_diagnostics(valid, allow_external_estimators=True).tensors[0]
    assert tensor.physical_bytes == 11  # ceil(4 quartets * 5 bits / 8) + 4 f16 scales
    assert tensor.code_scale_bpw == pytest.approx(5.5)

    invalid = TernaryLinear(4, 4, estimator=S34Estimator(False))
    with pytest.raises(ValueError, match="exactly one zero"):
        collect_diagnostics(invalid, allow_external_estimators=True)


def test_s34_canonical_ragged_tail_is_accounted_and_bad_tail_is_rejected():
    class RaggedS34Estimator(Estimator):
        algorithm_id = "test.s34-ragged"

        def __init__(self, values):
            super().__init__()
            self.values = values

        def project(self, master, *, context):
            del context
            trits = torch.tensor([self.values], dtype=torch.int8, device=master.device)
            scales = torch.ones(1, 1, dtype=torch.float16, device=master.device)
            decoded = trits.to(master.dtype) * scales.to(master.dtype)
            return TernaryProjection(
                dense=decoded + (master - master.detach()),
                planes=(
                    TernaryPlane(
                        trits,
                        scales,
                        master.shape[1],
                        structure="s34",
                    ),
                ),
                algorithm_id=self.algorithm_id,
                schema_version=1,
            )

    valid = TernaryLinear(5, 1, estimator=RaggedS34Estimator([0, 1, -1, 1, 1]))
    tensor = collect_diagnostics(valid, allow_external_estimators=True).tensors[0]
    assert tensor.physical_bytes == 4  # two five-bit groups plus one f16 scale

    invalid = TernaryLinear(
        6,
        1,
        estimator=RaggedS34Estimator([0, 1, -1, 1, 0, 0]),
    )
    with pytest.raises(ValueError, match="tail"):
        collect_diagnostics(invalid, allow_external_estimators=True)


def test_collection_budget_and_external_estimator_purity_boundary_preflight():
    class MutatingEstimator(Estimator):
        algorithm_id = "test.mutating"

        def __init__(self):
            super().__init__()
            self.register_buffer("calls", torch.zeros((), dtype=torch.int64))

        def project(self, master, *, context):
            del context
            self.calls.add_(1)
            trits = master.detach().sign().to(torch.int8)
            scales = torch.ones(master.shape[0], 1, dtype=torch.float16)
            decoded = trits.to(master.dtype) * scales.to(master.dtype)
            return TernaryProjection(
                dense=decoded + (master - master.detach()),
                planes=(TernaryPlane(trits, scales, master.shape[1]),),
                algorithm_id=self.algorithm_id,
                schema_version=1,
            )

    estimator = MutatingEstimator()
    model = TernaryLinear(4, 2, estimator=estimator)
    with pytest.raises(ValueError, match="max_latent_elements"):
        collect_diagnostics(model, max_latent_elements=4)
    assert estimator.calls.item() == 0
    with pytest.raises(ValueError, match="external estimator"):
        collect_diagnostics(model)
    assert estimator.calls.item() == 0
    collect_diagnostics(model, allow_external_estimators=True)
    assert estimator.calls.item() == 1


def test_additive_saturation_is_per_plane_and_nonfinite_norm_is_omitted():
    model = prepare_qat(
        torch.nn.Linear(4, 2),
        TernaryConfig.qat(estimator="salt-ste", planes=2),
    )
    model.weight.grad = torch.full_like(model.weight, float("inf"))

    tensor = collect_diagnostics(model).tensors[0]

    assert all(plane.saturation_rate is not None for plane in tensor.planes)
    assert tensor.saturation_rate is not None
    assert tensor.gradient_l2 is None
    assert tensor.gradient_finite is False


def test_path_selection_happens_before_projection():
    model = torch.nn.ModuleDict(
        {
            "left": prepare_qat(torch.nn.Linear(4, 2), TernaryConfig.qat()),
            "right": prepare_qat(torch.nn.Linear(4, 2), TernaryConfig.qat()),
        }
    )
    snapshot = collect_diagnostics(model, paths=("right.weight",))
    assert [tensor.path for tensor in snapshot.tensors] == ["right.weight"]
    with pytest.raises(ValueError, match="unknown diagnostics paths"):
        collect_diagnostics(model, paths=("missing.weight",))


def test_tied_latent_consumers_with_different_estimators_fail_closed():
    class DifferentProjection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = TernaryLinear(4, 2, estimator=AbsMeanSTE())
            self.right = TernaryLinear(
                4,
                2,
                estimator=SparseTernaryEstimator(target_sparsity=0.75),
            )
            self.right.weight = self.left.weight

    with pytest.raises(ValueError, match="share one estimator"):
        collect_diagnostics(DifferentProjection())


def test_tied_latent_consumers_with_different_training_modes_fail_closed():
    class DifferentMode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            estimator = AbsMeanSTE()
            self.left = TernaryLinear(4, 2, estimator=estimator)
            self.right = TernaryLinear(4, 2, estimator=estimator)
            self.right.weight = self.left.weight
            self.right.eval()

    with pytest.raises(ValueError, match="projection training mode"):
        collect_diagnostics(DifferentMode())


def test_finite_large_gradient_norm_does_not_overflow_float32_reduction():
    model = prepare_qat(torch.nn.Linear(2, 1), TernaryConfig.qat())
    model.weight.grad = torch.full_like(model.weight, 3e38)

    tensor = collect_diagnostics(model).tensors[0]

    assert tensor.gradient_finite is True
    assert tensor.gradient_l2 == pytest.approx(3e38 * math.sqrt(2), rel=1e-6)
    assert math.isfinite(tensor.gradient_l2)
