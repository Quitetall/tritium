"""Numerical contracts for bounded scale-only and hard-PV refinement."""

import pytest

torch = pytest.importorskip("torch")

from tritium.torch import RefinementConfig  # noqa: E402
from tritium.torch.projection import TernaryPlane  # noqa: E402
from tritium.torch.refinement_core import refine_weight_diagonal  # noqa: E402


def _planes(master, count=2):
    values = []
    residual = master.clone()
    for _ in range(count):
        scales = residual.abs().mean(dim=1, keepdim=True).to(torch.float16)
        trits = (
            (residual / scales.to(residual.dtype).clamp_min(1e-12))
            .round()
            .clamp(-1, 1)
            .to(torch.int8)
        )
        values.append(
            TernaryPlane(
                trits=trits,
                scales=scales,
                group_size=master.shape[1],
            )
        )
        residual -= trits * scales
    return tuple(values)


def test_scale_only_freezes_trits_and_never_worsens_stored_f16_objective():
    torch.manual_seed(211)
    master = torch.randn(7, 12)
    parent = _planes(master)
    metric = torch.rand(12).add_(0.1)
    result = refine_weight_diagonal(
        master,
        parent,
        metric,
        RefinementConfig.scale_only(),
        max_working_bytes=2048,
    )

    assert all(
        torch.equal(before.trits, after.trits)
        for before, after in zip(parent, result.planes)
    )
    assert result.refined_weighted_mse <= result.parent_weighted_mse
    assert all(plane.scales.dtype == torch.float16 for plane in result.planes)


def test_dense_hard_pv_alternates_assignments_and_scales_without_regression():
    torch.manual_seed(223)
    master = torch.randn(9, 16)
    parent = _planes(master, count=1)
    parent = (
        TernaryPlane(
            trits=torch.zeros_like(parent[0].trits),
            scales=parent[0].scales,
            group_size=16,
        ),
    )
    result = refine_weight_diagonal(
        master,
        parent,
        torch.ones(16),
        RefinementConfig.hard_pv(),
        iterations=3,
        max_working_bytes=4096,
    )

    assert not torch.equal(result.planes[0].trits, parent[0].trits)
    assert result.refined_weighted_mse <= result.parent_weighted_mse


def test_s34_hard_pv_emits_exact_one_zero_per_group_and_is_deterministic():
    torch.manual_seed(227)
    master = torch.randn(5, 12)
    parent = _planes(master)
    metric = torch.linspace(0.5, 1.5, 12)
    first = refine_weight_diagonal(
        master,
        parent,
        metric,
        RefinementConfig.hard_pv(structure="s34"),
        iterations=2,
        max_working_bytes=2048,
    )
    second = refine_weight_diagonal(
        master,
        parent,
        metric,
        RefinementConfig.hard_pv(structure="s34"),
        iterations=2,
        max_working_bytes=2048,
    )

    for one, two in zip(first.planes, second.planes):
        assert one.structure == "s34"
        assert torch.equal(one.trits, two.trits)
        groups = one.trits.reshape(5, 3, 4)
        assert torch.all(torch.count_nonzero(groups == 0, dim=2) == 1)
        assert set(one.trits.unique().tolist()) <= {-1, 0, 1}
