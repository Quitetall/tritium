from __future__ import annotations

import pytest

from tritium.torch import PlaneAllocation, allocate_planes


def test_native_rate_distortion_allocation_prefers_sensitive_group():
    result = allocate_planes(
        [4, 4],
        [1.0, 10.0],
        [[10.0, 9.0, 8.0, 7.0], [10.0, 1.0, 0.5, 0.0]],
        target_bpw=3.0,
    )
    assert isinstance(result, PlaneAllocation)
    assert result.plane_counts == (1, 2)
    assert result.total_weights == 8
    assert 0.0 < result.achieved_bpw <= result.target_bpw


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"group_sizes": [], "sensitivities": [], "error_curves": []}, "must not be empty"),
        (
            {"group_sizes": [4], "sensitivities": [], "error_curves": [[1, 0, 0, 0]]},
            "equal lengths",
        ),
        ({"group_sizes": [4], "sensitivities": [1], "error_curves": [[1, 0]]}, "t_max"),
    ],
)
def test_native_allocator_rejects_invalid_evidence(kwargs, message):
    with pytest.raises(ValueError, match=message):
        allocate_planes(target_bpw=3.0, **kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"group_sizes": [-1], "sensitivities": [1.0], "error_curves": [[1, 0, 0, 0]]},
            "positive integer",
        ),
        (
            {"group_sizes": [4], "sensitivities": [float("nan")], "error_curves": [[1, 0, 0, 0]]},
            "finite and nonnegative",
        ),
        (
            {"group_sizes": [4], "sensitivities": [1.0], "error_curves": [[1, -1, 0, 0]]},
            "finite and nonnegative",
        ),
    ],
)
def test_native_allocator_normalizes_boundary_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        allocate_planes(target_bpw=3.0, **kwargs)


def test_native_allocator_rejects_nonpositive_budget_and_range():
    evidence = {
        "group_sizes": [4],
        "sensitivities": [1.0],
        "error_curves": [[1, 0, 0, 0]],
    }
    with pytest.raises(ValueError, match="finite and positive"):
        allocate_planes(target_bpw=0.0, **evidence)
    with pytest.raises(ValueError, match="t_min must not exceed t_max"):
        allocate_planes(target_bpw=3.0, t_min=2, t_max=1, **evidence)
