"""Native output-aware dense-curvature estimator contract tests."""

from __future__ import annotations

import torch

from tritium import _tritium
from tritium.torch import DenseGroupFit, fit_dense_ternary_group


def _metric() -> torch.Tensor:
    return torch.tensor(
        [
            [2.0, 0.2, 0.0, 0.0],
            [0.2, 1.5, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.1],
            [0.0, 0.0, 0.1, 1.0],
        ],
        dtype=torch.float64,
    )


def test_native_dense_fit_is_deterministic_and_ternary():
    weights = [0.9, -0.8, 0.2, 0.1]
    metric = _metric().reshape(-1).tolist()
    first = _tritium.fit_joint_ternary_dense(
        weights,
        metric,
        2,
        16,
        1e-8,
        2,
        1e6,
        "f16",
        False,
        False,
    )
    second = _tritium.fit_joint_ternary_dense(
        weights,
        metric,
        2,
        16,
        1e-8,
        2,
        1e6,
        "f16",
        False,
        False,
    )
    assert first == second
    scales, trits, reconstruction, objective = first
    assert len(scales) == 2
    assert len(trits) == 2
    assert all(value in {-1, 0, 1} for plane in trits for value in plane)
    assert len(reconstruction) == 4
    assert objective >= 0.0


def test_python_dense_fit_wrapper_preserves_hard_decode():
    weights = torch.tensor([0.9, -0.8, 0.2, 0.1], dtype=torch.float32)
    result = fit_dense_ternary_group(weights, _metric(), planes=2, em_restarts=2)
    assert isinstance(result, DenseGroupFit)
    assert result.objective >= 0.0
    projection = result.projection
    assert projection.dense.shape == (1, 4)
    decoded = sum(
        plane.trits.to(torch.float32)
        * plane.scales.to(torch.float32).repeat_interleave(plane.group_size, dim=1)
        for plane in projection.planes
    )
    assert torch.equal(projection.dense, decoded)


def test_native_dense_fit_rejects_non_psd_metric():
    with torch.no_grad():
        try:
            _tritium.fit_joint_ternary_dense(
                [1.0, 2.0],
                [1.0, 2.0, 2.0, 1.0],
                1,
                16,
                1e-8,
                2,
                1e6,
                "f16",
                False,
                False,
            )
        except ValueError as error:
            assert "semidefinite" in str(error)
        else:
            raise AssertionError("non-PSD metric was accepted")
