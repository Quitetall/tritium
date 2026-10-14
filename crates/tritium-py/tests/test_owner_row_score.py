"""Actual shared-parameter autograd oracle for the common Rust row collector.

Tiny CPU tests are conformance, not curvature/quality/GPU/RSS qualification.
"""
import pytest
import torch
from tritium import _tritium


def _row(rows, columns, row, uses, *, observation=65536, workspace=65536):
    return torch.from_dlpack(_tritium._owner_row_score(
        rows, columns, row, uses, observation, workspace))


def _graph(weight):
    # Repeated lookup and output calls use the same actual owner. Position zero
    # has no local loss, yet influences later selected outputs through cumsum.
    first = torch.tensor([0, 1, 0], dtype=torch.int64)
    second = torch.tensor([2, 0], dtype=torch.int64)
    h1 = torch.nn.functional.embedding(first, weight)
    h2 = torch.nn.functional.embedding(second, weight)
    x1 = torch.tanh(h1.cumsum(0))
    x2 = torch.sin(h2 + h1[0])
    z1 = torch.nn.functional.linear(x1, weight)
    z2 = torch.nn.functional.linear(x2, weight)
    loss = z1[1:].square().sum() + 0.3 * z2.square().sum()
    return loss, [("gather", first, h1), ("matmul", x1, z1),
                  ("gather", second, h2), ("matmul", x2, z2)]


@pytest.mark.parametrize("scale", [1.0, 0.125, -2.0])
def test_actual_tied_nonlinear_repeated_calls_match_autograd_and_finite_difference(scale):
    weight = torch.tensor([[0.2, -0.3], [0.4, 0.1], [-0.5, 0.7]], requires_grad=True)
    loss, calls = _graph(weight)
    derivatives = torch.autograd.grad(scale * loss, [weight, *(call[2] for call in calls)])
    dense, *adjoints = derivatives
    uses = [(role, value.detach().contiguous(), adjoint.detach().contiguous())
            for (role, value, _), adjoint in zip(calls, adjoints)]
    actual = torch.stack([_row(3, 2, row, uses) for row in range(3)])
    assert actual.dtype == torch.float64 and actual.shape == (3, 2)
    torch.testing.assert_close(actual, dense.double(), rtol=3e-6, atol=3e-7)
    # Separate f64 finite-difference evaluation, not another PTQ implementation.
    direction = torch.tensor([[0.3, -0.4], [0.2, 0.5], [-0.1, 0.6]], dtype=torch.float64)
    w64 = weight.detach().double()
    step = 1e-6
    derivative = scale * (_graph(w64 + step * direction)[0] -
                          _graph(w64 - step * direction)[0]) / (2 * step)
    torch.testing.assert_close((actual * direction).sum(), derivative, rtol=5e-6, atol=5e-7)
    assert weight.grad is None
    assert adjoints[0][0].abs().sum() > 0  # The locally unsupervised position matters.


@pytest.mark.parametrize("second,expected", [(1.0, 4.0), (-1.0, 0.0)])
def test_shared_use_cross_terms_are_not_separate_consumer_grams(second, expected):
    uses = [("matmul", torch.ones(1, 1), torch.ones(1, 1)),
            ("gather", torch.tensor([0]), torch.tensor([[second]]))]
    score = _row(1, 1, 0, uses)
    assert score.square().item() == expected
    assert score.square().item() != 2.0


def test_f64_products_and_requested_row_budget():
    maximum = torch.finfo(torch.float32).max
    score = _row(1, 1, 0, [("matmul", torch.tensor([[maximum]]),
                            torch.tensor([[maximum]]))], observation=8, workspace=8)
    assert score.item() == float(maximum) ** 2
    uses = [("gather", torch.tensor([0]), torch.tensor([[1.0, 2.0]]))]
    torch.testing.assert_close(_row(1_000_000, 2, 999_999, uses, observation=16, workspace=16),
                               torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="Budget"):
        _row(1_000_000, 2, 0, uses, observation=16, workspace=15)
    with pytest.raises(ValueError, match="Budget"):
        _row(1_000_000, 2, 0, uses, observation=15, workspace=16)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_unrequested_row_and_unused_input_reject(bad):
    with pytest.raises(ValueError, match="Nonfinite"):
        _row(2, 1, 0, [("matmul", torch.ones(1, 1), torch.tensor([[0.0, bad]]))])
    with pytest.raises(ValueError, match="Nonfinite"):
        _row(1, 1, 0, [("matmul", torch.tensor([[bad]]), torch.zeros(1, 1))])


@pytest.mark.parametrize("index", [-1, 2, 2**63 - 1])
def test_invalid_signed_indices_reject(index):
    with pytest.raises(ValueError, match="Index"):
        _row(2, 1, 0, [("gather", torch.tensor([index]), torch.ones(1, 1))])


@pytest.mark.parametrize("uses", [
    [],
    [("unknown", torch.ones(1, 1), torch.ones(1, 1))],
    [("matmul", torch.ones(1), torch.ones(1, 1))],
    [("matmul", torch.ones(2, 1), torch.ones(1, 1))],
    [("gather", torch.tensor([[0]]), torch.ones(1, 1))],
    [("gather", torch.tensor([0.0]), torch.ones(1, 1))],
    [("matmul", torch.ones(1, 1, dtype=torch.float64), torch.ones(1, 1))],
    [("gather", torch.tensor([0]), torch.ones(1, 2))],
])
def test_invalid_geometry_dtype_and_role_reject(uses):
    with pytest.raises(ValueError):
        _row(1, 1, 0, uses)


def test_row_replay_order_is_stable_and_output_owns_its_storage():
    uses = [("matmul", torch.tensor([[2.0, 3.0]]), torch.tensor([[1.0, -2.0]])),
            ("gather", torch.tensor([1, 1]), torch.tensor([[4.0, 5.0], [6.0, 7.0]]))]
    second = _row(2, 2, 1, uses)
    torch.testing.assert_close(second, torch.tensor([6.0, 6.0], dtype=torch.float64))
    torch.testing.assert_close(_row(2, 2, 0, uses), torch.tensor([2.0, 3.0], dtype=torch.float64))
    torch.testing.assert_close(_row(2, 2, 1, uses), second)
    uses[0][1].zero_()
    assert second.tolist() == [6.0, 6.0]


@pytest.mark.parametrize("loss_scale", [1.0, 0.25])
def test_exact_output_curvature_columns_reproduce_tied_ggn_row_blocks(loss_scale):
    weight = torch.tensor([[0.2, -0.3], [0.4, 0.1], [-0.5, 0.7]], requires_grad=True)
    _, calls = _graph(weight)
    logits = torch.cat([calls[1][2][1:], calls[3][2]], dim=0)
    probabilities = logits.detach().double().softmax(-1)
    factors = []
    for p in probabilities:
        s = torch.diag(p.sqrt()) - torch.outer(p, p.sqrt())
        factors.append(s * loss_scale ** 0.5)
    root = torch.block_diag(*factors)
    jacobian = torch.stack([
        torch.autograd.grad(value, weight, retain_graph=True)[0].flatten().double()
        for value in logits.flatten()
    ])
    expected = jacobian.T @ (root @ root.T) @ jacobian
    blocks = torch.zeros(3, 2, 2, dtype=torch.float64)
    for column in root.T:
        probe = column.to(torch.float32).reshape_as(logits)
        adjoints = torch.autograd.grad((logits * probe).sum(),
            [call[2] for call in calls], retain_graph=True)
        uses = [(role, value.detach().contiguous(), adjoint.detach().contiguous())
                for (role, value, _), adjoint in zip(calls, adjoints)]
        for row in range(3):
            q = _row(3, 2, row, uses)
            blocks[row] += torch.outer(q, q)
    for row in range(3):
        torch.testing.assert_close(blocks[row], expected[row * 2:(row + 1) * 2,
            row * 2:(row + 1) * 2], rtol=5e-6, atol=5e-7)
    # Deliberately omitted inter-row blocks exist; this is not the full GGN.
    assert expected[:2, 2:4].abs().sum() > 0
