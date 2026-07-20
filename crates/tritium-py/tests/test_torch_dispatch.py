"""Dispatcher, autograd, fake/meta and compile gates for plan 0046."""

import pytest

torch = pytest.importorskip("torch")

from tritium.torch import ternary_linear  # noqa: E402
from tritium.nn import TernaryLinear  # noqa: E402


def test_ternary_linear_op_matches_literal_forward_and_masked_backward():
    weight = torch.tensor(
        [[-2.0, -0.4, 0.3, 2.1], [0.0, 1.0, -1.0, 0.2]], requires_grad=True
    )
    bias = torch.tensor([0.1, -0.2], requires_grad=True)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)

    output = ternary_linear(x, weight, bias)
    assert torch.allclose(output, torch.tensor([[3.7, -0.75]]), atol=1e-6)

    output.sum().backward()
    assert torch.allclose(
        weight.grad,
        torch.tensor([[0.0, 2.0, 3.0, 0.0], [1.0, 0.0, 0.0, 4.0]]),
        atol=1e-6,
    )
    assert torch.equal(bias.grad, torch.ones(2))
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_ternary_linear_opcheck_and_fullgraph_compile():
    torch.manual_seed(11)
    x = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    weight = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
    bias = torch.randn(5, dtype=torch.double, requires_grad=True)

    checks = torch.library.opcheck(
        ternary_linear,
        (x, weight, bias),
        atol=1e-8,
        rtol=1e-8,
    )
    assert set(checks.values()) == {"SUCCESS"}

    compiled = torch.compile(
        lambda a, w, b: ternary_linear(a, w, b),
        backend="eager",
        fullgraph=True,
    )
    assert torch.equal(compiled(x, weight, bias), ternary_linear(x, weight, bias))


def test_ternary_linear_supports_leading_dims_no_bias_and_cpu_autocast():
    x = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)

    output = ternary_linear(x, weight)
    assert output.shape == (2, 3, 5)
    output.square().mean().backward()
    assert x.grad is not None and weight.grad is not None

    with torch.autocast("cpu", dtype=torch.bfloat16):
        autocast_output = ternary_linear(x.detach(), weight.detach())
    assert autocast_output.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_ternary_linear_cuda_stays_resident_and_autocasts():
    x = torch.randn(8, 16, device="cuda", requires_grad=True)
    weight = torch.randn(12, 16, device="cuda", requires_grad=True)
    bias = torch.randn(12, device="cuda", requires_grad=True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        acc_events=True,
    ) as profile:
        output = ternary_linear(x, weight, bias)
        output.square().mean().backward()
        torch.cuda.synchronize()

    assert output.is_cuda
    assert x.grad is not None and x.grad.is_cuda
    assert weight.grad is not None and weight.grad.is_cuda
    assert bias.grad is not None and bias.grad.is_cuda
    event_names = {event.key.lower() for event in profile.key_averages()}
    assert not any("memcpy dtoh" in name or "memcpy htod" in name for name in event_names)

    with torch.autocast("cuda", dtype=torch.float16):
        autocast_output = ternary_linear(x.detach(), weight.detach(), bias.detach())
    assert autocast_output.dtype == torch.float16


def test_builtin_ternary_linear_module_compiles_as_one_full_graph():
    torch.manual_seed(17)
    layer = TernaryLinear(4, 3)
    sample = torch.randn(2, 4)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    assert torch.equal(compiled(sample), layer(sample))
