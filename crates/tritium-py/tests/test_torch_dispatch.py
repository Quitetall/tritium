"""Dispatcher, autograd, fake/meta and compile gates for plan 0046."""

import pytest

torch = pytest.importorskip("torch")

from tritium.torch import reference_ternary_linear, ternary_linear  # noqa: E402
from tritium.nn import TernaryLinear  # noqa: E402
from tritium import _tritium as _native  # noqa: E402


def test_ternary_linear_op_matches_literal_forward_and_masked_backward():
    _native._ternary_linear_cache_clear()
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
    assert _native._ternary_linear_cache_info()["hits"] == 1


def test_native_cpu_forward_caches_packed_weight_and_invalidates_on_mutation():
    _native._ternary_linear_cache_clear()
    weight = torch.tensor(
        [[-2.0, -0.4, 0.3, 2.1], [0.0, 1.0, -1.0, 0.2]], requires_grad=True
    )
    bias = torch.tensor([0.1, -0.2])
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    first = ternary_linear(x, weight, bias)
    first_info = _native._ternary_linear_cache_info()
    assert first_info == {
        "capacity": 4096,
        "entries": 1,
        "hits": 0,
        "invalidations": 0,
        "misses": 1,
    }

    second = ternary_linear(x, weight, bias)
    second_info = _native._ternary_linear_cache_info()
    assert torch.equal(second, first)
    assert second_info == {
        "capacity": 4096,
        "entries": 1,
        "hits": 1,
        "invalidations": 0,
        "misses": 1,
    }

    with torch.no_grad():
        weight[0, 1] = -1.4
    expected = torch.tensor([[1.55, -0.75]])
    third = ternary_linear(x, weight, bias)
    third_info = _native._ternary_linear_cache_info()
    assert torch.allclose(third, expected, atol=1e-6)
    assert third_info == {
        "capacity": 4096,
        "entries": 1,
        "hits": 1,
        "invalidations": 1,
        "misses": 2,
    }

    # Replacing storage through the legacy `.data` escape hatch does not bump
    # PyTorch's version counter. Native cache keys also bind the DLPack storage
    # pointer, so stale packed bytes still cannot survive.
    version_before_storage_swap = weight._version
    weight.data = torch.ones_like(weight)
    assert weight._version == version_before_storage_swap
    fourth = ternary_linear(x, weight, bias)
    assert torch.allclose(fourth, torch.tensor([[10.1, 9.8]]), atol=1e-6)
    assert _native._ternary_linear_cache_info() == {
        "capacity": 4096,
        "entries": 1,
        "hits": 1,
        "invalidations": 2,
        "misses": 3,
    }


def test_native_cpu_forward_keeps_composite_fallback_for_unsupported_layout_and_dtype():
    _native._ternary_linear_cache_clear()
    storage = torch.randn(4, 3)
    x = storage.transpose(0, 1)
    weight = torch.randn(5, 4)
    assert not x.is_contiguous()

    actual = ternary_linear(x, weight)
    expected = ternary_linear(x.contiguous(), weight)
    assert torch.allclose(actual, expected, atol=1e-6)

    double_actual = ternary_linear(x.double(), weight.double())
    double_expected = ternary_linear(x.double().contiguous(), weight.double())
    assert torch.equal(double_actual, double_expected)
    assert _native._ternary_linear_cache_info() == {
        "capacity": 4096,
        "entries": 1,
        "hits": 0,
        "invalidations": 0,
        "misses": 1,
    }


def test_native_cpu_cache_does_not_retain_dead_parameters():
    import gc

    _native._ternary_linear_cache_clear()

    def populate_cache():
        x = torch.ones(1, 4)
        weight = torch.ones(2, 4)
        ternary_linear(x, weight)
        assert _native._ternary_linear_cache_info()["entries"] == 1

    populate_cache()
    gc.collect()
    assert _native._ternary_linear_cache_info() == {
        "capacity": 4096,
        "entries": 0,
        "hits": 0,
        "invalidations": 0,
        "misses": 1,
    }


def test_native_cpu_uses_pytorch_scale_reduction_at_rounding_threshold():
    generator = torch.Generator().manual_seed(0)
    other = torch.randn(255, generator=generator) * torch.exp(
        torch.empty(255).uniform_(-8, 8, generator=generator)
    )
    sequential_sum = torch.tensor(0.0)
    for value in other:
        sequential_sum = sequential_sum + value.abs()
    boundary = sequential_sum / 511
    weight = torch.cat((boundary.view(1), other)).view(1, 256)
    x = torch.zeros(1, 256)
    x[0, 0] = 1.0

    expected = reference_ternary_linear(x, weight)
    actual = ternary_linear(x, weight)

    assert expected.item() > 0
    assert torch.equal(actual, expected)


def test_native_cpu_uses_pytorch_safe_scale_for_subnormal_rows():
    weight = torch.full((1, 4), 1.0e-40, requires_grad=True)
    x = torch.ones(1, 4, requires_grad=True)

    expected = reference_ternary_linear(x, weight)
    actual = ternary_linear(x, weight)
    assert torch.equal(actual, expected)
    assert actual.item() == 0.0

    actual.backward(torch.ones_like(actual))
    assert torch.equal(weight.grad, torch.ones_like(weight))


@pytest.mark.parametrize(
    ("x", "weight"),
    [
        (torch.tensor([[float("nan"), 2.0]]), torch.tensor([[0.0, 1.0]])),
        (torch.tensor([[1.0, 2.0]]), torch.tensor([[float("nan"), 1.0]])),
    ],
)
def test_native_cpu_falls_back_for_nonfinite_dense_semantics(x, weight):
    expected = reference_ternary_linear(x, weight)
    actual = ternary_linear(x, weight)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))


def test_native_cpu_cache_hit_profiles_without_projection_or_copy_ops():
    x = torch.randn(4, 16)
    weight = torch.randn(8, 16)
    bias = torch.randn(8)
    ternary_linear(x, weight, bias)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        acc_events=True,
    ) as profile:
        ternary_linear(x, weight, bias)

    event_names = {event.key.lower() for event in profile.key_averages()}
    forbidden = {
        "aten::_to_copy",
        "aten::abs",
        "aten::copy_",
        "aten::mean",
        "aten::to",
    }
    assert event_names.isdisjoint(forbidden)
    assert not any("memcpy" in name or "synchronize" in name for name in event_names)


def test_native_cpu_backward_reuses_packed_weight_without_projection_or_matmul():
    torch.manual_seed(23)
    _native._ternary_linear_cache_clear()
    x = torch.randn(3, 2, 16, requires_grad=True)
    weight = torch.randn(8, 16, requires_grad=True)
    bias = torch.randn(8, requires_grad=True)
    grad_output = torch.randn(3, 2, 8)

    expected_x = x.detach().clone().requires_grad_()
    expected_weight = weight.detach().clone().requires_grad_()
    expected_bias = bias.detach().clone().requires_grad_()
    reference_ternary_linear(expected_x, expected_weight, expected_bias).backward(grad_output)

    output = ternary_linear(x, weight, bias)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        acc_events=True,
    ) as profile:
        output.backward(grad_output)

    assert torch.allclose(x.grad, expected_x.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(weight.grad, expected_weight.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(bias.grad, expected_bias.grad, atol=1e-6, rtol=1e-6)
    assert _native._ternary_linear_cache_info() == {
        "capacity": 4096,
        "entries": 1,
        "hits": 1,
        "invalidations": 0,
        "misses": 1,
    }

    event_names = {event.key.lower() for event in profile.key_averages()}
    forbidden = {
        "aten::abs",
        "aten::clamp",
        "aten::matmul",
        "aten::mean",
        "aten::mm",
        "aten::round",
    }
    assert event_names.isdisjoint(forbidden)


@pytest.mark.parametrize("seed", [0, 1, 7, 23])
def test_native_cpu_backward_randomized_parity(seed):
    torch.manual_seed(seed)
    m = 1 + seed % 7
    n = 1 + (seed * 3) % 11
    k = 1 + (seed * 7) % 33
    x = torch.randn(m, k, requires_grad=True)
    weight = torch.randn(n, k, requires_grad=True)
    bias = torch.randn(n, requires_grad=True) if seed % 2 else None
    grad_output = torch.randn(m, n)

    expected_x = x.detach().clone().requires_grad_()
    expected_weight = weight.detach().clone().requires_grad_()
    expected_bias = bias.detach().clone().requires_grad_() if bias is not None else None
    reference_ternary_linear(expected_x, expected_weight, expected_bias).backward(grad_output)

    _native._ternary_linear_cache_clear()
    ternary_linear(x, weight, bias).backward(grad_output)

    assert torch.allclose(x.grad, expected_x.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(weight.grad, expected_weight.grad, atol=2e-5, rtol=2e-5)
    if bias is not None:
        assert torch.allclose(bias.grad, expected_bias.grad, atol=2e-5, rtol=2e-5)
    assert _native._ternary_linear_cache_info()["hits"] == 1


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or "cuda" not in _native.compiled_backends(),
    reason="CUDA-enabled Tritium extension unavailable",
)
def test_native_cuda_warm_forward_backward_avoids_composite_tensor_ops():
    torch.manual_seed(29)
    x = torch.randn(4, 3, 257, device="cuda", requires_grad=True)
    weight = torch.randn(33, 257, device="cuda", requires_grad=True)
    bias = torch.randn(33, device="cuda", requires_grad=True)
    grad_output = torch.randn(4, 3, 33, device="cuda")

    expected_x = x.detach().clone().requires_grad_()
    expected_weight = weight.detach().clone().requires_grad_()
    expected_bias = bias.detach().clone().requires_grad_()
    reference_ternary_linear(
        expected_x, expected_weight, expected_bias
    ).backward(grad_output)

    # First forward builds resident packed TQ2_0 state. Profile only warm
    # forward/backward, where projection and matrix contractions must be native.
    stream = torch.cuda.Stream(device=x.device)
    stream.wait_stream(torch.cuda.current_stream(x.device))
    with torch.cuda.stream(stream), torch.no_grad():
        ternary_linear(x, weight, bias)
    stream.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        acc_events=True,
    ) as profile:
        with torch.cuda.stream(stream):
            ternary_linear(x, weight, bias).backward(grad_output)
            torch.cuda._sleep(1)
        stream.synchronize()

    assert torch.allclose(x.grad, expected_x.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(weight.grad, expected_weight.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(bias.grad, expected_bias.grad, atol=2e-5, rtol=2e-5)
    event_names = {event.key.lower() for event in profile.key_averages()}
    forbidden = {
        "aten::abs",
        "aten::clamp",
        "aten::matmul",
        "aten::mean",
        "aten::mm",
        "aten::round",
    }
    assert event_names.isdisjoint(forbidden)
    assert not any("memcpy dtoh" in name or "memcpy htod" in name for name in event_names)
    native_kernel_names = {
        "tq2_projected_linear_forward",
        "tq2_projected_grad_a",
        "linear_grad_master_ste",
        "bias_backward",
    }
    native_resources = {
        event.device_resource_id
        for event in profile.events()
        if event.name in native_kernel_names
    }
    sentinel_resources = {
        event.device_resource_id
        for event in profile.events()
        if "spin_kernel" in event.name
    }
    assert len(native_resources) == 1
    assert native_resources == sentinel_resources


@pytest.mark.skipif(
    not torch.cuda.is_available() or "cuda" not in _native.compiled_backends(),
    reason="CUDA-enabled Tritium extension unavailable",
)
def test_native_cuda_tail_forward_backward_parity_for_memcheck():
    torch.manual_seed(37)
    x = torch.randn(4, 3, 257, device="cuda", requires_grad=True)
    weight = torch.randn(33, 257, device="cuda", requires_grad=True)
    bias = torch.randn(33, device="cuda", requires_grad=True)
    grad_output = torch.randn(4, 3, 33, device="cuda")
    expected_x = x.detach().clone().requires_grad_()
    expected_weight = weight.detach().clone().requires_grad_()
    expected_bias = bias.detach().clone().requires_grad_()

    output = ternary_linear(x, weight, bias)
    expected = reference_ternary_linear(expected_x, expected_weight, expected_bias)
    output.backward(grad_output)
    expected.backward(grad_output)

    torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(x.grad, expected_x.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(weight.grad, expected_weight.grad, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(bias.grad, expected_bias.grad, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or "cuda" not in _native.compiled_backends(),
    reason="CUDA-enabled Tritium extension unavailable",
)
def test_native_cuda_preserves_nonfinite_dense_semantics():
    cases = (
        (
            torch.tensor([[0.25, -0.5, 1.0], [1.0, 2.0, -3.0]]),
            torch.tensor([[float("inf"), 1.0, -1.0], [0.5, -0.5, 0.0]]),
        ),
        (
            torch.tensor([[float("nan"), -0.5, 1.0], [1.0, 2.0, -3.0]]),
            torch.tensor([[0.25, 1.0, -1.0], [0.5, -0.5, 0.0]]),
        ),
    )
    grad_output = torch.tensor([[0.5, -1.0], [2.0, 0.25]], device="cuda")

    for input_values, weight_values in cases:
        x = input_values.cuda().requires_grad_()
        weight = weight_values.cuda().requires_grad_()
        expected_x = x.detach().clone().requires_grad_()
        expected_weight = weight.detach().clone().requires_grad_()

        output = ternary_linear(x, weight)
        expected = reference_ternary_linear(expected_x, expected_weight)
        output.backward(grad_output)
        expected.backward(grad_output)

        torch.testing.assert_close(output, expected, equal_nan=True)
        torch.testing.assert_close(x.grad, expected_x.grad, equal_nan=True)
        torch.testing.assert_close(weight.grad, expected_weight.grad, equal_nan=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or "cuda" not in _native.compiled_backends(),
    reason="CUDA-enabled Tritium extension unavailable",
)
def test_native_cuda_cache_invalidates_on_mutation_and_storage_replacement():
    x = torch.tensor([[1.0, -2.0, 0.5]], device="cuda")
    weight = torch.tensor(
        [[0.1, 0.2, 0.3], [-0.75, 0.5, 0.25]], device="cuda"
    )

    first = ternary_linear(x, weight)
    with torch.no_grad():
        weight[0, 1] = -1.4
    second = ternary_linear(x, weight)
    torch.testing.assert_close(second, reference_ternary_linear(x, weight))
    assert not torch.equal(second, first)

    version_before_storage_swap = weight._version
    weight.data = torch.ones_like(weight)
    assert weight._version == version_before_storage_swap
    third = ternary_linear(x, weight)
    torch.testing.assert_close(third, reference_ternary_linear(x, weight))
    assert not torch.equal(third, second)


@pytest.mark.skipif(
    not torch.cuda.is_available() or "cuda" not in _native.compiled_backends(),
    reason="CUDA-enabled Tritium extension unavailable",
)
def test_native_cuda_cache_orders_cross_stream_pack_and_survives_owner_drop():
    import gc

    torch.manual_seed(31)
    x = torch.randn(8, 257, device="cuda")
    weight = torch.randn(33, 257, device="cuda")
    expected = reference_ternary_linear(x, weight)
    pack_stream = torch.cuda.Stream(device=x.device)
    consume_stream = torch.cuda.Stream(device=x.device)
    current = torch.cuda.current_stream(x.device)
    pack_stream.wait_stream(current)
    consume_stream.wait_stream(current)

    # Pack remains queued behind a long kernel. Cache-hit consumption on another
    # stream must wait for its private ready event without host synchronization.
    with torch.cuda.stream(pack_stream), torch.no_grad():
        torch.cuda._sleep(100_000_000)
        ternary_linear(x, weight)
    with torch.cuda.stream(consume_stream), torch.no_grad():
        output = ternary_linear(x, weight)

    # Dropping owner evicts hidden cache state while its raw-stream read remains
    # queued. Native binding's record_stream calls must prevent allocator reuse.
    del weight
    gc.collect()
    churn = [
        torch.empty((33, 132), dtype=torch.uint8, device=x.device)
        for _ in range(16)
    ]
    consume_stream.synchronize()
    assert churn
    torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)


def test_builtin_ternary_linear_module_compiles_as_one_full_graph():
    torch.manual_seed(17)
    layer = TernaryLinear(4, 3)
    sample = torch.randn(2, 4)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    assert torch.equal(compiled(sample), layer(sample))
