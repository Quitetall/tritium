"""Dispatcher, autograd, fake/meta and compile gates for plan 0046."""

import pytest

torch = pytest.importorskip("torch")

from tritium.torch import reference_ternary_linear, ternary_linear  # noqa: E402
from tritium.torch import ops as torch_ops  # noqa: E402
from tritium.nn import TernaryLinear  # noqa: E402
from tritium import _tritium as _native  # noqa: E402


def assert_no_cuda_synchronization_in_region(profile, region_name):
    events = profile.events()
    regions = [
        event
        for event in events
        if event.name == region_name
        and event.device_type == torch.autograd.DeviceType.CPU
    ]
    assert len(regions) == 1
    synchronization_names = {
        "cudadevicesynchronize",
        "cudastreamsynchronize",
        "cudaeventsynchronize",
        "cuctxsynchronize",
        "custreamsynchronize",
        "cueventsynchronize",
    }
    synchronizations = [
        event
        for event in events
        if event.name.replace(" ", "").lower() in synchronization_names
    ]
    region = regions[0]
    assert not any(
        synchronization.thread == region.thread
        and region.time_range.start
        <= synchronization.time_range.start
        < region.time_range.end
        for synchronization in synchronizations
    )


@pytest.mark.parametrize(
    "synchronization_name",
    [
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "cudaEventSynchronize",
        "cuCtxSynchronize",
        "cuStreamSynchronize",
        "cuEventSynchronize",
    ],
)
def test_cuda_synchronization_guard_rejects_same_thread_profiler_event(
    synchronization_name,
):
    class Event:
        def __init__(self, name, start, end, thread=7):
            self.name = name
            self.device_type = torch.autograd.DeviceType.CPU
            self.time_range = type("Range", (), {"start": start, "end": end})()
            self.thread = thread

    class Profile:
        def events(self):
            return [
                Event("warm", 1, 4),
                Event(synchronization_name, 2, 3),
            ]

    with pytest.raises(AssertionError):
        assert_no_cuda_synchronization_in_region(Profile(), "warm")


def test_cuda_synchronization_guard_ignores_other_thread_and_outside_region():
    class Event:
        def __init__(self, name, start, end, thread=7):
            self.name = name
            self.device_type = torch.autograd.DeviceType.CPU
            self.time_range = type("Range", (), {"start": start, "end": end})()
            self.thread = thread

    class Profile:
        def events(self):
            return [
                Event("warm", 2, 5),
                Event("cudaDeviceSynchronize", 1, 2),
                Event("cuCtxSynchronize", 3, 4, thread=8),
                Event("cudaStreamSynchronize", 5, 6),
            ]

    assert_no_cuda_synchronization_in_region(Profile(), "warm")


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


def test_native_cpu_cached_ste_mask_handles_zero_scale_and_word_tail():
    x = torch.ones(1, 65, requires_grad=True)
    weight = torch.zeros(2, 65, requires_grad=True)
    with torch.no_grad():
        weight[1, -1] = 1.0

    ternary_linear(x, weight).sum().backward()

    assert torch.equal(weight.grad[0], torch.zeros(65))
    assert torch.equal(weight.grad[1, :-1], torch.ones(64))
    assert weight.grad[1, -1].item() == 0.0


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
    expected_weight = weight.detach().clone().requires_grad_()
    reference_ternary_linear(x, expected_weight, bias).sum().backward()
    third.sum().backward()
    torch.testing.assert_close(weight.grad, expected_weight.grad)
    weight.grad = None

    # Replacing storage through the legacy `.data` escape hatch does not bump
    # PyTorch's version counter. Native cache keys also bind the DLPack storage
    # pointer, so stale packed bytes still cannot survive.
    version_before_storage_swap = weight._version
    weight.data = torch.ones_like(weight)
    assert weight._version == version_before_storage_swap
    fourth = ternary_linear(x, weight, bias)
    assert torch.allclose(fourth, torch.tensor([[10.1, 9.8]]), atol=1e-6)
    fourth.sum().backward()
    assert torch.equal(weight.grad, torch.zeros_like(weight))
    assert _native._ternary_linear_cache_info() == {
        "capacity": 4096,
        "entries": 1,
        "hits": 3,
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


def test_native_cpu_fast_path_preserves_public_validation_errors():
    x = torch.ones(1, 4)
    with pytest.raises(ValueError, match="ternary_linear input width"):
        ternary_linear(x, torch.ones(2, 3))
    with pytest.raises(ValueError, match="ternary_linear bias shape"):
        ternary_linear(x, torch.ones(2, 4), torch.ones(3))
    with pytest.raises(ValueError, match="share a device"):
        ternary_linear(x, torch.empty(2, 4, device="meta"))


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
        torch.ops.tritium.ternary_linear.default,
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


@pytest.mark.skipif(not hasattr(torch, "func"), reason="torch.func unavailable")
def test_ternary_linear_supports_functorch_grad_and_vmap():
    torch.manual_seed(17)
    x = torch.randn(2, 4)
    weight = torch.randn(3, 4)

    def objective(master):
        return ternary_linear(x, master).square().mean()

    actual_grad = torch.func.grad(objective)(weight)
    expected_grad = torch.func.grad(
        lambda master: reference_ternary_linear(x, master).square().mean()
    )(weight)
    torch.testing.assert_close(actual_grad, expected_grad)

    samples = torch.randn(5, 2, 4)
    actual = torch.func.vmap(lambda sample: ternary_linear(sample, weight))(samples)
    expected = torch.func.vmap(
        lambda sample: reference_ternary_linear(sample, weight)
    )(samples)
    torch.testing.assert_close(actual, expected)


def test_ternary_linear_supports_leading_dims_no_bias_and_cpu_autocast():
    x = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)

    output = ternary_linear(x, weight)
    assert output.shape == (2, 3, 5)
    output.square().mean().backward()
    assert x.grad is not None and weight.grad is not None

    x.grad = None
    weight.grad = None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        autocast_output = ternary_linear(x, weight)
        dispatcher_output = torch.ops.tritium.ternary_linear.default(
            x.detach(), weight.detach()
        )
    assert autocast_output.dtype == torch.bfloat16
    assert torch.equal(dispatcher_output, autocast_output)
    autocast_output.float().sum().backward()
    eager_x_grad = x.grad.detach().clone()
    eager_weight_grad = weight.grad.detach().clone()
    x.grad = None
    weight.grad = None

    compiled = torch.compile(
        lambda a, w: ternary_linear(a, w), fullgraph=True
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        compiled_output = compiled(x, weight)
    assert torch.equal(compiled_output, autocast_output)
    compiled_output.float().sum().backward()
    torch.testing.assert_close(x.grad, eager_x_grad)
    torch.testing.assert_close(weight.grad, eager_weight_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_ternary_linear_cuda_stays_resident_and_autocasts():
    torch.manual_seed(0)
    x = torch.randn(8, 16, device="cuda", requires_grad=True)
    weight = torch.randn(12, 16, device="cuda", requires_grad=True)
    bias = torch.randn(12, device="cuda", requires_grad=True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        acc_events=False,
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
        dispatcher_output = torch.ops.tritium.ternary_linear.default(
            x.detach(), weight.detach(), bias.detach()
        )
    assert autocast_output.dtype == torch.float16
    assert torch.equal(dispatcher_output, autocast_output)

    compiled = torch.compile(
        lambda a, w, b: ternary_linear(a, w, b), fullgraph=True
    )
    with torch.autocast("cuda", dtype=torch.float16):
        compiled_output = compiled(x.detach(), weight.detach(), bias.detach())
    assert torch.equal(compiled_output, autocast_output)


@pytest.mark.skipif(
    not torch.cuda.is_available() or "cuda" not in _native.compiled_backends(),
    reason="CUDA-enabled Tritium extension unavailable",
)
def test_native_cuda_compiled_autocast_preserves_master_cache_and_backward():
    torch.manual_seed(37)
    eager_x = torch.randn(8, 16, device="cuda", requires_grad=True)
    eager_weight = torch.randn(12, 16, device="cuda", requires_grad=True)
    eager_bias = torch.randn(12, device="cuda", requires_grad=True)
    compiled_x = eager_x.detach().clone().requires_grad_()
    compiled_weight = eager_weight.detach().clone().requires_grad_()
    compiled_bias = eager_bias.detach().clone().requires_grad_()
    grad_output = torch.randn(8, 12, device="cuda", dtype=torch.float16)

    with torch.autocast("cuda", dtype=torch.float16):
        eager_output = ternary_linear(eager_x, eager_weight, eager_bias)
    eager_output.backward(grad_output)

    compiled = torch.compile(
        lambda a, w, b: ternary_linear(a, w, b), fullgraph=True
    )
    torch_ops._CUDA_PACKED_CACHE.clear()
    with torch.autocast("cuda", dtype=torch.float16):
        warm_output = compiled(compiled_x, compiled_weight, compiled_bias)
    warm_output.backward(grad_output)
    compiled_x.grad = None
    compiled_weight.grad = None
    compiled_bias.grad = None

    cache_entries = list(torch_ops._CUDA_PACKED_CACHE.values())
    assert len(cache_entries) == 1
    warm_entry = cache_entries[0]
    warm_packed_storage = int(warm_entry.packed.untyped_storage()._cdata)
    assert warm_entry.owner() is compiled_weight
    assert warm_entry.dtype == torch.float32

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        acc_events=False,
    ) as profile:
        with torch.autocast("cuda", dtype=torch.float16):
            compiled_output = compiled(compiled_x, compiled_weight, compiled_bias)
        compiled_output.backward(grad_output)
        torch.cuda._sleep(1)
    torch.cuda.synchronize()

    assert torch.equal(compiled_output, eager_output)
    torch.testing.assert_close(compiled_x.grad, eager_x.grad)
    torch.testing.assert_close(compiled_weight.grad, eager_weight.grad)
    torch.testing.assert_close(compiled_bias.grad, eager_bias.grad)
    cache_entries = list(torch_ops._CUDA_PACKED_CACHE.values())
    assert len(cache_entries) == 1
    assert cache_entries[0] is warm_entry
    assert int(cache_entries[0].packed.untyped_storage()._cdata) == warm_packed_storage
    native_events = {
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    }
    assert {
        "tq2_0_add_mpgemm_tiled_f16_bias",
        "tq2_projected_grad_a_f16",
        "linear_grad_master_ste_autocast",
        "bias_backward_f16",
    }.issubset(native_events)
    event_names = {event.key.lower() for event in profile.key_averages()}
    assert event_names.isdisjoint(
        {
            "aten::abs",
            "aten::clamp",
            "aten::matmul",
            "aten::mean",
            "aten::mm",
            "aten::round",
        }
    )


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
        with torch.cuda.stream(stream), torch.profiler.record_function(
            "tritium_native_cuda_warm"
        ):
            ternary_linear(x, weight, bias).backward(grad_output)
            torch.cuda._sleep(1)
        stream.synchronize()

    assert torch.allclose(x.grad, expected_x.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(weight.grad, expected_weight.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(bias.grad, expected_bias.grad, atol=2e-5, rtol=2e-5)
    event_names = {event.key.lower() for event in profile.key_averages()}
    assert_no_cuda_synchronization_in_region(
        profile, "tritium_native_cuda_warm"
    )
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
def test_native_cuda_autocast_warm_forward_backward_uses_fp16_kernels():
    torch.manual_seed(31)
    x = torch.randn(4, 3, 257, device="cuda", requires_grad=True)
    weight = torch.randn(33, 257, device="cuda", requires_grad=True)
    bias = torch.randn(33, device="cuda", requires_grad=True)
    grad_output = torch.randn(4, 3, 33, device="cuda", dtype=torch.float16)

    expected_x = x.detach().clone().half().requires_grad_()
    expected_weight = weight.detach().clone().requires_grad_()
    expected_bias = bias.detach().clone().half().requires_grad_()
    scales = expected_weight.detach().abs().mean(dim=1, keepdim=True)
    normalized = expected_weight / scales.clamp_min(torch.finfo(torch.float32).tiny)
    projected = normalized.round().clamp(-1.0, 1.0) * scales
    mask = ((normalized.abs() < 1.0) & (scales > 0)).to(torch.float32)
    differentiable = (
        projected.detach()
        + (expected_weight - expected_weight.detach()) * mask
    )
    expected = torch.nn.functional.linear(
        expected_x.float(), differentiable, expected_bias.float()
    ).half()
    expected.backward(grad_output)

    stream = torch.cuda.Stream(device=x.device)
    stream.wait_stream(torch.cuda.current_stream(x.device))
    with torch.autocast("cuda", dtype=torch.float16), torch.cuda.stream(stream):
        # Tritium preserves the original fp32 master as cache owner, allowing
        # resident TQ2_0 state to warm before profiling.
        ternary_linear(x, weight, bias)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            acc_events=False,
        ) as profile:
            with torch.profiler.record_function("tritium_native_cuda_fp16_warm"):
                output = ternary_linear(x, weight, bias)
                output.backward(grad_output)
                torch.cuda._sleep(1)
        stream.synchronize()

    assert output.dtype == torch.float16
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x.grad, expected_x.grad.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        weight.grad, expected_weight.grad.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        bias.grad, expected_bias.grad.float(), atol=2e-2, rtol=2e-2
    )

    event_names = {event.key.lower() for event in profile.key_averages()}
    assert_no_cuda_synchronization_in_region(
        profile, "tritium_native_cuda_fp16_warm"
    )
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
        "tq2_projected_linear_forward_f16",
        "tq2_projected_grad_a_f16",
        "linear_grad_master_ste_autocast",
        "bias_backward_f16",
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
def test_native_cuda_fp16_tail_paths_for_memcheck():
    torch.manual_seed(35)
    subnormal_x = torch.ones(
        1, 3, device="cuda", dtype=torch.float16, requires_grad=True
    )
    subnormal_weight = torch.full(
        (1, 3), 2**-24, device="cuda", dtype=torch.float16, requires_grad=True
    )
    expected_subnormal_x = subnormal_x.detach().clone().requires_grad_()
    expected_subnormal_weight = (
        subnormal_weight.detach().clone().requires_grad_()
    )
    subnormal_output = ternary_linear(subnormal_x, subnormal_weight)
    expected_subnormal_output = reference_ternary_linear(
        expected_subnormal_x, expected_subnormal_weight
    )
    subnormal_output.backward(torch.ones_like(subnormal_output))
    expected_subnormal_output.backward(torch.ones_like(expected_subnormal_output))
    torch.testing.assert_close(subnormal_output, expected_subnormal_output)
    torch.testing.assert_close(subnormal_x.grad, expected_subnormal_x.grad)
    torch.testing.assert_close(
        subnormal_weight.grad, expected_subnormal_weight.grad
    )

    bad_master = torch.zeros(1, 3, device="cuda", dtype=torch.bfloat16)
    scales = torch.ones(1, device="cuda", dtype=torch.float32)
    packed = torch.empty(1, 66, device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="torch.float16"):
        _native._ternary_linear_cuda_pack(
            bad_master,
            scales,
            packed,
            1,
            3,
            66,
            2,
            torch.cuda.current_stream(),
            torch.cuda.current_device(),
        )

    x = torch.randn(2, 257, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(
        33, 257, device="cuda", dtype=torch.float16, requires_grad=True
    )
    bias = torch.randn(33, device="cuda", dtype=torch.float16, requires_grad=True)
    grad_output = torch.randn(2, 33, device="cuda", dtype=torch.float16)
    expected_x = x.detach().clone().requires_grad_()
    expected_weight = weight.detach().clone().requires_grad_()
    expected_bias = bias.detach().clone().requires_grad_()

    output = ternary_linear(x, weight, bias)
    expected = reference_ternary_linear(
        expected_x, expected_weight, expected_bias
    )
    output.backward(grad_output)
    expected.backward(grad_output)

    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x.grad, expected_x.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        weight.grad, expected_weight.grad, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(bias.grad, expected_bias.grad, atol=2e-2, rtol=2e-2)

    mixed_x = torch.randn(2, 257, device="cuda", requires_grad=True)
    mixed_weight = torch.randn(33, 257, device="cuda", requires_grad=True)
    mixed_bias = torch.randn(33, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        mixed_output = ternary_linear(mixed_x, mixed_weight, mixed_bias)
    mixed_output.backward(grad_output)
    assert mixed_output.dtype == torch.float16
    assert mixed_x.grad is not None and mixed_x.grad.dtype == torch.float32
    assert mixed_weight.grad is not None and mixed_weight.grad.dtype == torch.float32
    assert mixed_bias.grad is not None and mixed_bias.grad.dtype == torch.float32
    assert torch.isfinite(mixed_output).all()
    assert torch.isfinite(mixed_x.grad).all()
    assert torch.isfinite(mixed_weight.grad).all()
    assert torch.isfinite(mixed_bias.grad).all()


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

    dtype_swap_version = weight._version
    dtype_swap_storage = int(weight.untyped_storage()._cdata)
    dtype_swap_pointer = weight.data_ptr()
    shape = weight.shape
    stride = weight.stride()
    weight.data = weight.data.view(torch.float16).as_strided(shape, stride)
    assert weight._version == dtype_swap_version
    assert int(weight.untyped_storage()._cdata) == dtype_swap_storage
    assert weight.data_ptr() == dtype_swap_pointer
    assert weight.shape == shape
    fourth = ternary_linear(x.half(), weight)
    torch.testing.assert_close(fourth, reference_ternary_linear(x.half(), weight))


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
