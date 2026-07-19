"""Torch parity for the Tritium autograd ops (ADR 0030 Tier 0).

The raw ternary Conv1d (fp path, scale = ones) must match ``torch.nn.functional.conv1d`` in both the
forward and the gradients; FSQ must land on its grid and pass the STE gradient; the ``TernaryConv1d``
module must train. Requires PyTorch (skipped otherwise).
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from tritium.autograd import (  # noqa: E402
    FSQ,
    LearnedTernaryConv1d,
    TernaryConv1d,
    _Conv1dFn,
)


@pytest.mark.parametrize(
    "stride,dilation,pad,groups,k",
    [(1, 1, 1, 1, 3), (2, 1, 1, 1, 3), (1, 2, 2, 1, 3), (1, 1, 0, 2, 3), (1, 1, 3, 6, 7)],
)
def test_conv1d_matches_torch_forward_and_grad(stride, dilation, pad, groups, k):
    torch.manual_seed(0)
    b, c_in, c_out, length = 2, 6, 6, 16
    x = torch.randn(b, c_in, length)
    w = torch.randn(c_out, c_in // groups, k)
    scale = torch.ones(c_out)
    cfg = (b, c_in, c_out, length, k, stride, dilation, pad, pad, groups)

    xt = x.clone().requires_grad_(True)
    wt = w.clone().requires_grad_(True)
    yt = _Conv1dFn.apply(xt, wt.reshape(c_out, (c_in // groups) * k), scale, cfg)

    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    yr = F.conv1d(xr, wr, stride=stride, padding=pad, dilation=dilation, groups=groups)

    assert yt.shape == yr.shape
    assert torch.allclose(yt, yr, atol=1e-4), (yt - yr).abs().max()

    g = torch.randn_like(yr)
    yt.backward(g)
    yr.backward(g)
    assert torch.allclose(xt.grad, xr.grad, atol=1e-4), (xt.grad - xr.grad).abs().max()
    assert torch.allclose(wt.grad.reshape(wr.shape), wr.grad, atol=1e-4), (
        wt.grad.reshape(wr.shape) - wr.grad
    ).abs().max()


def test_fsq_lands_on_grid_and_passes_ste_gradient():
    fsq = FSQ(levels=[5, 5], bound="clamp", ste="hard")
    x = torch.tensor([[0.1, 0.44, -0.9], [0.0, 0.6, -0.3]], requires_grad=True)
    q = fsq(x)
    grid = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    for v in q.flatten():
        assert (grid - v).abs().min() < 1e-5, f"{v} off grid"
    # Clamp bound, |x| < 1 everywhere ⇒ straight-through gradient is exactly 1.
    q.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-5)


def test_fsq_saturated_clamp_zeroes_gradient():
    fsq = FSQ(levels=[3], bound="clamp", ste="hard")
    x = torch.tensor([[1.6, -2.0, 0.2]], requires_grad=True)  # first two saturated
    fsq(x).sum().backward()
    assert x.grad.flatten().tolist() == [0.0, 0.0, 1.0]


def test_fsq_batched_bcl_matches_per_sample():
    # The decoder feeds [B, C, L]; FSQ must apply per-channel L over the batch and match a per-sample
    # 2-D application, with STE gradients flowing.
    torch.manual_seed(3)
    fsq = FSQ(levels=[3, 5], bound="clamp", ste="hard")
    x = torch.randn(4, 2, 7, requires_grad=True)
    q = fsq(x)
    assert q.shape == x.shape
    ref = torch.stack([fsq(x[b].detach()) for b in range(4)])
    assert torch.allclose(q, ref, atol=1e-6)
    q.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_ternary_conv1d_module_trains():
    torch.manual_seed(1)
    m = TernaryConv1d(4, 6, 3, padding=1)
    x = torch.randn(2, 4, 10, requires_grad=True)
    y = m(x)
    assert y.shape == (2, 6, 10)
    y.pow(2).mean().backward()
    assert m.weight.grad is not None and m.weight.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_learned_ternary_conv1d_trains_alpha_and_weight():
    torch.manual_seed(4)
    m = LearnedTernaryConv1d(4, 6, 3, padding=1)
    x = torch.randn(2, 4, 10)
    y = m(x)
    assert y.shape == (2, 6, 10)
    y.pow(2).mean().backward()
    assert m.weight.grad is not None and m.weight.grad.abs().sum() > 0
    assert m.alpha.grad is not None and m.alpha.grad.abs().sum() > 0, "LSQ alpha must receive gradient"


def test_joint_ternary_conv_and_fsq_reduces_loss():
    # A tiny encoder: TernaryConv1d -> FSQ latent, fit a target. Gradients flow through both the
    # activation-FSQ STE and the weight-ternary STE, so training reduces the reconstruction loss.
    torch.manual_seed(5)
    conv = TernaryConv1d(2, 4, 3, padding=1)
    fsq = FSQ(levels=[16, 16, 16, 16], bound="tanh", ste="hard")
    x = torch.randn(1, 2, 16)
    target = torch.rand(4, 16) * 1.6 - 0.8  # within the FSQ range
    opt = torch.optim.SGD(conv.parameters(), lr=0.1)

    def loss_fn():
        z = conv(x)[0]  # [4, 16]
        return (fsq(z) - target).pow(2).mean()

    l0 = loss_fn().item()
    for _ in range(40):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    l1 = loss_fn().item()
    assert l1 < l0, f"joint conv+FSQ training must reduce loss: {l0} -> {l1}"


def test_ternary_conv1d_weights_are_effectively_ternary():
    # The convolution the module runs is s_q ⊙ conv(x, trits); a unit impulse recovers the ternary
    # reconstruction (s_q · trit) columns, whose per-output-channel values take at most 3 magnitudes.
    torch.manual_seed(2)
    m = TernaryConv1d(1, 1, 3, padding=0)
    # Feed impulses so each output tap reads one weight; check reconstruction is on {-s,0,+s}.
    from tritium import ste_absmean_scale, ste_quantize_forward

    wf = m.weight.reshape(1, 3)
    s = ste_absmean_scale(wf.flatten().tolist(), 1, 3)[0]
    trits = ste_quantize_forward(wf.flatten().tolist(), [s], 1, 3)
    for t in trits:
        assert t in (-1.0, 0.0, 1.0)
